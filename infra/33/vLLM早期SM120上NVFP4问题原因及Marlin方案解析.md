# vLLM早期SM120上NVFP4问题原因及Marlin方案解析

## 🧑‍💻 User

很久以前的vllm nvfp4在sm120上是有问题的，要把cutlass gemmkernel换成marlin才能跑，是怎么回事

---
## 😀 默认助手 (7)

## TL;DR

这不是“RTX 5090 不支持 FP4”的问题，恰恰相反：这是 vLLM 在 Python 层把 12.0 当成“≥ 10.0 所以也支持”放行了，但真正编进 `.so` 的只有 **SM100(tcgen05/UMMA) 版** 的 CUTLASS NVFP4 kernel；SM120 **没有 tcgen05**，FP4 走的是另一套 warp 级 `mma.sync` 指令，于是要么 `no kernel image`，要么内部错误/静默算错。

Marlin 能跑，是因为它**根本不是 FP4 kernel**，而是 W4A16 的“4bit 反量化 + fp16 tensor core”GEMM，只要求 sm80，天生 arch-agnostic。

---

## 一、根因：SM100 和 SM120 上的“NVFP4”是两套不同的硬件路径

NVFP4 的数值格式两边一样（E2M1 元素 + 16 元素一块的 FP8‑E4M3 block scale + FP32 per‑tensor scale），但**执行单元完全不同**：

| | SM100（B200/GB200，CC 10.0） | SM120（RTX 50 / RTX Pro，CC 12.0） |
|---|---|---|
| FP4 矩阵指令 | `tcgen05.mma`（UMMA，第 5 代 TC） | `mma.sync m16n8k64 kind::mxf4nvf4`（warp 级，需 `sm_120a`） |
| A/B 操作数 | 直接从 SMEM/TMEM 读 | 必须 **SMEM → 寄存器** |
| Scale factor | 存在 **TMEM** 里，有专门 TMEM 布局 | 走 SMEM/寄存器，布局要求不同 |
| CUTLASS collective | `sm100_blockscaled_mma_warpspecialized` | `sm120_blockscaled_mma_tma`（`MainloopSm120TmaWarpSpecializedBlockScaled`） |
| 编译 arch | `arch=compute_100a` | `arch=compute_120a`（**`a` 后缀必须有**） |
| Cluster / 2SM MMA | 支持 | 不支持 |

CUTLASS 里这两个 mainloop 是**两个文件、两套代码**，不能互相复用。`sm120_blockscaled_mma_tma.hpp` 里甚至直接 `static_assert` 抱怨“SM120 mainloop 必须指定 smem→rmem 的 copy atom”，就是因为它没有 UMMA 那种直接从 SMEM 喂 MMA 的能力。

所以：**SM100 的 kernel 二进制在 SM120 上物理上跑不了**。

---

## 二、vLLM 当时具体错在三个地方（层层叠加）

### 1️⃣ 能力判断用的是数值大小，而 arch 是离散的

```cpp
bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability) {
  cudaRuntimeGetVersion(&runtimeVersion);
  return cuda_device_capability >= 100 && runtimeVersion >= 12080;
}
```

`120 >= 100` → Python 侧 `cutlass_fp4_supported()` 返回 `True`，`ModelOptNvFp4LinearMethod` 兴高采烈地选 `backend = "cutlass"`，然后在 5090 上撞墙。

### 2️⃣ CMake 只把 FP4 源码编进 `10.0a`

```cmake
cuda_archs_loose_intersection(FP4_ARCHS "10.0a" "${CUDA_ARCHS}")
...
"csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
```

entry 里当时是无条件的：

```cpp
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  return cutlass_scaled_fp4_mm_sm100a(D, A, B, A_sf, B_sf, alpha);
#endif
```

→ 在 5090 上就是那个经典报错（vLLM issue **#21274**，2025‑07，8×5090 机器）：

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

`no kernel image` 这个措辞非常“诚实”：**host stub 存在，但 fatbin 里没有 sm_120 的 cubin**。

### 3️⃣ 加了 SM120 编译器分支后，又写成了编译期 `#if/#elif`

PR **#21309**（2025‑08，`nvfp4_scaled_mm_sm120_kernels.cu`）补上了 SM120 kernel，但分派写成了：

```cpp
#if defined ENABLE_NVFP4_SM100 && ENABLE_NVFP4_SM100
  return cutlass_scaled_fp4_mm_sm100a(...);
#elif defined ENABLE_NVFP4_SM120 && ENABLE_NVFP4_SM120
  return cutlass_scaled_fp4_mm_sm120a(...);
#endif
```

这意味着：只要一个 `.so` 同时带 `10.0a` 和 `12.0a`（官方 wheel 的常态），**SM100 分支永远优先**，5090 还是被送进 SM100 的 kernel → `RuntimeError: Error Internal`（issue **#22783**，v0.10.1.dev，Qwen3‑32B‑NVFP4）。顺带还有个缺 `CUDAGuard` 的坑（多卡时拿错 device 的指针）。

修法是 PR **#29711**，改成**运行时分派 + 设备守卫**：

```cpp
const c10::cuda::OptionalCUDAGuard device_guard(device_of(A));
const int32_t sm = get_sm_version_num();
if (sm >= 100 && sm < 120) { cutlass_scaled_fp4_mm_sm100a(...); return; }
if (sm >= 120 && sm < 130) { cutlass_scaled_fp4_mm_sm120a(...); return; }
TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 mm kernel for SM ", sm, ...);
```

注意 `sm >= 100 && sm < 120` 这个上界——这就是当时“把 SM120 从 SM100 的管辖范围里摘出去”的那一刀。

---

## 三、为什么换成 Marlin 就好了

Marlin 的定位写在源码注释里，非常直白：

```python
# For GPUs that lack FP4 hardware support, we can leverage the
# Marlin kernel for fast weight-only FP4 quantization
def apply_fp4_marlin_linear(...):
    ... ops.gptq_marlin_gemm(..., scalar_types.float4_e2m1f ...)

def is_fp4_marlin_supported():
    return current_platform.has_device_capability(80)   # ← 只要求 Ampere+
```

关键点：

- **权重保持 packed（2 个 FP4/byte）**，在 kernel 内部用 `dequant` 指令反量化成 fp16/bf16 寄存器值，然后走普通 `mma.sync m16n8k16` + `cp.async` 多级流水。
- 只依赖 sm80 就有的东西（`cp.async`、`ldmatrix`、fp16 TC），**完全不需要 FP4 tensor core**，所以 SM80/89/90/100/120 一视同仁。
- 它还很“抠”地把 FP8‑E4M3 的 block scale **重排并改写成 S0E5M3 格式**（`nvfp4_marlin_process_scales` 里那个 `*.view(torch.int16) << 1` 的技巧，前提是 scale ≥ 0），把全局 scale 折进 exponent bias，省掉一次额外乘法。
- 限制：group_size 只支持 16（`FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16]`）。

**代价**（也就是你看到的那句 warning 的意思）：

> Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.

同样的 MAC 数要吃 4 倍的 fp16 tensor core 周期，等效峰值大约只有原生 FP4 的 $\tfrac{1}{4}$ 量级（再叠上反量化指令和寄存器压力）。**decode 阶段是搬权重瓶颈，所以几乎不吃亏；prefill / compute-bound 场景掉得很明显。**

---

## 四、当时的开关与时间线

<details>
<summary><b>展开：从“只能 Marlin”到“自动选原生 SM120 kernel”</b></summary>

| 时间 | 状态 |
|---|---|
| 2025 春夏（v0.8–v0.9） | 只有 `10.0a` 的 CUTLASS kernel；5090 上 `no kernel image`（#21274）。可用方案：改用 W4A16 的 nvfp4 权重 + Marlin（W4A16 路径本来就走 marlin），或 `VLLM_USE_NVFP4_CT_EMULATIONS=1` |
| 2025‑08，PR #21309 | 新增 `nvfp4_scaled_mm_sm120_kernels.cu`：`cutlass::arch::Sm120` + `OpClassBlockScaledTensorOp`，CMake 里 `12.0;12.0a` → `ENABLE_NVFP4_SM120` |
| v0.11–v0.14 | `VLLM_NVFP4_GEMM_BACKEND` 的合法值只有 `flashinfer-cudnn / flashinfer-trtllm / flashinfer-cutlass / cutlass`，**没有 marlin** |
| v0.15 前后 | 加入 `marlin` 选项，注释原文：`"marlin": use marlin GEMM backend (for GPUs without native FP4 support)`；后来又加 `"emulation"`（反量化成 bf16 跑 bf16 GEMM） |
| 2025‑12，PR #29711 | 修好编译期 `#if/#elif` + 缺 device guard → 真正按 `sm` 运行时分派 |
| v0.24+ | `VLLM_NVFP4_GEMM_BACKEND` 整个被删掉，改成自动选择（`VLLM_STRICT_QUANT_SCHEME` 之类的机制用来约束回退） |

也就是说：**“把 cutlass gemm 换成 marlin” 在某个版本区间里是官方支持的做法**（`VLLM_NVFP4_GEMM_BACKEND=marlin`），后来随着 SM120 原生路径稳定，这个开关就被回收了。

</details>

---

## 五、现在怎么判断你遇到的是不是同一类问题

| 现象 | 含义 |
|---|---|
| `CUDA error: no kernel image is available for execution on the device` | 编译时 arch 列表没带 `12.0`/`12.0a`（或 wheel 与 GPU 不匹配） |
| `No compiled nvfp4 mm kernel for SM 120` | 同上，新版本会给出这个明确报错（老版本会静默落到别的 backend） |
| `RuntimeError: Error Internal` | CUTLASS 内部 check 失败，典型是 arch/SF 布局不匹配 |
| 日志出现上面那句 Marlin warning | 已经回退到 W4A16 Marlin 了——注意这条 warning 在新版本被认定是**误导性文案**（GPU 其实有 FP4），相关 issue：**#47749**、#53328、#56603 |

自查两行：

```python
import torch; print(torch.cuda.get_device_capability())   # (12, 0)
print(torch.version.cuda)                                  # 需要 >= 12.8 才有 12.0a 的 FP4 MMA
```

还要注意 **`sm_120` ≠ `sm_120a`**：`mma.sync ... kind::mxf4nvf4` 属于 “arch-accelerated” 特性，目标必须是 `compute_120a`。这也是 vLLM CMake 里写 `"12.0;12.0a"` 的原因。

---

## 六、顺带一提：SM120 的 NVFP4 至今仍是“坑最多的一条路”

即使原生 CUTLASS kernel 现在已经能跑，历史上和现在围绕 SM120 的 FP4 issue 一直没断过，都是同一类“布局/对齐/分派”问题：

- per‑rank K 不是 32 对齐（TP8）时 torch.compile 下**静默垃圾输出**（SM120 block scale padding 问题）
- SM120 的 NVFP4 CUTLASS GEMM **epilogue 里没有 bias**
- `flashinfer_cutlass` 的 FP4 MoE 在 SM120/121 上**静默算错**，需要显式拒绝
- NVFP4 KV cache 在 SM120 上长期没有走 FlashInfer 的 kernel

一句话概括你的记忆：**“SM120 ≠ 小号 SM100”**。凡是把 `capability >= 100` 当判据、把 CUTLASS 的 SM100 block‑scaled mainloop 当通用实现的代码路径，在 5090 上都会翻车；Marlin 之所以是万金油，正因为它绕开了 FP4 硬件本身。

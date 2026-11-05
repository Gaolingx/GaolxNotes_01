# 从“跑通流程”到“构建 Flywheel”：infra 优化的结构性方法论

## 🧑‍💻 User

以vllm/sglang为例，一个模型部署起来了，想要优化，如何自己去看框架源码，搜索可能的优化点，挖掘一些图优化点空间，自己写kernel，补入框架验证

---
## 😀 默认助手 (7)

# 从“跑起来”到“优化进去”：vLLM / SGLang 源码级优化工作流

先说结论性的方法论，后面展开：

> **不要从头读源码。用 profiler 反查调用链；先分类瓶颈；先排除配置问题；再做 microbench；最后才改框架。**
> 90% 的“优化空间”在真正动手写 kernel 之前就能用 profile + 配置 + 图分析找到，而且大部分不需要写 CUDA。

---

## 总纲：一个可重复的优化循环

```
0. 定义 workload & 基线（含配置调优，排除伪瓶颈）
1. 定位瓶颈类别（CPU-bound / launch-bound / memory-bound / compute-bound / 同步-barrier）
2. 反查源码：hot kernel → python 调用栈 → module → 图
3. 提出假设 + 算理论收益（roofline / 访存下界 / launch 数）
4. 独立 microbench 验证假设（脱离框架，shape sweep）
5. 写 kernel / 改图 / 改调度
6. 以最低侵入度补进框架
7. 端到端 + 正确性 + 长跑回归
8. 若端到端无收益 → 回到 1，说明被别的瓶颈掩盖
```

**第 3 步的"算理论收益"是最容易被跳过、也最关键的一步**。不算清楚上界，你会花两周优化一个只占 3% 时间的热点。

---

## Step 0：先定义 workload，再排除"配置伪瓶颈"

优化的收益高度依赖 workload 形状。先量化：

| 维度 | 要记下来的东西 |
|---|---|
| 输入/输出长度分布 | `input_len` / `output_len` 的 p50 / p95，是否多轮、有 prefix 共享 |
| 并发 | `max_num_seqs`、实际 concurrency 曲线 |
| 指标 | TTFT（prefill+queue）、TPOT/ITL（decode）、tokens/s、goodput（满足 SLO 的吞吐） |
| 精度 | bf16 / fp8 / int4，量化方案 |
| 硬件拓扑 | GPU 型号、TP/PP/EP 大小、NVLink/IB |

然后**先把免费的可调项试完**，很多"优化需求"其实是配置问题：

- `--max-num-batched-tokens`、`--max-num-seqs`、`--gpu-memory-utilization`
- chunked prefill 的 chunk 大小、是否开 prefix caching（命中率多少？）
- attention backend 选择、page/block size（16/32/64 对 paged attention kernel 效率影响很大）
- CUDA graph 模式（full / piecewise / none）、`cuda-graph-max-bs`
- 量化、TP vs EP（MoE 模型上 EP 常有数量级差异）

```bash
# vLLM
vllm bench serve --model <m> --dataset-name sharegpt --num-prompts 1000 \
  --max-concurrency 64 --profile
# SGLang
python -m sglang.bench_serving --backend sglang --num-prompts 1000 --max-concurrency 64
python -m sglang.bench_one_batch --model <m> --batch-size 1 --input-len 1024 --output-len 512
```

**先把这些扫一遍，画出 latency-throughput 曲线**，后面所有 kernel 级优化都要在这条曲线上对比，而不是单点对比。

---

## Step 1：瓶颈定位 —— 四类问题，四种工具

| 现象 | 判据 | 工具 | 典型根因 |
|---|---|---|---|
| **CPU-bound** | GPU 利用率低、kernel 间有大段空隙、nsys timeline 上 CPU 行比 GPU 行满 | `nsys profile` + `torch.profiler(with_stack=True)` | Python 侧 attention metadata 构造、`_prepare_inputs`、scheduler、`.item()` 同步、list/Dict 分配 |
| **launch-bound** | 单 kernel < 5–10 μs 且数量极多，launch gap ≈ kernel 时长 | nsys timeline 放大 | 小算子串联（decode 时 M 很小），应靠 fusion / CUDA graph 解决 |
| **memory-bound** | DRAM throughput 接近峰值带宽，SM/tensor core 利用率低 | `ncu --set full` memory chart | decode 阶段几乎所有 GEMM、norm、activation、KV 读写 |
| **compute-bound** | SM occupancy 高、tensor core 忙、DRAM 有余量 | ncu Compute Workload Analysis | prefill 大 GEMM、长序列 attention、MoE |

**一个非常好用的量化判据**（decode 阶段）：

$$ I = \frac{2 \cdot N \cdot B}{W} \approx B \quad \text{FLOP/byte（bf16 权重 } W = 2N\text{）} $$

H100 的 $\text{FLOPs}/\text{BW} \approx 989\text{ TF} / 3.35\text{ TB/s} \approx 295$。所以 **decode batch < ~295 时必然 memory-bound**，此时算力优化（更快的 GEMM）几乎无用，只有"减少字节数"（量化、融合减少 round-trip、减少 KV 读取）和"减少 launch"有效。这一条能帮你砍掉一半错误的优化方向。

```bash
# 用 nsys 看 timeline：先看 GPU 空转在哪
nsys profile -t cuda,nvtx,osrt -o prof --force-overwrite true python -m vllm.entrypoints.openai.api_server ...
# 用 ncu 对单个热点 kernel 做深度分析
ncu --set full -k "regex:.*gemv.*|.*attention.*" -c 20 --launch-skip 100 python ...
```

---

## Step 2：源码地图 & "反查式"读法

### 读法（重要）

不要从 `main()` 往下读。**从上往下读是 O(repo)，从热点往上读是 O(path)。**

1. 用 `torch.profiler` 抓到 hot kernel 名（比如 `_ZN...fused_add_rms_norm...`）
2. `grep -rn "fused_add_rms_norm" python/` → 找到 Python wrapper
3. 看 wrapper 被谁调 → 上一层 module（`RMSNorm.forward`）→ 再上一层（`LlamaDecoderLayer.forward`）→ `model.forward`
4. 一次 forward 的骨架只需要看 **一个** decoder layer，模型同理

### vLLM（v1 架构）

```
vllm/entrypoints/                      # API 层，与性能基本无关
vllm/v1/engine/core.py                 # EngineCore：调度循环（CPU 开销大户）
vllm/v1/engine/core_client.py          # 进程间通信、ZMQ
vllm/v1/core/sched/scheduler.py        # ★ 调度决策：batch 组成、chunked prefill、抢占
vllm/v1/core/kv_cache_manager.py       # 前缀共享、block 分配
vllm/v1/worker/gpu_model_runner.py     # ★★ 性能核心：_prepare_inputs / execute_model / _dummy_run
vllm/v1/attention/backend.py           # ★ AttentionBackend / Metadata / MetadataBuilder 抽象
vllm/v1/attention/backends/            # flash_attn / flashinfer / triton_attn / mla / flex_attention
vllm/model_executor/models/llama.py    # ★ 一个模型 = 一个文件，先读这个
vllm/model_executor/layers/            # linear / layernorm / rotary_embedding / activation / fused_moe / quantization
vllm/compilation/                      # ★ torch.compile 接入：decorators.support_torch_compile / backends.VllmBackend
vllm/v1/cudagraph_dispatcher.py        # ★ CUDA graph 捕获与分派
vllm/_custom_ops.py                    # ★ torch.library 注册的所有 custom op
csrc/                                  # CUDA kernel（attention/ moe/ quantization/ cache_kernels.cu）
vllm/plugins/                          # 无需 fork 的注入点
```

**三个最值得精读的文件**：`vllm/v1/worker/gpu_model_runner.py`（一次 step 的全貌）、`vllm/v1/attention/backend.py`（换 attention backend 的接口契约）、`vllm/compilation/decorators.py`（什么样的一段 forward 能被编译/捕获）。

### SGLang

```
python/sglang/srt/managers/scheduler.py        # ★ 调度 + radix cache 交互
python/sglang/srt/managers/tp_worker.py
python/sglang/srt/model_executor/model_runner.py  # ★★ prepare_for_extend / prepare_for_decode / forward
python/sglang/srt/model_executor/cuda_graph_runner.py  # ★
python/sglang/srt/mem_cache/radix_cache.py     # 前缀树（SGLang 的招牌）
python/sglang/srt/mem_cache/memory_pool.py     # KV pool / token→slot 映射
python/sglang/srt/layers/                      # layernorm / rotary_embedding / activation / linear / moe/ / attention/
python/sglang/srt/layers/attention/            # flashinfer_backend / triton_backend / fa3 / mla
python/sglang/srt/speculative/                 # EAGLE / draft model
sgl-kernel/                                    # 独立 CUDA 仓库（python 侧改不用重编，csrc 改动要重编）
```

SGLang 的 `prepare_for_decode`（构造 `req_to_token`、`out_cache_loc`）和 `prepare_for_extend` 是 CPU 开销与 kernel 融合的双重热点，值得逐行读。

---

## Step 3：优化点清单（按层扫，别乱试）

<details>
<summary><b>3.1 算子级（最安全，收益最常见）</b></summary>

- **融合 chain**：`residual_add + RMSNorm (+ quantize)`、`SiLU/SwiGLU + mul (+ quantize)`、`RoPE + KV cache store`、`QK-norm + RoPE`
  - 判据：连续 2+ 个 elementwise kernel 且中间张量只被消费一次 → 融合能省一次 HBM 往返
- **减少 round-trip**：`hidden_states` 在 $B \times H$ 上的一次读写 = $2BH \cdot \text{sizeof(dtype)}$ bytes，乘层数乘步数往往可观
- **小 M GEMM**：decode 的 `M = B`（1~256）。检查是否用了针对小 M 的 kernel（split-K / GEMV / cutlass 特化），padding 浪费多少
- **量化 fusion**：per-token / per-group scale 的计算是否和主 kernel 融合，scale 的 reduce 是否多了一次 kernel
- **采样**：top-k/top-p 是否 fused；spec decode 的 rejection sampling 是否 CPU 同步（`.item()` 是重罪）
- **数值**：bf16 累加、fp8 scale 的 amax 计算方式，往往"优化"和"精度"在这里打架
</details>

<details>
<summary><b>3.2 Kernel 级</b></summary>

- **Paged attention**：`block_table` 的 gather 是额外访存；page size 16 vs 64 对该 kernel 的 L2 行为差异明显；GQA 下 KV head 复用是否被利用
- **Split-KV / flash-decoding**：decode 长上下文 + 小 batch 时，把 KV 沿 seq 切分并行是标准手段；检查当前划分的 `num_splits` 是否自适应
- **MLA**：是否做了 weight absorb（$W_{UK}$ 吸收进 $W_Q$）、absorb 后 dtype/精度损失
- **MoE**：grouped GEMM 的 tile 是否按 expert token 数自适应；shared expert 是否融合；topk 的 softmax+scatter 是否 fused
- **通信**：TP all-reduce 是否 overlap；EP all-to-all 是否有 SM 争用；NCCL chunk size
- **访存模式**：用 ncu 看 uncoalesced 和 sector 利用率，stride 是个常见坑
</details>

<details>
<summary><b>3.3 图级（下面单独展开）</b></summary>
</details>

<details>
<summary><b>3.4 调度级</b></summary>

- batch 组成策略：decode 与 prefill 混批比例、chunked prefill 的 chunk 大小对 TPOT 的冲击
- 前缀缓存命中率（SGLang radix cache / vLLM APC）：命中率提升 10% 常常比任何 kernel 优化都值钱
- 抢占与 recompute 策略、KV block 碎片率
- spec decode 的 draft 长度自适应（draft 太长在低接受率时是负收益）
- 多步调度（multi-step scheduling）摊薄 CPU 开销
</details>

---

## Step 4：图优化空间怎么挖（这是你说的“图优化点空间”）

**先让图"可见"**，这是没有捷径的一步：

```bash
# 1. 看有多少 graph break（每个 break = 一段回退 eager 的 python）
TORCH_LOGS="+graph_breaks" python -m vllm.entrypoints...

# 2. 看 inductor 生成的 triton kernel 源码 —— 数 kernel 个数、看每个 kernel 的访存
TORCH_LOGS="output_code" python ... 2>&1 | tee inductor_dump.py

# 3. 看有没有反复重编译（dynamic shape 处理不当的典型症状）
TORCH_LOGS="recompiles" python ...

# 4. 看 CUDA graph 覆盖情况
VLLM_LOGGING_LEVEL=DEBUG python ... 2>&1 | grep -i cudagraph
```

拿到 `inductor_dump.py` 之后，按这个清单审：

| 审查项 | 怎么判断有问题 | 优化动作 |
|---|---|---|
| **pointwise kernel 数量** | 一段 norm+act+quant 生成了 4~6 个 kernel | 写一个 fused custom op 顶掉整段；或确保 inductor 能 fuse（避免 `item()`、避免 data-dependent 分支） |
| **pad 到 capture size 的浪费** | decode 实际 batch=7 被 pad 到 8/16/32 | 加密 capture size 档位；或解码时按档位对齐 batched-token 预算 |
| **dynamic shape 特化缺失** | prefill 与 decode 共用一张图，或在同一 `mode` 下反复 recompile | 分开编译（vLLM 对 uniform decode 有独立路径）；显式 `dynamic=True` + `mark_dynamic` |
| **graph break 落点** | break 出现在 attention metadata 构造、自定义 op 边界 | 用 `torch.library.custom_op` + `register_fake` 把整段包成 opaque op，既有融合又能编译 |
| **通信不在图里** | all-reduce 在 cudagraph 外 | 让 NCCL 调用被捕获（需固定 shape + 无 host 同步） |
| **多余的类型/dtype 转换** | 图里出现 `to(float32)` → `to(bfloat16)` 往返 | 在 module 层统一 dtype |
| **layout 转换** | 反复 transposing / `contiguous()` | 改层内 layout 约定，或在 kernel 里直接支持 stride |
| **fallback 到 eager 的 op** | `aten::` 出现在 triton 输出中 | 换实现或加 decomposition |

**三个最高性价比的图级动作**：

1. **把 CPU 侧 metadata 构造"预算化"**：paged attention 需要的 `block_table`、`seq_lens`、`slot_mapping` 如果每步都在 Python 里拼 list 再 `torch.tensor()`，代价巨大。改成预分配 pinned buffer + kernel 内写（vLLM/SGLang 都在往这个方向走），这是 CPU-bound 场景的头号收益点。
2. **用 custom op 划定"编译边界"**：把「一串小 kernel + Python glue」声明为单个 custom op。好处：(a) 允许手工融合，(b) 缩小 inductor 图的面积、减少编译时间和重编译风险，(c) 避免 graph break。
3. **扩大 CUDA graph 覆盖**：任何 host 侧同步（`.item()`、`if tensor.any()`、动态 `num_splits`、动态 grid 依赖 tensor 值）都会把该段排除出图，收益随之消失。

---

## Step 5：自己写 kernel —— 选型、验收、骨架

### 5.1 技术选型

| 场景 | 首选 | 理由 |
|---|---|---|
| elementwise / reduction / norm / quant / sampling | **Triton** | 迭代快，vLLM 与 SGLang 的绝对主力；调试用 `TRITON_INTERPRET=1` |
| 需要极致 GEMM / attention tile 控制 | **CUTLASS / CuTe** 或直接改 sgl-kernel | 访存与 swizzle 精细控制 |
| MoE grouped GEMM、稀疏 | CUTLASS grouped GEMM / 现成库 | 形状动态，手写成本高 |
| 快速验证"融合有没有用" | 先手写 Triton 版 | 收益为正再工程化 |
| 只想少一次读写 | 先试 `torch.compile` + 手改 module | 零 CUDA 成本 |

**建议顺序永远是：改 module 让 inductor fuse → Triton → CUDA。** 很多"需要写 kernel"的场景在第一步就解决了。

### 5.2 验收标准（写之前就定好）

- **正确性**：对 fp64 参考实现 + 对原实现，测 `max_abs_diff` / `allclose`，覆盖边界（`n=0`、非 2 的幂、极端 scale、全零、inf/nan 传播）
- **性能**：报 **GB/s** 和 **TFLOPs**，并和理论下界比较。例如纯 elementwise 融合 kernel 的下界是 $\frac{\text{bytes moved}}{\text{BW}_{peak}}$，只报"快 1.3x"没有意义
- **shape sweep**：必须覆盖真实 workload 分布 —— decode 的 $M \in \{1,2,4,8,16,32,64,128,256\}$、prefill 的 $M \in \{512, 2048, 8192\}$、head_dim、GQA 比例、dtype、TP 切分后的形状
- **与基线同条件**：同 GPU、同 clock（`nvidia-smi -lgc` 锁频）、多次 warmup 取中位数、用 `triton.testing.do_bench` 或 `torch.utils.benchmark`

<details>
<summary><b>5.3 骨架示例：fused residual + RMSNorm + fp8 动态量化</b></summary>

```python
import torch, triton, triton.language as tl

@triton.jit
def _fused_add_rmsnorm_quant_fp8(
    X, R, W, Y, SCALE,                    # x, residual, weight, out, per-token scale
    stride_x, N, eps, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    s = x + r                                     # residual add
    tl.store(R + row * stride_x + offs, s, mask=mask)   # 原地写回 residual（后续层还要用）
    var = tl.sum(s * s, axis=0) / N
    rs = s * tl.rsqrt(var + eps)                  # rmsnorm（fp32 归约）
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    o = (rs * w).to(tl.float32)
    amax = tl.max(tl.abs(o), axis=0)              # per-token 动态 scale，无额外 kernel
    scale = amax / 448.0
    inv = 1.0 / tl.maximum(scale, 1e-12)
    tl.store(Y + row * stride_x + offs, (o * inv).to(tl.float8e4nv), mask=mask)
    tl.store(SCALE + row, scale)
```

要点：所有中间量留在寄存器/smem；scale 的 amax 归约和主 kernel 融合（省掉一次全量读写）；`BLOCK` 用 `triton.next_power_of_2(N)`；行 stride 支持非连续张量以便无缝替换。

**注意**：这个 pattern 在 vLLM（`fused_add_rms_norm` + `scaled_fp8_quant`）和 SGLang 里都已有实现。**写它的价值是把验证流程跑通**，然后你才会知道"融合一个 elementwise chain 到底能拿几个百分点"——通常是 0.5%~3% 端到端，而一次 launch 缩减在图里的收益可能更大。用这个校准你的预期。
</details>

---

## Step 6：补进框架 —— 五个侵入等级，从低到高

**永远从最低等级开始。** 只有 microbench 证明有收益、且低等级做不到时，才升级。

<details>
<summary><b>Level 0：纯配置（零代码）</b></summary>

backend 选择、compilation level、cudagraph 模式、chunk size、page size、quant。先确认这些组合的最优解。
</details>

<details>
<summary><b>Level 1：Monkey patch / plugin（不 fork，验证最快）</b></summary>

vLLM 有 plugin entry point（`vllm.general_plugins`），SGLang 可以直接改 python（python 层改动**不需要重新编译**，这是快速迭代的关键）。

```python
# my_ext/__init__.py —— 通过 entry_points 注册，不改 vllm 源码
def register():
    from vllm.model_executor.layers.layernorm import RMSNorm
    from my_ext.ops import fused_add_rmsnorm_quant_fp8
    _orig = RMSNorm.forward
    def forward(self, x, residual=None):
        if residual is not None and self._my_flag:
            out = torch.empty_like(x)
            fused_add_rmsnorm_quant_fp8(x, residual, self.weight, out, ...)
            return out, residual
        return _orig(self, x, residual)
    RMSNorm.forward = forward
```

适合：**2 小时内验证"这个融合端到端到底有没有用"**。不适合提交。
</details>

<details>
<summary><b>Level 2：改 module（用 custom op 包裹，保持可编译）</b></summary>

直接改 `vllm/model_executor/layers/xxx.py`，但必须：

- 用 `torch.library.custom_op`（vLLM 里是 `direct_register_custom_op`）+ **`register_fake`**，否则 inductor 会 graph break 或在 fake tensor 上崩
- 保持 module 被 `@support_torch_compile` 装饰时的可追踪性
- 不要在图里做 host 同步、不要动态分配、不要 python 控制流依赖 tensor 值

```python
from torch import Tensor
import torch

@torch.library.custom_op("myext::fused_add_rmsnorm_quant_fp8", mutates_args={"out", "residual"})
def fused_add_rmsnorm_quant_fp8(x: Tensor, residual: Tensor, w: Tensor, out: Tensor, eps: float) -> None:
    _launch_triton(x, residual, w, out, eps)

@fused_add_rmsnorm_quant_fp8.register_fake
def _(x, residual, w, out, eps):
    return None                      # fake impl：不分配、不计算
```
</details>

<details>
<summary><b>Level 3：新 attention backend（收益最大，工作量也最大）</b></summary>

按 vLLM 的契约实现三件套（SGLang 类似，看 `layers/attention/*_backend.py`）：

```python
class MyAttnBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str: return "MY_ATTN"
    @staticmethod
    def get_impl_cls(): return MyAttnImpl
    @staticmethod
    def get_metadata_cls(): return MyMetadata
    @staticmethod
    def get_builder_cls(): return MyMetadataBuilder
    @staticmethod
    def get_kv_cache_shape(...) -> tuple: ...
    @staticmethod
    def get_supported_head_sizes() -> list[int]: ...
    @classmethod
    def supports_head_size(cls, hd): ...
```

然后注册到 backend 枚举，用 `VLLM_ATTENTION_BACKEND=MY_ATTN` 切换。**务必将 metadata 构造（`seq_lens`、`block_table`、`slot_mapping`）改成 kernel/pinned-buffer 写入**，这一项常常比 attention kernel 本身带来更多收益。
</details>

<details>
<summary><b>Level 4：C++/CUDA + csrc 注册</b></summary>

在 `csrc/` 加 `.cu` + `csrc/ops.h` 声明 + `torch_bindings.cpp` 注册，`pip install -e .` 重编（首次 20–40 min，增量可接受）。SGLang 侧对应 `sgl-kernel/`。仅当 Triton 表达不了（需要 async copy pipeline、cluster、warp specialization、TF32/FP8 细微控制）时才走这里。
</details>

### 兼容性 checklist（不满足就不要往下推）

- [ ] 无 host 侧同步（`.item()`、`.cpu()`、`if tensor.all()`、`print(tensor)`）
- [ ] 无 data-dependent 的动态 grid / `num_splits`（或者提供一个保守静态上界）
- [ ] 不每次调用新建 workspace（用预分配 pool），否则 CUDA graph replay 会崩或退化
- [ ] fake tensor 下能跑（`register_fake` 正确）
- [ ] 支持非连续输入 / 任意 stride，或显式 `.contiguous()` 但计入成本
- [ ] padding 区域不产生 nan/inf 污染（capture size 对齐时必然有）
- [ ] TP 切分后的形状（可能不是 2 的幂）已覆盖测试

---

## Step 7：验证与回归

### 正确性（三道关）

1. **Logits 对比**：同 prompt、teacher forcing，`max_abs_diff` 与 top-1 token 一致率（允许极小差异但要量化）
2. **任务指标**：`lm-eval`（gsm8k / mmlu）与原实现对比，多数情况下差异应在噪声内；fp8 融合类改动必须看
3. **边界扫描**：超长 prompt、`B=1`、`B=max`、极端 concurrency、多轮对话（前缀缓存路径）、以及 **CUDA graph capture/replay 两路都跑一遍**

### 性能（四个视角，缺一不可）

| 视角 | 目的 |
|---|---|
| **kernel microbench** | 证明 kernel 本身变快了（GB/s 接近下界） |
| **单层/单步 profile** | 证明该 kernel 在真实 batch 下确实变快（小 shape 上 Triton 可能反而更慢！） |
| **端到端曲线** | throughput-latency 曲线整体左移/上移，而不是单点偶然 |
| **消融（开/关）** | 用 `--enforce-eager` / 关闭 fusion 的 flag 隔离收益来源，避免"看起来快了但其实是因为顺便改了别的东西" |

**必看：如果端到端收益 < microbench 收益的一半，通常意味着被别的瓶颈掩盖了**（比如你优化了一个 GEMM，但 decode 是 CPU-bound）。此时回到 Step 1。

### 稳定性

- 长跑 2–4 小时持续请求，看 RSS / HBM 是否单调增长（workspace 泄漏、graph 泄漏）
- 首 token 延迟（TTFT）是否退化 —— 融合大 kernel 常让 prefill 变差
- 不同 GPU（H100 / A100 / L40S）、不同 TP 大小、不同 `max_num_seqs`
- 并发抖动：`weight_version`、recompile 次数（`TORCH_LOGS=recompiles` 在压测中应为常数）

---

## 避坑清单

1. **用单点 benchmark 做决策**。decode B=1 的收益在 B=64 时常常归零甚至为负。
2. **在小 shape 上盲目用 Triton**。$M=1$ 的 GEMV 上，高度调优的 CUDA/CUTLASS kernel 往往赢 Triton，因为 launch 与 occupancy 弹性差。
3. **忘了 CPU 开销**。vLLM v1 / SGLang 的 scheduler + `_prepare_inputs` + metadata 构造在很多真实负载里占 30%+ 的 step 时间，且它免费、不需要写 kernel。
4. **破坏 CUDA graph**。任何一处 host 同步，会让你前面所有融合收益作废（因为回退到逐 kernel launch）。
5. **精度悄悄退化**。fp8 融合 `amax` 的归约顺序、bf16 累加、softmax 的数值稳定性，都要显式测。
6. **只改 python 却重编了 csrc**。python 层改动 `pip install -e .` 后直接生效，别浪费时间等编译。
7. **没对齐上游**。vLLM/SGLang 迭代很快，你优化的位置可能下周就被上游 fuse 掉了 —— **动手前先看 `main` 分支和近期 PR**，很多时候答案已经在那里，你的工作是复现 + 调参而不是重写。

---

## 建议的第一次实操路径（一周）

| 天 | 做什么 | 产出 |
|---|---|---|
| 1 | 定 workload + 跑配置 sweep + 建 e2e 基准曲线 | 一条可复现的 latency-throughput 曲线 |
| 2 | nsys + torch.profiler 抓 timeline；判定瓶颈类别 | 一张"时间去向"表（scheduler / metadata / GEMM / attn / norm / sample 各占多少） |
| 3 | `TORCH_LOGS=output_code` dump 图，数 pointwise kernel、找 graph break | 一份"图问题清单"（几处 break、几个可融合 chain） |
| 4 | 挑**最大的那一项**，写 Triton microbench + shape sweep | kernel 级收益数字 + 与 roofline 下界的差距 |
| 5 | Level 2 custom op 接入，跑正确性三关 | 端到端收益（大概率 1%~5%） |
| 6–7 | 换第二个点（通常是 CPU 侧 metadata），压测 + 长跑 | 组合收益 + 回归测试 |

**一句话总结**：先把"时间去哪了"变成一张表，把"能不能快"变成 roofline 上的一个数，再去写 kernel。这套流程走顺了，你会发现在 vLLM/SGLang 上真正能挖到的图优化点，绝大多数集中在 **CPU 侧 metadata 构造、CUDA graph 覆盖度、以及 elementwise chain 的 round-trip**——而不是 attention 或 GEMM 本身。

---
## 🧑‍💻 User

业务场景的case很多，我举个简单的例子 为了让线上的推理服务更细粒度调度 我会单独起一个持久化sm 作为scheduler 往不同请求launch的算子里面插入sleep或者barrier，这样就可以在指令级调度pd要怎么去做
当然还有其他的上下文 单卡多卡多机 写法也都不一样
这个基础上还可以看看怎么做megakernel（我们在做megakernel的线上serving） 还有不同的序列长度 可以搜索的情况就更多了 如果按照ai那种暴力搜索 能搜一个月
这种情况你到底要怎么去goal怎么去做research啊

---
## 😀 默认助手 (7)

## 先说一句不太顺耳的结论

**你现在遇到的并不是“搜索空间太大”的问题，更根本的问题是把四类研究对象（配置、结构设计、控制策略、实现细节）混进同一个 search loop 里。**

暴力搜索的问题不只在于慢：它在**原理上**就是错的，理由有三条：

1. **噪声下限**。e2e 服务的 run-to-run 相对标准差是 2%~5%。要检测 1% 的效应，每组需要重复几十次；10⁴ 个点就是几十万次压测，相当于搜一个月，最后只能搜到“噪声的最大值”。
2. **非平稳性**。你搜到最优点的当天，线上流量构成已经变了，所以搜出来的 optimum 是过期的。
3. **它不产生知识**。搜完你只知道“这组参数好”，不知道“为什么好”，于是在下一个 case（多卡、不同序列长度、megakernel）上要从零重来。

**“一搜一个月”的解法，是让大部分维度根本不需要搜，而不是搜得快。**

---

## 一、把“搜索问题”改成“证伪一个机制”

`goal` 的粒度决定了你要不要搜一个月。对比一下：

| ❌ 不能叫 goal | ✅ 能叫 goal |
|---|---|
| “优化 PD 混部调度” | “在 `out_len≥1k, B≤8, TP=1` 这一 regime 下，让 decode 期空闲的 DRAM 带宽被 prefill 填补，使 goodput 提升 ≥15%（上界 22%），且 TPOT p99 不退化 >5%” |
| “探索 megakernel” | “在已开 CUDA graph 的前提下，megakernel 相对 `cudagraph+compile` 在 `B=1..8` 上把 TPOT 降 ≥8%（launch 上界只有 3%，所以必须来自跨层权重预取重叠）” |
| “支持单卡多卡多机” | “同一 cost model 在 3 种拓扑下预测出的最优粒度分别为 warp / CTA / chunk，实测误差 <20%” |

**每一个 goal 必须包含四件事：**

```
Claim:     在 regime R 下，机制 M 把指标 Y 改善 ≥ X%
Precondition: 前提 P（能用一个下午的实验证伪的那个）
Upper bound: 由 roofline / launch 计数算出的 ≤ U%
Kill criterion: 若 P 不成立，或 U < 阈值，或 cost 曲线形状不对 → 立刻停
```

**研究品味的一大半就在 `Precondition` 上：找到那个最便宜、最能杀死你想法的实验。** 例如你说“指令级 PD 混部”这个想法，它有一个致命前提：

> decode 阶段真的有空闲资源可以被 prefill 填吗？

用 ncu 测 decode 稳态的 DRAM BW util、SM issue slot util，一个下午就能做完。如果 DRAM 已经 **>85%**，那么插入 prefill 只会抢带宽，P 和 D 一起变差，整个“指令级调度”项目就应该当场死掉，省下你三个月。如果 DRAM 是 40% 而 SM issue 只有 20%，那这个前提成立，并且告诉你“应该往哪个方向切”。

**这就是做 research 的方式：先花一天杀死它，杀不死才值得投入。**

---

## 二、先用上界砍掉 95%（这一步几乎不花时间）

对每个候选优化，先算这样一个数：

$$
U_i = \frac{t_i}{T_{\text{total}}} \cdot \left(1 - \frac{t_i^{\min}}{t_i}\right)
$$

其中 $t_i$ 是该组件当前耗时，$t_i^{\min}$ 是它的物理下界（$\text{bytes}/\text{BW}$ 或 $\text{FLOPs}/\text{peak}$）。**如果 $U_i < 2\%$，就不做实验，也不用看源码，砍掉。**

现实里这一条会砍掉 95% 的候选。剩下 5% 才是你要花时间的。

<details>
<summary><b>用 megakernel 现场演示这个计算（这类计算能救你一个季度）</b></summary>

**上界 1：launch 开销**

$$
T_{\text{launch}} \approx N_k \cdot \ell,\qquad N_k = L \cdot k
$$

稠密模型 $L=80$、每层约 $k\approx10$ 个 kernel → $N_k \approx 800$。

- eager：$\ell \approx 2\text{–}3\,\mu s$ → $T_{\text{launch}} \approx 1.6\text{–}2.4\text{ ms}$
- **CUDA graph 开启后**：$\ell \approx 0.3\text{–}0.5\,\mu s$ → $T_{\text{launch}} \approx 0.24\text{–}0.4\text{ ms}$

8B 模型 bf16、B=1 的 decode step 下界：$16\text{ GB} / 3.35\text{ TB/s} \approx 4.8\text{ ms}$，实测通常 6–8 ms。

所以：

| 场景 | launch 占比 | megakernel 的理论天花板 |
|---|---|---|
| eager | 20%~25% | 值得做 |
| **已开 CUDA graph** | **3%~5%** | **仅靠省 launch 不值得做** |

**结论：如果你们线上已经开了 CUDA graph，megakernel 的 launch 收益只剩 3%–4%。你必须从别的机制去 justify 这个项目**，也就是下面这一条：

**上界 2：跨层重叠**

megakernel 剩下唯一可能的大收益，是**把 layer $i+1$ 的权重预取隐藏在 layer $i$ 的计算/等待之后**（软件流水），以及在 MoE 里把 dispatch/combine 的 HBM 往返变成片上传递。这个收益的上界由“权重加载的串行尾部”决定，不是由带宽决定。**先测一下当前每层结尾 SM 的空闲时间占比（ncu 的 kernel tail / drain 分析），那个数就是你的天花板。**

如果 tail 占比 2%，那 megakernel 在你这个 regime 下的总上界是 5%~7%，加上巨大的工程复杂度，**结论应该是不做，或者只在 MoE + 小 batch 的局部 regime 做**。

**顺便一个战略洞察**：megakernel 需要的 persistent kernel + work queue + 软件屏障，和你那个“持久化 SM 做 scheduler”是**同一套基础设施**。这两个项目应该合并成一个“persistent 执行框架”，而不是两个项目。这就是“把 case 变成 regime”带来的复用。
</details>

---

## 三、噪声地板：为什么"搜一个月"必然失败

要检测相对效应 $\delta$，在相对噪声 $\sigma_r$ 下的最小重复次数大约是：

$$
n \gtrsim 2\left(\frac{2\sigma_r}{\delta}\right)^2
$$

取 $\sigma_r = 2\%$：

| 想检测的效应 | 需要的重复次数 | 10⁴ 个点需要的压测次数 |
|---|---|---|
| 10% | 1 | 10⁴ |
| 5% | ~2 | 2×10⁴ |
| 2% | ~16 | 1.6×10⁵ |
| 1% | ~32 | 3.2×10⁵ |

**推论：1%~2% 级别的搜索，用 e2e 压测在物理上就不可行。** 所以正确的做法是**分层测量**：

- **探索阶段全部放在低噪声的代理指标上**：kernel 级 GB/s、SM issue rate、单 kernel 耗时、nsys 上的 gap 占比。这些的 $\sigma_r$ 可以做到 0.3%~1%，同样的预算能分辨 5 倍更小的效应。
- **e2e 只用于最终确认**，每个决策点 ≤3 次重复。
- 永久锁频（`nvidia-smi -lgc`）、独占卡、固定 clock 和 power limit。**测量环境的噪声水平本身就是一项基础设施资产**，比任何 kernel 都值钱。

---

## 四、把空间按“轴的类型”切开，四类四种方法

这是解决你这个问题的核心。**不要把它们放进同一个 loop。**

| 轴类型 | 例子 | 空间规模 | 正确方法 | 时间 |
|---|---|---|---|---|
| **① 静态配置** | batch size、chunk size、page size、backend、dtype、capture size 档位 | 10²–10³ | OAT 筛选 → top-3 全因子 → 确认 | 几小时 |
| **② 结构设计** | megakernel 的分区与 sync 位置、SM 预算、融合边界、PD 空间切分 | 10⁴⁺，但**由依赖图约束** | **不搜**。从依赖图推导可行分区，用成本模型排序 | 几天（建模） |
| **③ 控制策略** | 抢占策略、时间片长度、SLO 反馈控制器 | 连续/策略空间 | **不搜**。建模 + 控制论 + 稳定性证明 | 周级 |
| **④ 实现细节** | tile size、warps、vector width、num_stages | 10²/kernel | 交给 autotune（Triton/CUTLASS 自带） | 免费 |

**你的 persistent-SM scheduler 和 megakernel 属于 ② 和 ③，它们不是搜索问题。** 把它们当搜索做，就是你感觉“要搜一个月”的根源。它们需要的是**成本模型**。

<details>
<summary><b>② 结构设计为什么不靠搜：以 megakernel 分区为例</b></summary>

Megakernel 的设计变量是“哪些东西必须同时驻留、同步点放哪里”。这不是 10⁴ 个组合，而是**依赖图上的一个割（cut）**：

- 相邻 layer 的 activation 依赖 → 若不割，可留在 smem/register，省 HBM 往返
- attention 有 data-dependent 控制流 → 必须走 work queue，不能静态展开
- MoE 路由 → 同上
- 权重预取深度 → 受 smem 容量约束

**可行解的数量其实很小（个位数到几十），由 smem 容量和依赖图决定。** 你要做的是写出这些约束（一个小的 ILP 或直接手推），而不是枚举。搜出来的“最优分区”往往不可实现或不可证。

正确产出物：**一张依赖图 + 一组割 + 每个割的上界**，而不是一个参数字典。
</details>

---

## 五、序列长度：不要枚举，找 breakpoint

序列长度维度上，性能曲线是**分段常数/分段光滑**的，断点由硬件常数决定。找出断点，段内只需 2–3 个点验证单调性。

典型 breakpoint 来源（每项都能解析算出）：

| 来源 | 断点位置 |
|---|---|
| arithmetic intensity 交叉 | decode $I\approx B$，H100 上 $295$ → $B^*\approx295$ 之后转 compute-bound |
| tile 量化 | 若 tile M=128，则 $M \bmod 128 = 0$ 是阶梯；$M$ 跨过 128/256 会跳变 |
| wave 量化 | $N_{\text{CTA}} \bmod (SM \times \text{per-SM-CTA}) = 0$ |
| L2 容量 | KV working set 超过 L2（H100 50MB）后 attention 性能掉崖 |
| 调度器预算 | `max_num_batched_tokens`、chunk size、`max_num_seqs` 的整数边界 |
| paged 结构 | block size 与 `seq_len` 的对齐关系；block table 长度跨页 |
| 通信 | 消息大小跨过 NCCL 的算法切换阈值（小消息用 tree，大消息用 ring） |

**离散化后的测试点通常是每个轴的 15–30 个，不是几千个。** 5 个轴 = 10²~10³，且大部分可以 OAT。

---

## 六、“单卡多卡多机写法都不一样” —— 这是最该先消灭的一类 case

这句话本身暴露了真正的问题：**你有 N 份实现，而不是一份被参数化的实现。** 这是研究时间最大的黑洞，因为任何优化都要乘 N。

正确的处理是：**让拓扑成为 cost model 里的参数，让实现成为它的函数。**

<details>
<summary><b>用你自己的场景推一遍：为什么粒度会随拓扑变化（这就是统一的方法）</b></summary>

你有一个“插入 sleep/barrier 做细粒度调度”的机制。切换粒度为 $g$（每多少工作量切一次），单次切换成本 $c(g)$，单位工作量的总开销：

$$
\text{overhead}(g) = \underbrace{\frac{c(g)}{g}}_{\text{切换频率}\times\text{单次成本}} \qquad
\text{benefit}(g) = \text{被隐藏的 stall}
$$

最优粒度：

$$
g^* = \arg\min_g \left[\, \frac{c(g)}{g} + \frac{\text{residual stall}(g)}{\text{work}}\, \right]
$$

现在关键：**$c(g)$ 的构成随拓扑变化，$g^*$ 就自动变化。**

| 拓扑 | $c(g)$ 主导项 | 量级 | 推出的 $g^*$ |
|---|---|---|---|
| 单卡，SM 内 | 软件 barrier（spin/arrive-wait），跨 SM 走 L2 | ~100–500 ns | **warp / 单 tile 级** |
| 单机多卡 | NVLink barrier + SM 间可见性 | ~1–3 μs | **CTA / chunk 级** |
| 多机 | 网络 RTT + 序列化 | ~5–50 μs | **请求 / micro-batch 级** |

**这意味着“写法不一样”不是工程债，而是同一个模型在不同参数区间的必然产物。** 你要做的不是统一代码，而是：

1. 把 $c(g)$ **实测出来**（一次 microbench，测 $c$ vs $g$ vs 拓扑，得到一条曲线；这条曲线复用到所有 case）
2. 用这条曲线**解析地**算出每个拓扑的 $g^*$
3. 只实现 $g^*$ 附近的 2–3 个取值

这样多卡/多机从“三个要各自优化的 case”变成“一条曲线的三个采样点”。**搜索维度从 3 变成 1（一条曲线）。**
</details>

---

## 七、你的 persistent-SM scheduler 应该怎么做 research

这是你项目里最有研究价值的部分，但它的形态**不是参数搜索，而是机制 + 成本模型 + 控制器**。

<details>
<summary><b>四步，每步的产出物和 kill criterion</b></summary>

**Step 1 · 验证前提（半天，最重要）**

测 decode 稳态的：DRAM BW util、SM issue slot util、tensor core util、SM active warps。
- 若 DRAM >85% → **kill**（没有免费带宽，插入 prefill 只有互相伤害）
- 若 SM issue <30% 而 DRAM 40%~60% → 前提成立，且告诉你应该切“带宽友好的 prefill 片段”（低 arithmetic intensity 的层）而不是整个 prefill

**Step 2 · 测切换成本曲线（2–3 天）**

Microbench：固定 $g$，让 kernel 在 $g$ 处 hit barrier，测 $c(g)$，扫 $g \in \{$warp-iter, CTA, tile, layer$\}$ × 拓扑。
产出 **$c(g)$ 曲线**——这是你所有后续决策的唯一依据，也是你论文里最值一张图。

**Step 3 · 解析求 $g^*$（1 天，不用搜）**

把 Step 2 的曲线代入上面的 $\text{overhead}(g)$，配一个 stall 模型（由 Step 1 的 util 反推可隐藏时间），一维求极小。
**空间从 10^k 塌缩成 1 维。**

**Step 4 · 把静态策略变成控制器（1–2 周）**

线上流量非平稳，$g^*$ 会漂。所以真正要部署的不是常数，而是反馈律：

$$\text{action}_{t} = f\big(\text{TPOT } p99_t - \text{SLO},\ \text{DRAM util}_t\big)$$

这里的研究点是**稳定性**：比例控制会不会振荡？切换抖动（thrashing）怎么抑制？死锁/活锁的边界在哪（persistent kernel 上这是真风险）？
产出物：一个带稳定性论证的策略 + 一条“SLO 违约率 vs 负载”曲线。

**这样整个项目是 4 个 artifact、一个成本模型，而不是一次 10⁶ 点搜索。**
</details>

---

## 八、那到底怎么排期：把实验当预算来分配

假设你有 4 周 GPU 时间。别平均分。

**1. 目标函数先写出来（这一步省掉 80% 的 case）**

$$
J = \sum_{r \in \text{regimes}} w_r \cdot \frac{\text{goodput}_r}{\text{cost}_r},\qquad \sum w_r = 1,\ w_r = \text{线上该 regime 的流量占比}
$$

线上的 case 分布极度长尾：**通常 3–5 个 regime 覆盖 85%+ 的流量和 90%+ 的成本**。用 $w_r$ 加权后，$w_r < 2\%$ 的 regime **不做优化，用默认配置兜住**。case 数量从 N 变成有效 3–5 个。这是回答你“case 很多”最直接的答案。

**2. 实验设计：OAT 筛选 → top-k 全因子 → 确认**

- **OAT 筛选**：从基线出发，每个轴单独动一次。$k$ 个轴只要 $O(k)$ 次测量，得到每个轴的灵敏度 $\partial Y/\partial x_i$
- **只对灵敏度 top-3 做全因子**（$3^3 = 27$ 次）。硬件性能空间的交互项很少且已知（batch×KV 占用、tile×warp、$g$×拓扑）
- **确认**：3 次重复

总测量数从 $O(10^4\sim10^6)$ 降到 $O(10^2)$。**这是几个小时到一两天的事。**

**3. 时间账（4 周）**

| 分配 | 用途 |
|---|---|
| 20% | 测量基础设施：锁频、低噪声 harness、kernel microbench 模板、regime 定义与流量加权 |
| 50% | **一个**上界最大的机制（大概率是 persistent 执行框架，把 scheduler 和 megakernel 合并做） |
| 20% | 探索性/反直觉的东西（必须留，否则你永远只做增量） |
| 10% | 回归与线上验证 |

**70/20/10 的铁律：不要开第二个主线项目，直到第一个给出了 cost model。**

---

## 九、什么时候该换问题

这是 research 里最难、也最值钱的部分。触发条件：

1. **上界 < 3%** —— 换问题。你已经用一天算出来了，别再花一个月。
2. **成本模型的斜率不对** —— 例如测出 $c(g)$ 在网络拓扑下是 μs 级，说明“指令级”在多机上根本不存在。**这不是失败，这是你的研究结论**：它告诉你细粒度调度只在单机 NVLink 域内成立。这个负结论比一个 +2% 的 kernel 值钱得多。
3. **两个机制在抢同一个资源** —— 例如 megakernel 的 SM 分区和 scheduler 的 SM 预算在抢同一批 SM。这说明它们必须统一建模（这也往往是最大机会）。
4. **你在优化第 4 个 regime，而 $w_4=3\%$** —— 停。用默认配置。

---

## 总结成一句话

> **不要 goal 成“把这个框架优化 X%”，要 goal 成“证明/证伪一个机制 M 在 regime R 下能拿到 ≥X% 的收益，上界 U，成本模型 cost(g)，kill criterion K”。**
>
> 空间不是用来搜的，是用来**塌缩**的：上界砍 95%，轴分层砍 4/5，breakpoint 离散化砍 10²，拓扑参数化把 N 个 case 变成 1 条曲线，流量加权把长尾 case 砍掉，成本模型把结构/控制空间从 10⁴ 变成 1 维。
>
> **最后剩下真正要做的实验，是一周一个可证伪的 claim。**
>
> 你感觉要搜一个月，是因为你在用“搜”这个动作去替代“建模”这个思考。硬件性能的可预测性远超你的直觉：**任何一个 gap，都能算出一个上界；任何一条曲线，只要量出斜率，微积分就替你搜完了。**

---
## 🧑‍💻 User

未来的模型也有新的infra问题 我认为新的infra问题增长的速度会比模型能力快
那么infra的问题实际上不止来自于架构变动 还有序列长度 专家数量 等等 还有agent那边的各类请求 思考强度 这会带来更多的问题 而且他们是乘算关系
problem的数量级是引入的参数的笛卡尔积 所以一定会很多很多 你不可能把这么多case都试一遍
有没有什么迭代效率更高的infra flywheel design？

---
## 😀 默认助手 (7)

## 先纠正一个前提：显式轴的笛卡尔积不等于真实的乘法关系

你说“问题数量是引入参数的笛卡尔积”。**在显式参数化下这是对的，但它只是参数化方式造成的假象，并不代表问题本身的真实关系。**

先看一张表，并检查你举的那几个轴：

| 你列举的"不同问题" | 本质上的同一个问题 |
|---|---|
| MoE 路由偏斜 | 每 token 的计算量不同 |
| Speculative decoding 接受率 | 每 token 需要验证的长度不同 |
| Thinking budget / 自适应推理强度 | 每请求生成长度自适应 |
| Early exit / cascade | 每 token 提前退出 |
| Chunked prefill | 每次调度的计算粒度可变 |
| Continuous batching | 同上 |

**这六个轴本质上都是同一个原语：`per-token 可变计算量 + 工作队列 + 分配与公平策略`。** 一套机制（可变工作量队列、cost model 驱动的 allocator 和预算控制器）就能覆盖全部。

同理你另外几个轴：

| 轴 | 实际是什么 |
|---|---|
| 序列长度 | KV 工作集大小、局部性 |
| 专家数量 | 有效参数量、访存字节数、token 到专家的映射偏斜 |
| Agent 请求类型 | 到达过程的突发性、共享前缀比例、长度分布的形状 |
| 多卡多机 | 通信量与计算量的比值、拓扑半径 |

**结论：你感觉到的「8 个轴 → 10⁸ 个 case」，其中大部分是约 7 个潜变量（latent dimension）的投影。** 真实的坐标维度不高，是参数化重复了。

这是整件事的关键：**infra flywheel 的核心任务，是找一个坐标变换，让显式笛卡尔积塌缩；单纯“搭更好的搜索框架”解决不了问题。**

物理学的做法就是把一群流场问题压缩成一条以雷诺数 $\mathrm{Re}$ 为横轴的曲线。你的“序列长度 × 专家数 × batch × dtype × TP”也应该压缩成一组无量纲数上的少量曲线。

---

## Flywheel 的设计目标，可以写成一个导数

不要用“迭代更快”这种模糊说法。真正要优化的量是：

$$
\boxed{\ \frac{\partial \ln N_{\text{exp}}}{\partial a} \approx 0 \quad \text{（每引入一个新轴 } a\text{）}\ }
$$

朴素做法是 $\frac{\partial \ln N_{\text{exp}}}{\partial a} = \ln n_a > 0$，面积**乘性增长**。
Flywheel 的目标是让面积**加性甚至零增长**。

三个必须同时成立的结构性质：

<details>
<summary><b>性质 1 · 封闭性：机制基是有限的（~10 个）</b></summary>

所有 infra 优化，无论 case 多花哨，最终只落在约 10 个机制上：

| # | 机制 | 主要收益来源 |
|---|---|---|
| 1 | 融合（fusion） | 减 launch、减 HBM 往返 |
| 2 | 量化/压缩 | 减字节 |
| 3 | Tiling / pipelining | 提利用率 |
| 4 | 重叠（overlap） | 隐藏通信、阶段、访存 |
| 5 | 缓存与复用 | prefix、权重、编译产物 |
| 6 | 批处理/调度/公平性 | 利用率、SLO |
| 7 | 抢占/迁移 | 适配性（adaptivity） |
| 8 | 投机/近似 | 用算力换延迟 |
| 9 | 分片/分布 | 容量扩展 |
| 10 | 特化（codegen） | 形状专属 |

**模型能力是开放的，infra 机制是封闭的。** 这是 infra 唯一的、也是决定性的结构性优势：无论模型怎么变，机制集合不会膨胀。**所以：flywheel 必须建在机制上，绝不能建在 case 上。**

推论：真正要维护的不是“case 库”，而是一张 **机制 × 适用 regime 的适用性映射**。它的面积是 $10 \times |\text{regimes}|$，不是 $10^{\,k}$。case → 机制的映射是分类任务，成本很低。
</details>

<details>
<summary><b>性质 2 · 投影性：新轴必须以潜变量的形式进入</b></summary>

这是你问题中真正的核心。给出我建议的潜空间（数量级：7）：

| 潜维度 | 无量纲形式 | 吸收掉的显式轴 | 断点从哪来 |
|---|---|---|---|
| 算术强度 $I$ | $I / I^*$，$I^*=\pi/\beta$ | batch、seq len、dtype、模型大小 | roofline 交叉点 |
| 局部性 $W$ | 工作集 / 容量（L2 / HBM / smem） | seq len、KV dtype、prefix 复用率、page size | $W/C \approx 0.5, 1$ |
| 形状量化 $Q$ | $M \bmod \text{tile}$、wave 余数 | batch、seq len、每专家 token 数、TP 切分 | 整数边界 |
| 通信计算比 $\rho$ | $\rho/\rho^*$ | TP/PP/EP 度、节点数、batch | NCCL 算法切换阈值 |
| 精密度 $b$ | bytes/element | dtype、量化方案 | 离散 |
| 变异度 $\kappa$ | $\mathrm{CV}=\sigma/\mu$ of per-token work | MoE 偏斜、spec 接受率、thinking 长度 | 调度器阈值 |
| 时变性 $\lambda$ | 利用率、burstiness | concurrency、PD 混比、请求类型 | 队列/控制阈值 |

**关键操作**：每引入一个新轴，第一件事是问“它投影到哪几个潜维度上、怎么无量纲化”，**而非“要加多少个测试格”**。如果答不出来，说明你还没理解这个轴，此时做任何 benchmark 都是浪费。

你列的那四个轴投影结果：

- 序列长度 → $W$（主）、$I$（次）
- 专家数量 → $I$（有效字节）、$\kappa$（偏斜）、$Q$（每专家 token 数）
- Agent 请求类型 → $\lambda$、$W$（共享前缀比例）
- Thinking 强度 → $\kappa$、$I$

**“8 个轴”最终只落在 4 个潜维度上。空间从 $\prod n_i$ 变成潜空间里的 200~500 个点。**
</details>

<details>
<summary><b>性质 3 · 闭环性：产物必须是 policy，不是 config</b></summary>

如果产物是静态配置，新轴 = 新 case。
如果产物是**控制器**，新轴 = 新的输入信号。变的是函数签名，而非测试矩阵。

$\text{action} = \pi(\text{observable state})$，其中 state 包含 thinking budget、queue 状态、KV 占用、DRAM util。

这就是所谓的面向未来设计：**把 adaptivity 放进系统里，让运行时自己去覆盖 case 空间，而不是让人在实验室里枚举它。** 你那个 persistent-SM scheduler 天然就是这个形态——它本来就是个控制器。
</details>

---

## 具体的 Flywheel：七个环节

```
 ┌────────────────────────────────────────────────────────────┐
 │  ① 轴收编        新参数进来 → 先投影到潜空间，不许直接建 case │
 │       ↓                                                      │
 │  ② 区域判定      潜坐标 → 落在哪个 regime（决策树/分类器）    │
 │       ↓                                                      │
 │  ③ 机制查表      regime → 机制组合 + 适用性置信度（资产本体）  │
 │       ↓                                                      │
 │  ④ 代理模型      预测收益上界 U 与最优粒度 g*（毫秒级，不测）  │
 │       ↓                                                      │
 │  ⑤ 主动采样      只测「模型不确定 × 决策影响大」的点          │
 │       ↓                                                      │
 │  ⑥ 参数化实现    一个实现 + 编译期参数覆盖整片 regime         │
 │       ↓                                                      │
 │  ⑦ 线上影子      真实流量回灌 → 写入语料 → 模型误差下降        │
 └────────────────────────────────────────────────────────────┘
```

每一环的**产出物**和**失败模式**：

| 环 | 产出物 | 失败模式（务必警惕） |
|---|---|---|
| ① 轴收编 | 一个轴接入 spec（见下） | 把新轴当新 case，面积开始乘性增长 |
| ② 区域判定 | regime 决策树 + breakpoint 生成规则 | regime 划太细 → 退化成枚举 |
| ③ 机制查表 | **机制 × regime 适用性矩阵**（真正的资产） | 矩阵里全是“待验证”，等于没有 |
| ④ 代理模型 | cost model + 上界计算器 | 模型精度没预算，永远在拟合 |
| ⑤ 主动采样 | 一批最大信息量实验 | 均匀网格搜索（信息效率最低的做法） |
| ⑥ 参数化实现 | 一份代码 N 个参数化 | 每个拓扑一份代码（你正在踩的坑） |
| ⑦ 线上影子 | 标注好的测量语料 | 语料不版本化，硬件一换全部作废 |

---

## 量化：这套东西能把面积压到多少

设显式轴 $k$ 个，每个轴取 $n_i$ 个水平。

$$
N_{\text{naive}} = \prod_{i=1}^{k} n_i
$$

取 $k=8$、$n_i=6$ → $6^8 \approx 1.7\times10^6$。

Flywheel 下：

$$
N_{\text{fly}} \approx \underbrace{\sum_{i=1}^{r} n_i}_{\text{潜维 OAT + 断点}} \;+\; \underbrace{CA(r)}_{\text{潜维两两覆盖}} \;+\; \underbrace{|\mathcal{I}_3|}_{\text{已知三阶交互，手列}}
$$

$r=7$、$n_i\approx6$：

- OAT + 断点：$\sim 40$
- 两两覆盖阵列 $CA(N;2,7,6)$：$N \approx 40\text{–}60$
- 已知三阶交互（如 batch × dtype × topology）：手列 5~10 个

$\Rightarrow N_{\text{fly}} \approx 100$。

**$1.7\times10^6 \to 10^2$，四个数量级。** 而这个压缩并非靠猜测：它建立在两个真实成立的技术假设上：

1. **断点离散化**：性能曲线分段光滑，断点由硬件常数解析给出（上一轮讨论过）。段内不需要采样。
2. **交互稀疏性**：性能空间里高阶交互罕见，已知的少数可手列。这是组合测试（combinatorial testing）的根基——先用两两覆盖，再对已知三阶补点。

**如果你不接受“交互稀疏”这个假设，那你就只能枚举——所以这个假设必须被当作一项可验证的 research 命题，而不是当成假设默认成立。** 验证方法很便宜：随机抽 20 个三阶交互三元组，测交互项的方差贡献。这一实验结果决定你后续 90% 的时间怎么花。

---

## 三件最该先建的东西

<details>
<summary><b>(A) 轴接入 SDK —— 让"新轴成本"固定为 ≤10 个测试</b></summary>

每个新轴（thinking budget、专家数变化、agent 请求类型）都走同一个 5 步清单：

1. **投影**：它落到哪几个潜维？无量纲式写出来。
2. **断点规则**：给定硬件常数，解析地算出它的断点在哪。若算不出来 → **拒绝接入**，回去建模。
3. **观测**：运行时怎么读到它（telemetry / 请求元数据 / 编译期常量）？
4. **控制器输入**：它进哪个 policy 的 state（抢占？批处理？路由？）
5. **回归集**：≤10 个用例，直接取断点 ± 边界。

> 验收标准：**接入前后测试矩阵面积不增长。** 如果增长了，说明你在做 case patch，打回去。

这条规矩的威力在于它**从制度上禁止面积乘性增长**。这是 flywheel 里唯一真正会“自动复利”的部分：每个轴接入的成本被固定住了，而收益线性累加。
</details>

<details>
<summary><b>(B) 代理模型 + 主动采样 —— 把"测"换成"猜 + 验"</b></summary>

目标不是准确预测性能（那是做不到的），而是**准确预测「值不值得测」**。

最小可用版本：
- 上界计算器：给定潜坐标，输出 $U$（roofline + launch 计数 + 通信量的解析式）。这部分是确定性的，不需要学。
- 残差模型：一个轻量模型拟合「解析上界与实测之间的比」，输入是潜坐标。几百个样本就够。
- **采样策略**：选择 $\arg\max_x \big[\,\text{Var}(x) \times |\partial J/\partial U|\,\big]$，即“预测最不确定、且决策最敏感”的点。

**关键指标：TTFA（time-to-first-answer）**——一个新 case 从进来，到系统能给出一句“这需要特殊处理，机制是 M，上界 U%”的时间。

- 当前大概是：几周到几个月
- 目标：**小时级**，靠模型回答；只有模型残差大的点才进实验室。

这就是 flywheel 的“齿轮比”：不是让实验跑得快，而是**让 95% 的实验根本不必跑**。
</details>

<details>
<summary><b>(C) 机制适用性矩阵 —— 你唯一的长期资产</b></summary>

一个 $10 \times R$ 的表（$R$ = regime 数，控制在 20 以内），每格存：

```
适用性: 是 / 否 / 有条件
条件:   潜坐标的谓词（e.g. W/C > 0.8 且 rho/rho* < 1）
收益上界: 解析式
成本模型: c(g) 的拟合式
失败记录: 在哪些坐标上失效、为什么 ← 最值钱
```

**“失败记录”比“成功记录”值钱得多**，因为它永久性地剪枝未来空间。一个团队最缺的资产是“哪些路走不通、以及为什么”——而不是 benchmark 数字。

这个矩阵还带来一个组织收益：**它是一份可以被质疑、被复用、被继承的文档**，而“搜出来的最优配置字典”三个月后就没人看得懂。
</details>

---

## 从源头限制笛卡尔积：把“加轴”定价

前面全是“消化”新轴。但更强的一招是**减少新轴的供给**。你的判断“infra 问题增长快于模型能力”有一个真实的机制：

> 模型团队可以免费地引入新轴（新的注意力变体、新的路由、新的推理模式），而 infra 团队承担全部的组合成本。

这是典型的**成本外部化**。解法是给它定个价：

**任何引入新轴的模型侧变更，必须随 PR 附带：**
1. 参考实现（哪怕很慢）
2. 形状/参数分布 spec（这个轴的实际取值分布，带线上权重）
3. 降级路径（不支持时退化成什么）
4. 该轴投影到哪个潜维、预期落在哪个 regime

这不是流程官僚，而是**把笛卡尔积的成本在源头显性化**。做得好的话，模型团队会自发地把新轴设计成“落在已有 regime 里”的形式——这才是最高杠杆的优化：**让问题不产生**。

极端形态是硬件/系统感知的模型设计（hardware-aware co-design）：infra 团队交给模型团队的不是 kernel，而是**一个“这个 op 集、这个 layout、这个 dtype 在系统上便宜”的契约**。模型在契约内创新，infra 的组合空间被契约封死。这在一线大厂是真实存在的组织杠杆，价值超过任何一个 kernel。

---

## Flywheel 的 KPI 与反模式

**KPI（只看这四个，别看吞吐数字）**

| KPI | 目标 |
|---|---|
| $\partial(\text{测试矩阵面积}) / \partial(\text{新轴})$ | ≈ 0 |
| TTFA（新 case 到第一句可靠结论） | 小时级 |
| 代理模型的残差分布 | 90% 的点残差 < 20% |
| 机制矩阵的“已定性”覆盖率 | > 80% |

**反模式（这五个会让 flywheel 变成飞轮形状的墓碑）**

1. **建一个通用 autotuner** —— 会死。autotune 已被编译器解决（Triton/CUTLASS 自带），你别重复造。
2. **建一个大网格搜索 DB** —— 会很快过期，且无人能读懂。
3. **建一个没有精度预算的模拟器** —— 永远在拟合，永远不可信。
4. **建一个所有人都能刷分的 benchmark 套件** —— 半年后它测的东西和线上无关。
5. **为每个 case 打 patch** —— 面积乘性增长，flywheel 反向转。

**必须有的机制：失效期管理。** 语料和模型都要带硬件/框架版本标签，硬件换代时按残差而不是按时间重验。否则你的复利资产会变成负资产。

---

## 一句话

> 你的乘法关系是**参数化**造成的，不是**现实**造成的。infra 的 flywheel 不是“搜得更快”，而是**换坐标系**：把 $k$ 个显式轴投影到 $\sim 7$ 个无量纲潜维（此时面积从 $\prod n_i$ 掉到 $10^2$），把产物从 config 变成 policy（此时新轴成为输入而非 case），把机制集合当作封闭基来经营（$10 \times R$ 的适用性矩阵），并且**从制度上禁止 case patch**，让新轴接入成本恒定为 10 个测试。
>
> 模型能力是开放的，机制空间是封闭的。**这是你唯一的结构性优势，flywheel 必须整个建在这个优势上。** 你现在感觉要被 case 淹没，是因为你在开放维度上建流程，而那里永远赢不了。

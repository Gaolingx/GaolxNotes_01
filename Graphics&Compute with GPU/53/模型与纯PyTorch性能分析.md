# 模型与纯PyTorch性能分析

### 🧑‍💻 User

gemm kernel中，前一个gemm的output过激活到下一个op的input swizzle layout这种算子有哪些模型会用到，纯torch实现会遇到性能瓶颈吗？

---

### 🤖 Assistant

# GEMM→激活→GEMM + swizzle layout：模型与纯 PyTorch 性能分析

## 1. 这种"算子"具体指什么

你描述的其实是一个 **producer-consumer 融合 + epilogue** 的组合模式：

$$\underbrace{\text{GEMM}_1}_{\text{Linear/up\_proj}} \;\longrightarrow\; \underbrace{\sigma(\cdot)}_{\text{GELU/SiLU/ReLU}} \;\longrightarrow\; \underbrace{\text{GEMM}_2}_{\text{Linear/down\_proj}}$$

其中中间结果 $H = \sigma(XW_1)$ 不以标准 row-major 布局写回显存，而是以 **swizzle（tile 重排）布局** 存储，供下一个 GEMM 读取。这里实际包含两个独立但常被一起做的优化：

| 层面 | 含义 | 收益 |
|---|---|---|
| **Epilogue fusion**（激活融合） | GEMM₁ 的 epilogue 里顺带算激活，激活结果不再"写回再读出" | 省掉激活张量的 HBM 往返 |
| **Tile swizzle**（tile 顺序/布局重排） | GEMM₁ 写 tile 的顺序与 GEMM₂ 读 tile 的顺序对齐（如 boustrophedon/zigzag、128×128 swizzle） | 提升 L2 命中率，减少 HBM 流量，producer/consumer 可重叠 |

注意区分：**共享内存 swizzle**（防 bank conflict，单 kernel 内部）和**全局内存 tile swizzle**（跨 op 的数据布局，L2 locality）是两回事——你问的是后者。

---

## 2. 哪些模型会用到

几乎所有带"两级 MLP/FFN"的网络都会命中这个模式：

| 模型家族 | 结构 | 激活 |
|---|---|---|
| BERT / GPT-2 / GPT-3 / ViT | `down(GELU(up(x)))` | GELU |
| T5、部分 ResNet 风格 | `down(ReLU(up(x)))` | ReLU |
| **LLaMA / Qwen / Mistral / DeepSeek / Gemma** | SwiGLU：`down(SiLU(gate(x)) ⊙ up(x))` | SiLU + 乘 |
| GPT-3 / PaLM / GLM | GeGLU：`down(GELU(gate(x)) ⊙ up(x))` | GELU + 乘 |
| MoE（Mixtral、DeepSeek-MoE、Switch） | 每个 expert 就是上面的 FFN | 同上 |
| DLRM / 推荐系统 | `Linear→ReLU→Linear` 的 bottom/top MLP | ReLU |
| CNN | 1×1 conv（im2col 后即 GEMM）+ 激活 + 1×1 conv | ReLU |
| Attention（广义） | $QK^\top \to \text{softmax} \to PV$，softmax 是 row-wise pointwise，FlashAttention 正是这个融合 + block/swizzle 布局 | softmax |

**实际落地这些优化的框架/内核**：CUTLASS（epilogue + `ThreadblockSwizzle`）、cuBLASLt、TensorRT、ONNX Runtime CUDA EP、FlashAttention-2/3、DeepSeek 的 Hopper TMA + warp specialization 内核、Liger-Kernel / Unsloth / vLLM fused MLP / FlashInfer 等。

---

## 3. 纯 PyTorch 会有性能瓶颈吗？——会，而且很明显

eager 模式下，上面模式会退化成 3 个独立 kernel：

```python
h = F.linear(x, W1)       # GEMM1 (cuBLAS)
h = F.gelu(h)             # 独立 elementwise kernel
out = F.linear(h, W2)     # GEMM2 (cuBLAS)
```

### 3.1 激活无法融合 → 内存带宽成为瓶颈

以 LLaMA-7B 为例：hidden=4096，FFN intermediate=11008，fp16。

对 $N$ 个 token，中间张量大小约 $N \times 11008 \times 2\ \text{bytes}$。

- **不融合**：GEMM₁ 写 $H$（1 次写）→ 激活 kernel 读 $H$ + 写 $H$（2 次访问）→ GEMM₂ 读 $H$（1 次读），共 **4 次 HBM 访问**。
- **融合**：GEMM₁ 在寄存器/共享内存里算完激活再写 $H$（1 次写）→ GEMM₂ 读（1 次读），共 **2 次访问**。

即融合省掉激活张量的 **2 次完整 HBM 往返**。取 $N=4096$：中间张量 $\approx 4096\times11008\times2 \approx 90\ \text{MB}$，省下 $\approx 180\ \text{MB}$/层；H100 带宽约 3.35 TB/s，折合约 54 μs/层，32 层累计约 1.7 ms。这在**短序列推理 / 小 batch 解码**（算术强度低）时占比更高，因为 GEMM 本身已经 memory-bound，激活的额外流量是纯开销。

### 3.2 PyTorch 无法原生表达 swizzle 布局

PyTorch 张量默认是 row-major contiguous。想要 swizzle 布局只能：

- `view/reshape + permute` 得到逻辑视图（stride 变化），但 `F.linear`/cuBLAS 要求 contiguous，最终会触发 `.contiguous()` **拷贝 kernel**——额外一次 HBM 往返，swizzle 变成纯负收益；
- 或把非连续 stride 直接喂给 cuBLAS，cuBLAS 内部要么走 strided batched GEMM，要么先 copy，性能更差。

所以 swizzle 布局只有在 **producer/consumer 共设计的内核**（CUTLASS/Triton/TensorRT）里才划算，eager PyTorch 里做不了也做不划算。

### 3.3 kernel launch 开销 + cuBLAS 不暴露 epilogue

- 3 个 kernel vs 2 个融合 kernel，短序列时 launch 开销（每个 ~5–10 μs）不可忽略；
- `torch.nn.functional.linear` 走 cuBLAS，**不暴露 epilogue**（bias 有，但 GELU/SiLU 激活融合没有），即使 `torch.backends.cuda.preferred_linalg_library` 切到 cuBLASLt 也主要影响 bias/算法选择，激活 fusion 仍不开放。

### 结论

> 纯 eager PyTorch 在这类模式上会同时吃 **内存带宽（激活往返）+ 额外拷贝（swizzle 无法表达）+ launch 开销** 三份损失，规模越大、序列越短、模型越深，损失越明显。GEMM 本身（cuBLAS）很快，瓶颈在 GEMM **之外**的数据搬运。

---

## 4. 怎么解决

| 方案 | 能做什么 | 局限 |
|---|---|---|
| `torch.compile` + Inductor | 能把 `mm + gelu/silu` 融合成 Triton kernel（epilogue fusion），消除激活往返 | 跨 op 的 swizzle tile 布局 co-design 基本未暴露，融合覆盖有限 |
| CUTLASS / cuBLASLt | 完整 epilogue（bias + 激活 + swizzle store） + `ThreadblockSwizzle` | 需要 C++ 写，集成成本高 |
| TensorRT / ORT CUDA EP | Gemm+Activation 自动融合 + swizzled 中间布局，L2 locality 优化成熟 | 图优化器，不适合训练 |
| 手写 Triton | Liger-Kernel（fused linear+SiLU/GELU）、Unsloth（fused MLP）、vLLM `fused_moe`/`silu_and_mul`、FlashInfer | 灵活但需自己处理 tile 顺序、L2 调度 |
| Hopper TMA + warp specialization | FlashAttention-3 / DeepSeek / FlashMLA 的 TMA + swizzled shared memory + 异步 epilogue | 面向 H100+，工程复杂度最高 |

**实践建议**：如果是推理场景优先考虑 TensorRT/ORT/vLLM 现成融合；训练场景用 Liger-Kernel 或 `torch.compile` 先拿掉激活往返；要极致性能（尤其 MoE 或大 hidden 的 LLM）再上 CUTLASS epilogue + 全局 tile swizzle 的定制内核。

---

### 🧑‍💻 User

通过编写custom triton kernel能不能提高性能？

---

### 🤖 Assistant

# Custom Triton kernel 能不能提高性能？

**能，但要分场景看——Triton 的赢点不在"裸 GEMM"本身，而在融合和布局控制。** 这正好对应上一轮分析的三个瓶颈。

---

## 1. 先明确：Triton 在哪些地方能赢

| 上一轮的瓶颈 | Triton 能否解决 | 说明 |
|---|---|---|
| 激活无法融合 → HBM 往返 | ✅ 核心收益 | 把激活写进 GEMM 的 **epilogue**，寄存器/共享内存里算完直接写回 |
| swizzle 布局无法表达 | ✅ 可以 | `program_id` 重映射成 tile 顺序，完全可控 |
| launch 开销 / contiguous 拷贝 | ✅ 可以 | 多算子融合成单 kernel，中间张量甚至不出全局内存 |

### 1.1 激活融合（epilogue fusion）—— 最直接的收益

```python
# eager: 3 个 kernel，激活张量 H 来回读写
h = F.linear(x, W1)          # 写 H
h = F.gelu(h)                # 读 H + 写 H  ← 纯内存流量
out = F.linear(h, W2)        # 读 H
```

```python
# Triton: GEMM 的 epilogue 里直接算 gelu，H 只写一次、读一次
@triton.jit
def fused_linear_gelu(x, w, bias, out, M, N, K, ...):
    ...
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(...)
        b = tl.load(...)
        acc += tl.dot(a, b)          # tensor core MMA
    acc += bias
    acc = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))  # GELU 融合
    tl.store(out_ptrs, acc.to(out_dtype), mask=...)
```

省掉的正是中间张量的 **2 次 HBM 访问**。这一条对 **短序列推理 / 小 batch 解码**（算术强度低、memory-bound）收益最明显。

### 1.2 swizzle / tile 顺序控制

Triton 里 tile 遍历顺序完全由 `pid` 的映射决定，可以手写 swizzle 让 producer 写序和 consumer 读序对齐，提升 L2 命中率：

```python
# 把一维 pid 映射成 (group_m, group_n)，组内 swizzle，组间 rasterize
pid = tl.program_id(0)
num_pid_n = tl.cdiv(N, BLOCK_N)
group_m = pid // (num_pid_n * GROUP_M)
pid_mn = pid % (num_pid_n * GROUP_M)
pid_m = (pid_mn % GROUP_M) + group_m * GROUP_M   # 行方向分组
pid_n = pid_mn // GROUP_M                        # 列方向遍历
```

这就是 FlashAttention 里 `GROUP_M` 那种做法的思想：**相邻 pid 处理相邻的 tile，吃 L2**。纯 PyTorch 完全做不到这一步。

### 1.3 更进一步：整段 MLP 融合成 persistent kernel

如果中间维度不大，可以把 **GEMM₁ + 激活 + GEMM₂ 放进同一个 kernel**，中间结果停在 L2 甚至共享内存，只对输入/输出做 HBM 访问。vLLM 的 `fused_moe`、Unsloth 的 fused MLP、Liger-Kernel 走的就是这个路线。

---

## 2. 但 Triton 在"裸 GEMM"上通常打不过 cuBLAS

这是必须诚实说的一点：

- cuBLAS/CUTLASS 的 GEMM 有十几年调优积累：最优 tile size、split-K、software pipelining、`cp.async`、Hopper 上的 TMA + `wgmma`、warp specialization、混合精度策略。
- Triton 的 `tl.dot` 已经很强，但在**大矩阵、高算术强度**（大 batch prefill、大 hidden）场景，裸 GEMM 吞吐通常仍比 cuBLAS **慢 5%–20%**（fp8/bf16 上差距可能更大）。

**所以策略是：**

> 如果只是"把 `F.linear` 换成 Triton GEMM，激活还是单独一个 kernel"——大概率**变慢**（GEMM 输给 cuBLAS，融合收益又没拿到）。

---

## 3. 决策表：什么场景值得写

| 场景 | 纯 PyTorch vs 手写 Triton | 结论 |
|---|---|---|
| 只换 GEMM，不融合 | Triton 略慢 | ❌ 不值得 |
| GEMM + 激活 epilogue 融合 | 省激活往返，通常净收益 10%–30%+ | ✅ 值得 |
| 整个 MLP（up/gate/down + SiLU·up）融合 | memory-bound 时收益最大 | ✅✅ 很值得 |
| 大 batch prefill（compute-bound） | cuBLAS 本身已够快，融合收益小 | ⚠️ 收益有限 |
| MoE（小 expert、多路由） | fused_moe 比 eager 快很多 | ✅✅ 很值得 |
| 训练（Liger 风格 fused linear+act） | 省显存 + 省带宽，实测 ~20% 吞吐提升 | ✅ 值得 |

---

## 4. 落地建议

1. **别从零写通用 GEMM**。先用现成轮子验证收益：
   - 推理：vLLM `fused_moe` / `silu_and_mul`、FlashInfer；
   - 训练：Liger-Kernel（fused `linear + SiLU/GELU`、fused `SwiGLU`）、Unsloth；
   - 编译兜底：先试 `torch.compile` + Inductor，很多时候能把 `mm + gelu` 自动融掉，成本为零。

2. **必须 autotune**。Triton 性能对参数极敏感：
   - `BLOCK_M / BLOCK_N / BLOCK_K`、`num_warps`、`num_stages`（pipeline 深度）；
   - swizzle 的 `GROUP_M` / tile 顺序；
   - 是否用 `tl.make_tensor_descriptor`（TMA，Hopper）+ `wgmma`。

3. **目标硬件不同，写法不同**：
   - Ampere（A100）：`tl.dot` + `cp.async`（Triton 自动生成）；
   - Hopper（H100/H200）：TMA + `tl.dot_scaled`（fp8）才有机会逼近甚至反超 cuBLAS。

4. **注意数值与正确性**：epilogue 里融合激活要用 fp32 accumulator，别在 fp16 上直接算 GELU/SiLU；SwiGLU 的 `SiLU(gate) ⊙ up` 也要在 fp32 下乘完再 cast。

---

## 结论

**能提高，而且针对你问的这个"GEMM→激活→GEMM + swizzle"模式，手写 Triton 是性价比最高、最可控的方案之一**——因为瓶颈本就不在 GEMM 的 FLOPs，而在数据搬运和布局，而这恰恰是 Triton 能显式控制的。唯一要记住的边界是：**不要指望 Triton 在纯矩阵乘法上打赢 cuBLAS，把它用在"融合 + 布局 + persistent kernel"上才对。**
# Tri Dao causal conv1d算子详解

## 🧑‍💻 User

tridao causal conv1d是什么算子

---
## 😀 默认助手 (7)

## 一句话概括

**Tri Dao 的 `causal-conv1d` 是一个“因果深度可分离一维卷积”（causal depthwise conv1d）的手写 CUDA 算子**，仓库是 `Dao-AILab/causal-conv1d`，PyTorch 接口为 `causal_conv1d_fn`。它是 Mamba 系列状态空间模型（SSM）里的标准组件，也被 vLLM 和 SGLang 用来做推理。

## 数学定义

对每个通道独立做卷积（depthwise，`groups = dim`），并且**只在左侧 padding**，保证因果性：

$$
y_{b,c,t} \;=\; \sum_{k=0}^{K-1} w_{c,k}\, x_{b,c,\,t-(K-1)+k} \;+\; \beta_c,
\qquad t-(K-1)+k < 0 \text{ 时视为 } 0
$$

其中 $x \in \mathbb{R}^{B\times D\times L}$，权重 $w \in \mathbb{R}^{D\times K}$（每个通道一组，K 通常为 4），可选再接 $\mathrm{SiLU}$。也就是说，位置 $t$ 的输出只看 $t-K+1 \dots t$，绝不看未来。

官方 README 给的等价写法非常直白：

```python
F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
```

## 为什么要专门写一个算子

| 维度 | PyTorch 原生做法 | `causal_conv1d` |
|---|---|---|
| padding | 显式 `F.pad` 再切掉尾部，多一次访存 | kernel 内部处理越界，读写各一次 |
| 卷积实现 | cuDNN/cuBLAS，K=4 这种极小 kernel 效率低 | 手写 kernel，按 channel 并行 + 共享内存做滑窗 |
| 激活 | 单独一次 SiLU kernel | 前向融合 `activation="silu"`, 反向也融合 |
| 变长 batch | 需自己 mask，容易跨序列泄漏 | `seq_idx` 参数直接在 kernel 内隔离 |
| 状态 | 无 | `initial_states` / `final_states` 支持分块训练与流式推理 |
| 推理单步 | 不支持 | `causal_conv1d_update` 支持 KV-cache 式增量更新 |

关键点：**K 很小（2、3、4）、通道数很大（几千）**，这是典型的“访存受限”形状，通用卷积库发挥不出性能，所以要用专用 kernel 把“卷积 + 激活 + 状态管理”全部融合掉。

## 三个 API 分别对应什么场景

```python
# 1) 训练 / prefill：整段序列
causal_conv1d_fn(x, weight, bias=None, seq_idx=None,
                 initial_states=None, return_final_states=False,
                 final_states_out=None, activation=None)

# 2) 流式 decode：每次一个 token（或几个）
causal_conv1d_update(x, conv_state, weight, bias=None,
                     activation=None, cache_seqlens=None,
                     conv_state_indices=None)
```

- **`x: (B, D, L)`, `weight: (D, K)`, `bias: (D,)`**；支持 fp32/fp16/bf16。
- **`seq_idx: (B, L)`**：packed 变长序列（如 flash-attn 的 varlen），同一条 batch 里多条序列不会互相“看见”。
- **`initial_states / final_states: (B, D, K-1)`**：把长序列切成 chunk 训练，或把“滑动窗口的最后 K-1 个输入”当缓存传给下一段，等价于 RNN 的 hidden state。
- **`causal_conv1d_update`** 里的 `conv_state` 是环形缓冲区，`cache_seqlens` 指向写入位置，`conv_state_indices` 支持 **continuous batching**（vLLM/SGLang 里 Mamba 推理的核心）。

## 在 Mamba 里的位置

```
x → in_proj (线性) → causal_conv1d (K=4, depthwise) → SiLU → selective scan (SSM) → out_proj
                     ↑ 就是这个算子
```

Mamba/Mamba-2、Zamba、Jamba、Codestral Mamba 等混合模型的 `conv1d` 层都是它；`mamba_ssm` 包在安装时也依赖 `causal_conv1d`。

## 局限与注意事项

- 只支持 **kernel size 2、3、4**，不是通用卷积。
- 必须编译（CUDA 或 ROCm；ROCm 6.0 需要打 patch），有预编译 wheel 但常与 torch/CUDA 版本耦合，装不上是常见问题。
- 对 **channel-last 布局**（`x.stride(1) == 1`）最友好，`return_final_states` 仅在 channel-last 下支持。
- 反向也是手写 kernel，支持融合下游 chunk 的 `dx` 预分配，这是它比“pad+conv1d”快数倍的原因。

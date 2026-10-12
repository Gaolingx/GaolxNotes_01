# 现代LLM图谱不适合用Netron查看

## 🧑‍💻 User

以hf transformers为例，为什么现代llm（hybridattn/moe/mla）里边计算图已经不能再简答用Netron查看了

---
## 😀 默认助手 (7)

# 为什么现代 LLM（hybrid attention / MoE / MLA）的计算图不能用 Netron「看一眼」

## 一句话结论

Netron 的设计前提是：**模型 = 一个已经序列化好的、固定的、无状态的静态 DAG 文件**。而现代 LLM 恰恰把这三个前提全部打破了：

| Netron 的前提 | 现代 LLM 的现实 |
|---|---|
| 图是**固定的** | 图是**输入相关的**（MoE 路由、稀疏注意力选块、动态 shape） |
| 图是**无状态的** | 图是**有状态迭代的**（KV cache / SSM 状态 / `.generate()` 循环 / 投机解码） |
| 图**就是执行** | 图只是**中间产物**，真正执行的是运行时 dispatch + JIT 编译 + CUDA Graph |

再加上一个工程层面的硬伤：**文件已经大到 Netron 自己打不开**。下面逐条展开，都用 HF `transformers` 的代码和事实做锚点。

---

## 一、先把 Netron 的能力边界钉死

Netron 是**格式查看器**，不是**程序分析器**。它实际能做的只有：

- 读 `.onnx` / TorchScript `.pt` / `.pte` / TFLite / CoreML 等**序列化静态图**；
- **一次性**对全图做分层布局（ELK / graphviz 类算法），复杂度随节点数增长；
- 把 `If` / `Loop` / `Scan` 的子图**塞在节点属性里**显示得极其糟糕甚至忽略；
- **不做**任何数据流推断、不做 shape 求解、不执行任何算子。

关键推论：**Netron 的图里没有"循环"和"复用"这两个抽象**。ONNX 没有宏/子例程引用，所以"60 层相同的 Transformer block"必须被**展开复制 60 份**。信息不是被压缩，而是被冗余放大。这一点是后面所有"爆炸"的根源。

---

## 二、五条断裂

### 1. 动态控制流：tracing 只能捕获「一条路径」

HF 的 forward 里到处都是**数据相关或 Python 对象相关的分支**：

```python
# 示意：现代 modeling_*.py 里随处可见
if past_key_values is not None:            # 对象状态分支
    ...
if attention_mask is not None and not torch.all(attention_mask == 1):
    ...                                     # 数据相关分支
if cache_position.shape[0] != 1:
    ...
```

`torch.onnx.export` 的老路径是 **trace**：它靠**真的跑一遍**来记录图。于是：

- `attention_mask` 全 1 的快速路径会被**固化**进图，mask 逻辑消失；
- prefill（seq_len=N）和 decode（seq_len=1）是**两条完全不同的图**，optimum 只能导成 `model.onnx` + `decoder_model_merged.onnx`，用 `If` 节点在中间劈一刀；
- `if cache_position...` 这类分支要么被固化，要么变成 ONNX 的 `If`（Netron 里只能看到一个三角节点）。

而 `torch.export` / Dynamo 时代更进一步：图里出现的是**高阶算子**——`torch.ops.higher_order.cond`、`while_loop`、`map`、`effects`，甚至把 Python 函数当作属性塞进节点（见下面的 FlexAttention）。**Netron 没有这些算子的 reader，也没有办法渲染"节点属性里挂着一个 Python callable"的语义。**

### 2. 有状态：`.generate()` 是一个 Python 程序，不是一张图

HF 的推理主入口是：

```python
model.generate(input_ids, max_new_tokens=..., do_sample=True, temperature=...)
```

这里面装着：`while` 循环、`StoppingCriteria`（依赖已生成 token 的 Python 判断）、`LogitsProcessor`（top-k / top-p / repetition penalty / 语法约束 / JSON 约束）、beam search 的 `BeamSearchScorer`，以及投机解码的「草稿模型 + 接受/拒绝」循环。

**这些控制流一行都不在图里。** 而且 `past_key_values` 本身是个 Python 对象：

```python
# DynamicCache 一代：每步 cat
key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)
# StaticCache 一代：往预分配 buffer 里做 index_copy_（原地状态变更）
```

`torch.cat` / index-assign 是**状态变更**，不是纯函数。要让 `torch.export` 导出，HF 不得不额外提供 `TorchExportableModuleWithStaticCache` 这类 wrapper，把 cache 展平成**额外 4×L 个输入输出张量**。也就是说：

> **能导出的那张图，和你 `.generate()` 实际跑的那个程序，已经不是同一个东西了。**

### 3. MoE：输入相关的稀疏 + 展开爆炸

MoE 是"图必须依赖输入"的最纯粹的例子。看 HF 的 `MixtralSparseMoeBlock`（示意，结构稳定）：

<details>
<summary>展开看 MoE 的关键代码结构</summary>

```python
router_logits    = self.gate(hidden_states)                       # [T, E]
routing_weights  = F.softmax(router_logits, dim=1, dtype=torch.float)
routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
final_hidden_states = torch.zeros_like(hidden_states)
expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

for expert_idx in range(self.num_experts):        # ← Python for，且每次走的是动态索引
    idx, top_x = torch.where(expert_mask[expert_idx])
    current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
    current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
    final_hidden_states.index_add_(0, top_x, current_hidden_states)   # ← 数据相关 scatter
```

</details>

三个致命点：

- **`torch.topk` + `torch.where` + `index_add_` 是数据相关索引**。导出后你看到的是 `TopK`、`NonZero`、`Gather`、`ScatterND` 这些**通用算子**——路由语义（哪几个专家、权重多少、负载均衡 loss、`e_score_correction_bias`）**在图里完全不可见**。
- **真实执行的子图数量是天文数字**。DeepSeek-V3 有 **256 个路由专家、top-8**，理论上的激活组合：
$$\binom{256}{8}\approx 4.4\times 10^{14}$$
每层就有这么多可能子图，58 个 MoE 层叠起来是它的 58 次幂。**没有任何静态 DAG 能"表示"这个计算。**
- **专家并行 / all-to-all** 引入 `all_to_all`、`index_select` 跨设备通信，`device_map="auto"` 或 `tp_plan` 的切分是**图外**的行为。

### 4. MLA：同一数学，至少两套图；而能跑的那套只有 kernel 能表达

MLA（DeepSeek 的 Multi-head Latent Attention）把 KV 压成低秩潜向量：

$$c_t^{KV} = W^{DKV}h_t,\quad k_t = W^{UK}c_t^{KV},\quad v_t = W^{UV}c_t^{KV}$$

**关键事实：存在两种数学等价但图完全不同的实现。**

- **朴素形式**（HF `DeepseekV2Attention` 走的路线）：真的把 $k_t, v_t$ 物化出来，`kv_b_proj` 后 `view/split` 成每头 K/V，然后 SDPA。缓存是 $2n_h d_h$，显存大。
- **吸收形式**（vLLM / SGLang / FlashMLA 走的路线）：把 $W^{UK}$ 折叠进 $W^{UQ}$：

$$q_t^\top k_s = \big(W^{UQ}c_t^{Q}\big)^\top\big(W^{UK}c_s^{KV}\big) = c_t^{Q\top}\underbrace{\big(W^{UQ\top}W^{UK}\big)}_{\text{可预折叠}} c_s^{KV}$$

  缓存只留 $c^{KV}$，但代价是：解耦 RoPE 只作用在 $d_r$ 维上（`q_pe`/`k_pe` 单独算）、大量**依赖 shape 常量**的 `reshape/view/slice`、以及最终被 `flash_mla` 之类的**单节点 kernel** 吞掉。

后果：

- **图里有大量硬编码 shape 的 `Reshape`/`Slice`**，只要 `num_heads`、`max_position_embeddings`、`q_lora_rank` 变一点就失效；
- 你导出的（HF eager 版）**不是线上跑的那个模型**——线上一个节点就是 FlashMLA，图里什么都看不见；
- 打开动态 shape 后，Netron 里会出现**一整片"算 shape 的张量算子"扇出**（`Shape`→`Gather`→`Concat`→`Reshape`），可读性归零。

### 5. 算子被熔合、被运行时替换：图既太粗又太细

这是最"元"的一条。

**太粗（语义被 kernel 吞掉）**——现代 HF 默认路径下，注意力的全部算法内容就是一个节点：

```python
# attn_implementation="sdpa"（HF 默认）
attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, is_causal=...)
```

这一个 `aten._scaled_dot_product_flash_attention` 节点里藏了：因果掩码、滑动窗口、GQA 的 `repeat_kv`、softmax 的 fp32 upcast。同理：

- `flash_attn_varlen_func`（变长打包）、`xformers`、`liger_kernel`
- **FlexAttention**：mask 是一个 **Python 函数**，被 Inductor 编成 Triton kernel。图里是 `torch.ops.higher_order.flex_attention`，**程序本体是一个被编译的 callable**。
- `bnb.matmul_4bit`（bitsandbytes NF4）、GPTQ/AWQ 的 `QuantLinear`、HQQ 的 codebook `gather`、torchao 的 fp8 block scaling
- MoE 的 `GroupedGEMM` / `fused_moe`

**太细（展开后爆炸）** —— 如果你想"绕过熔合"，用 `attn_implementation="eager"` 手动展开，节点数直接涨一个量级，而且 `RepeatKV`、`masked_fill`、`softmax` 全变成几十个原子算子，Netron 里是一团毛线。

**而且同一份权重，图会随环境变**：

```python
torch.backends.cuda.sdp_kernel(...)   # 运行时选 backend
# H100 + fp16 → FlashAttention-2；A100 + fp32 → math 路径
# pip install flash_attn / causal_conv1d / mamba_ssm 是否成功，直接决定图里是 1 个节点还是 20 个节点
```

Hybrid 架构把这个矛盾推到极致（Qwen3-Next 的 Gated DeltaNet + Gated Attention、Jamba/Zamba/Nemotron-H 的 Mamba + Attention、Gemma-2/3 的 5:1 局部/全局交替、DeepSeek-V3.2 的稀疏注意力）：

- **`layer_types` 是 PyTorch tensor 之外的配置列表**，`if layer_types[i] == "linear_attention"` 是图外分支；
- 线性注意力的 **chunked scan / recurrent state** 是 `while` 循环或自定义 Triton kernel；Mamba 的 `conv_state`/`ssm_state` 是**与 KV cache 异构的第二类状态**；
- Gemma-2 的 sliding window 只体现为**一个 mask**，图中的算子和全注意力**长得一模一样**——**Netron 里看不出语义差异**；
- HF 为此专门引入了 `HybridCache`，`cache.layers[i]` 每层类型不同——这是一个**Python 对象图**，不是张量图。

**最终杀手：CUDA Graph。** `torch.compile` + `reduce-overhead`、vLLM 的 piecewise CUDA graph 把整段计算**捕获成一次 graph launch**。此时**运行时根本不存在算子级 DAG**。Netron 要看的那张图，物理上已经不存在了。

---

## 三、量级：为什么 Netron 会直接崩

| 模型 | 展开后节点量级 | 权重文件（fp16） |
|---|---|---|
| Llama-3-8B（SDPA 熔合） | ~10³ | ~16 GB |
| Llama-3-8B（eager 展开） | ~2×10³ | ~16 GB |
| Mixtral-8x7B（8 专家 × 32 层） | ~10⁴ | ~93 GB |
| Qwen3-Next-80B（512 专家 + hybrid） | ~10⁵ | ~160 GB |
| DeepSeek-V3（256 专家 × 58 MoE 层） | **~10⁵–10⁶** | **~1.34 TB** |

两个工程事实：

1. ONNX 的 protobuf 序列化有 **2 GB 上限**，必须 `save_as_external_data=True` 拆成外部权重文件；Netron 打开时仍要建立全部 initializer 的索引。
2. Netron 的**全局分层布局是 O(n) 起步但常数极大**，在 10⁵ 节点上基本是 OOM 或卡死。实践中大 ONNX 加载数分钟到不可用是常态。

---

## 四、那现在用什么「看图」

Netron 的位置已经被四个不同层次的工具取代，因为**你需要的是四个不同的"图"**：

| 你想回答的问题 | 应该用的工具 |
|---|---|
| 模型结构 / 模块层级是什么 | `print(model)`、`modeling_*.py` 源码、`config.layer_types` |
| 导出的**语义图**长什么样、cache 被拍平成什么输入 | `torch.export.export(...)` → `ExportedProgram.graph_module.print_readable(False)` |
| 为什么没编成一张图（graph break / recompile） | `TORCH_LOGS="graph_breaks,recompiles,output_code"`、`torch._dynamo.explain()` |
| 编译器实际生成了什么 kernel / 融合了什么 | `TORCH_COMPILE_DEBUG=1` → `ir_pre_fusion.txt`、`output_code.py`（Triton IR） |
| **真实执行**了什么、每种 kernel 花了多久 | `torch.profiler` → Perfetto / `chrome://tracing`、Nsight Systems |
| 权重切分 / 通信 | `DTensor`、`accelerate` 的 `device_map`、NCCL profiler |

你会发现一个规律：**越靠"程序"越有用，越靠"图片"越没用。** 现代 LLM 的正确分析对象是 *FX graph 文本 + Triton IR + profiler trace*，而不是一张拓扑图。

---

## 五、什么时候 Netron 仍然有用

不必走向另一个极端，它在这几种场景依然是首选：

- **小模型 + 明确 opset 的 ONNX 导出**，检查算子覆盖度（比如能不能喂给 TensorRT / OpenVINO / RKNN）；
- **量化 DAG 检查**：看 Q/DQ 节点插在哪、`Cast` 有没有漏、dequant 有没有被提到循环外；
- **导出 bug 定位**：某个 `Reshape` 的常量 shape 错了、某个 `Slice` 的轴反了，这类**局部**问题用图看最直观；
- **教学 / 汇报**：给人看一张 Llama-3-8B（SDPA 熔合版，~700 节点）的结构图，依然清晰。

结论是：**它可以验证一张已经导出的图，但不能用来理解一个现代 LLM 是怎么算的。**

---

## 六、心智模型的转变

<details>
<summary>一句话总结这十年的范式迁移</summary>

Netron 属于 **静态图编译器时代**（TensorFlow 1.x / Caffe / ONNX-TensorRT freeze）。那时候：

$$\text{Graph} \;=\; \text{Program} \;=\; \text{Execution}$$

现代 LLM 的三层解耦把等号全部打断了：

$$\underbrace{\text{Python 代码}}_{\text{生成图}} \;\to\; \underbrace{\text{可被 JIT 重写的中间图}}_{\text{FX / HOP / Triton IR}} \;\to\; \underbrace{\text{运行时 dispatch + CUDA Graph 的 kernel 流}}_{\text{真正的执行}}$$

MoE 让图**依赖输入**，KV cache / SSM 让图**依赖历史**，MLA 让图**依赖实现选择**，hybrid attention 让图**逐层异构**，量化与融合 kernel 让图**丢失语义**，`torch.compile` 让图**每台机器都不一样**。

所以正确的判断不是"Netron 不好了"，而是：**"图"这个隐喻本身已经不足以描述现代 LLM 的计算了。** 该看的是程序、IR 和运行轨迹。

</details>

# flex_attention 自动生成 Triton 代码 省显存

## 🧑‍💻 User

flex_attention是如何自动生成triton code并达成mem efficiency的

---
## 😀 默认助手 (7)

I'll ground this in the actual PyTorch implementation before writing.

The template path moved. Let me locate it.

# flex_attention 的 Triton 代码生成与显存效率

一句话：`flex_attention` 把用户写的 `score_mod`/`mask_mod` 当作**纯 pointwise 函数**，用 Dynamo 抓成嵌套 FX 子图，再由 Inductor 把子图降成"没有调度、只有点式表达式"的 IR，最后通过 Jinja 模板里的 `{{ modification(...) }}` 钩子把这段 IR **内联**成主 FlashAttention 循环里的 `@triton.jit` 辅助函数；配合 `BlockMask` 的块稀疏元数据（哪些 KV 块完全为空、哪些块完全不需要 mask），实现"不物化 $N\times N$ score、跳过整块、mask 只在必要处求值"的显存/带宽节约。

---

## 1. 生成流水线：从 Python 函数到 Triton 源码

### 1.1 捕获（frontend + Dynamo）

用户写：

```python
def score_mod(score, b, h, q_idx, kv_idx):
    return score + bias[q_idx]          # bias 是闭包捕获张量

flex_attention(q, k, v, score_mod=score_mod, block_mask=bm)
```

`torch/nn/attention/flex_attention.py` 的公开入口最终调用 HOP：

```python
torch.ops.higher_order.flex_attention(
    query, key, value,
    subgraph,                  # score_mod 抓出来的 FX GraphModule
    block_mask.as_tuple(),     # main 上是 17 元组（见 1.4）
    scale, kernel_options,
    score_mod_other_buffers,   # 闭包捕获张量（bias 等）
    mask_mod_other_buffers,
)
```

Dynamo 在 `torch/_dynamo/variables/higher_order_ops.py` 里特判这个 HOP：`score` 用 fake tensor（形状 `[B,H,Q,KV]`）充当，索引 `b,h,m,n` 是符号整数——**整张 $M\times N$ 的 score 在 tracing 阶段只有形状意义，永远不会真的分配**。抓出来的子图输入顺序固定为 `(score, b, h, m, n, *captures)`，这个约定后面在 lowering 里被硬编码复用。

### 1.2 Lowering：把子图降成"可内联的点式表达式"

`@register_lowering(torch.ops.higher_order.flex_attention)` 落在 `torch/_inductor/kernel/flex/flex_attention.py`（2.5~2.8 是 `torch/_inductor/kernel/flex_attention.py`）。核心三行：

```python
placeholder_inps = [create_placeholder(name, dtype, dev) for name, dtype in
                    [("score", q.dtype), ("b", torch.int32), ("h", torch.int32),
                     ("m", torch.int32), ("n", torch.int32)]]
subgraph_buffer = build_subgraph_buffer(placeholder_inps + list(score_mod_other_buffers), subgraph)
freeze_irnodes(subgraph_buffer)
```

而 `build_subgraph_buffer`（`kernel/flex/common.py`）用的是：

```python
pw_subgraph = PointwiseSubgraphLowering(
    graph_module, root_graph_lowering=V.graph,
    allowed_mutations=OrderedSet([torch.ops.flex_lib.zeros_and_scatter.default]),
    additional_lowerings={...: zeros_and_scatter_lowering},
)
with V.set_graph_handler(pw_subgraph):
    pw_subgraph.run(*args)
```

`PointwiseSubgraphLowering`（`torch/_inductor/subgraph_lowering.py`）是一个 `torch.fx.Interpreter`，直接复用**主图**的 lowering 表，但：

- **禁止建 buffer**（`register_buffer` 直接抛 `SubgraphLoweringException`）→ 你的 `score_mod` 里不可能偷偷产生一个中间张量；
- 得到的 `ComputedBuffer` **不做 schedule**，它的 `data` 只是一棵树状的 `ops.load / ops.where / ops.mul / ...` 表达式，等价于"一段待内联进 kernel 的代码"；
- 任何没有 lowering 的算子（例如归约、非点式 op）都会报错，这是 flex_attention 只允许 pointwise 的强制点。

### 1.3 模板渲染：`modification` 钩子把 IR 拼成 Triton

模板本身是 `torch/_inductor/kernel/flex/templates/{flex_attention,common,utilities}.py.jinja`，由

```python
flex_attention_template = TritonTemplate(
    name="flex_attention", grid=flex_attention_grid,
    source=load_flex_template("flex_attention") + load_flex_template("utilities")
           + load_flex_template("common"),
)
```

组装。主循环里真正的"自动生成"发生在这两处：

```jinja
{{ modification(subgraph_number=0, output_name="post_mod_scores",
                score="qk", b="off_z", h="off_h", m="m", n="n", out="qk",
                input_shapes={"score": ("BLOCK_M","BLOCK_N"),
                              "m": ("BLOCK_M","1"), "n": ("1","BLOCK_N")},
                input_dtypes={"b":"index","h":"index","m":"index","n":"index"}) }}

if not IS_FULL_BLOCKS:
    {{ modification(subgraph_number=1, output_name="mask_mod_output", score="qk", ...) }}
    post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
```

`modification(subgraph_number=i)` 取出 `subgraphs=[subgraph_buffer, mask_graph_buffer]` 的第 $i$ 个子图，把它的 `inner_fn` 用 Triton 的 ops 生成器渲染成一个 `@triton.jit` 辅助函数（`score_mod_*` / `mask_mod_*`），实参就是把 tile 变量 `qk / off_z / off_h / offs_m / offs_n` 按 `input_shapes` 广播后传入；`input_dtypes={"b":"index"}` 告诉 codegen 这些是整型索引。同一模板里还有 `{{def_kernel(...)}}`、`{{gen_argdefs()}}`、`{{gen_defines()}}`、`{{store_output(...)}}`、`{{stride(...)}}/{{size(...)}}` 等钩子，负责指针参数签名、把捕获张量提升为额外 kernel 参数、以及输出 store。

生成出来的核心数学（`common.py.jinja::forward_block_mn`，略去边界处理）：

```python
qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION)
if not PRESCALE_QK: qk *= SM_SCALE
# ... score_mod 内联展开（读 bias / 相对位置 / where ...）
# ... mask_mod 内联展开，然后 where(mask, s, -inf)
if not PRESCALE_QK: post_mod_scores *= RCP_LN2      # 换成 exp2，快且少一次乘法
m_ij  = tl.maximum(m_i, tl.max(post_mod_scores, 1))
alpha = tl.math.exp2(m_i - m_ij_masked)
p     = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])
l_i   = l_i * alpha + tl.sum(p, 1)
acc   = acc * alpha[:, None]
acc   = tl.dot(p.to(MATMUL_PRECISION), v.to(q.dtype), acc, ...)
m_i   = m_ij
```

### 1.4 autotune 与特化

```python
configs = V.choices.get_flex_attention_fwd_configs(head_dim, seq_len_q, dtype, dev)
for conf in configs:
    flex_attention_template.maybe_append_choice(
        choices=choices,
        input_nodes=[query, key, value, logsumexp, max_scores,
                     kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices],
        layout=layout, subgraphs=[subgraph_buffer, mask_graph_buffer],
        mutated_inputs=[logsumexp, max_scores], call_sizes=query.get_size(),
        **cur_kernel_options)
input_gen_fns = {5: create_num_blocks_fake_generator(kv_indices), 6: create_indices_fake, ...}
out, _ = autotune_select_algorithm("flex_attention", choices, ..., input_gen_fns=input_gen_fns)
```

每个 config 渲染出**一份不同源码**（`BLOCK_M/BLOCK_N/num_warps/num_stages/USE_TMA` 全是 constexpr），再实机 benchmark 选最快。`input_gen_fns` 用"看起来合理"的伪造稀疏索引/块计数来构造 benchmark 输入——因为真实 `BlockMask` 的稀疏张量形状受 symint 约束，直接 materialize 代价高。

关键特化开关（都由 lowering 自动推断，也是"同一个模板生成一堆高度专用 kernel"的来源）：

| 选项 | 作用 |
|---|---|
| `IS_DIVISIBLE` | 序列长度能被 block 整除 → 完全省掉边界 masking 分支 |
| `SAFE_HEAD_DIM` / `QK_HEAD_DIM_ROUNDED` | head_dim 非 2 的幂时启用 `load_checked_2d` 的 masked load |
| `PRESCALE_QK` | 把 $1/\sqrt{d}$ 折进 Q，省掉每块 $BLOCK_M\times BLOCK_N$ 的乘法（数值误差略大） |
| `ROWS_GUARANTEED_SAFE` | 保证每行至少一个有效元素 → 跳过 `-inf` 行检查 |
| `BLOCKS_ARE_CONTIGUOUS` | 块掩码连续 → 用 `tl.multiple_of` 递推指针，不做间接跳转 |
| `USE_TMA` | `tl.make_tensor_descriptor` + `load_tensor_descriptor`，异步/少地址计算/低寄存器压力 |

---

## 2. 显存效率从哪来

$$ \text{eager: } \underbrace{O(BHMN)}_{\text{score 矩阵}} + \underbrace{O(BHMN)}_{\text{P/softmax 中间}} \quad\longrightarrow\quad \text{flex: } O(BHMD)+O(BHND)+O(BHM) $$

<details>
<summary><b>展开：6 条具体机制（含代码证据）</b></summary>

**① 块稀疏：整块为空的 KV 块根本不进循环**

`BlockMask` 只把 `mask_mod` 在 **block 粒度**（默认 128×128，自适应）上求值一次，得到每个 query 行"非空的 KV 块索引表"：

```python
kv_indices = KV_IDX + sparse_kv_idx_offset
kv_start   = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE
kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE,
                         tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
acc, l_i, m_i = forward_inner(..., 0, block_n_end, ..., IS_FULL_BLOCKS=False)
```

循环上界由 `kv_num_blocks` 决定：全空块既不 load K/V、也不做 dot、也不求 mask。因果掩码下算力/带宽从 $O(N^2)$ 降到 $O(N^2/2)$，滑动窗口/文档掩码能降到近线性。元数据本身很小（`[B,H,M/128,max_blocks]` int32），且在 L2 里被同一批 head 复用。

**② full / partial 块分离：mask 只在"对角线附近"求值**

模板里有两段循环，先跑 partial 块再跑 full 块：

```jinja
# ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~  (IS_FULL_BLOCKS=False → 会用 mask_mod)
# ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~  (IS_FULL_BLOCKS=True  → 不生成 mask_mod 代码)
```

`IS_FULL_BLOCKS` 是 constexpr，所以 full 分支里**连 `mask_mod` 的 Triton 代码都不会生成**（`if not IS_FULL_BLOCKS` 在编译期被消掉）。因果掩码下 partial 块只有 $O(N/block)$ 量级，mask 求值开销从 $O(N^2)$ 降到 $O(N\cdot block)$。

**③ online softmax：结构上不物化 score**

`acc[m_i, l_i, m_i] = forward_block_mn(...)` 的累加器只有 `acc[BLOCK_M, V_HEAD_DIM]`、`l_i[BLOCK_M]`、`m_i[BLOCK_M]`，全在寄存器里做 rescale（`alpha = exp2(m_i - m_ij)`）。前向只为反向多写两个 $\mathbb{R}^{B\times H\times M}$ 的小张量 `LSE`、`MAX`（强制 fp32），而不是 $\mathbb{R}^{B\times H\times M\times N}$ 的概率矩阵。

**④ score_mod/mask_mod 内联：零额外 kernel、零显存往返**

bias、相对位置、`where` 分支全部变成 tile 上的寄存器/SMEM 运算，没有 `[B,H,M,N]` bias 物化，也没有 kernel launch。捕获张量以指针参数透传（`gen_argdefs`），索引 load 在 tile 上做；代价是同一个 bias 会被多个 Q tile 重复 load（靠 L2 命中），这是拿带宽换显存。

**⑤ tile 与 header 的"多对一"设计降低元数据与寄存器压力**

- `SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M`，多个 Q tile 共享同一行稀疏元数据（`q_start // SPARSE_Q_MULTIPLE` 索引），元数据 load 次数减少；
- `SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N`，用窄 tile 迭代宽稀疏块，`kv_offset` 递推、`BLOCKS_ARE_CONTIGUOUS` 时避免逐块 `tl.load` 索引；
- 静态断言 `SPARSE_*_BLOCK_SIZE % BLOCK_* == 0`，并且只有单个 config 时把 `BLOCK_M/N` 裁到 ≤ 稀疏块，保证 TileLang 式幂次块假设成立。

**⑥ 输出与布局最小化**

输出 layout 由 query 的 stride 反推（`infer_dense_strides`）并强制 `last dim stride == 1` 以便向量化 store；GQA 用 `GQA_SHARED_HEADS` 在 kernel 内做 head 映射，KV 在多个 Q head 间复用（L2），不复制 K/V；batch 广播（`ZKV == 1`）同样零拷贝。

</details>

---

## 3. 代价与限制

- **只允许 pointwise**：`score_mod` 里不能有跨 tile 的归约、不能有数据依赖的 Python `if`（要用 `tl.where`）、不能用会产生 buffer 的 op；否则子图 lowering 抛 `SubgraphLoweringException`。
- **捕获张量会被重复读**：B 方向的 broadcast load 靠 L2，稀疏/不规则 gather 可能成为瓶颈。
- **反向的额外机制**：`flex_attention_backward` 用 `fw_graph + joint_graph(dscore) + mask_graph` 三个子图，`delta = sum(out * grad_out)`（有 `grad_logsumexp` 时再减 `grad_logsumexp * log2e`）；对捕获张量的梯度用 `torch.ops.flex_lib.zeros_and_scatter` → `scatter_mode="atomic_add"` 在同一个 kernel 里累加（`allowed_mutations` 白名单），再 cast 回原 dtype。
- **多索引捕获的限制**：同一个张量被索引两次（`bias[q] + bias[kv]`）在反向不支持，需 `clone()`（见 lowering 里的 `validate_joint_graph`）。

<details>
<summary><b>文件地图 / 其他后端</b></summary>

| 阶段 | main | ≤2.8 |
|---|---|---|
| 公开 API、`BlockMask`、`_create_sparse_block_mask` | `torch/nn/attention/flex_attention.py` | 同 |
| HOP 定义 / Dynamo 变量 | `torch/_higher_order_ops/flex_attention.py`、`torch/_dynamo/variables/higher_order_ops.py` | 同 |
| HOP lowering | `torch/_inductor/kernel/flex/flex_attention.py` | `torch/_inductor/kernel/flex_attention.py` |
| 子图降级 | `torch/_inductor/subgraph_lowering.py`、`kernel/flex/common.py` | `subgraph_lowering.py` |
| Triton 模板 | `torch/_inductor/kernel/flex/templates/*.py.jinja` | 内嵌在 `flex_attention.py` |
| 解码专用 kernel | `kernel/flex/flex_decoding.py` | `kernel/flex_decoding.py` |
| 配置候选 | `torch/_inductor/heuristics/template/triton.py` | `torch/_inductor/template_heuristics/flex_attention.py` |

main 上 lowering 入口现在按后端分派：Triton 模板、Triton decode（短 Q 的 split-KV）、CuteDSL/Flash（`flex_flash_attention.py`，可对 trivial score/mask 子图直接走 flash-attn 路径）、以及 CPU（CPP）与 MPS（Metal）专用模板。

前向 grid 为 $(\lceil M/B_M\rceil, B, H_q)$，反向 grid 为 $(\lceil M/B_{M2}\rceil \cdot (H_q/H_{kv}) + \lceil N/B_{N1}\rceil, B, H_{kv})$——反向把"算 K/V 的 dK/dV"和"算 dQ"拼在同一个 kernel 里，避免单独一趟 reduce。

</details>

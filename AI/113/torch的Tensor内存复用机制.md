# torch的Tensor内存复用机制

### 🧑‍💻 User

我记得torch有一个tensor碎片复用的机制，通过复用已经释放的tensor地址来减少新tensor malloc，所以tensor用完不会瞬间gc，会有一个生命周期，这套机制是怎么work的？复用buffer，只有在shape更大的时候扩增吗？

---

### 🤖 Assistant

# PyTorch 的 Tensor 内存复用机制（Caching Allocator）

你说的这套机制叫 **Caching Allocator（缓存分配器）**，最典型的是 CUDA 版本（`CUDACachingAllocator`，源码在 `c10/cuda/CUDACachingAllocator.cpp`），CPU 侧也有一个类似实现（`CPUAllocator`，基于 `posix_memalign`）。下面以 CUDA 版为准讲原理。

## 1. 核心数据结构：Segment / Block / Size Class

分配器不按"tensor 的 shape"管理内存，只看**字节数 + 对齐 + 设备**。它维护三层结构：

- **Segment**：通过 `cudaMalloc` 从驱动拿到的大块内存（一次拿一大片）。
- **Block**：Segment 内部被切成的块，状态是 `allocated`（在用）或 `free`（空闲），每个 Block 有 `prev/next` 指针，支持**切分（split）**和**合并（merge/coalesce）**。
- **Size Class（大小类）**：请求的大小会被**向上取整**到一个粒度。默认小分配（<1MB）取整到 512B 的倍数，大分配（≥1MB）取整到 2MB 的倍数。所以请求 100B 和 511B 会落到同一个 512B 的块上。

Free block 被放进按 size 排序的 free list（`BlockPool`）里，等待复用。

## 2. 分配流程（什么时候复用、什么时候真正 malloc）

```text
allocate(request_size):
    size = round_up(request_size)          # 取整到 size class
    block = free_list.find(size)           # 先找"同大小类"的空闲块
    if block: 直接复用, return

    block = free_list.find_any(>= size)    # 再找"更大"的空闲块
    if block: split(block, size)           # 切成两块：size 拿去用，剩余塞回 free list
              return

    segment = cudaMalloc(big_chunk)        # 缓存里没有够大的块 → 才真正向驱动申请
    return carve(segment, size)
```

关键点：**复用是"按字节大小"做的，和 shape/dtype 无关**。同一个缓存块这次可能是 `(512, 512)` 的 float tensor，下次可能是 `(65536,)` 的 int8 tensor，只要字节数对得上就行。

## 3. 释放流程（为什么不会瞬间 GC）

```text
free(block):
    block.state = free                      # 只改状态，不还给驱动
    if next 或 prev 也是 free 且同 stream:
        merge(block, neighbor)              # 相邻空闲块合并成更大的块
    free_list.insert(block)                 # 放回缓存池
```

所以 tensor 析构后，内存只是回到缓存池，**并没有 `cudaFree`**。只有当：

- 显式调用 `torch.cuda.empty_cache()`，
- 或分配器销毁/进程退出，
- 或触发了 GC 阈值（见第 5 节），

才会真正把 Segment 还给驱动。

## 4. 直接回答你的问题：只在 shape 更大时扩增吗？

**不完全是。** 更准确的说法是：

> 只有当"缓存池里没有任何一个空闲块 ≥ 请求大小（取整后）"时，才会真正 `cudaMalloc` 扩增。

三种情况：

| 新请求 vs 空闲块 | 行为 |
|---|---|
| 正好同大小类有空闲块 | 直接复用，不 split 不 malloc |
| 有空闲块但更大 | **split**：切出需要的大小，剩余部分继续留在缓存。即**更小的 tensor 可以复用更大的缓存块** |
| 所有空闲块都不够大 | 才 `cudaMalloc` 拿新 Segment（扩增） |

所以"扩增"的触发条件不是"shape 变大"，而是"**缓存中找不到足够大的块**"。反过来，即使新 tensor 比旧的大，只要缓存里躺着更大的空闲块，也不会 malloc。

## 5. Stream 语义（生命周期里最容易被忽略的一环）

每个 Block 都绑定一个 stream。释放时会在这个 stream 上**记录一个 event**，表示"这块内存安全可复用"。

- 同一 stream 上释放的块，后续分配可以直接复用。
- **不同 stream** 想复用这个块时，分配器会让新 stream **先等待那个 event**，保证旧 kernel 已经跑完，避免 data race。

这就是为什么在多 stream 场景下要 `tensor.record_stream(s)` 告诉分配器"这个 tensor 实际是在 s 上用完的"，否则可能过早被复用导致数据损坏。

## 6. 观测与配置

```python
torch.cuda.memory_snapshot()   # 看 segments / blocks 的 allocated / free 状态
torch.cuda.memory_stats()      # 看 allocated_bytes / reserved_bytes / 复用情况
torch.cuda.empty_cache()       # 强制把所有空闲块还给驱动
```

环境变量 `PYTORCH_CUDA_ALLOC_CONF` 常用项：

<details>
<summary>点击展开配置项说明</summary>

| 配置项 | 作用 |
|---|---|
| `max_split_size_mb` | 允许 split 的最大块大小。现代版本默认"无限制"（即任何大块都能切）。历史上大块默认不切，导致碎片化 OOM，这个参数就是当时引入的修复手段 |
| `roundup_power2_divisions` | 小分配是否取整到 2 的幂 |
| `garbage_collection_threshold` | 空闲缓存占总内存的比例超过该值就自动触发回收，默认 0（关闭，只靠 `empty_cache`） |
| `expandable_segments` | 用 `cudaMallocAsync`/虚拟内存做可伸缩 Segment，可真正释放未用的物理页，减少碎片 |

</details>

**一句话总结**：这是一套"**从驱动拿大块 → 内部按字节大小类切块 → 释放只回池不还驱动 → 复用优先、不够大才 split、split 也不行才真正 malloc**"的缓存分配器。shape 只是决定最终字节数，复用与否完全由字节数和空闲池决定。

补充一点：CPU 侧，即使没走 PyTorch 自己的 `CPUAllocator`，glibc 的 `malloc/free` 本身也是缓存分配器（释放的内存不会立刻还给内核），所以"tensor 用完不瞬间 GC"在 CPU 上同样成立，只是它不受 `torch.cuda.empty_cache()` 控制。

---

### 🧑‍💻 User

是的，因为小变大是没法利用碎片的，大变小甚至这个碎片可以复用好几次，然后复用还有个好处就是好做cudagraph（cudagraph要求就是复用），但是训练的话可能不好做，训练要存全状态tensor，这个地方要怎么去刻意优化？

---

### 🤖 Assistant

你抓住了关键矛盾：**allocator 的复用对"瞬态"内存很有效，但训练里最占内存的恰恰是"常驻"状态（参数/梯度/优化器状态/激活），它们生命周期=整个训练，根本不进缓存池**。所以训练内存优化的思路就两条腿：

1. **把常驻态砍小**（优化器状态、激活、梯度）；
2. **把瞬态复用"确定化"**，让每次迭代地址稳定——这正好也是 CUDA graph 能 capture 的前提。

下面拆开讲。

## 一、先分清：常驻 vs 瞬态

| 类型 | 内容 | 能否被 allocator 复用 |
|---|---|---|
| 常驻态 | 参数、梯度、Adam 的 m/v、master weight、backward 要用的激活 | 不能（一直占着） |
| 瞬态 | 前向中间结果、autograd saved tensors、临时 buffer | 能（释放后回池） |

所以"刻意优化"的重点，**第一刀砍常驻态，第二刀让瞬态复用确定化**。

## 二、砍常驻态（大头在优化器状态）

Adam 是隐形的内存大户：`fp32 参数 + fp32 master weight + m + v` ≈ **参数的 3~4 倍**。几个直接手段：

- **8-bit 优化器**（bitsandbytes 的 `AdamW8bit`）：m/v 用 int8 存，优化器状态省约 75%。
- **bf16 训练 + fp32 master weight**：激活和梯度用 bf16，只有 master weight 保持 fp32。
- **ZeRO 分片**（DeepSpeed / 原生 FSDP）：
  - Stage 1：分片优化器状态（每卡只存 `1/N` 的 m/v）
  - Stage 2：再分片梯度
  - Stage 3：连参数也分片，**每层用时 gather 出完整副本，用完立即释放回池，下一层复用同一块 buffer**。

第三点非常关键——**FSDP 的本质就是把"常驻的参数"变成"瞬态的临时副本"，主动制造复用**。这正好回应你说的"训练要存全状态 tensor 不好优化"：不需要每张卡都常驻全量状态，按层 gather/释放，让缓存池去复用。

激活侧：

- **Gradient checkpointing**（`torch.utils.checkpoint`）：故意不存中间激活，backward 时重算，用算力换内存——这是"复用"的反面，但很有效。
- **FlashAttention 等 fused kernel**：不物化完整 `QK^T` 矩阵，激活峰值大幅下降。
- FSDP 还能把 checkpointing 和参数释放联动（`activation_checkpointing` 时连 gathered 参数一起释放）。

梯度侧：

```python
# 默认：grad buffer 常驻，每 step 原地 memset 清零
optimizer.zero_grad()

# 优化：把 grad tensor 释放回缓存池（省一次 memset，eval 阶段内存可被复用）
optimizer.zero_grad(set_to_none=True)
```

注意梯度累积时只在**累积周期结束**时 `set_to_none=True`，中间 micro-batch 不清。

## 三、让瞬态复用"确定化"——这是 CUDA graph 的地基

你观察到"复用对 cudagraph 友好"，反过来说也成立：**CUDA graph 要求地址稳定，而地址稳定只能靠"分配序列完全确定"来保证**。做法：

- **固定 shape / 固定 batch size**：每层激活的字节数、分配顺序每轮一致，allocator 每次都命中同一个缓存块。
- **预分配 workspace**：`buf = torch.empty(size)` 一次，之后 `buf.copy_()` / 直接写，别每轮 `torch.empty`。
- **in-place / `out=`**：减少新 tensor 的产生，让块只复用不新建。

CUDA graph 还有个专门机制配合：capture 期间所有分配进**私有 graph memory pool**（`torch.cuda.graph_pool_handle()`），replay 之间这个池**不释放**，所以地址天然固定，且不会和普通 tensor 的分配互相踩。

## 四、训练里落地 CUDA graph 的正确姿势

难点是 backward 动态、有 lazy init、有 CPU sync。解法是 **warmup 后把 `forward + backward + optimizer.step()` 整段 capture 成图**，之后每轮只 `copy_` 输入 + `replay()`。

<details>
<summary>点击展开：完整 capture 训练循环示例</summary>

```python
import torch

# 前提：静态 shape、AdamW(capturable=True)
model = ...                          # 固定输入形状
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, capturable=True)

static_x = torch.randn(B, ...).cuda()   # 静态输入 buffer，地址永远不变
static_y = torch.randn(B, ...).cuda()

# 1) warmup：触发 cuDNN 选算法、lazy init、autotune
for _ in range(5):
    out = model(static_x)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# 2) capture：forward + backward + step 整段进图
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    out = model(static_x)
    out.sum().backward()
    optimizer.step()

# 3) 训练循环：只换数据，不产生新分配
for x, y in dataloader:
    static_x.copy_(x)
    static_y.copy_(y)
    g.replay()
    optimizer.zero_grad(set_to_none=True)
```

要点：

- `zero_grad` 在**图外**做；`step` 在图内原地更新参数，所以参数地址不变。
- `capturable=True` 让 AdamW 用无 CPU sync 的实现，否则 capture 会失败。
- 输入必须 `copy_` 进静态 buffer，不能直接给新 tensor（会破坏地址稳定性）。

**省力方案**：`torch.compile(model, mode="reduce-overhead")` 会自动给编译产物套 CUDA graph，训练时直接用即可，不用手写上面这套。

</details>

## 五、检查清单

<details>
<summary>点击展开：训练内存优化 checklist</summary>

1. **优化器状态**：8-bit / bf16 / ZeRO-1,2,3（或 FSDP）—— 这是省最多的地方。
2. **激活**：gradient checkpointing、FlashAttention、`torch.no_grad()`+`inference_mode` 处理 eval。
3. **梯度**：`zero_grad(set_to_none=True)`；累积时只在周期末清。
4. **确定性**：固定 shape/batch size，预分配 workspace，多用 in-place / `out=`。
5. **CUDA graph**：warmup → `capturable=True` → capture step → `copy_` + `replay`；或直接 `torch.compile(mode="reduce-overhead")`。
6. **极端手段**：优化器状态 CPU offload、激活 offload（DeepSpeed offload / FSDP `cpu_offload`）。

</details>

**一句话总结**：allocator 的复用解决的是"瞬态"内存的 malloc 开销和碎片问题；训练真正要刻意优化的，是把**常驻状态变少/变瞬态**（8-bit、ZeRO 分片、checkpointing、`set_to_none`），再把剩余瞬态分配**确定化**（静态 shape + 预分配 + 图内 replay），这样既省内存又天然满足 CUDA graph 的地址稳定要求。

---

### 🧑‍💻 User

还有一个凹点就是尽量用inplace计算，因为torch内部是不存在fuse的，你在不开一些compile之类东西的情况下最好还是inplace，但是这个也要看实际情况
比方说
(B N) = (B N) + (1 N) + (1 N)
这种情况，你最好给后面两个加个括号
这种你可以写成
(B N) += (1 N)
(B N) += (1 N)
但是显然存在两次B N的计算，但是前面不用inplace的话只存在一次B N和一次1 N的计算
如果你的B很大的话会浪费不少算力
一些框架会把这个计算融合成
BroadcastAddBroadcastAdd

所以计算优先顺序也是一个凹点
当然如果是fuse的情况，那就不用考虑这个问题了
但是大多数框架都是没法做这种fuse的
binary unary shape操作 都是可以fuse的
能不能讲解下这个凹点具体怎么实践？

---

### 🤖 Assistant

你戳到了 eager 模式下最经典的"凹点"。先把本质点破：这里其实是**两根独立的轴**在打架，很多人（包括直觉上）会把它们混为一谈——

1. **大张量的 pass 数**（决定访存带宽，`B×N` 越大越疼）；
2. **临时 tensor 的分配**（决定峰值内存）。

`inplace` 只解决第 2 根轴；而你这个例子的真正大头是第 1 根轴。再加上前面聊过的：**PyTorch 有缓存池，临时分配本来就很便宜**，所以为了省临时去 inplace、反而多一次 pass，通常是亏的。

## 一、四种写法的精确对比

设 `x:(B,N)` 大，`a,b:(1,N)` 小：

| 写法 | 大张量读写 | 临时分配 | kernel 数 |
|---|---|---|---|
| `x += a; x += b` | 4·B·N（2 读 2 写） | 0 | 2 |
| `x = (x + a) + b`（默认左结合） | 4·B·N | 2 个 (B,N) | 2 |
| `x = x + (a + b)`（**先算小项**） | **2·B·N** | 1 个 (B,N) | 2 |
| `x = x + a + b` + `torch.compile` | 2·B·N | 0 | **1** |

结论很反直觉：**inplace 链（`+=` 两次）并没有省带宽，只是省了临时**，反而把大张量读写翻倍到 4·B·N。真正省算力的是"重结合"——先把 `(1,N)` 的小项合并（`a+b` 只花 2·N 的成本），再对大张量做**一次**广播加。

## 二、实践规则

### 规则 1：先小后大（eager 下手动重结合）

决策流程：

```python
# 有小项（标量 / (1,N) / (C,1,1)）→ 先合并小项，只对大张量做一次 pass
x = x + (a + b)          # 2·B·N，而不是 4·B·N

# 想连临时也省掉 → 用 out= 原地写回（一个 pass + 复用 x 的 buffer）
x = torch.add(x, a + b, out=x)   # 注意 shape/dtype 必须一致

# 所有操作数都是大张量 → inplace 链反而是合理的（pass 数不变，省临时）
x += a; x += b           # a、b 也是 (B,N) 时，这个不亏
```

核心判断就一句：**数一下"大张量被完整读写了多少次"，让它最少。**

### 规则 2：优先用框架自带的 fused 算子（不用 compile 也 fused）

eager 模式下每个 `+`/`*`/`relu` 都是独立 kernel，但 PyTorch 已经把很多常见组合做成**单个 kernel**了，别手写拆开：

| 手写反模式 | fused 替代 |
|---|---|
| `x * sigmoid(x)` | `F.silu(x)` |
| `x * 0.5 * (1 + erf(x/√2))` | `F.gelu(x)` |
| `input + v * t1 * t2` | `torch.addcmul(input, t1, t2, value=v)` |
| `input + v * t1 / t2` | `torch.addcdiv(...)` |
| `start + (end - start) * w` | `torch.lerp(start, end, w)` |
| `x @ W.t() + b` | `F.linear(x, W, b)`（addmm） |
| `x.var()` + `x.mean()` | `torch.var_mean(x)`（一次 pass） |
| 手写 mean/var/norm | `nn.LayerNorm`、`nn.BatchNorm`（内部 fused） |
| 手写 QK^T/softmax/V | `F.scaled_dot_product_attention`（FlashAttention） |

`log_softmax` + `nll_loss` 这种，`F.cross_entropy` 也已经融合好了。

### 规则 3：inplace 的正确边界

inplace 只在这两种情况划算：

- **单算子、旧值不再需要**：`x = F.relu(x)` → `x.relu_()`（pass 数不变，省一个临时，缓存更友好）；
- **残差**：`x = x + residual` → `x += residual`。

但要注意 **autograd 的版本计数器**：原地修改一个 backward 还会用到的 tensor，会直接报 `modified by an inplace operation`，或者静默算错梯度。所以 inplace 尽量用在 `no_grad` 里、或明确知道下游不再依赖旧值的地方。

### 规则 4：通用融合交给编译器

你提到的 `BroadcastAddBroadcastAdd` 这种"任意 binary/unary/shape 组合成一个 kernel"的通用融合，eager 是做不到的，只能：

```python
model = torch.compile(model, mode="reduce-overhead")  # 或 max-autotune
```

Inductor 会把 `x = x + a + b` 整条 pointwise 链生成**一个** Triton kernel：读 `x/a/b` 各一次、写 `x` 一次（2·B·N + 2·N），而且**不管你怎么加括号**都融成一个 pass——这正是你说的 `BroadcastAddBroadcastAdd`。开了 compile 之后你写最自然的 `x = x + a + b` 就行，不用手动重结合。

## 三、两个必须知道的坑

<details>
<summary>点击展开：浮点结合律 & autograd 风险</summary>

1. **浮点加法不满足结合律**：`(x + a) + b` 和 `x + (a + b)` 结果**不 bitwise 相等**（最后一两个 ulp 会差）。手动加括号 = 悄悄改了数值。训练里通常无所谓，但如果你在做严格复现/对拍，要意识到这点。

2. **inplace 对 autograd 不是免费的**：`x += a` 会 bump `x` 的版本号，如果 backward 还需要旧 `x`，要么报错要么要额外存一份旧值（内存反而可能变多）。所以 inplace 的"省内存"是有条件的。

3. **shape 操作本来就免费**：`view/reshape/transpose/permute/broadcast` 只是改 stride 元数据，不产生 kernel 也不拷贝，所以它们"天然可 fuse"，你随便 `.view()` 不会付出代价，真正的成本都集中在 binary/unary 的 pass 上。

</details>

## 四、一句话总结

> eager 模式下优化 elementwise：**先合并小项 → 减少大张量 pass 数 → 用现成 fused 算子 → 单算子才考虑 inplace → 复杂链直接 `torch.compile`**。

判断标准永远只有一句：**大张量被完整读写了多少次？** 让它最小，临时分配交给缓存池兜底。这也和前面 allocator 的话题闭环了——正因为临时 tensor 会被缓存池复用、`cudaMalloc` 几乎不发生，所以"省临时"这件事在 PyTorch 里优先级远低于"省 pass"。
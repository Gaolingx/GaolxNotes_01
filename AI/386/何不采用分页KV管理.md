# 何不采用分页KV管理

### 🧑‍💻 User

众所周知，llama.cpp的多session是走slot机制的，也就是说并发数是下一个batch的并发数，然后走完一个batch走下一个batch，他的kv cache是unified，也就是说等于是不同请求的kv上下文扔到一个池子里，然后下一次推理从池子里面捞，所以他存多少kv cache是看你的-ctx开多少，如果session>空闲slot则超出的session会阻塞等待。也就是说N个并发vram是需要1份权重和N份kv cache的，即使slot空闲，且slot占用的显存都是预分配的，slot数量不能动态扩容，在生产环境中这会导致严重的显存碎片和浪费，为什么要这样设计？

---

### 🤖 Assistant

我先核实一下 llama.cpp 当前的实现细节（KV cache 池、slot 机制、相关选项），确保回答准确。

README 拿到了，server 文档路径变了，再抓一下新的位置。

文档基本拿到了，关键选项都确认了。再抓一下核心源码确认 KV 池大小的计算逻辑（unified 模式下总池到底是 `n_ctx` 还是 `n_ctx × n_parallel`）。

源码逻辑基本确认了（unified 时 `n_ctx_seq = n_ctx`，非 unified 时 `n_ctx / n_seq_max`）。再确认一下 server 端 slot 的 per-slot context 上限是怎么定的。

server 端 slot 的上下文预算逻辑在 `server-context.cpp` 里，抓一下确认。

先说我查到的结论：**你这个批评里有一半是基于旧版设计的印象，另一半是对的，而且"为什么这么设计"其实有非常明确的工程理由**。我拉了最新的源码和文档核实过，下面分四部分讲清楚。

---

## 0. 先纠正一个关键前提：unified 下并不是 "N 份 KV cache"

你描述的是 llama.cpp **很久以前**（每个 session 一个独立 `llama_context`、各自带一份 KV）的设计。unified KV cache 恰恰是为了消灭这个浪费而引入的。现在的实现是：

- **整个 KV 是一个共享池**，总 cell 数 = `n_ctx`（你 `-c` 开的那个值），跟 `-np`（slot 数）无关；
- 池子按 layer 组织，每个 layer 一块连续的 K、V tensor；任何 session 的任意 token 可以落在池子里任意空闲 cell；
- slot 不是一个内存分区，它只是一个 **seq_id + 会话状态容器**（prompt 缓存、采样参数、生成的 token）。

源码证据（`src/llama-context.cpp`）：

```cpp
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
if (cparams.kv_unified) {
    cparams.n_ctx_seq = cparams.n_ctx;            // unified：单序列理论上可用整个池
} else {
    cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;  // 非 unified：每序列固定切片
}
```

以及 `decode()` 里：

```cpp
const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;
```

server 文档也印证了池的大小逻辑：`--kv-unified-per-slot N` 的说明写着 *"when set without -c/--ctx-size, the shared KV pool is sized to n_parallel*N"*——反过来说，**不设这个 flag 时共享池就按 `-c` 一个值来定，而不是 N 份**。

所以显存账目（unified 默认情况）是：

$$M_{KV} = n_{ctx} \times n_{layer} \times \left(2 \times n_{kv\_head} \times d_{head}\right) \times b \qquad (b=\text{每元素字节数，f16=2})$$

**一份权重 + 一个共享 KV 池**，N 个并发会话只是往这个池子里各占一些 cell。你观察到的"N 份 KV"只在 `--no-kv-unified`（旧的非统一模式，每 slot 固定分 `n_ctx/n_parallel` 个 cell）或更老的每上下文一份 KV 的设计里成立。

---

## 1. 这个设计到底想解决什么（历史视角）

llama.cpp 的演进顺序是：

1. 最早是**单请求 CLI**：一个 `llama_context` = 一个会话 + 一份 KV，起多个上下文跑多路就是你说的"N 份 KV"。
2. 后来加了多 sequence 支持，把 KV 并成一个 **unified buffer**（这是针对"N 份 KV 浪费"的优化）：任意 token 放任意 cell，用 `find_slot()` 在环形 buffer 里找空闲 cell（`src/llama-kv-cache.h` 里 `v_heads` 注释写得很清楚：*"the current index from where we start searching for a free slot in the ring buffer of KV cells"*）。
3. server 在它上面加了 **slot 机制**：一个 session 对应一个 slot，slot 复用同一份 KV 池，这样多轮对话的系统提示词才能留在 cache 里，实现 prompt caching。

也就是说：**slot 机制本身就是"在单上下文引擎上加多用户支持"的最省事方案**——它没有重写推理核心，只是给 seq_id 包了一层会话外壳。这才是它看起来"土"的真正原因。

---

## 2. 为什么是"预分配 + 固定数量 + 排队"（核心理由）

### 2.1 确定性内存预算，运行时永不 OOM

这是最重要的一条。整个 KV 池在**启动时一次性分配一大块连续显存**，因此：

- 启动时就知道"装得下 / 装不下"，装不下直接报错，绝不运行到一半爆显存；
- 运行期零分配、零碎片——一个大块连续 allocation 恰恰是**反碎片**的（VRAM 碎片化来自大量小块的反复 alloc/free，这个设计里根本没有）；
- 对本地用户（笔记本、树莓派、Mac）这是刚需：你要的是一个"设好 `-c` 就能跑、跑不跑得动一眼可知"的工具，而不是一个需要调优的 serving 引擎。

### 2.2 跨后端可移植性

llama.cpp 要跑在 CPU / Metal / CUDA / Vulkan / SYCL / OpenCL / RISC-V 等十几个后端上。**"一层 layer 一块连续 K/V tensor，attention kernel 直接按 cell 索引切片"是最简单、所有后端都能高效实现的形式**。vLLM 式的 paged KV（虚拟内存式分页）需要每个后端写一套自定义 attention kernel、维护页表、处理重映射——这是几千行 CUDA 专属代码才能换来的，和 llama.cpp 的"零依赖、到处能跑"目标完全冲突。

### 2.3 内核与调度器都假设固定上界

- 计算图在初始化时按最坏情况 `n_seq_max` 个 sequence 预留（`sched_reserve()`，日志里 `worst-case: n_tokens = ..., n_seqs = ...`）；
- 输出缓冲、采样缓冲按 `n_seq_max` 分配；
- 调度循环就是"收集所有可运行 slot → 拼一个 batch → decode 一步"。

**slot 数 = 并发上界，它同时是图的预留上界。** 动态增删 slot 意味着运行期重协商 GPU 缓冲（昂贵、可能失败），还要把已占用的 KV cell 挪来挪去——这正是 llama.cpp 刻意避免的"运行期不确定性"。

### 2.4 slot 的定位：它不是显存单元，是"会话身份 + 并发预算"

一个 slot 本质上是：

- 一个 `seq_id`（对应 KV 池里的 cell 集合）；
- 一份会话状态（prompt 缓存、采样器、生成进度）；
- 一个**API 可见的实体**（`id_slot`、`/slots` 端点、slot save/restore）。

它**不是一块内存**。你 `-np 4` 并不会把 KV 池乘 4，只会把"最大并发序列数"设为 4，并相应预留 compute/output 缓冲。

---

## 3. 你指出的三个问题：哪些对，为什么，现状如何

| 你的观察 | 准确性 | 说明 |
|---|---|---|
| 池按 `-c` 预分配，空闲也占着 | ✅ 对 | 这是**预留（reservation）**，不是碎片。代价是"池子空转也付钱" |
| slot 数量固定、不能动态扩容 | ✅ 对 | 见 2.3：图、缓冲、API 都以固定 slot 数为前提 |
| 并发 > 空闲 slot 会阻塞等待 | ✅ 对 | `server-context.cpp` 里 `get_available_slot()` 找不到空闲 slot 就 `queue_tasks.defer(...)`，任务排队 |
| N 并发 = N 份 KV cache | ❌ 错（新设计） | unified 下是 1 个共享池；你描述的是 2023 年之前的旧设计 |
| "严重的显存碎片" | ❌ 用词不准 | 一个大连续池恰恰是**反碎片**的；真正的浪费是**超量预留**（reservation），以及 slot 满时即使池里有空闲 cell 也会排队（slot 是并发上限，不是内存上限） |

还有一个你没提到但真实存在的约束：**KV 池容量本身也会卡请求**。当 batch 塞不进池子时，`decode()` 会走 `try_clear_idle_slots()` 先清掉空闲 slot 腾 cell，不行就把 batch 减半重试，再不行才报 "Context size has been exceeded"（`server-context.cpp` 里这段逻辑是显式写着的）。所以实际有**两个**排队/失败来源：slot 数（并发上界）和池容量（token 上界）。

---

## 4. 生产环境怎么缓解（这些都是现成开关，不是饼）

| 手段 | 说明 |
|---|---|
| `--cache-idle-slots` + `--cache-ram N` | **直击你吐槽的点**：新任务进来时，把空闲 slot 的 KV 存入 RAM prompt cache，unified 模式下**清空它的 cell**，池子立刻被复用（`process_single_task` 里那段 `if (params_base.cache_idle_slots)` 逻辑） |
| 池满时的自动清理 | `try_clear_idle_slots()` 在 KV 压力下自动 purge 空闲 slot，无需干预 |
| `-ctk/-ctv q8_0|q4_0|iq4_nl|...` | KV 量化，把每 cell 从 2 字节降到 ~1/0.5 字节，池子等效翻倍/翻四倍（量化 V 需配合 `-fa`） |
| `--context-shift` | 单 slot 长对话：到上限时丢掉中间一段并做 KV shift，**cell 滚动复用**，池子不用扩也能无限续聊 |
| `--cache-reuse N` | 跨请求前缀复用（KV shifting），多用户共享系统提示词时省 prefill |
| `-np` 按真实并发峰值设 | 别超配；宁可少配 slot，配合排队也比空转占池强 |
| `--kv-unified-per-slot N` | 给每个 slot 固定预算，池按 `n_parallel × N` 开，行为可预期 |
| `--fit` / `--sleep-idle-seconds` | 自动按显存调 `-c`；空闲时整模型+KV 卸载 |
| 多模型 router 模式 | 每个模型实例各自一套 slot，不会为一个模型超配拖累整机 |

调度上再补一句：`--cont-batching`（默认开）下 prefill 和 decode 是交错的——decode 一步收集所有 generating slot，prefill 分块塞进 batch 剩余空间。所以你说的"走完一个 batch 走下一个"基本准确，但不是先全部 prefill 再全部 decode 的静态批处理。

---

## 5. 和 vLLM 的本质区别（也是这个问题的最终答案）

| | llama.cpp | vLLM |
|---|---|---|
| KV 管理 | 启动时一次性预留整池 | paged attention：按页动态分配 |
| 内存效率 | 低（预留制） | 高（按需，空闲即释放） |
| 确定性 | 启动即知成败，运行期零失败 | 运行期需要 preemption/swap 处理争用 |
| 复杂度/可移植性 | 一个连续 buffer 适配所有后端 | 需要每后端自定义 kernel + 页表，基本是 GPU-only |
| 定位 | 本地/边缘/跨平台推理库 | 高吞吐 serving 引擎 |

**"为什么要这样设计"的答案是一句话：llama.cpp 用"预留全部资源、启动时一次性确定成败"换取了确定性、简单性和跨平台可移植性；你要求的"按需分配、动态扩容、不排队"恰恰是它用"运行期重协商资源 + 自定义内核 + 抢占调度"换来之后才能给你的东西，那是 vLLM 的路。** 而且你点名批评的"N 份 KV"正是它多年前就主动干掉的设计——现在的统一池已经是那次的优化结果，剩下的"预留浪费"和"固定 slot"则是它为了保持"永远不运行期失败"这个承诺而保留的代价，并在 `--cache-idle-slots`、KV 量化、context shift 这些开关里做了尽量多的补救。

> 补充一句实测提示：如果你在压生产并发，最有性价比的组合是 `-ctk q8_0 -ctv q8_0 -fa`（池子减半）＋ `--cache-idle-slots --cache-ram 4096`（空闲 slot 让位）＋ `-np` 设成你观测到的并发 P95 而不是理论最大值。这样基本能把"空转占池"的浪费压到最低，又不需要放弃 llama.cpp 的确定性优势。

---

### 🧑‍💻 User

当前设计的kv cache manage motivation 就是：
- 当前 unified KV 是连续 `[n_ctx × n_seq_max]` 思路；
- 长短请求混合时存在内部内存浪费；
- 最大并发受最大 context 而不是平均 context 限制；
- 希望改成固定大小 block pool；
- sequence 按需申请 block；
- 空闲 block 可以回收；
- 后续甚至支持 CoW、prefix cache、CPU swap。

理想中的kv cache manage：
初始化阶段 自动填充kv cache pool直到最大memory usage，根据session动态扩充/移除slot，如果pool已满且slot无法release则新session开始阻塞+排队，直到有空闲slot则按照lru算法立即清空旧page
动态维护一个tree匹配ctx的hash来hit kv cache
维护一个动态的request graph，而不是遍历所有 active slots 来组 batch，当session开始alloc buffer，结束自动release，同时清理显存碎片，同时更方便做Continuous Batching

为什么llama.cpp不采用**Paged KV cache and scheduler**设计？

---

### 🤖 Assistant

这个问题值得用代码和社区讨论来回答，我先查证两件事：llama.cpp 社区里关于 paged KV / block pool 的讨论记录，以及 vLLM 官方对 PagedAttention 设计的描述（用来做准确对比）。

vLLM 那个文档路径跳转了。换个思路：直接用 GitHub API 搜 llama.cpp 里关于 paged/block KV 的 issue，同时抓 vLLM 官方 PagedAttention 博客。

GitHub API 搜索请求被重复编码导致 422，vLLM 博客路径也变了。先看一下 hub 上还有哪些工具可用（比如 shell 或专门的搜索），再决定怎么查。

`CherryFetchFetchJson` 更合适调 GitHub API，试一下（不加多余参数，避免编码问题），同时抓 vLLM 的 PagedAttention 设计文档源码。

fetch 工具对 GitHub API 的重编码问题绕不开。改用浏览器工具（带 JS 渲染），直接开 GitHub issue 搜索页和 vLLM 设计文档目录。

浏览器搜索拿到了关键证据——llama.cpp 里关于 paged KV 的历史非常有说服力（多次尝试全部被放弃/驳回，而 ggerganov 自己走了另一条路）。我再抓三个关键材料：PR #14363（high-throughput mode，这是官方"替代方案"）、PR #18747（当前正在做的 block tracking 基础设施）、以及 vLLM 的 prefix caching 文档做对比。

材料齐了。llama.cpp 官方讨论里对这个问题的态度非常明确，而且有完整的**尝试—拒绝—转向**的时间线可查。下面把"为什么"讲透。

---

## 0. 一句话结论

不是"llama.cpp 没想到 Paged KV cache"——它从 2023 年就开始被反复请求，**多次实现尝试全部被驳回/放弃**；官方（ggerganov）选择了另一条路线（stream 拆分 + 4D attention，见 #14363），并同意把"block tracking 基础设施"作为未来铺垫（#18747，仍在开放中）。拒绝的核心理由只有一句话：**paging 不减少 per-token KV 内存，只换来复杂度，而 llama.cpp 的全部设计目标就是"确定性 + 可移植性 + 可维护性"，与 serving 引擎是两种生物**。

---

## 1. 证据时间线（都来自 ggml-org/llama.cpp 仓库）

| 时间 | 编号 | 内容 | 结果 |
|---|---|---|---|
| 2023-06 | issue **#1955** | "Investigate PagedAttention KV-cache memory management" | 关闭，从未实现 |
| 2025-06 | PR **#14070** | "feat: Implement Paged KV Cache and CUDA Paged Attention" | 当天关闭（abandoned） |
| 2025-07 | PR **#14363** | "llama : add high-throughput mode"（ggerganov 亲写） | **merged** ← 官方路线 |
| 2025-11 | PR **#17579** | "Add PagedAttention support (**experimental, CUDA only**)" | 关闭（abandoned） |
| 2026-01 | PR **#18747** | "KV cache size limiting + block tracking infrastructure" | 仍 open；被维护者尖锐批评 |
| 2026-03 | PR **#20026** | "enable Paged Attention for Apple Silicon" | 当天关闭 |
| 2026-04 | PR **#22569** | "[Draft] feat: implement paged KV cache and attention" | draft，未合入 |

三个最有信息量的原始评论：

> **qnixsynapse**（collaborator，评 #18747）："I think paging will not reduce its 'per token' effective size no matter what type of chunking/paging is done... **It only avoids fragmentation / wasted KV allocation.**"

> **JohannesGaessler**（维护者，评 #18747）："Whether or not the code works correctly on this commit is irrelevant, **80% of the work is maintenance.**"

> **ngxson**（collaborator）："all the benefits you mentioned already exist to some extent in llama.cpp. **if paged attention was that good, we should have already implemented it a long time ago, just like how flash attention was added.**"

连 #18747 的作者最终都自己承认了：

> "There is no per-token memory improvement... True PagedAttention memory benefits require: **1. Sparse tensor allocation — would require ggml core changes; 2. Memory sharing across sequences (prefix caching) — not implemented here; 3. Better fragmentation handling — marginal benefit at best.**"

这句话其实就是整件事的答案——**llama.cpp 的维护者早就精确算过这笔账**。

---

## 2. 为什么拒绝：一笔笔把"价值账"算给你看

### 2.1 paging 不减少每 token 的 KV 内存

这是最根本的一点。一个 token 的 K/V 是模型结构决定的：

$$M_{token} = n_{layer} \times (2 \times n_{kv\_head} \times d_{head}) \times b$$

分页、block pool、动态分配**都不改变这个数**。它只解决两类问题：

1. **预留浪费**（你上一条吐槽的"池子按 -c 预分配，空转也占显存"）；
2. **前缀共享**（多个请求共享相同 system prompt 的 KV）。

而 llama.cpp 对这两个问题**分别有更便宜的解法**：

| 问题 | vLLM 的解法 | llama.cpp 的解法 |
|---|---|---|
| 预留浪费 | paged block pool，按需分配 | **KV 量化** `-ctk/-ctv q8_0/q4_0/...`（每 token 2B→1B/0.5B，等效池子翻倍/翻四倍）；`--cache-idle-slots`（空闲 slot 的 KV 存 RAM 并**清空池 cell**，见上一轮的 `try_clear_idle_slots()`）；`--kv-cache-demand-paged`（`mmap(MAP_NORESERVE)`，OS 级惰性分配，#18747 里提出） |
| 前缀复用 | APC（hash block 表 + LRU） | `--cache-reuse`（KV shift 复用，跨请求共享前缀）；`--cache-ram` prompt cache；多轮会话系统提示词留在 slot 的 KV 里不重算 |
| 碎片 | block 粒度回收 | **一个大连续 buffer 本身就是零碎片**；cell 级有 `find_slot()` + defrag |

结论：**paging 在 llama.cpp 里能省下的东西，很大一部分已经被 KV 量化和共享池用更低成本拿走了**。剩余的部分（动态 slot、CoW、CPU swap）只在高并发 serving 场景才值得付出那套复杂度。

### 2.2 每后端一套 kernel + 间接寻址，与 ggml 的执行模型冲突

vLLM 的 PagedAttention 是**几千行 CUDA 专属代码**（block table、重映射、`paged_attn` kernel、`reshape_and_cache`…）。llama.cpp 要跑在 CUDA / Metal / CPU / Vulkan / SYCL / OpenCL 等十几个后端上。证据就在 PR 标题里：

- #17579 自己写着 **"experimental, CUDA only"**；
- #20026 是专门给 **Apple Silicon** 补的一份。

也就是说，社区里已经有人把"CUDA-only paged"和"Metal paged"各做了一版，**都放弃了**。原因正如 JohannesGaessler 说的：能跑通只占 20% 工作量，剩下 80% 是让它在 15 个后端上永久可维护。

更深一层的冲突：ggml 的模型是**静态计算图**——整个图在启动时一次性分配 buffer，运行期**零分配、零失败**。paged 需要"运行时按需稀疏分配 block"（#18747 作者明确说这需要 ggml core 改动，超出该 PR 范围）。这直接打破 llama.cpp 最核心的不变式："启动即确定成败，运行期永不 OOM"。对本地/边缘用户，这个不变式比内存效率值钱得多。

### 2.3 官方选择的方向：先解决 attention 的"计算浪费"，而不是内存浪费（#14363）

这是最关键的一手资料——ggerganov 本人写的 high-throughput mode PR 里，对 unified KV 的吐槽比你还狠：

> "One significant drawback of the unified KV cache is that it leads to performing a lot of **unnecessary computation in the attention** when the unified buffer is shared between many large independent sequences. The reason is that we have to view this buffer continuously and therefore **we end up computing large portions of 'cross-sequence attention' which we then simply discard.**"

他的解法不是分页，而是：

- 引入 **"streams"（虚拟序列）**：把 unified KV 按序列拆成多个 buffer（`ggml_set_rows` 支持）；
- 把 batch 按 stream 拆分，attention 只对**本 stream 自己的 token** 做，不再计算然后丢弃其他序列的 KV；
- 把 `ggml_flash_attn_ext` / `ggml_soft_max_ext` 扩展到 **4D（带 n_seq 维）**，CUDA 侧由 #14628 补上；
- 两种模式可选：`kv_unified=1`（单 stream，适合共享大前缀）vs split（每序列一个 stream，适合相互独立的请求）。

实测数据（PR 自带，M2 Ultra，`-np 32`，PP=4096 总吞吐）：

| 模型 | master | #14363 | 提升 |
|---|---|---|---|
| Qwen2.5-Coder 3B Q8_0 | 1689 t/s | 2706 t/s | **≈1.60x** |
| Gemma-3 4B Q8_0 | 1720 t/s | 2353 t/s | **≈1.37x** |

**为什么这个方向比 paging 更适合 llama.cpp**：它把"unified buffer 必须按连续视图处理"这个病根治了，改动是**给 kernel 加一个维度**（n_seq），而不是**加一层间接寻址**（block table）。所有后端都能共享收益，不需要每个后端重写一套分页 kernel。这是典型的 ggerganov 风格：找问题的最小可移植解，而不是抄 serving 引擎的架构。

### 2.4 你理想方案里的每一项，llama.cpp 都有对应物（或明确拒绝理由）

| 你的理想设计 | llama.cpp 的现实 |
|---|---|
| 固定 block pool，sequence 按需申请 | 缺失，但 **#18747 正在铺路**（block pool + sequence→block 表 + O(1) alloc/free，测试里甚至模拟了 CoW）。维护者没否决这套基础设施，只否决了"现在就用它" |
| 动态扩充/移除 slot | **半误解**：unified 下 slot 不是内存分区（是 seq_id + 会话状态），真正占内存的是池。slot 数固定是因为**计算图按 `n_seq_max` 预留**，动态改 = 运行期重协商 GPU 缓冲 = 违背确定性原则 |
| 池满按 LRU 清空旧 page | 已有等价物：`try_clear_idle_slots()`（池满自动 purge 空闲 slot）、`--context-shift`（超长对话滚动复用 cell）、SWA 前视 cell 复用（#18747 里的优化，94% 显存节省） |
| tree hash 前缀命中 KV | 已有 `--cache-reuse`（KV shifting 复用前缀）+ `--cache-ram` prompt cache；粒度比 vLLM 的 hash-block APC 粗，但机制是同一个目标 |
| request graph 组 batch，而非遍历 slot | 这是 **serving engine 的概念**（vLLM 的 `v1/core/sched` + `request_queue` + DAG），服务于 speculative decoding、beam search、并行采样、优先级抢占。llama.cpp 的定位是推理库，slot 遍历 + cont-batching 在本地/中小并发下足够 |
| CoW | #18747 测试里已模拟 CoW（引用计数块共享）；`llama_memory_seq_cp()` 在 multi-stream 模式就是真实 buffer copy |
| CPU swap | 真缺失。vLLM 的 `kv_offload`（LRU/ARC + swap_blocks）是它多年的核心资产，也是 CUDA 专属。llama.cpp 只有层级 offload（`-ngl`/`-nkvo`） |

---

## 3. 更深一层：这是两种"生物"的定位差异

| | llama.cpp | vLLM |
|---|---|---|
| 本质 | 跨平台推理**库** + 本地工具 | 高吞吐 **serving 引擎** |
| 目标函数 | 可移植性 × 可维护性 × 确定性 | 显存利用率 × 吞吐 × 批处理灵活性 |
| 内存哲学 | 启动时一次性预留全部，运行期零失败 | 运行期按需分配、可抢占、可 swap |
| 核心资产 | ggml 图执行 + 量化 kernel（十几后端） | PagedAttention + scheduler + block pool（CUDA 为主） |
| 换内存效率的方式 | KV 量化、共享池、context shift、cache-reuse | 分页、APC、CoW、offload |

社区维护风格也说明问题：#18747 因为 AI 生成代码 + 夸大宣传被维护者当场拆穿并拉黑，紧接着 #18765 直接**ban 掉 AI 生成的 PR 描述**。这个项目把"每一个合入的东西都要能被少数核心维护者永久维护"看得比功能多寡重得多。在这样一个项目里，合入 vLLM 级别的 paged scheduler，等于引一个需要专职团队维护的子系统。

---

## 4. 那你的生产场景到底该怎么办

务实结论：

- **如果**你的场景就是"高并发、长上下文、多租户 serving"，用 vLLM / SGLang / TensorRT-LLM 是对的——它们就是把你想的那套（block pool、LRU、APC、CoW、CPU swap、request DAG）全部实现了，且为此雇佣了全职团队。**不要**在 llama.cpp 上等它。
- **如果**你的场景是"本地 / 边缘 / 显存有限 / 需要确定性内存"，llama.cpp 是对的，而且它的 `-ctk/-ctv` 量化 + `--cache-idle-slots` + `--cache-reuse` + `-np` 按实际并发配已经能把"预留浪费"压到很低。
- **llama.cpp 不采用 Paged KV 不是缺陷，是定位选择**。它已经用"更便宜的机制"覆盖了 paging 实际能解决的痛点（碎片、前缀复用），剩下的部分（动态分配、抢占、swap）正是它故意放弃的——因为那些东西换来的收益，在它的目标用户群里不值它付出的复杂度。你如果认同"启动即确定、运行期永不失败"这个不变式，就会理解这其实是一种**纪律**，而不是短视。
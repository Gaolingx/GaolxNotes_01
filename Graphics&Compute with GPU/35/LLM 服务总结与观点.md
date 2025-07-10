# LLM 服务总结与观点

### 🧑‍💻 User

![](./vLLM%20First%20SF%20Meetup%20Slides%20(Public).pptx)
总结下面的文章，给出总结、摘要、观点三个部分内容，其中观点部分要使用列表列出，使用 Markdown 回复

---

### 🤖 Assistant

好的，这是根据您提供的演示文稿总结的内容。

### 总结

[cite_start]该演示文稿是 vLLM 项目于 2023 年 10 月 5 日在旧金山举办的首次技术分享会的内容 [cite: 1, 2][cite_start]。内容主要分为四个部分：首先由 Woosuk Kwon 对 vLLM 项目进行概述，介绍了其核心目标、关键技术 PagedAttention，以及不断扩大的社区和行业采用情况 [cite: 16, 18, 23, 101][cite_start]；接着由 Zhuohan Li 详细解读了 vLLM 的系统架构和工作流程，解释了从请求接收到分布式处理的完整步骤 [cite: 135, 138][cite_start]；之后，Woosuk Kwon 回顾了项目发布以来的主要进展，包括新模型支持、性能优化、正确性保障和易用性提升 [cite: 329, 331][cite_start]；最后，两位主讲人共同展望了项目的未来路线图，重点阐述了在延迟优化、量化支持和底层 CUDA 核心优化等方面的规划 [cite: 595, 600]。

### 摘要

[cite_start]这份演示文稿全面介绍了 vLLM，一个旨在成为最快、最易用的开源大语言模型（LLM）推理服务引擎 [cite: 19, 20, 21][cite_start]。项目由加州大学伯克利分校的博士生 Woosuk Kwon 和 Zhuohan Li 主导 [cite: 5, 6, 8, 9][cite_start]。vLLM 的核心创新是 PagedAttention 算法，它借鉴了操作系统中虚拟内存管理的技术，通过将 KV 缓存（KV cache）分页存储在非连续的内存空间中，有效解决了内存碎片化问题，从而显著提升了吞吐量 [cite: 24, 26]。

[cite_start]文稿详细拆解了 vLLM 的系统架构，包括作为中央控制器的 LLMEngine、负责任务调度的 Scheduler 以及执行模型计算的分布式 Worker [cite: 168, 171, 177, 187, 189][cite_start]。vLLM 提供了包括离线批量推理和与 OpenAI 兼容的 API 服务器在内的多种接口 [cite: 77, 87][cite_start]，并已被 NVIDIA、Meta、Anyscale、Databricks 等众多公司和开源项目采用 [cite: 101, 105, 106]。

[cite_start]自项目启动以来，团队在支持更多新模型（如 Mistral、Falcon）、提升性能（通过高效的去符号化和矢量化采样器将吞吐量提升高达 66%）、保障准确性以及改善用户体验（如提供预编译的 CUDA 二进制文件以实现快速安装）等方面取得了显著进展 [cite: 331, 345, 503, 560, 562]。

[cite_start]展望未来，vLLM 的路线图将优先聚焦于延迟优化，计划通过引入轻量级运行时、多进程架构和推测解码等技术来降低延迟 [cite: 601, 717][cite_start]。同时，项目还将加强对各类量化技术的支持（如 GPTQ、GGML），并持续优化底层的 CUDA 内核 [cite: 602, 603, 721][cite_start]。项目承诺将始终保持 Apache 2.0 协议下的开源模式 [cite: 597, 598]。

### 观点

* [cite_start]**核心创新是高效的内存管理**：vLLM 的基石是 PagedAttention 算法 [cite: 23, 24]。它解决了传统 LLM 服务中因 KV 缓存管理不善而导致的严重内存浪费和碎片化问题，这是实现高吞吐量的关键。
* [cite_start]**性能是首要目标**：项目致力于成为“最快”的推理引擎 [cite: 19][cite_start]。通过矢量化采样器和高效的去符号化等优化手段，v0.2.0 版本的吞吐量较旧版提升了高达 66% [cite: 503]，显示了其对性能优化的持续追求。
* [cite_start]**易用性是扩大影响力的关键**：除了追求高性能，项目同样重视“易用性” [cite: 20, 21][cite_start]。提供 `pip` 一键安装 [cite: 72][cite_start]、预编译二进制文件 [cite: 560][cite_start]、兼容 OpenAI 的 API [cite: 87] [cite_start]以及与 Ray Serve、Triton 等多种服务框架的集成 [cite: 567]，极大地降低了用户的使用门槛。
* [cite_start]**开源和社区驱动是发展的动力**：vLLM 不仅是一个开源项目 [cite: 597][cite_start]，还积极与社区互动，吸纳了众多贡献者 [cite: 131, 133]。这种开放的模式吸引了大量公司和开发者采用和贡献，形成了良性生态。
* [cite_start]**未来聚焦于降低延迟**：在解决了吞吐量瓶颈后，vLLM 的下一个主要目标是优化延迟 [cite: 601][cite_start]。计划通过 C++ 轻量级运行时、多进程架构和推测解码等多种手段，解决 Python 开销带来的延迟问题 [cite: 621, 633, 639, 663]。
* [cite_start]**模型和硬件的广泛兼容性是未来方向**：为了覆盖更广泛的应用场景，路线图包含了对更多量化方法（如 GPTQ、GGML/GGUF） [cite: 695, 696, 697][cite_start]、LoRA [cite: 708][cite_start]、专家混合模型（MoE） [cite: 710] [cite_start]以及对 AMD GPU 的支持 [cite: 709]，展示了其成为通用 LLM 服务基础设施的雄心。

---

### 🧑‍💻 User

Optimization (1): Efficient de-tokenization
Optimization (2): Vectorized sampler
这两个部分能不能单独展开分析下？

---

### 🤖 Assistant

好的，我们来分别对 vLLM 的这两项性能优化进行展开分析。

### 1. 高效的去符号化 (Efficient de-tokenization)

[cite_start]这项优化旨在降低模型在生成文本过程中，将符号（Token ID）转换回文本（Text）时产生的 CPU 开销 [cite: 413]。

* **问题背景**：在流式生成文本的场景中，模型每次只生成一个或几个新的 Token ID。为了向用户展示完整的文本流，系统需要将当前所有的 Token ID 序列转换成人类可读的文本。传统的做法是在每一步都对完整的 Token ID 序列进行一次完整的转换，这会造成大量的重复计算，尤其当序列变长时，CPU 开销会非常大。

* [cite_start]**vLLM 的解决方案**：vLLM 采用了一种**增量式更新和缓存**的策略来解决这个问题 [cite: 413]。
    1.  [cite_start]**缓存（Caching）**：系统会缓存已经生成并转换好的文本部分 [cite: 411, 412]。
    2.  **增量更新（Incremental Updates）**：当模型生成一个新的 Token ID 时，系统并不会从头转换整个序列，而仅仅处理这个新增的 Token ID，然后将其对应的文本追加到缓存的文本末尾。
    3.  [cite_start]**处理复杂情况**：该机制甚至能处理更复杂的情况。例如，当一个新生成的 Token 会与前一个 Token 合并成一个新的单词时（如 "promise" + "ing" -> "promising"），系统也能高效地更新缓存，而不是重新计算整个序列 [cite: 434, 435, 436, 437, 438]。

* [cite_start]**优化效果**：通过缓存和增量更新，这项优化极大地减少了冗余的 CPU 计算。根据性能测试图表，该优化使得 LLaMA-2 7B 模型在 A100 80GB 上的吞吐量提升了 18%（从 5.1 reqs/s 提升至 6.0 reqs/s） [cite: 503, 504]。

### 2. 矢量化采样器 (Vectorized sampler)

这项优化旨在利用 GPU 的并行计算能力，一次性为批处理中的所有序列完成采样，而不是逐个进行。

* [cite_start]**问题背景**：在服务多个并发请求时，系统通常会把它们组成一个批次（batch）进行处理。然而，每个请求可能有不同的采样参数（例如，有的需要 top-k 采样，有的需要返回 `log_probs`，有的需要生成多个候选序列 `n > 1`）。之前的实现（被称为“迭代式”）是通过一个循环，在批次中逐个处理每个序列的采样逻辑 [cite: 449]。这种迭代方式无法充分利用 GPU 的并行能力，效率较低。

* [cite_start]**vLLM 的解决方案**：vLLM 设计了一个**矢量化**的采样器，将整个批次的采样操作合并成一个单一的大型操作 [cite: 450]。
    1.  [cite_start]**统一处理**：它将批处理中所有序列的模型输出（logits）收集起来，整合成一个大的张量（tensor） [cite: 479]。
    2.  [cite_start]**并行采样**：然后，它根据所有序列的采样参数，通过如 `torch.multinomial()` 或 `torch.topk()` 这样的矢量化 PyTorch 操作，一次性为所有序列生成下一个 Token [cite: 493, 494]。这意味着，无论每个序列的采样参数多么不同，采样过程都是并行完成的。

* [cite_start]**优化效果**：这种矢量化方法消除了 Python 循环带来的开销，并充分发挥了 GPU 的并行计算优势，显著提升了处理速度。性能测试图表显示，在高效去符号化的基础上再加入矢量化采样器，吞吐量又进一步提升了 42%（从 6.0 reqs/s 提升至 8.5 reqs/s） [cite: 503, 505]。

---

### 🧑‍💻 User

在Future Roadmap——Optimizing latency (1): Lightweight runtime、Optimizing latency (2): Multi-process architecture、Optimizing latency (3): Speculative decoding
这三个部分能不能单独展开分析下？

---

### 🤖 Assistant

好的，我们来分别对 vLLM 未来路线图中旨在优化延迟的三个关键策略进行展开分析。

### 1. 优化延迟 (1): 轻量级运行时 (Lightweight runtime)

这项策略旨在解决由 Python 和 PyTorch 框架自身带来的额外开销问题。

* [cite_start]**问题背景**: 当前 vLLM 的模型执行逻辑运行在 Python 环境中。在 GPU 执行计算任务（如矩阵乘法 `matmul`、激活函数 `gelu` 等）的间隙，CPU 需要通过 Python/PyTorch 框架来调用和调度下一个 CUDA 核心（kernel）。这个调度过程本身存在开销（overhead），在处理小批量请求时尤其明显。根据演示文稿，这部分开销最高可占据整个端到端延迟的 50% [cite: 621][cite_start]。这意味着 GPU 有近一半的时间处于空闲状态，等待 CPU 发出下一个指令 [cite: 619, 620]。

* [cite_start]**vLLM 的解决方案**: 为了消除这种开销，vLLM 计划引入一个轻量级的运行时，绕过高开销的 Python 调度。具体方案有两种 [cite: 635]：
    1.  [cite_start]**CUDA Graphs + PyTorch**: 利用 CUDA Graphs 技术将一系列的 CUDA 操作预先录制成一个固定的图，之后便可以一次性启动，从而绕过 Python 逐个操作调度的开销 [cite: 634]。
    2.  [cite_start]**C++ 运行时**: 使用 C++ 来重写模型的执行和调度逻辑，完全取代 Python 在这个环节中的作用，从而实现更底层的控制和更高的效率 [cite: 634]。

* [cite_start]**预期效果**: 通过轻量级运行时，CPU 可以一次性将整个计算图（或多个操作）提交给 GPU，从而让 GPU 核心不间断地连续执行任务，最大化利用率并显著降低延迟 [cite: 633]。

### 2. 优化延迟 (2): 多进程架构 (Multi-process architecture)

这项策略旨在通过将系统功能模块化和进程化，来解决单进程内不同任务互相阻塞的问题。

* [cite_start]**当前架构 (单进程)**: 目前，vLLM 在一个进程中处理所有任务，包括请求处理、文本的符号化（Tokenization）、模型执行、去符号化（De-tokenization）以及结果的流式返回 [cite: 640, 642, 643, 644, 645, 646]。这意味着，当模型在 GPU 上进行密集计算时，CPU 可能无法及时处理新的请求或对已生成的结果进行去符号化，反之亦然，造成了瓶颈。

* [cite_start]**未来架构 (多进程)**: 路线图提出将不同的功能拆分到独立的进程中，并通过异步通信（Async. send）连接 [cite: 647, 658, 659, 660, 661][cite_start]。具体拆分如下 [cite: 648, 649, 651, 652, 654, 656, 657]：
    * **请求处理器 (Request handler)**: 专门接收和分发用户请求。
    * **符号化器 (Tokenizer)**: 负责将输入文本转换为 Token ID。
    * **模型执行器 (Model Executor)**: 核心进程，专注于在 GPU 上执行模型计算。
    * **去符号化器 (De-tokenizer)**: 负责将模型输出的 Token ID 转换回文本。
    * **流式处理器 (Token Streamer)**: 负责将文本结果流式返回给客户端。

* **预期效果**: 这种架构使得 CPU 密集型任务（如符号化/去符号化）和 GPU 密集型任务（模型执行）可以真正地并行执行，避免了相互等待和阻塞，从而提升整体的响应速度和系统吞吐量。

### 3. 优化延迟 (3): 推测解码 (Speculative decoding)

这项策略利用一个“小模型”来辅助“大模型”，从而打破传统自回归生成（一次生成一个 token）的限制，实现一次性生成并验证多个 token。

* [cite_start]**问题背景**: 大语言模型通常采用自回归（Autoregressive）方式生成文本，即必须生成完第 N 个词之后才能开始生成第 N+1 个词，这是一个串行过程，限制了生成速度 [cite: 674]。

* [cite_start]**vLLM 的解决方案**: 推测解码的原理如下 [cite: 691]：
    1.  [cite_start]**起草 (Draft)**: 使用一个参数量远小于主模型（例如，小 10 倍）的轻量级小模型，以非自回归的方式（并行地）一次性快速生成一个包含多个 token 的“草稿” [cite: 664, 675, 689]。
    2.  [cite_start]**验证 (Verification)**: 然后，让参数量大的主模型接收这个草稿，并一次性地对草稿中的所有 token 进行并行验证，判断这个草稿是否是它自己会生成的内容 [cite: 677, 690]。
    3.  **采纳与修正**: 如果草稿被主模型接受，那么就相当于一次性生成了多个 token，大大加快了速度。如果主模型在某个位置拒绝了草稿，则从该位置起重新开始生成过程。

* **预期效果**: 这种方法利用小模型的速度优势来起草，利用大模型的并行验证能力来把关，最终实现在保持生成质量的同时，用一个步骤完成多个 token 的生成，从而成倍地降低生成延迟。

---

### 🧑‍💻 User

为什么vllm的prompt processing用的flashattention，token generation用的pagedattention？

---

### 🤖 Assistant

这个问题的答案可以在演示文稿的系统架构部分找到。vLLM 针对推理过程中的两个不同阶段（Prompt Processing 和 Token Generation）的计算特性，分别选用了最适合的注意力（Attention）实现方案。

简而言之：**两个阶段的瓶颈不同**。Prompt 处理阶段是**计算密集型**的，而 Token 生成阶段是**内存密集型**的。

以下是详细分析：

### 1. Prompt Processing (Prefill) 阶段：使用 FlashAttention

* **计算特性**: 这个阶段需要一次性处理用户输入的整个提示（Prompt）。提示通常很长，包含大量 Token。因此，这里的计算瓶颈在于对这批长序列进行一次大规模的注意力计算。
* [cite_start]**为什么用 FlashAttention?**: FlashAttention 是一个高度优化的注意力算法，它通过 Kernel Fusion（核函数融合）等技术，专门用于加速对长序列的注意力计算，同时能显著减少显存的占用。它非常适合这种“一次性”的、计算量大的场景。因此，vLLM 选择它来高效地处理初始提示，为后续的生成做好准备 [cite: 281]。

### 2. Token Generation (Decoding) 阶段：使用 PagedAttention

* **计算特性**: 这个阶段是自回归的，即模型一次只生成一个新 Token。这里的计算瓶颈不再是单次注意力计算的规模，而是如何高效地管理和访问不断增长的 **KV 缓存**（存储了所有历史 Token 的注意力键值对）。每次生成新的 Token，都必须将它的注意力信息与之前所有 Token 的信息进行计算。
* [cite_start]**为什么用 PagedAttention?**: 这正是 vLLM 的核心创新所在。PagedAttention 并非一个独立的注意力算法，而是一套**结合了注意力计算的内存管理机制**。它通过将 KV 缓存分页存储在不连续的物理显存块中，解决了传统方法导致的内存碎片和浪费问题 [cite: 23, 24][cite_start]。vLLM 为这个生成阶段专门编写了 PagedAttention CUDA 核心（kernel），该核心能够高效地从分页的、非连续的显存中读取所需的 KV 缓存，并完成注意力计算 [cite: 282]。这使得系统可以支持更大的批处理规模（batch size），从而实现更高的吞吐量。

### 总结

vLLM 的设计哲学是为每个特定任务选择最优的工具：

* 对于**初始的长文本处理**，其瓶颈在于计算，因此使用为该场景优化的 **FlashAttention**。
* 对于**后续的逐词生成**，其瓶颈在于 KV 缓存的内存管理和访问，因此使用 vLLM 自身发明的、专为解决此问题的 **PagedAttention** 核心。
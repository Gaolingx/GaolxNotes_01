# DeepInfer代码结构与核心技术分析

## 🧑‍💻 User

你是一个软件设计师，深度分析DeepInfer的代码结构及格模块功能及核心技术并形成一份分析报告，https://github.com/deepseek-ai/DeepInfer

---
## 😀 默认助手 (7)

我先去获取仓库信息。

# DeepInfer 源码结构与核心技术深度分析报告

> 分析对象：`https://github.com/deepseek-ai/DeepInfer`（`main` 分支，commit tree `774ce24`）
> 元数据：Python 3.12 / Apache-2.0 / 11.0k stars / 968 forks / 创建于 2026-10-16 / 约 3.2 MB 源码
> 定位（官方描述）："DeepSeek 大规模生产服务的高性能推理框架，本次开源提供**离线推理**版本"

---

## 0. 阅读覆盖度声明（先说清楚证据边界）

本报告基于**实际拉取并通读**的 30 个文件（约 40 万字符源码）＋ 全仓库 328 个文件的完整文件清单（jsDelivr flat API）。已完整通读的核心文件：

| 层 | 已通读文件 |
|---|---|
| 接入/配置 | `README.md`、`pyproject.toml`、`.gitmodules`、`build.sh`、`scripts/build_vendor.sh`、`deepinfer/__init__.py`、`environ.py`、`config.py`、`engine/arg_utils.py`、`engine/engine_type.py` |
| 引擎 | `engine/llm_engine.py`(45KB)、`entrypoints/llm.py` |
| Worker | `worker/worker.py`、`worker/base.py`、`worker/model_runner.py`、`worker/cache_engine.py`(49KB) |
| 模型 | `models/deepseek.py`、`models/mtp/dflash.py`、`v4_modules/attn.py`(46KB)、`v4_modules/index_attn.py`、`model_metadata_utils.py`、`input_metadata/prepare_tensor_input_zos.py`、`layers/sampler.py`(30KB) |
| 内核/调度（C++） | `csrc/engine/scheduler.hpp`(66KB)、`mtp_varlen.hpp`、`zos.hpp` |
| 其他 | `eplb/__init__.py`、`structure_output/matcher.py`、`examples/README.md` |

**未通读**（仅凭命名、调用点、交叉引用推断）：`v4_modules/{moe,mlp,mla_attn,compress,engram,hc,mhc_modeling,yoco,cross_layer}.py`、`parallel_utils/*`、`ve/*`、`tilelang_ops/*`（78 个内核文件）、`csrc/engine/{block_manager,flops_estimator,prepare_tensor_input_v4,sequence}.hpp/cpp`、`eval/*`、`tests/*`。报告中凡属推断的内容均已标注「（推断）」。

⚠️ 技术说明：抓取管线把 HTML 当作文本处理，**C++ 头文件中的模板参数 `<...>` 被吞掉**（如 `std::tuple<...>` 显示为 `std::tuple,`）。因此下文引用的 C++ 签名是**语义还原**而非逐字原文；Python 侧则基本无损。

---

## 1. 结论速览（TL;DR）

<details open>
<summary><b>五个判断</b></summary>

1. **它不是"又一个 vLLM 复刻"，而是一次"引擎内核重写 + 模型侧深度特化"**。Python 外壳（`LLMEngine`/`Worker`/`CacheEngine`/`Sampler`）明显派生自 vLLM（文件头统一标注 `Derived from vLLM; modified by DeepSeek`），但**调度器、KV 布局、注意力、采样、投机解码全部被替换为 DeepSeek 自研实现**，且调度器被搬进了 C++（`csrc/engine/scheduler.hpp`，67 KB 单文件）。

2. **核心设计主线是"用零开销调度（ZOS）把 CPU 调度、CPU 后处理、GPU forward、GPU sample 四件事完全流水化"**。这是把吞吐推向极致的结构性手段，代价是引入了"CPU 侧元数据滞后一步 → GPU 侧修正"的一整套精密不变量（`_correct_seqlens_qk`、`advance_optimistically`/`commit_processed_tokens`、`DeferredRelease`）。

3. **模型侧的关键词是"压缩 + 稀疏 + 分层复用"**：KV 压缩比 $C_1/C_2$、FP4/UE8M0 量化的 indexer、SWA 滑动窗口、YoCo 跨层 KV/Index 复用（share-kv / share-index）、Engram n-gram 记忆、HC 超连接。这些都不是通用框架能"配"出来的，而是**为 V4.1 架构与硬件共同设计**的。

4. **投机解码是"置信度驱动的变长验证"（DSpark/DFlash + MTPVarlen）**，本质是一个**在线吞吐最优化的控制问题**：用置信度头在线估计接受率曲线，并按实测 forward/sampler 耗时曲线求 $\arg\max$ 吞吐的 batch 与每请求 draft 长度，而不是固定 draft 长度。这在工程上比 EAGLE/Medusa 系列更激进。

5. **工程上最值得学习的是"契约式零拷贝"**：`SchedulerOutputs` 借用 scheduler 内部 `std::vector` 的存储并以 numpy"空 capsule"回传给 Python，调用方必须在下一次 `schedule()` 前消费完；CUDA Graph 场景用 `get_or_create_stacked` 构造等间距行视图实现零拷贝切批。这类做法换来的是极致性能，也带来极高的维护风险。

</details>

---

## 2. 仓库全景

### 2.1 目录骨架与体量分布

```text
DeepInfer/
├── csrc/                      # C++ 引擎内核（pybind11 暴露）
│   ├── engine/                # ★ 调度器、序列、BlockManager、Flops 估计、MTP varlen
│   │   ├── scheduler.hpp      # 66 KB，全仓库最重的单文件
│   │   ├── sequence.hpp/.cpp  # 17 KB
│   │   ├── block_manager.hpp  # 7 KB
│   │   ├── flops_estimator.hpp# 18 KB
│   │   ├── mtp_varlen.hpp     # 12 KB，DSpark 的在线吞吐决策
│   │   ├── zos.hpp/.cpp       # 15 KB，ZOS 输入装配
│   │   └── input_metadata/    # prepare_tensor_input_{base,v4}.cpp（共 45 KB）
│   ├── graph_timer/           # CUDA Graph 内计时（CUDA/Ascend 双实现）
│   └── utils/                 # lockable / object_pool / log
├── deep_cluster/              # 硬件与拓扑抽象（nvidia / ascend / cpu）
├── deepinfer/                 # Python 主包（详见 §3）
├── doc/images/                # 仅两张 Pareto 图，无设计文档 ⚠️
├── eval/                      # AIME / GPQA / HLE 评测（pass@k + LLM judge）
├── examples/                  # offline_inference / benchmark / sweep / yoco
├── scripts/                   # build.sh / build_vendor.sh / format_csrc / gen_stub
├── tests/                     # 20 个测试文件（多为 GPU 依赖）
└── vendor/                    # 6 个 git submodule（算子库）
```

**体量特征（按字节数粗估）**：`deepinfer/model_executor/layers/tilelang_ops/` 是最大子系统——约 **78 个文件**，单文件普遍 10–50 KB（`fused_transform.py` 47 KB、`sampler.py` 49 KB、`fused_sample.py` 38/45 KB），其中 `cuda/` 与 `asc/` 两份几乎一一对应。**内核层体量 ≈ 引擎层 × 2**，这本身就说明：这是一个"kernel-first"的推理栈。

### 2.2 依赖的私有算子栈（vendor submodule）

| Submodule | 作用（依据调用点） | 硬件 |
|---|---|---|
| `FlashMLA` | MLA 注意力（`flash_mla.fused_norm_rope_attn_rope_cast`） | NVIDIA |
| `DeepGEMM` | FP8/FP4 GEMM、FP8 einsum、**sparse MQA logits**（`fp8_fp4_sparse_mqa_logits`）、scale 布局变换 | NVIDIA |
| `DeepEP` / `DeepEP-Ascend` | MoE all-to-all 分发 | 双栈 |
| `DeepGEMM-Ascend` | NPU GEMM 对偶 | Ascend |
| `DeepSelect` | 高性能 `topk`（`deep_select.interface.topk`） | 双栈 |
| `TileKernels` (pip 依赖) | `tile_kernels.quant.per_token_cast`、`engram_hash` | 双栈 |

`build_vendor.sh` 按 `DEEPINFER_ACCELERATOR` 切换包集合与构建目标（`CUDA` / `ASCEND`），并对 NCCL 做了固定版本覆写（`nvidia-nccl-cu13==2.32.3`）——这暗示其集合通信对 NCCL 行为有版本级依赖。

### 2.3 与 vLLM 的血缘关系（代码考古）

| 文件 | 血统 | 改造程度 |
|---|---|---|
| `engine/llm_engine.py`、`worker/base.py`、`worker/cache_engine.py`、`entrypoints/llm.py`、`config.py`、`layers/sampler.py` | vLLM | **骨架保留、语义重写**（引擎类名/参数名/文档字符串沿用 vLLM 风格，内部逻辑已完全不同） |
| `csrc/engine/scheduler.hpp`、`mtp_varlen.hpp`、`tilelang_ops/*`、`v4_modules/*`、`eplb/*`、`structure_output/*` | **纯自研** | 无 vLLM 对应物 |
| `model_defaults`/`SamplingParams` | vLLM 影子 + 自研（`so_args`、`seq_seed`、`golden_token_ids`） | 显著扩展 |

---

## 3. 分层架构

### 3.1 逻辑分层

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ L5 接入层   entrypoints/llm.py  (LLM.generate)                            │
│             ├─ 离线批处理语义（tqdm、tokenizer decode、RequestOutput clone）│
│             └─ 无 HTTP server（README 明确：在线服务包装不在此次开源范围）  │
├──────────────────────────────────────────────────────────────────────────┤
│ L4 引擎层   engine/llm_engine.py  (LLMEngine)                             │
│             ├─ Scheduler          ← C++ (csrc/engine) ★核心               │
│             ├─ BlockManager ×2    (KV block, State block)                 │
│             ├─ SequenceMetadataPool / RequestOutputPool  (C++ 对象池)     │
│             ├─ DeferredRelease    (跨 step 延迟释放)                      │
│             ├─ SOMatcherBuilder   (xgrammar 编译线程池)                   │
│             ├─ StatsLogger / ProfileManager (HTTP) / FlopsEstimator       │
│             └─ EngineType: NORMAL / PROMPT / COMPLETION (PD 分离预留)     │
├──────────────────────────────────────────────────────────────────────────┤
│ L3 执行层   worker/worker.py  (Worker : WorkerBase)                       │
│             execute_model_step = Forward → Post(prev) → Sample → Sync     │
│             ├─ ModelRunner      (input metadata / CUDA Graph / sample)    │
│             ├─ CacheEngine      (KV tensor + State tensor 布局)            │
│             ├─ CudaGraphRunner / DFlashCudaGraphRunner                    │
│             └─ DecodePerfCache  (decode 性能曲线缓存, TTL 1h)              │
├──────────────────────────────────────────────────────────────────────────┤
│ L2 模型层   model_executor/models/                                        │
│             DeepseekForCausalLM                                           │
│             ├─ DeepseekModel (emb → N×DecoderLayer → HC head → norm)      │
│             ├─ v4_modules: attn / index_attn / compress / moe / mlp       │
│             │               engram / hc / mhc_ops / yoco / cross_layer    │
│             ├─ mtp/dflash: DFlash (draft layers + ConfidenceHead)         │
│             └─ layers: sampler / linear / layernorm / rope / embedding    │
├──────────────────────────────────────────────────────────────────────────┤
│ L1 内核层   tilelang_ops/{cuda,asc}  +  vendor/*                          │
│             量化/存储/加载/采样/投机验证/结构输出/Gather/编译期元数据        │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 一次 `step()` 的完整调用链（ZOS 开启时）

```text
LLMEngine.step()                                    # engine/llm_engine.py
├─ scheduler.schedule(sequence_metadata_pool)       # → C++ Scheduler::_schedule()  (释放 GIL)
│    ├─ 引擎步型决策（NORMAL: 两轮 DP 投票）
│    ├─ _schedule_prefill() / _schedule_decode()
│    └─ 构造 SequenceMetadata[] + SchedulerOutputs（借用内部 vector 存储）
├─ _run_workers("clone_and_zero_states_and_blocks_on_default_stream")
├─ post_schedule(...)
├─ _execute_model_async_step(...)                   # → Worker.execute_model_step
│    │
│    │  ┌────────────────────── Worker ──────────────────────┐
│    ├──┤ 1. _correct_seqlens_qk()   # 用上一步采样结果在 GPU 修正 seqlen_q/k
│    │  │ 2. forward(N)  launch      # 必须零 cuda sync（forbid_cuda_sync 守卫）
│    ├──┤ 3. _post_process_step(N-1) # sync sample_end_evt → 后处理 → engine.post_process
│    │  │ 4. sample(N)     launch
│    │  │ 5. pre_sampler_sync()      # DeferredRelease.release() + SO pump + finalize_seq
│    │  │ 6. 决定 post(N) 当场做（prompt/非ZOS）还是推迟（ZOS decode）
│    │  └───────────────────────────────────────────────────┘
├─ deferred_release.collect_request_outputs(...)    # 下一 step 的 pre_sampler_sync 才真正释放
└─ return request_outputs                           # uint64 指针数组（零拷贝回传）
```

关键洞察：**第 2 步（forward N 的 launch）与第 3 步（后处理 N-1）在时间上重叠**——GPU 跑 forward(N) 的同时 CPU 在跑上一轮的 Python 后处理。这就是 "zero-overhead" 名字的真实含义：**不是调度开销为零，而是把调度与后处理的 CPU 开销塞进了 GPU 的空隙**。

### 3.3 模块职责表（Python 主包）

| 包/文件 | 职责 | 关键类/函数 |
|---|---|---|
| `__init__.py` | 平台探测、`_so/<accelerator>-<arch>` 动态加入 `__path__`、延迟初始化 `PLATFORM`（避免早于 `ASCEND_RT_VISIBLE_DEVICES` 的初始化） | `_so_platform_tag`、`__getattr__` |
| `environ.py` | **60+ 环境变量的描述符式懒解析**（`EnvVar` 非数据描述符 + 线程锁 + `rearm()` 热重载） | `envs`、`EnvVar.choice` |
| `config.py` | `ModelConfig` / `CacheConfig` / `ParallelConfig` / `SchedulerConfig`；**`runtime_max_model_len` 推导**（含 nextn + ZOS 冗余位置） | `_get_runtime_max_model_len` |
| `engine/llm_engine.py` | 引擎主循环、Worker 编排、请求生命周期、SO 异步构建、统计 | `LLMEngine`、`DeferredRelease` |
| `engine/arg_utils.py` | 参数 → 四元组配置；根据 `EngineType` 自动设置 ZOS/多模态池默认值 | `EngineArgs.create_engine_configs` |
| `engine/profile_server.py` | 运行时 HTTP profiling 接口（默认端口 6070） | `ProfileManager` |
| `engine/stats_logger.py` | 滑动窗口吞吐/MFU/时延统计 | `StatsLogger`、`StepStats` |
| `worker/base.py` | 进程/分布式/模型加载/KV 分配/slave RPC 循环 | `WorkerBase`、`set_pdeathsig` |
| `worker/worker.py` | 单步执行四段式流水 | `Worker`、`EngineCallbacks` |
| `worker/model_runner.py` | 输入元数据装配、CUDA Graph、采样、profile、decode 性能曲线标定 | `ModelRunner`、`MainModelOutput` |
| `worker/cache_engine.py` | **KV/State 物理布局算法**（本仓库最烧脑的 Python 文件） | `KVCacheManager`、`StateBlock`、`GPUCacheInterface` |
| `worker/zos_status.py` | ZOS 跨步状态载体 | `MetadataZOS`、`WorkerStepState` |
| `worker/decode_perf_cache.py` | decode Pareto 曲线磁盘缓存 | `DecodePerfCache` |
| `model_executor/model_metadata_utils.py` | 从 HF config 派生的**运行时元数据总表**（压缩率、topk、engram、yoco、VL…） | `ModelMetadata`、`derive_compress_ratios` |
| `model_executor/input_metadata/` | prefill/decode/ZOS 三套元数据装配（CPU 侧 vs GPU 侧 kernel） | `prepare_decode_v4_zos`、`prepare_v4_decode_metadata` |
| `model_executor/eplb/` | 专家负载均衡（策略 + 权重 shuffle） | `ExpertLoadBalancer`、`rebalance_experts_global` |
| `model_executor/structure_output/` | 结构化输出（xgrammar 封装 + 异步构建 + mask） | `XGrammarMatcher`、`SOMatcherBuilder` |
| `model_executor/parallel_utils/` | 进程组、CP、userbuffer（对称内存/UB）、通信 op | `PG`、`userbuffer`、`cp_broadcast_multi_tensor` |
| `model_executor/ve/` | 多模态前端（图像 processor / VL metadata / schema） | `MultimodalProcessor`、`prepare_input_metadata_vl` |
| `npu_ops/` | Ascend 适配补丁（`patch_imports`、`deep_select`、`vit_attn`） | — |
| `deep_cluster/` | 硬件能力（`get_bf16_tflops_limit`）+ NUMA/拓扑绑定 | `bind_numa`、`which_numa` |

---

## 4. 核心技术深度剖析

### 4.1 ZOS：零开销调度（Zero-Overhead Scheduling）

**问题**：常规紧耦合引擎每步必然出现"CPU 调度 → GPU forward → GPU sample → CPU 后处理"的串行气泡，`bsz=1` 时气泡占比可达 30%+。

**DeepInfer 的解法**：把 CPU 侧的"**上一轮**后处理 + **下一轮**调度"全部塞进当前 GPU 计算窗口，代价是**CPU 侧状态永久滞后一步**，于是必须补两处"事后修正"。

**(a) 序列长度修正**（`worker.py::_correct_seqlens_qk`）

CPU 在调度第 $N$ 步时只知道第 $N-2$ 步的接受结果，因此 `seqlen_q`（本轮 query 数 = $1 + $ 参与验证的 draft 数）与 `seqlen_k` 是**乐观估计**。修正机制：

1. `prepare_zos_inputs(seq_metadata_list, prev_seq_metadata_list, n_seq_with_padding, nextn)` 产出 `(remap, seqlens_q, seqlens_k, has_overlap)`；
2. 若与上一步 batch 无交集 → 无需修正（快速路径）；
3. 否则把 `remap / seqlens_q / seqlens_k` H2D，用上一步 GPU 上的 `num_accepted_tokens` 就地重算：
   - `compute_seqlens_k_zos(...)` 修 `seqlen_k`
   - `compute_seqlens_q_by_confidence_rates(...)` 在 **MTP varlen** 模式下按上一步实际置信度重写 `seqlen_q`
4. 仅 varlen 模式需要额外一次 **D2H** 把修正后的 `seqlen_q` 拉回 CPU 供 sampler 元数据使用（其余情况修正完全留在 GPU）。

**(b) KV 计数两阶段提交**（`scheduler.hpp::process_outputs`）

```cpp
seq.cache.advance_optimistically(scheduler_outputs.seqlens_q[i]);   // 乐观推进
...
if (!prompt_run) seq.cache.commit_processed_tokens(token_cnt);      // 按实际接受数回填
```

因为 draft token 被拒绝后不占 KV，所以在途阶段按上限推进、出结果后回退，避免"申请块→释放块"的抖动。

**(c) 资源延迟释放**（`DeferredRelease`）

step 产出的 `SequenceMetadata` / `RequestOutput` 是**借用内部对象池**的，调用方（entrypoint）还没读完，因此释放被推迟到下一步的 `pre_sampler_sync`，与 GPU 计算重叠。其中 `clone_unpins` 还挂了一个 `cuda.Event`，保证 `unpin_blocks` 在 D2H 拷贝完成后才发生。

**(d) 零拷贝契约**（三处）

| 位置 | 机制 | 风险 |
|---|---|---|
| `SchedulerOutputs.clone_states/clone_blocks/zero_states` | scheduler 每次 `_schedule()` 先 `clear()` 复用 vector，pybind 以 const 引用暴露为 numpy | Python 必须在下一次 `schedule()` 前消费 |
| `process_outputs` 返回的 `first_token_latencies` | `static std::vector` + `py::array(py::buffer_info(...), py::capsule([]{ }))` | 下一次调用即失效 |
| CUDA Graph 输入 buffer | `get_or_create_stacked` 用 `as_strided` 构造等间距行视图，大 batch 捕获后 `_cut` 前缀零拷贝 | capture 流程一改就断言失败 |

> 这类"契约式零拷贝"让 Python/C++ 边界几乎没有序列化开销，但把所有正确性负担转移给了注释与断言。对二次开发者而言，这是**最大的心智负担来源**。

---

### 4.2 调度器内核（`csrc/engine/scheduler.hpp`）

**(a) 三引擎类型 + 步型一致性投票**

`EngineType = {NORMAL, PROMPT, COMPLETION}`（PD 分离预留位）。NORMAL 引擎的难点：**DP 组内所有 rank 必须同时做 prefill 或同时做 decode**，否则集合通信错配。解法是两轮 `allreduce(MIN)` 投票：

```text
ALL 点（每 prefill_attempt_interval=70 个 decode step）：全员 want=1 才 prefill（防陪跑）
ANY 点（每 prefill_any_attempt_interval=100 步）：任一 want=1 即 prefill（防饿死）
第二轮投票：确认真有 rank 排出 prefill，否则回退 decode
```

第二轮投票解决了一个真实的**死锁场景**：`want=true` 只说明 waiting 非空，若所有 rank 都因 slot/块不足而空手，硬做空 prompt step 会清零 `decode_step`，下一步继续投票、继续空手 → 无限循环。

**(b) Prefill 调度的多重硬约束链**（`_schedule_prefill`）

```text
队头可调度性(单调) → prompt 长度上限 → block+state 联合可分配性(OK/LATER/NEVER)
→ max_num_prompts → prefill_max_sum_seqlen_k → chunked 切分(cp_align 对齐)
→ chunk_checkpoints 精确边界 → 多模态图片边界修正 → lookahead_tokens 下限
```

细节亮点：
- **`cp_align = cp_size * 2`**（×2 是 mirror CP），切分时先向下对齐再切，保证 CP all-gather 形状合法；
- **`_adjust_chunk_size`** 保证不在图片内部切开（`MM_TOKEN_IMAGE + MM_TOKEN_IMAGE_PAD*N` 的哨兵区间），并支持"往前收缩/往后扩张"两个方向；
- **`_can_allocate_with_state`** 把 KV block 与 state slot 联立判断——这是"state 也需要分配"这一非标准设计的直接后果；
- **prefix caching 命中**（`block_table` 非空或 `branch_id` 以 `_` 开头）走 fork 路径，需 `_clone_state` + 尾块 copy（`clone_blocks`）。

**(c) Decode 调度：LCFS + 最短输出优先抢占**

```text
prompted → running（fork 需克隆 state + 尾块）
sort by output_len DESC  → 从右端（output 最短）开始分配
block 不足 → _preempt(running[l])  ← 从左侧（output 最长）牺牲
```

即：**保护长输出（已投入的成本），牺牲短输出（重算代价低）**。抢占策略二选一：`stop_preempted_request=true` 直接终止；否则 `reset_for_recompute()` + 放回 waiting 队头重算。**带结构化输出的请求无法重算**（`so_blocks_recompute`），强制终止。

**(d) MTP info slot 池**：slot 0 保留为 padding，真实请求用 $[1, \text{max\_num\_seqs}]$，decode 后断言"已用 slot 数 == running 队列长度"。

**(e) 可观测性内建**：调度期即计算 `attn_tflops` / `total_tflops`（`FlopsEstimator`，支持 YoCo 感知），`min/max_seqlen_k`、`sum_seqlen_qxk`，直接喂给 `StatsLogger` 算 MFU。

---

### 4.3 KV Cache 与 State Cache 布局（`worker/cache_engine.py`）

这是全仓库**信息密度最高的 Python 文件**。核心抽象是把"每层 KV"和"每层滚动状态（SWA ring + 压缩 state）"统一成两张大 uint8 张量：

```python
blocks     : [num_gpu_blocks, block_bytes]          # 分页 KV
gpu_states : [num_state_slots, state_bytes]         # 每 seq 的 StateBlock
num_state_slots = int(max_num_seqs * 1.2) + 1       # DEEPINFER_NUM_STATE_SLOTS_RATIO
```

**每 token 的 KV 字节数**（`_kv_bytes_per_token`）：

| 格式 | data | scale | 合计 | 说明 |
|---|---|---|---|---|
| MXFP8 (`V41_FORMAT`) | 512 B | 16 B (UE8M0, 每 32 通道 1 个) | **528 B** | 主压缩 KV / SWA |
| FP4 (`V41_FP4_FORMAT`) | 256 B | 32 B (每 16 通道 1 个) | **288 B** | `use_fp4_sparse_kv` 时 |
| Index K (FP4) | $\text{index\_head\_dim}/2$ | $\text{index\_head\_dim}/32$ | — | indexer 的 K cache |

由此可反推 $kv\_lora\_rank + qk\_rope\_head\_dim = 512$。

**StateBlock 的分区哲学**（注释写得非常清楚）：

```text
[ stored region: SWA(需落盘) ] [ dropped region: YOCO consumer SWA | compression state ]
                                 ↑ 不落盘，但 PE→CE 要传
state_size_to_disk = 仅 stored SWA 的 inference 窗口
```

- **SWA ring 长度**：`inference = window_size + MAX_BACKTRACKING_TOKENS`（要支持投机回退），`kvstore = inference + PREFILL_STORE_EXTRA_SIZE(128) + YoCo 余量`；
- **压缩 state ring 长度**：`compress_ratio + max_backtracking - 1`；
- **落盘粒度 `KVSTORE_GRAIN = 128`**：只在 128 的倍数处落盘，所以才需要"多存一段"保证多轮命中；
- **YoCo consumer SWA 被放进 dropped 区**——因为"consumer SWA 的恢复依赖 prefill 重算"，不需要落盘。

**对齐约束**：`kvcache_align_to = 128`（MXFP8/FP4 sparse 的 block_stride 要求），`finalize()` 取 `lcm(128, bytes_per_token)`；SWA 每层按 32 B pad；`_alloc_gpu_uint8` 走 `cuda_utils.alloc_tensor`（`cuMemAlloc`，**不计入 torch 显存统计**，因此 profile 时要单独扣 state 大小 ×2）。

**共享 KV 与共享 Index 解耦**：`layer_info_shared_cfg.{layer_kv_map, layer_index_module_map, layer_index_k_map}` 三张映射表，允许"KV 从 A 层复用、index K 从 B 层复用"，且强制"只能指向已处理过的前序层"。

---

### 4.4 注意力子系统：MLA + 稀疏 Index + SWA + 跨层复用

**(a) 分层设计**（`v4_modules/attn.py`）

```text
V4AttentionBase   ── 共享：Q/KV 投影、RMSNorm、RoPE、W_v/O 投影（FP8 einsum）
   ├── V4Attention      ── 主注意力：压缩 KV / 稀疏 index / SWA / YoCo / share-kv
   └── DFlashAttention  ── draft 注意力：独立 draft KV（每步覆写）
```

**(b) 稀疏索引注意力（`index_attn.py::MQA`）——V4 最关键的结构创新**

两阶段 top-k：

```text
① 计算 relu(q·k) * per-head weight  →  p_sum（DeepGEMM fp8_fp4_sparse_mqa_logits）
② 若启用 candidate 机制：block 级 top-k（candidate_topk_blocks × block_size）
   → 只在候选块内做 token 级 top-k（restore_candidate_indices_inplace 还原全局下标）
```

三个工程细节：
- **Index Q/K 恒为 FP4**（`FP4_INDEX_KV_FORMAT`, E2M1 + UE8M0），weights/psum 用 BF16；
- **按内存上限切 chunk**：`compute_chunk_size = min(8192, INDEX_ATTN_MEMORY_LIMIT / psum_row_bytes / 8 * 8)`，注释说明"chunk 太大时 MQA 会降频，需要切块并叠加 topk 才最快"；
- **topk 的 `end` 必须传**：OOB 的 p_sum 未清零，`end=topk_lens` 是正确性前提；decode 侧还要 `convert_kv_index` 把逻辑下标就地转成 kvcache 物理 slot。

**(c) SWA（滑动窗口）+ attention sink**：`sink_mode == "swa"`，`use_swa=False` 仅允许用于 packed-KV（YoCo consumer），代码断言强制。Prefill 时 KV 拼装顺序为 `[swa_cache | 当前 swa kv | 压缩 cache]`，且**必须先 `load_paged_kv_cache` 再 `store_kv_cache`**（slot 可能重叠）。

**(d) 跨层复用（`cross_layer.py` 推断 + `attn.py` 实证）**：`CrossLayerRuntime` 跨层携带 `shared_hidden / pre_rope_kv / topk_indices`，`SharedLayerTopology` 决定谁是 source。Prefill 时 source 层把 gather 的 KV 缓存进 `loaded_prefill_kv`，后续 consumer 复用同一 buffer（按"段"管理：新 source 开始时释放上一段）。

**(e) YoCo**：`n_prefill_layer < num_hidden_layers` 时，后半段是 consumer 层，其 `kv_swa` 由**共享 hidden** 投影（`yoco_swa_share_hidden`），并与 `Q` 解耦，因此不能融合投影。

---

### 4.5 DSpark + 置信度调度：投机解码的在线最优化

**(a) DFlash 结构**（`models/mtp/dflash.py`）

```text
主模型 R 个 target layer 的 hidden  → projection (R·h → h) → main_kv_proj
                                     → 写入 main KV（随主模型 CUDA Graph 一起 replay）
decode: golden_token + noise_token 填充到 dflash_block_size
        → N×DFlashDecoderLayer(draft) → post_norm → lm_head → Markov bias → sample
        → ConfidenceHead(每步独立温度) → 置信度
```

要点：
- **`noise_token_id`** 填充 draft 位置（并行预测 $k$ 个 token，一次 forward）；
- **Markov 头**（`markov_w1/w2`, rank>0 时）为 draft 概率加一阶 token 依赖偏置；
- **`ConfidenceHead` 用 fused kernel 单独输出每步接受概率**，温度支持按步配置（`DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` 可为 5 元组）；
- **主 cache 与 draft cache 分离**；被拒绝的 draft 位置会在 `forward_drafts` 中被覆写，因此**无需按接受结果屏蔽写入**——一个漂亮的"用写入顺序代替条件控制"的技巧。

**(b) 置信度的数学**（`process_MTPInfo`）：每 token 独立置信度 $c_j$，累乘为累积接受率

$$a_j = \prod_{k=1}^{j} c_k ,\qquad j = 1..n_{\text{next}}$$

遇 EOS 时把下一轮所有 draft 置信度置 0（`masked_fill_`），自然抑制无效 draft。

**(c) MTPVarlen：把 draft 长度与 batch 大小当成联合优化变量**（`mtp_varlen.hpp`）

对候选 batch（把所有请求按"当前可用 draft 数"逐个加回），计算

$$\text{throughput}(b) \;=\; \frac{b + \sum_{i \in \text{sel}(b)} a^{(i)}}{\;T_{\text{fwd}}\big(g(b)\big) + T_{\text{smp}}(b)\;},\qquad T_{\text{fwd}}(g) = \text{baseline}(g) + \text{slope}\cdot g$$

取 $\arg\max$，再从"按 accept_rate 降序"的队列里把 $\text{best\_bsz} - n_{\text{seqs}}$ 个额外 query 分配给接受率最高的请求（每请求最多 +1 直到 nextn）。

三个精妙之处：
1. **两代 accept_rate 的分工**：用第 $N-2$ 轮估计 best_bsz（避免"用未来信息"），用第 $N-1$ 轮分配每请求 seqlen_q。代码用 `schedule_cnt` / `process_cnt` 双计数器表达"配对/滞后一步"两种状态，并 `TORCH_CHECK` 二者之差 ∈ {0,1}；
2. **在线回归实测曲线**：`ForwardRuntimeEstimator` 用"两层中位数"（按 graph size 聚合的中位数 ∘ 各 graph 中位数）估计 `slope`（窗口 5000、每 graph 最少 10 样本、`max(0, slope)` 防负外推）；`SamplerRuntimeEstimator` 用 9 样本滑动中位数按 `num_seqs` 分桶；
3. **启动曲线来自离线标定**：`profile_decode_perf()` 遍历所有 CUDA Graph batch size，warmup 1 + 测量 5 + 截尾均值（去首尾），跨 EP 组 all_reduce 取均值，结果写 `DecodePerfCache`（TTL 3600 s）。

> 这意味着 **varlen 投机解码的收益依赖 5 个统计估计量的在线收敛**（forward slope、sampler 中位数、两份 accept_rate、decode 曲线）。冷启动前 ~10 步会偏保守，这与 `_min_samples` 的设计一致。

---

### 4.6 YoCo 有界重放（SWA bounded replay）

**动机**：长上下文 prefill 时，后半段（consumer 层）的 query 数量是成本的线性因子。若这些层只依赖"滑窗 + 少量回溯"，则可以只重放**有界后缀**。

**两种模式**（`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`）：

| 模式 | 做法 | 影响面 |
|---|---|---|
| `mask` | 保留全部 query，只收窄 SWA 可见域为「原滑窗 ∩ 末尾 $K$ token 矩形」 | 无额外布局切换，保守 |
| `compact` | 真正移除截断区 query，metadata 切成 compact 视图 | 省算力，但需 `get_consumer_view()` + full metadata 经 `CrossLayerRuntime` 传给 full-source 路径 |

**正确性约束**（代码里全是断言）：

$$K_{\text{keep}} \ge \text{window\_size} + \text{MAX\_BACKTRACKING\_TOKENS}$$

否则投机回退后 SWA 覆盖不足。同时：
- compact 下 **DFlash target 层必须全在 consumer 半区或全不在**；
- compact 下 **consumer 层不支持 Engram**；
- compact 下 `prompt_logprobs` 被强制关闭（compaction 无法计算被裁行），`mask` 模式的 logp 对矩形前 token 退化；
- compact 的完整 KV 由 `_store_yoco_full_kv` 用 **full 长度 hidden** 提前写入，compact query 只算 top-k。

`examples/yoco/yoco_equiv.py` (52 KB) + `run_yoco_equiv_l2.sh` 专门做 mask/compact 与 off 的等价性验证——**这是该特性唯一的回归防线，也说明其风险等级**。

---

### 4.7 MoE / EP + EPLB

**(a) 并行拓扑**（`config.py::ParallelConfig`）

```text
model_parallel_size = context_parallel_size     # "Attention DP / 均衡 CP"
dp_size  = max(1, ep_size // cp_size)           # 未指定时反推
world_size = max(mp * dp, engram_parallel_size)
约束: (mp*dp) % ep == 0,  world % (mp*dp) == 0,  world % enp == 0
```

即 **注意力按 CP 切、MoE 按 EP 切**，两套并行域在同一 world 上叠合；`engram_parallel_size` 是第三种域（Engram 记忆表的并行）。

**(b) EPLB 运行时**（`eplb/__init__.py`）

```text
冗余专家: num_extra = round_up_to_divisible(ep_size, min_num_extra)   # 默认 32 下界
三张映射: physical_to_logical / logical_to_physical(含 -1 无效位) / logical_count
计数:     expert_count[L, local_physical] 每 step 清零，滚入 window(默认 100)
重排:     每 rearrangement_interval 步（首次提前到 1/4 周期）→ all_reduce → rebalance
shuffle:  单机 NVL 域走 symm_mem 镜像 IPC（每层复用一份 buffer），跨机走 P2P
统计:     每 128 步 all_gather 出 moe_load_balance_ratio(avg, max)
```

工程细节：**同构布局分组**（按 `(num_logical, num_physical, ep_group, (shape,dtype)…)` 分桶，异构层共用一个 balancer），FP8 权重 shuffle 前需把 `.sf.mT` 转回连续布局。`DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE` 允许把重排推迟到 prefill 之后，避免在长 prefill 前抖动。

**(c) 与 benchmark 的耦合**：`examples/README.md` 明确"与 MoE LB flush 碰撞的采样区间要丢弃"，且 benchmark 把 `DEEPINFER_MOE_LB_STAT_INTERVAL` 设为 10 以便 prefill 段也能读到真实 `moe_lb`——**这是把"可观测性副作用"当成一等公民来设计**。

---

### 4.8 采样器：投机验证 + 结构化输出 + 批不变

**(a) 统一入口**（`layers/sampler.py`）

```text
forward()
├─ [prompt]  gather 最后 token → logits → eos 抑制 → SO mask → _prefill_golden_gen
└─ [decode]  _verification()
              ├─ prepare_verification_metadata(ftp, seqlen_q, total_q, max_MTP)
              ├─ apply_token_bitmask_by_step_inplace  (每 step 一份 mask)
              ├─ process_sampling(...)  # top-k/top-p/min-p + 与上轮 draft logprob 比对
              ├─ get_accept_result(...) # 接受前缀 + golden 覆盖 + is_final
              └─ finalize_verification(...) # sampled/verified 打包、置信度误差统计
            然后 mtp_module.decode_forward() 生成下一轮 draft
```

关键设计：**flat 布局**。所有结果按 `[total_q]` 扁平排布，用 `first_token_positions`（每请求在 flat 数组里的起点）+ `seqlens_q` 做切片，避免 Python 侧 per-request 循环。验证结果同时产出 `sampled_tokens` 与 `verified_tokens` 两条流，`num_generated_tokens / num_verified_tokens / batch_confidence_error` 均为 GPU tensor，通过 `post_funcs` 延迟到 post_process 才 `.tolist()`。

**(b) 结构化输出**（`structure_output/matcher.py`）

```text
SO 生命周期:  add_request 时 so_args → SOMatcherBuilder.submit（线程池, 绑 idle CPU）
              → 编译与入队解耦（enqueue 回调），超时 20 s → 请求级中止（aborted_matcher）
每 step:      so_status 状态机 0(未开始) / 1(待启用) / 2(约束中) / -1(判死)
              accept_token 逐步推进；draft token 试探接受，失败则 rollback(k)
              fill_vocab_mask 产出 [bsz, vocab] bitmask（pinned）→ 独立 copy stream → event
投机兼容:      MAX_ROLLBACK_TOKENS = nextn + 2，_progress_history 环形记录
              → 支持"接受 k 个 draft 后回滚 k" 的精确 FSM 状态恢复
特殊能力:      trigger_tokens + bound_token_id（KMP 失配函数驱动），实现
              "先自由生成、命中触发词后才开始约束"的延迟约束
鲁棒性:        任何 matcher 异常只杀单请求（so_status=-1），绝不 raise 到采样主路径
```

> 这是**投机解码与语法约束 FSM 的正确耦合**——大多数框架在这点上要么禁用 SO，要么放弃 rollback。`_progress_history` 上限 `nextn+2` 且超出即抛异常（而非静默错误），是"宁可失败不可出错"的取值。

**(c) 批不变（Batch Invariance）与确定性**

- `DEEPINFER_BATCH_INVARIANT=1`：影响 kernel 选择（`decode_mla_mqa_sparse(batch_invariant=...)`）与**融合路径禁用**——注意 `use_fused_norm_rope_attn_rope_cast` 的条件里明确写了 `not envs.DEEPINFER_BATCH_INVARIANT`（PROMPT 引擎时），说明融合 kernel 在批不变模式下不可用（归约顺序不同）。
- 确定性采样：`BATCH_INVARIANT=1` + `MTP_VARLEN=0` + 固定 `seq_seed`（`process_sampling` 里 `SamplingKernelInputs(seq_seeds, sample_positions)`），NVIDIA only。
- 校验手段：`tests/models/test_batch_invariant.py`、`test_offline_inference.py` 对 `tests/models/offline_decode_ppl.json` 做**逐样本精确比对**（不是容差比对），跨节点用文件锁 + 按 rank 分片。

---

### 4.9 多模态（VE）通路

- **图像在词表空间里的哨兵**：`MM_TOKEN_IMAGE / MM_TOKEN_IMAGE_PAD = vocab_size + MMTokenOffset.*`，`vocab_size` 校验时放宽 `NUM_SPECIAL`；
- **异步 CPU 预处理**：`MultimodalProcessor` 在 worker 进程里跑（`DEEPINFER_MULTI_MODAL_POOL_SIZE_GB=8`），`fetch()` 每 step 拉取，失败即 abort 该请求（日志明确"通常是损坏的图片/视频"）；
- **chunk 边界不能切图**：调度器侧 `_adjust_chunk_size`；
- **共享池句柄**：`free_multi_modal_pool_handles` / `free_deferred_pool_handles` 在 DeferredRelease 里统一回收；
- **VE 三态引擎**：`sweep/ve_sweep.py` 只加载 VE 权重做纯编码吞吐扫描，`--prompt_mode image` 走端到端。

---

### 4.10 双栈可移植性（NVIDIA SM100 / Ascend 950）

| 机制 | 实现 |
|---|---|
| 平台标签 | `_so/<accelerator>-<uname -m>` 加入 `__path__`，**同 arch 的 nvidia/ascend 产物可共存于同一代码树** |
| 构建切换 | `DEEPINFER_ACCELERATOR` → vendor 包集合 + `BUILD_TARGET=CUDA/ASCEND` |
| 内核双份 | `tilelang_ops/{cuda,asc}/` 一一对应（24 : 24），共享 `cache_common.py` 的格式定义 |
| 硬件抽象 | `deep_cluster/hardware/{nvidia,ascend,cpu}.py`、`PLATFORM.is_gpu / is_npu / fp8_sf_type` |
| 规避点 | 大量 `**({} if PLATFORM.is_npu else {...})` 条件参数（如 `use_unaligned_ks`、`use_ascend_kv_layout`） |
| 未支持项 | `symm_mem` EP shuffle（NPU 无）、部分 CUDA Graph 路径、batch-invariant（NVIDIA only） |
| 初始化顺序 | `__init__.py` 严禁 import torch；NPU 初始化必须在 `ASCEND_RT_VISIBLE_DEVICES` 设置后（`PLATFORM` 懒加载） |

这种"**同一份 Python、两套 kernel、平台能力运行时分派**"的设计，比"维护两个分支"更可持续，但也导致条件分支遍布模型代码。

---

### 4.11 工程化基建

<details>
<summary><b>构建、JIT、CUDA Graph、内存、可观测性（点击展开）</b></summary>

**构建**：`build.sh` → 装 `requirements-<accel>.txt` → 固定 NCCL → `build_vendor.sh`（submodule → `bdist_wheel` → 离线 pip install）→ `scripts/build.sh build`（编译 `csrc`，nanobind/pybind 生成 `_C` stub：`_C/engine.pyi` 30 KB）。

**进程级调优**（`__init__.py` + `base.py`）：
- `CUDA_SCALE_LAUNCH_QUEUES=4x`（多发射队列）、`PYTORCH_NVML_BASED_CUDA_CHECK=1`（避免提前初始化 CUDA 导致 `CUDA_VISIBLE_DEVICES` 失效）、`TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES=1`（EPLB）
- `expandable_segments:True`、`torch.set_num_threads(8)`、`gc.disable()` + `gc.freeze()`（slave worker）+ 每 1000 步 `gc.collect()`
- `set_pdeathsig(SIGKILL)`（父进程死则子进程被杀）
- NUMA 绑定（`bind_numa` / `which_numa`），失败降级为 first-touch

**CUDA Graph**：`CudaGraphRunner(max_batch_size = max_num_seqs × (1+nextn), input_on_device=ZOS)`，`batch_sizes` 步长 8（`CUDAGRAPH_BATCH_SIZE_INTERVAL`）；`DFlashCudaGraphRunner` 与主图**共享 memory pool 与 stream**，把 DFlash 的广播+元数据+draft+采样整体捕获；`destroy_cuda_graphs` 显式销毁（规避 NCCL 依赖导致 process group 销毁挂死）。

**同步纪律**：`forbid_cuda_sync()` 上下文管理器守卫 forward/sample 的 launch 段，`DEEPINFER_DISABLE_SYNC_GUARD` 可关闭——**用运行时断言防止有人写出隐式同步**，这是把性能约束代码化的范例。

**计时**：`cpu_timer`（分阶段 `cpu_timer.record("schedule:cpp")` 等）、`cuda_timer`（事件对）、`cuda_graph_timer`（图内计时，`csrc/graph_timer/`）——三层计时统一命名（`launch_forward_N{n}_B{b}`、`decode_core_attn-part-{layer}`），且只对"代表性层"启用并按同类层数 `scale` 放大，避免打点本身成为开销。

**Profile HTTP**：`ProfileManager` 在 driver GPU 上起 HTTP（默认 6070），每 step `profile_manager.step()` 采样。

</details>

---

## 5. 关键设计模式总结

| 模式 | 体现 | 价值 |
|---|---|---|
| **契约式零拷贝** | borrow vector → numpy empty-capsule；`as_strided` 行视图 | 边界零序列化 |
| **滞后一步 + 事后修正** | ZOS：`_correct_seqlens_qk`、乐观推进/回填 | 换取 CPU/GPU 完全重叠 |
| **延迟释放** | `DeferredRelease` + 下一 step 的 `pre_sampler_sync` | 生命周期与计算重叠 |
| **两阶段/两轮投票** | prefill 步型 ALL/ANY + 二次确认 | 分布式步型一致且不死锁 |
| **在线回归替代静态假设** | MTPVarlen 的 slope/sampler 估计 + 启动曲线 | 投机收益自适应 |
| **能力探测而非编译期分支** | `PLATFORM.is_gpu/is_npu`、`PLATFORM.fp8_sf_type` | 双栈单代码树 |
| **描述符式懒配置** | `EnvVar` + `rearm()` | 环境变量热重载、无初始化顺序陷阱 |
| **失败隔离** | SO 异常只杀单请求；`ignored_seqs` 带 `finish_reason` 回传 | 单请求故障不连坐 NCCL |
| **断言即契约** | 遍布 `TORCH_CHECK` / `assert` + 中文解释 + `SPRINT_VARS` 打印上下文 | 在无类型约束处保住不变量 |
| **可观测性一等公民** | 调度期算 TFLOPS、EPLB 指标、`cpu_timer` 命名规范 | 性能问题可定位 |

---

## 6. 风险、局限与改进建议

### 6.1 作为开源框架的**功能缺口**（README 自述 + 代码印证）

| 缺口 | 影响 | 代码线索 |
|---|---|---|
| 无在线服务前端（HTTP/gRPC） | 只能离线批处理 | `entrypoints/llm.py` 只有一个离线 `LLM` |
| 无 KV cache 系统/落盘服务 | `state_size_to_disk`、`KVSTORE_GRAIN`、`select_state_slice_to_store` 全是「挂点」，需外部实现 | `cache_engine.py` 大量 kvstore 语义但无 store 实现 |
| PD/VE 分离仅"大部分实现" | `EngineType.PROMPT/COMPLETION`、`send_processed_prompts`、`pe_*_save_kvcache` 均为空实现或虚函数 | `llm_engine.py` 里全是 `"虚函数，..."` docstring |
| 无 DP 副本间负载均衡 | 多副本需外部路由 | — |
| 模型单一 | 只支持 `DeepSeek-V4.1-Flash`，`ModelMetadata` 里全是 V4.1 特有断言 | `assert moe is not None, "只支持 MoE 模型（V4.1）"` |

### 6.2 工程风险

1. **硬件门槛极高**：SM100 / CANN 9.2 + GCC 13 + CUDA 13 + Python 3.12，社区几乎无法复现。
2. **维护性**：`scheduler.hpp` 67 KB 单文件、`cache_engine.py` 49 KB 单文件、`tilelang_ops/sampler.py` 49 KB；`attn.py` 46 KB 中一个 `forward` 跨 ~80 行且含 6 条互斥分支。**建议**：拆分 scheduler 为 `PreemptPolicy` / `ChunkPlanner` / `DpVoter` 三个可单测组件。
3. **隐式契约过载**：借用存储、`static std::vector`、`torch.Tensor` 生命周期、`Movable` 包装、`record_stream`——任何一处被破坏都是"偶发数值错误"而非崩溃。**建议**：为三类零拷贝接口引入 debug 模式（拷贝副本 + 校验），CI 用 `DEEPINFER_STRICT_ZEROCOPY=1` 跑一遍。
4. **环境变量爆炸**：60+ 个 env（见附表），且部分互相约束（`YOCO_KEEP ≥ window + backtracking`、`EXPERT_COUNT_WINDOW ∈ (0, interval]`）。**建议**：收敛为单一 TOML/YAML 配置 + 启动期 schema 校验。
5. **文档贫瘠**：`doc/` 只有两张 PNG；核心设计（ZOS 不变量、StateBlock 布局、MTPVarlen 目标函数）只存在于**中文代码注释**里。注释质量很高但**不可被外部检索/索引**。
6. **测试覆盖**：20 个测试文件，`ci` / `gpu` / `long_context` 标记齐全，但**无 CI 配置文件**（无 GitHub Actions），且大量测试需 8×SM100。`test_state_cache_ops.py` 36 KB 是最大单测，说明 state 布局被重点保护。
7. **中英混排注释**：对内高效、对外不友好。

### 6.3 若由我主导，优先级最高的三件事

1. **补 vitest 级"不变量测试矩阵"**：把 ZOS 的 6 条不变量（seqlen 单调、slot 配对、乐观推进上界、借用存储未越界、抢占跳过记账、MTP 计数配对）写成可在 CPU 上模拟的模型测试——这些是目前只能靠 `TORCH_CHECK` 事后发现的错误。
2. **抽出 `KVCacheLayoutPlanner` 为纯函数 + golden 文件**：`StateBlock` 的字节级布局完全可以用 JSON 快照锁定，现在它靠 `finalize()` 里的 3 个 assert 交叉验证。
3. **把 MTPVarlen 的估计器做成可回放的离线仿真器**：给定 accept_rate 轨迹与耗时曲线，离线验证吞吐决策是否最优——目前只能在真卡上"跑着看"。

---

## 7. 附：关键配置速查表

<details>
<summary><b>环境变量（摘自 <code>environ.py</code>，共 60+，此处列核心）</b></summary>

| 分类 | 变量 | 默认 | 作用 |
|---|---|---|---|
| 平台 | `DEEPINFER_ACCELERATOR` | `nvidia` | `nvidia` / `ascend` |
| 调度 | `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | 按 EngineType | ZOS 开关 |
| 调度 | `DEEPINFER_PREFILL_ATTEMPT_INTERVAL` | 70 | ALL 投票点间隔 |
| 调度 | `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL` | 100 | ANY 保底间隔 |
| 调度 | `DEEPINFER_PREFILL_MAX_SUM_SEQLEN_K` | 无限 | prefill 批的 Σseqlen_k 上限 |
| 投机 | `DEEPINFER_USE_NEXTN_DECODE` | 5 | 最大 draft 数（0 关闭） |
| 投机 | `DEEPINFER_ENABLE_MTP_VARLEN` | off | 变长验证（置信度调度） |
| 投机 | `DEEPINFER_MAX_BACKTRACKING_TOKENS` | ≥ nextn+1 | SWA/state ring 回退余量 |
| 投机 | `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` | 1.0 | 支持按步 5 元组 |
| YoCo | `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | `off` | `off`/`mask`/`compact` |
| YoCo | `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` | 0 | 必须 ≥ window + backtracking |
| KV | `DEEPINFER_NUM_STATE_SLOTS_RATIO` | 1.2 | state slot 数比例 |
| KV | `DEEPINFER_PREFILL_STORE_EXTRA_STATE` | off | kvstore ring 重排 |
| KV | `DEEPINFER_PREFILL_STORE_EXTRA_SIZE` | 128 | 额外存储 token 数 |
| 注意力 | `DEEPINFER_INDEX_ATTN_MEMORY_LIMIT` | 10 GiB | indexer chunk 大小上限 |
| EPLB | `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` | 0(关) | 重排周期 |
| EPLB | `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | 100 | 计数窗口 |
| EPLB | `DEEPINFER_MOE_LB_STAT_INTERVAL` | 128 | LB 统计 all_gather 周期 |
| EPLB | `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` | — | 冗余专家下界（32） |
| 图 | `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL` | 8 | graph batch 步长 |
| 图 | `DEEPINFER_ENFORCE_EAGER` | off | 关闭 CUDA Graph |
| 图 | `DEEPINFER_DECODE_PERF_CACHE_TTL_SEC` | 3600 | decode 曲线缓存 TTL |
| 采样 | `DEEPINFER_SAMPLER_FORCE_TOPK` | 0 | 只在 top-k 上 softmax |
| 采样 | `DEEPINFER_BATCH_INVARIANT` | off | 批不变（NVIDIA only） |
| 采样 | `DEEPINFER_COMPUTE_SUM_SMALLER_PROBS` | off | 额外输出 sum_smaller_probs |
| SO | `DEEPINFER_XGRAMMAR_TIMEOUT` | 20 | 编译超时（超时杀请求） |
| SO | `DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES` | 10 GiB | 语法编译缓存 |
| SO | `DEEPINFER_SO_BUILDER_CPUS` | 4 | 编译线程数 |
| 多模态 | `DEEPINFER_MULTIMODAL_POOL_SIZE_GB` | 8 | 图像缓冲池 |
| 调试 | `DEEPINFER_DISABLE_SYNC_GUARD` | off | 关闭 `forbid_cuda_sync` |
| 调试 | `DEEPINFER_PROFILE_HTTP_ENABLE` | off | 开启 HTTP profiling |

</details>

<details>
<summary><b>关键类关系（继承/组合）</b></summary>

```text
LLM ──has──> LLMEngine ──creates──> Scheduler(C++) / BlockManager(C++) ×2
                    │                     └── MTPVarlen (在线吞吐决策)
                    ├──creates──> Worker(driver) : WorkerBase
                    │                   ├── ModelRunner
                    │                   │     ├── CudaGraphRunner
                    │                   │     ├── DFlashCudaGraphRunner
                    │                   │     └── DecodePerfCache
                    │                   └── CacheEngine ──> GPUCacheInterface
                    │                                          ├── KVCacheManager
                    │                                          └── StateBlock
                    └──spawns──> Worker(slave) × (cp_size-1)   [gloo RPC 循环]

DeepseekForCausalLM
├── DeepseekModel
│    ├── VocabParallelEmbedding
│    ├── DeepseekDecoderLayer × N
│    │     ├── self_attn : V4Attention | DFlashAttention
│    │     │     ├── MQA (稀疏 indexer)
│    │     │     └── Compressor (KV 压缩 + 状态环)
│    │     ├── mlp : DeepseekMoE ──> eplb.ExpertLoadBalancer
│    │     └── engram : EngramRuntime (可选)
│    ├── HCHead (可选超连接)
│    └── output_norm
├── DFlash (MTP)  ──> DFlashDecoderLayer × num_nextn_predict_layers
│                 └── ConfidenceHead
└── Sampler
```

</details>

---

## 8. 一句话总结

**DeepInfer 是一份"把推理引擎当作分布式实时系统来设计"的样本**：它把调度器下沉到 C++、把 CPU 后处理全部藏进 GPU 气泡（ZOS）、把 KV 缓存重构成"分页 KV + 每序列 state block"的双张量模型、把投机解码变成一个带在线回归估计的吞吐最优化问题，并用 78 个 TileLang kernel 支撑一条 NVIDIA/Ascend 双栈。它的**架构思想值得逐条借鉴**；它的**隐式契约密度和硬件门槛，也应当作为反面教材被认真记录**。

---
## 🧑‍💻 User

总结精简 重点保留推理后端和上下文管理的技术细节

---
## 😀 默认助手 (7)

# DeepInfer 精简技术摘要

> 只保留两块：**推理后端（执行/调度/内核/投机）** 与 **上下文管理（KV & State 生命周期）**。

---

## 一、推理后端

### 1.1 线程/进程骨架与一次 step

```text
LLMEngine.step()                                   # engine/llm_engine.py
├─ Scheduler::_schedule()   ← C++，GIL 释放，持 BlockManagerMutex
│    → SequenceMetadata[] + SchedulerOutputs（借用内部 vector 存储）
├─ workers.clone_and_zero_states_and_blocks_on_default_stream()
├─ Worker.execute_model_step()                     # worker/worker.py
│    ① forward(N)  launch        ← 零 cuda sync（forbid_cuda_sync 守卫）
│    ② post_process(N-1)         ← sync sample_end_evt → Python 后处理
│    ③ sample(N)   launch
│    ④ pre_sampler_sync         ← DeferredRelease.release / SO pump / finalize_seq
│    ⑤ prompt 或非 ZOS：当场 post_process(N)；ZOS decode：推迟
└─ 返回 uint64 指针数组（零拷贝）
```

**关键点：②③④ 与 ① 在时间上重叠** —— GPU 跑 forward(N) 时 CPU 在处理第 N-1 轮。

进程模型：driver worker 在引擎进程内；其余 CP rank 由 `SpawnProcessWithCrashHeader` 拉起，进入 `cp_broadcast_object` 的 slave RPC 循环，`prctl(PR_SET_PDEATHSIG, SIGKILL)` 防孤儿。

### 1.2 ZOS（零开销调度）：滞后一步 + 三处事后修正

CPU 在调度第 N 步时只有第 N-2 步的接受结果，因此引入三类补偿：

| 补偿 | 位置 | 机制 |
|---|---|---|
| **序列长度修正** | `worker.py::_correct_seqlens_qk` | `prepare_zos_inputs()` 产出 `(remap, seqlens_q, seqlens_k, has_overlap)`；H2D 后用上一步 GPU 上的 `num_accepted_tokens` 就地重算 `compute_seqlens_k_zos` / `compute_seqlens_q_by_confidence_rates`。仅 MTP-varlen 模式需一次 D2H 回传 |
| **KV 计数两阶段提交** | `scheduler.hpp::process_outputs` | `advance_optimistically(seqlen_q)` 乐观推进 → `commit_processed_tokens(token_cnt)` 按实际接受数回填（被拒 draft 不占 KV，避免申请/释放抖动） |
| **资源延迟释放** | `llm_engine.py::DeferredRelease` | `SequenceMetadata` / `RequestOutput` 借用对象池，下一 step 的 `pre_sampler_sync` 才释放；`clone_unpins` 挂 `cuda.Event` 保证 D2H 完成后再 `unpin_blocks` |

**三处零拷贝契约（务必记住）**：
- `SchedulerOutputs.{clone_states,clone_blocks,zero_states}`：scheduler 每次 `clear()` 复用，pybind 以 const 引用 → numpy"空 capsule"，Python 必须在下一次 `schedule()` 前消费；
- `process_outputs` 返回的 `first_token_latencies`：`static std::vector` + 空 capsule；
- CUDA Graph 输入：`get_or_create_stacked()` 用 `as_strided` 构造等间距行视图，大 batch 捕获后 `_cut` 前缀零拷贝。

### 1.3 C++ 调度器（`csrc/engine/scheduler.hpp`，66 KB）

**步型一致性投票（NORMAL 引擎）**：DP 组内必须同步做 prompt 或 decode。

```text
ALL 点（每 DEEPINFER_PREFILL_ATTEMPT_INTERVAL=70 步）：全员 want=1 才 prefill（防陪跑）
ANY 点（每 DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL=100 步）：任一 want=1 即 prefill（防饿死）
第二轮 allreduce(MIN)：确认真有 rank 排出 prefill，否则回退 decode
```
第二轮是为规避真实死锁：`want` 只看 waiting 非空，若全员空手，空 prompt step 会清零 `decode_step` → 无限投票循环。

**Prefill 约束链（顺序即优先级）**：
```text
队头可调度性(单调) → prompt 长度上限 → block+state 联合可分配(OK/LATER/NEVER)
→ max_num_prompts → Σseqlen_k 上限 → chunked 切分(cp_align) → chunk_checkpoints
→ 多模态图片边界修正 → lookahead_tokens 下限
```
- `cp_align = cp_size * 2`（×2 为 mirror CP），切分时先向下对齐；
- `_adjust_chunk_size()` 保证不切在 `MM_TOKEN_IMAGE + MM_TOKEN_IMAGE_PAD*N` 区间内，支持前收缩/后扩张；
- `_can_allocate_with_state()`：KV block 与 state slot **联立**判断（state 也要分配，这是非标准设计）。

**Decode 调度：LCFS + 短输出优先牺牲**
```text
按 output_len DESC 排序 → 从右端（输出最短）分配 slot
block 不足 → _preempt(最左侧，输出最长者) 保护已投入成本
```
抢占二选一：`stop_preempted_request=true` 直接终止；否则 `reset_for_recompute()` 放回 waiting 队头重算。**带结构化输出的请求禁止重算**，强制终止。

**MTP info slot 池**：slot 0 保留给 padding，真实请求用 $[1,\text{max\_num\_seqs}]$；decode 后断言"已用 slot 数 == running 长度"。

**调度期即算 FLOPs**（`FlopsEstimator`，支持 YoCo 感知），产出 `attn_tflops/total_tflops/min|max_seqlen_k/sum_seqlen_qxk` 供 MFU 统计。

### 1.4 执行层

- `ModelRunner`：`_prepare_input()`（driver 建元数据并 `broadcast_tensor_dict` 给 CP follower）→ `execute_main_model()`（CUDA Graph replay 或 eager）→ `execute_sample()`。所有 launch 段包 `forbid_cuda_sync()`，`DEEPINFER_DISABLE_SYNC_GUARD=1` 可关。
- **CUDA Graph**：`CudaGraphRunner(max_batch = max_num_seqs × (1+nextn))`，batch 步长 8；`DFlashCudaGraphRunner` 与主图**共享 memory pool + stream**，把"广播 + 固定 stride 元数据 + draft 层 + 采样（含 pool 写）"整体捕获为一张图；`destroy_cuda_graphs()` 显式销毁以规避 NCCL 依赖导致 process group 挂死。
- **启动期标定**：`profile_decode_perf()` 遍历全部 graph batch size，warmup 1 + 测量 5 + 截尾均值，跨 EP all_reduce，写 `DecodePerfCache`（TTL 3600 s）。
- **内存**：`profile_num_available_blocks()` 峰值测量 → `available = total*util − peak − state_size×2`，`num_gpu_blocks = available // block_bytes`，跨 rank 取均值并向下取偶数。
- **计时三层**：`cpu_timer`（阶段）/ `cuda_timer`（事件对）/ `cuda_graph_timer`（图内），命名规范如 `launch_forward_N{n}_B{b}`、`decode_core_attn-part-{layer}`，只对代表性层打点并按同类层数 `scale` 放大。

### 1.5 内核层

- `tilelang_ops/{cuda,asc}/`（约 78 文件，cuda:asc ≈ 1:1）：`fused_transform`(47 KB)、`fused_sample`、`sampler`(49 KB)、`store/load_paged_kv_cache`、`prepare_v4_decode_metadata`、`compress`、`norm`、`gather_*_zos`、`structure_output`、`random_sampling`、`sum_smaller_probs`、`kv_index_cat`、`swa_index`、`candidate_blocks`。
- 核心融合：`fused_norm_rope_hadamard_cast_store_kvcache()` —— **RMSNorm + RoPE + Hadamard + 量化(FP8/FP4, UE8M0) + 写 KV cache** 一体化，是 attention 主路径的标准入口。
- vendor：`FlashMLA`(MLA)、`DeepGEMM`(FP8/FP4 GEMM、`fp8_einsum`、`*_mqa_logits`、`get_*_schedule_metadata`)、`DeepEP`(MoE all2all)、`DeepSelect`(topk)、`TileKernels`(`per_token_cast`, `engram_hash`)。

### 1.6 投机解码（DSpark / DFlash + MTPVarlen）

**DFlash 结构**：主模型 R 个 target layer 的 hidden → `projection` → `main_kv_proj` → 写 main KV（**随主模型 CUDA Graph 一起 replay**）；decode 时 `golden_token` + `noise_token` 填到 `dflash_block_size`，一次 forward 并行出 $k$ 个 draft（可选 Markov 一阶偏置 `markov_w1/w2`），`ConfidenceHead` 每步独立温度输出接受概率。被拒 draft 的位置由后续 `forward_drafts` 覆写 → **无需按接受结果条件屏蔽**。

**置信度数学**（`process_MTPInfo`）：每 token 独立 $c_j$，累乘

$$a_j=\prod_{k=1}^{j}c_k,\quad j=1..n_{\text{next}}$$

遇 EOS 把下一轮 draft 置信度置 0。

**MTPVarlen（`csrc/engine/mtp_varlen.hpp`）：把 draft 长度与 batch 大小联合优化**

$$\text{throughput}(b)=\frac{b+\sum_{i\in\text{sel}(b)}a^{(i)}}{\;\text{baseline}\big(g(b)\big)+\text{slope}\cdot g(b)+T_{\text{smp}}(b)\;}$$

取 $\arg\max$ 得 `best_bsz`，再把 $\text{best\_bsz}-n_{\text{seqs}}$ 个额外 query 按 accept_rate 降序分配给各请求（每请求至多 +1，上限 nextn）。

- **两代 accept_rate 分工**：第 $N-2$ 轮估 `best_bsz`、第 $N-1$ 轮分配 `seqlen_q`；用 `schedule_cnt`/`process_cnt` 双计数器表达"配对/滞后"，断言差值 $\in\{0,1\}$；
- **在线回归**：`ForwardRuntimeEstimator` 两层中位数（按 graph size 聚合 ∘ 各 graph 中位数），窗口 5000、每 graph ≥10 样本、`max(0, slope)` 防负外推；`SamplerRuntimeEstimator` 9 样本滑动中位数按 `num_seqs` 分桶，不足则回退全局。

### 1.7 采样与结构化输出

- **flat 布局**：所有结果按 `[total_q]` 排列，用 `first_token_positions` + `seqlens_q` 切片，杜绝 per-request Python 循环；`num_generated/verified_tokens`、`batch_confidence_error` 均为 GPU tensor，经 `post_funcs` 延迟到 post_process 才 `.tolist()`。
- **decode 验证流**：`prepare_verification_metadata` → `apply_token_bitmask_by_step_inplace`（每 step 一份 mask）→ `process_sampling`（top-k/top-p/min-p + 与上轮 draft logprob 比对）→ `get_accept_result`（接受前缀 + golden 覆盖）→ `finalize_verification`。
- **SO（xgrammar）**：`SOMatcherBuilder` 线程池异步编译（超时 20 s → 只杀该请求）；`so_status` 状态机 0/1/2/-1；draft 试探接受 + `rollback(k)`，`MAX_ROLLBACK_TOKENS = nextn + 2` 且 `_progress_history` 环形记录，超限**抛异常而非静默错误**；支持 `trigger_tokens + bound_token_id`（KMP 失配函数）实现"自由生成→命中触发词后才约束"。任何 matcher 异常只置 `so_status=-1`，绝不 raise 到采样主路径。

---

## 二、上下文管理（Context / KV）

### 2.1 双张量模型（`worker/cache_engine.py`）

```python
blocks     : [num_gpu_blocks, block_bytes]     # 分页 KV（所有层叠在一起）
gpu_states : [num_state_slots, state_bytes]    # 每序列的滚动状态 StateBlock
num_state_slots = int(max_num_seqs * 1.2) + 1  # DEEPINFER_NUM_STATE_SLOTS_RATIO
```

- `tokens_per_block = 128`（`ModelMetadata.resolved_tokens_per_block`，不再是可调 block_size）；
- `kvcache_align_to = 128`（MXFP8 / FP4 sparse 的 block_stride 要求），`finalize()` 取 `lcm(128, bytes_per_token)`；
- 每层 KV 起点 `% 32 == 0`（层内 32 B pad，`padding_bytes` 会打日志）；
- 分配走 `cuda_utils.alloc_tensor`（`cuMemAlloc`，**不计入 torch 统计**，profile 时需单独扣 state×2）。

### 2.2 每 token 字节与量化格式

| 用途 | 格式 | data | scale | 合计 |
|---|---|---|---|---|
| 主压缩 KV / SWA | MXFP8 `V41_FORMAT` | 512 B | 16 B（UE8M0，每 32 通道） | **528 B** |
| 主压缩 KV（`use_fp4_sparse_kv`） | FP4 `V41_FP4_FORMAT` | 256 B | 32 B（每 16 通道） | **288 B** |
| Index K（indexer） | FP4（E2M1+UE8M0） | $\text{index\_head\_dim}/2$ | $\text{index\_head\_dim}/32$ | — |

由 528/288 反推 $kv\_lora\_rank + qk\_rope\_head\_dim = 512$。Index Q/K 恒 FP4，weights/psum 用 BF16。

### 2.3 StateBlock 分区（上下文状态的核心设计）

```text
[ stored region: SWA(需落盘) ][ dropped region: YOCO consumer SWA | compression state ]
                                  ↑ 不落盘，但 PE→CE 要传
state_size_to_disk = 仅 stored SWA 的 inference 窗口
```

| 区域 | 长度 | 依据 |
|---|---|---|
| SWA inference ring | $\text{window\_size}+\text{MAX\_BACKTRACKING\_TOKENS}$ | 投机回退要余量 |
| SWA kvstore ring | inference $+\ (\text{PREFILL\_STORE\_EXTRA\_SIZE}+ \text{YoCo 余量})$ | 只在 `KVSTORE_GRAIN=128` 的倍数落盘，需多存保证多轮命中 |
| 压缩 state ring | $\text{compress\_ratio}+\max(0,\text{backtracking}-1)$ | 环形 + 回退 |
| SWA 每层 pad | 32 B | 整个 block 按 $\mathrm{lcm}(32,\text{swa\_bytes\_per\_token})$ 对齐 |

**YoCo consumer SWA 被放进 dropped 区**——因为它的恢复依赖 prefill 重算，不需要落盘。

### 2.4 压缩与共享

- **压缩率仅 $\{C_1, C_2\}$**（`ModelMetadata` 断言，C4/C128 已移除；`compress_overlap`、`use_ape` 也已移除）；每 block 存 `block_size / ratio` 个 token；
- **三张共享映射**解耦 KV 与 index：`layer_info_shared_cfg.{layer_kv_map, layer_index_module_map, layer_index_k_map}`，强制"只能指向已处理的前序层"，`ica_cache` / `ca_cache` / `state` 可分别复用；
- **`StateBlock` 与 `KVCacheManager` 的交叉校验**：`finalize()` 用 3 个 assert 验证 `process()` 累加的 offset 与预计算 `total_bytes_*` 一致（防两处逻辑 drift）。

### 2.5 上下文生命周期

```text
add_request → Sequence（prompt / compressed_prompt 双份；Engram 需压缩）
   ├─ block_table.reserve((len + max_tokens + block - 1)/block + 1)   # 预留容量
   └─ SO 异步编译（可延后入队）
schedule
   ├─ 首次分配：_can_allocate_with_state → acquire.allocate_seq
   ├─ fork/prefix-cache：clone_states/clone_blocks + _clone_state（尾块 copy）
   └─ decode：_append_slot（多退少补 block_table）
process_outputs → advance_optimistically → append tokens → commit_processed_tokens
preempt → 终止 or reset_for_recompute（清 cache.num_tokens / inflight / MTP 计数）
free_seq → release_mtp_info_slot + free state + free blocks
```

**Prefix caching / fork**：`block_table` 非空或以 `_` 开头的 `branch_id` 视为命中；fork 时 `pin_blocks` + 克隆 state，尾块（写了一半）必须 copy；`prompt_incomplete` 时进 `waiting.emplace_front`，否则进 `prompted`（等待 PD 的 CE 侧 admit）。

### 2.6 Prefill 侧上下文构造

- **chunked prefill**：`max_num_batched_tokens` 默认 $\max(\text{max\_model\_len}, 4096, \text{max\_num\_seqs}\times(1+n_{\text{next}}))$；chunk 大小对齐 `cp_align`；
- **`chunk_checkpoints`**：外部指定精确切点（供 KV 落盘/PD 迁移在语义边界对齐）；
- **`runtime_max_model_len`**：$\text{max\_model\_len} + \underbrace{n_{\text{next}}}_{\text{draft}} + \underbrace{n_{\text{next}}+1}_{\text{ZOS 调度}} + \underbrace{n_{\text{next}}}_{\text{draft}}$，保证 RoPE cache 等按 position 索引的 buffer 够用（调度期即断言 `position_end ≤ runtime_max_model_len`）；
- **KV 拼装顺序**（prefill attention）：`[swa_cache 已缓存 | 当前 swa | 压缩 cache]`，且**必须先 `load_paged_kv_cache` 再 `store_kv_cache`**（slot 可能重叠）；
- **CP**：mirror CP 布局，`mirror_cp_index` / `mirror_cp_reverse_index` 做重排与回排，`cp_mirror_len` 是行数基准。

### 2.7 上下文计算量削减：YoCo 有界重放

$n_{\text{prefill\_layer}} < n_{\text{layer}}$ 时后半段为 consumer 层，其 `kv_swa` 由**共享 hidden** 投影（`yoco_swa_share_hidden`），与 Q 解耦（不能融合投影）。

| 模式 | 做法 | 代价 |
|---|---|---|
| `mask` | 保留全部 query，只把 SWA 可见域收窄为「原滑窗 ∩ 末尾 $K$ 矩形」 | 不省 query 计算 |
| `compact` | 真正删掉截断区 query，metadata 切 `get_consumer_view()`，full metadata 经 `CrossLayerRuntime.consumer_full_metadata` 交给 full-source 路径 | 省算力，但 full KV 需由 `_store_yoco_full_kv()` 用 full 长度 hidden 提前写入 |

正确性铁律：

$$K_{\text{keep}} \ge \text{window\_size} + \text{MAX\_BACKTRACKING\_TOKENS}$$

外加：compact 下 DFlash target 层必须全在 consumer 半区或全不在；consumer 层不支持 Engram；compact 强制关闭 `prompt_logprobs`。`examples/yoco/yoco_equiv.py`(52 KB) 是唯一等价性防线。

### 2.8 上下文外溢（落盘 / PD）挂点

| 抽象 | 作用 |
|---|---|
| `KVSTORE_GRAIN = 128` | 只在 128 token 倍数处落盘 |
| `state_size_to_disk` | 只含 stored SWA 的 inference 窗口，按长度截断即排除整个 dropped 区 |
| `select_state_slice_to_store()` | 从 state cache 取出要落盘的一段；`TO_CE=True` 时额外产出 dropped 区（consumer SWA 按 ring 重排，compression state 直拷） |
| `load_state_from_store_to_prefill_cache()` | 反向灌回 |
| `EngineType.PROMPT/COMPLETION` + `pe_*_save_kvcache` / `send_processed_prompts` | PD 分离虚函数，**本次开源为空实现** |

多模态上下文：图像以 `vocab_size + MMTokenOffset.{IMAGE, IMAGE_PAD}` 哨兵写入 token 流，`MultimodalProcessor` 在 worker 进程异步预处理（池 8 GB），`fetch()` 每 step 拉取，失败即 abort 该请求。

---

## 三、必须在改动前知道的 8 条不变量

1. `SchedulerOutputs` 的三个 vector 是**借用存储**，Python 必须在下一次 `schedule()` 前消费完。
2. ZOS 下 CPU 侧 `seqlen_q/k` **永久滞后一步**，任何新增的 CPU 侧依赖都需走 `_correct_seqlens_qk`。
3. `cache.num_tokens` 走"乐观推进 → 回填"，中途值 ∈ $[\text{len}, \text{len}+n_{\text{next}}]$。
4. 在途被抢占的 seq（`status == WAITING` 且 decode 步）必须**跳过全部记账**（`process_outputs` 与 `process_MTPInfo` 两处判据必须一致）。
5. MTP 不变量：`schedule_cnt − process_cnt ∈ {0,1}`；`process_cnt` 不得抢跑。
6. StateBlock 的 `process()` 累加 offset 必须与 `StateBlock` 预计算一致（`finalize()` 三个 assert）。
7. CUDA Graph 的输入 buffer 必须是**等间距行视图**，否则 `get_or_create_stacked` 断言失败。
8. 带结构化输出的请求**不可抢占重算**（FSM 状态无法重建）。

---

## 四、主要风险（浓缩）

- 单文件过重：`scheduler.hpp` 66 KB、`cache_engine.py` 49 KB、`tilelang_ops/sampler.py` 49 KB、`attn.py` 46 KB。
- 隐式契约密集（借用存储 / static buffer / `Movable` / `record_stream`），破坏后表现为**偶发数值错误**而非崩溃。
- 硬件门槛：SM100 或 CANN 9.2 + CUDA 13 + GCC 13 + Python 3.12。
- 60+ 环境变量且互相有约束（`YOCO_KEEP ≥ window+backtracking`、`EXPERT_COUNT_WINDOW ∈ (0, interval]`）。
- `doc/` 只有两张 PNG；核心设计只存在于中文代码注释中。
- 缺口：无在线服务前端、无 KV store 实现、PD/VE 分离为虚函数、无 DP 副本间负载均衡、仅支持 V4.1-Flash。

---
## 🧑‍💻 User

似乎跟主流推理框架也没很大的差异？

---
## 😀 默认助手 (7)

你的直觉**基本正确**——如果只看"有哪些模块"，这份代码和 vLLM V1 / SGLang / TRT-LLM 的骨架是同一张图。我应该更早把"哪些是标配、哪些是真差异"标清楚。下面重新校准。

---

## 一、先说"确实一样"的部分（占 80%）

| 能力 | DeepInfer 实现 | 主流对应物 |
|---|---|---|
| 分页 KV + block_table | `block_manager.hpp` | vLLM PagedAttention |
| 连续批处理 / chunked prefill | `_schedule_prefill` | vLLM / SGLang |
| Prefix caching / fork | `branch_id` 前缀 + `pin_blocks` | vLLM prefix cache、SGLang RadixAttention |
| CUDA Graph 按 batch 分桶 + padding dummy seq | `CudaGraphRunner`、`_placeholder_seq_metadata` | vLLM / SGLang / TRT-LLM 都是这套 |
| 调度与输出处理重叠 | ZOS | SGLang overlap scheduler；vLLM V1 也以 zero-overhead scheduling 为设计目标（据我所知） |
| 结构化输出 bitmask | xgrammar 封装 | vLLM / SGLang 都直接调 xgrammar |
| MoE EP + EPLB + 冗余专家 | `eplb/` | vLLM/SGLang 均已有 EPLB；`DeepEP` 本来就是 DeepSeek 自己开源的 |
| PD 分离钩子 | `EngineType.PROMPT/COMPLETION` 虚函数 | vLLM / SGLang / Mooncake / Dynamo |
| 批不变 + 确定性采样 | `DEEPINFER_BATCH_INVARIANT` | vLLM 有同名 `VLLM_BATCH_INVARIANT`（据我所知） |
| 动态 draft 长度 | MTPVarlen | vLLM 已有 dynamic num_speculative_tokens（据我所知） |
| FP8 KV / 量化 cache | `V41_FORMAT` | TRT-LLM / vLLM |
| 稀疏注意力 indexer | `MQA` | vLLM/SGLang 已支持 DeepSeek 的 sparse attention（V3.2 起） |

**这些都是"入场券"，不是差异。** 我之前把它们逐条列出来，容易让人误以为每一条都是贡献。

---

## 二、真正不一样的地方（只有 5 条，且分三档）

### 档 1：结构层面真的少见（2 条）

**① 调度器是 C++ 单点真理，且与 Python 之间是"借用存储"契约**

主流（vLLM V1、SGLang）调度器都在 Python；TRT-LLM 有 C++ executor，但走的是正规序列化/IPC。DeepInfer 的做法是：

```cpp
// scheduler 内部 vector 每步 clear 复用，SchedulerOutputs 以 const 引用借用
// pybind 侧再包成 numpy"空 capsule" → Python 侧零拷贝、零生命周期保护
```

代价是所有正确性负担压在注释与断言上。**这是"为了抹掉最后一次 Python↔C++ 拷贝而愿意欠债"的取舍**，方向罕见。

**② State 作为一等公民，且有 `stored / dropped` 分区**

```text
[ stored: SWA ring (落盘) ][ dropped: YOCO consumer SWA | compression state (不落盘) ]
```

主流框架把 SWA 当"另一种分页层"，Mamba 类状态才用 per-sequence state block。这里的独特之处是把**有状态层（SWA ring + 压缩 state ring）与"落盘粒度 `KVSTORE_GRAIN=128`"绑定设计**——ring 长度 = 窗口 + 回退余量 + 落盘对齐余量 + YoCo 余量，且 dropped 区在 PE→CE 时要"consumer SWA 按 ring 重排、compression state 直拷"。**"上下文状态怎么在存储/迁移边界上保持可用"这个问题，主流开源基本没解**。

### 档 2：半步领先（2 条）

**③ YoCo 有界重放（prefill 截断）**

利用"后半段层感受野有界"直接**跳过截断区的 prefill 计算**，分 `mask`（只收窄可见域）与 `compact`（真删 query，metadata 切视图）两档。类似的"少算 prefill"思路（MInference、稀疏 prefill）存在，但把它做成**可切换 + 有等价性回归脚本（`yoco_equiv.py` 52 KB）+ 与投机解码回退余量强耦合**的工程闭环，我没在其他框架见过同等完成度。

**④ 投机解码当作"在线吞吐最优化"而非"固定 draft 长度"**

$$\text{throughput}(b)=\frac{b+\sum a^{(i)}}{\text{baseline}(g(b))+\text{slope}\cdot g(b)+T_{\text{smp}}(b)}$$

差异不在"动态调 draft 数"（vLLM 也有），而在：**batch 大小与每请求 draft 长度联合优化** + **两代 accept_rate 严格分工**（第 N-2 轮定量、第 N-1 轮分配，用 `schedule_cnt/process_cnt` 双计数器表达滞后）+ 两层中位数在线估计 slope。这是控制论式的做法。

### 档 3：只是工程细节，但决定成败（1 条）

**⑤ 两轮 DP 投票 + ALL/ANY 双间隔**

```text
ALL 点(70 步)：全员有活才 prefill  → 防"一个 rank 干活、其余陪跑"
ANY 点(100 步)：任一有活就 prefill → 防孤立 rank 饿死
第二轮投票：确认真排出 prefill，否则回退 decode → 防空 prompt step 清零 decode_step 导致的死循环
```

"DP 组步型必须一致"是共识（vLLM DP attention 也要同步），但**用两轮投票把"饥饿"和"死锁"同时消掉**，这个具体构造我没在别处见过。这类东西看着不起眼，实盘一崩就是 NCCL 超时。

---

## 三、为什么"看起来差不多"——三个真实原因

**1. 差异化部分被有意识地扣留了。**
README 明写"构成完整服务所必需、但本次未包含"：**KV cache 系统、VE/Prefill/Decode 分离、DP 副本间负载均衡**。而这三件恰恰是生产环境里最难、最能拉开差距的。**开源释放的是 vLLM 形状的骨架 + 模型特化内核，把"服务层护城河"留下了。** 所以看到"和主流差不多"是结构性必然，不是错觉。

**2. 它的差异主要落在"常数与顺序"，而不是"模块图"。**
`70 / 100 / 0.9 / 1.2 / 128 / 8`、LCFS 从哪端牺牲、先 load 后 store、`cp_align = cp_size*2`……这些读起来像普通工程，但**性能就在这些常数里**。框架之间 90% 同形，胜负在那 10% 的细节；这份代码的细节密度异常高，但呈现形态不"性感"。

**3. 大量"差异"其实是模型特化，被误读成框架能力。**
FP4 indexer、`engram_hash`、`hc/mhc`、`markov_w1/w2`、压缩率只支持 $C_1/C_2$ —— 这些换个模型就全废。**它们是"为 V4.1 把 kernel 写到极致"，不是"通用框架设计"**。这也是为什么它只有 3.2 MB / 约 330 文件：它是**瘦而深的专用栈**，不是 vLLM 那种胖平台。

---

## 四、一个可验证的判据

| 场景 | 会不会真的更快 |
|---|---|
| 长上下文 + 投机解码 + 高并发 decode | **会**，ZOS 重叠 + MTPVarlen 联合优化 + SWA/state ring 三者叠加 |
| 超长 prefill（128k+） | **会**，YoCo compact 直接砍掉 consumer 层截断区算力 |
| 单请求、短上下文、无投机 | **基本无差别**，差异全在常数级 |
| 换模型（非 V4.1） | **不可用**，断言会直接拦死 |

---

## 五、结论（修正版）

一句话：**架构上它是"vLLM V1 骨架 + TRT-LLM 式的 C++ 关键路径 + DeepSeek 私有模型特化"，没有范式创新；真正的差异化是"上下文状态的可迁移性设计"和"把 CPU 开销彻底抹掉的决心"，而不是"有没有某个模块"。**

如果要"抄"，值得抄的只有三条：

1. **State ring 与落盘粒度的联合设计**（`stored/dropped` 分区 + `state_size_to_disk` 截断语义）—— 这是为有状态层准备上下文迁移路径，主流开源是空白；
2. **两轮 DP 投票消除饥饿 + 死锁**；
3. **投机解码的"两代 accept_rate 分工"**（避免用未来信息做当前决策的严谨性）。

其余部分：读它是为了知道"极致优化后长什么样、要付什么代价"，不是为了学"新架构"——**它没有新架构**。

---
## 🧑‍💻 User

README 明确说了不开源：完整 KV cache 系统、P/D 编排、DP 副本间负载均衡。而这三块恰恰是 Mooncake、Dynamo、SGLang 这些"框架"真正的护城河所在——KV 池化、路由策略、跨副本调度。DeepInfer 开源的是"计算侧"（kernel、模型、调度器），把"系统侧"藏起来了。所以拿它和 vLLM 比，当然像。


最后一句可能更重要：推理框架的差异从来不在框架本身，而在模型-框架协同设计。SWA replay 之所以能成立，是因为 V4.1 用了 SWA + YOCO；DSpark 能成立，是因为模型自带 MTP head 和 confidence head。vLLM 拿到同样的模型也能做，但它不会为一个模型改架构。DeepSeek 是自己造模型自己造框架，所以能做这种"贴身定制"——这才是 155 万亿 token/天、38 美元/GPU·小时背后的东西，代码只是结果。

那么我感觉的”没差异“，很大程度是因为有差异的部分没开源吗？

---
## 😀 默认助手 (7)

## 直接回答：是，但只占一半左右；你第二段其实给出了更重要的那一半

我先把你的论断拆成可称重的三层，因为"没差异"是三个原因叠加的**合成错觉**，而且它们的性质完全不同。

| 原因 | 权重 | 性质 | 能否用"多开源点代码"消除 |
|---|---|---|---|
| ① 系统侧被扣留（KV 池化 / 路由 / 跨副本调度） | ~40% | **可见性缺失** | 能 |
| ② 模型-框架协同的差异"长得不像差异" | ~40% | **认知偏差** | 不能 |
| ③ 发布版对计算侧本身也做了裁剪 | ~20% | **又一个可见性缺失** | 能 |

---

### ① 你说对了，而且代码里有"墓碑"作证

不只是 README 那三行"not included"。更硬的证据是我在 `model_metadata_utils.py` 里读到的**主动移除**：

```python
if not set(self.kv_compress_ratio_list) <= {0, 1, 2}:
    raise ValueError("Only C1/C2 compression is supported; C4 and C128 have been removed")
if attn.cfg.get("compress_overlap", False):
    raise ValueError("Compression overlap is no longer supported")
if any(self.kv_compress_ratio_list) and cfg.get("use_ape", True):
    raise ValueError("Compression APE is no longer supported; use an APE-free checkpoint")
```

`C4/C128` 的压缩路径、`compress_overlap`、`APE` —— 这些不是"没写"，是"写了又删，还留了断言拦人"。再加上 PD 一堆空实现的虚函数（`"虚函数，..."` 的 docstring）、`EngineType.COMPLETION` 的存在但无对端、`offline_inference` 这条唯一通路。

**所以准确说法不是"开源了完整的计算侧、藏了系统侧"，而是"开源了一个可离线跑通的计算侧子集，系统侧与一部分计算侧能力一起被削掉了"。** 这比你的判断还要更悲观一点——但恰恰更支持你的结论。

---

### ② 但"贴身定制"这一层的差异，本质上是不可见的（这才是关键）

这是你第二段讲对的东西，我想把它讲得更狠一点：**协同设计的收益，在代码里的表现形式是"删掉泛化"，而"少写的东西"不产生创新感。**

具体到这份代码：

- 因为 V4.1 保证 $kv\_lora\_rank + qk\_rope\_head\_dim = 512$，所以每 token 528 B 可以**硬编码**、128 B 对齐可以**写死**、kernel 只需支持**一种** KV 布局；
- 通用框架必须支持几十种 `(dtype × layout × quant × head_dim)` 组合 → 只能上多态/模板/dtype dispatch → **每一层都多一次分支、每一次 layout 变化都多一次 gather**；
- 因为模型自带 MTP head 和 confidence head，`DSpark` 才敢用 $\prod c_k$ 做在线接受率估计——**vLLM 拿到同一个 checkpoint 也能做，但它不能假设"一定有 confidence head"，所以只能退回固定 draft 长度**；
- 因为 V4.1 是 SWA + YoCo，`bounded replay` 才成立——**框架不能假设"第 40 层之后感受野有界"，这是模型给的承诺**。

这类收益在代码审阅里是**隐形的**：你看到的是"断言、常数、被删掉的分支"，而不是"新算法"。所以它读起来像普通工程，实际上每一行 assert 背后都是一个"我们不需要支持 X"的成本节约。

**"看起来普通"本身就是深度定制化的证据。** 一个为通用性买了单的框架，代码反而更"丰富"、更"有设计感"。

---

### ③ 用可证伪的方式检验这个判断

如果想验证"差异到底在哪"，有三件事可以做，都不靠感觉：

1. **数通用性代码的占比**：grep `PLATFORM.is_gpu`、`dtype ==`、`layout=`、`if.*use_fp4` 这类分派点的密度。这份代码的分派几乎全部收敛在 `PLATFORM`（双栈）而不是 `model × dtype × layout`（多组合），说明它买的是"两个后端"而不是"任意组合"。
2. **找"墓碑"**：所有 `has been removed` / `no longer supported` / `只支持 V4.1` 的断言，数量与"定制深度"正相关。这份代码里非常多。
3. **比 kernel 调用序列**：拿同一 checkpoint 在 vLLM 上抓 nsys，比 kernel 数量与融合边界。**差异不在"用了哪个模块"，而在"多少个 kernel 被合成一个、元数据在哪算"**（比如 `prepare_v4_decode_metadata` 整个 decode 元数据由**一个 GPU kernel** 产出，而不是 Python 侧拼）。

---

### 一个必要的校准（关于你引的数字）

"155 万亿 token/天、38 美元/GPU·小时"这两个数字我无法核实，而且 155T tokens/天 与公开讨论过的 DeepSeek 服务规模相比高出若干数量级——建议别把它当论据用，容易让整个论证被一个错误数字带偏。**"自造模型 + 自造框架 ⇒ 允许贴身定制"这条因果链本身是成立的，不需要靠那两个数字撑。**

---

### 最后：你的问题问对了一半

- **"没差异"的 40% 确实来自"有差异的部分没开源"**——系统侧 + 被裁剪的计算侧；
- 但另外 40% 来自一个更根本的东西：**推理框架的差异分层，越靠近集群越像"系统"，越靠近模型越像"常数"**。前者你藏起来了，后者你看见了却认不出来。

所以更准确的说法是：**你感觉到的"没差异"，一半是因为没给你看，一半是因为给你看了你也不会觉得那是差异。** 而这两半同源——**正因为是自己造模型，才敢把通用性删干净；而删掉通用性恰好是唯一"显眼"的工程量。** 留下的常数、断言、顺序，才是那 155…（咳，那笔账）真正所在的地方。

---
## 🧑‍💻 User

精读以下文件（7 个）：
csrc/engine/scheduler.hpp
csrc/engine/block_manager.hpp
csrc/engine/sequence.hpp
csrc/engine/mtp_varlen.hpp
csrc/engine/zos.cpp
deepinfer/model_executor/eplb/policy.py
deepinfer/model_executor/eplb/shuffle.py
deepinfer/structure_output/compiler.py

回答调度与上下文管理的心脏、序列/KV 语义、变长验证、EPLB 落地、涉及 worker 执行、kernel、模型结构的部分逐条分析

---
## 😀 默认助手 (7)

# 七文件精读：逐条分析

> 说明：抓取管线会吞掉 C++ 中的模板实参（`std::optional>` 实为 `std::optional<std::vector<int>>`，`std::tuple,` 实为 `std::tuple<...>`），下文按语义还原并标注。凡属推断均标「（推断）」。

---

## 0. 七文件的关系：一条"上下文状态"的单向通道

```text
                 ┌──────────────────────────────────────────────┐
                 │ sequence.hpp   状态容器（唯一真理）           │
                 │  Sequence{ request,input,generation,          │
                 │            cache,mtp,structured_output,       │
                 │            schedule }                          │
                 └───┬───────────────────┬──────────────────┬────┘
        borrow(指针/引用)│                  │ 读改            │ 读改
                 ┌───▼─────────┐   ┌──────▼──────┐   ┌───────▼────────┐
                 │block_manager│   │ scheduler   │   │   zos.cpp      │
                 │  分页分配   │◄──┤ 心脏/记账   ├──►│ 输入装配+修正  │
                 └─────────────┘   └──────┬──────┘   └───────┬────────┘
                                          │                  │
                                  ┌───────▼──────┐   ┌───────▼────────┐
                                  │ mtp_varlen   │   │ worker/kernel  │
                                  │ 决定 seqlen_q│   │ 消费 pinned buf│
                                  └──────────────┘   └────────────────┘

  (旁路，不进调度热路径)
  eplb/policy.py ──决策──► eplb/shuffle.py ──搬运──► MoE kernel 读映射表
  structure_output/compiler.py ──编译──► matcher.py ──mask──► sampler
```

**核心结论先说**：这 7 个文件构成一条"**单写者、多借用者、两次修正**"的状态流水线——`Sequence` 是唯一可变状态；`SequenceMetadata` 是它的**只读投影**（借用指针/引用）；`BlockManager` 只管物理块与引用计数；`Scheduler` 是唯一写者与唯一记账者；`zos.cpp` 负责把"被 GPU 推翻的 CPU 假设"重新注入。

---

## 一、调度与上下文管理的心脏

### 1.1 三队列状态机（`scheduler.hpp`）

```cpp
std::deque<SequencePtr>             waiting;    // 未分配/被抢占重算/多模态不可调度
std::deque<pair<SequencePtr,bool>>  prompted;   // PE 完成、待 CE admit；bool = forked
std::vector<SequencePtr>            running;     // 已分配、可 decode
```

**状态转移（完整，逐条）**：

| 从 | 到 | 触发点 | 关键副作用 |
|---|---|---|---|
| — | `waiting` | `add_sequence()` | `arrival_time` 覆盖；`reserve_output_tokens()`；`block_table.reserve(_block_table_capacity())` |
| `waiting` | `running` | `_schedule_prefill` 非 chunked 分支 | `status=RUNNING`；`allocate_mtp_info_slot()` |
| `waiting` | `waiting`（队头） | chunked prefill 未完 | `process_outputs` 里 `waiting.push_front(seq_ptr)` |
| `waiting` | `ignored_seqs` | 长度超限 / 不可调度 / VE 失败 / SO 失败 | `FINISHED_IGNORED` / `FINISHED_SO_FAILED` |
| `waiting` | `running` | `process_outputs` 的 `if (prompt_run) running.push_back()` —— 在途被 abort 的 prompt 请求 | 目的是让下一 decode 步的 `free_finished_seqs()` 收尸 |
| `prompted` | `running` | `_schedule_decode` 的 `admit` lambda | forked 需先克隆 state + 尾块 |
| `running` | `finished_seqs` | `free_finished_seqs()`（decode 每步开头） | `release_mtp_info_slot()` |
| `running` | `waiting`（队头） | `_preempt` + `!stop_preempted_request` | `reset_for_recompute()` + `free_seq_nolock()` |
| `running` | `ignored_seqs` | `_preempt` + `stop_preempted_request`（或 SO 请求） | `FINISHED_PREEMPTED` |

**`waiting` 队列的双重清理**（这是精读才看得到的防守）：`_schedule_decode` 开头会**全队列扫描**清掉 `is_finished()` 的 seq，注释明确给出理由——

> prefill 只在本 rank `want=true` 时才走到；某 rank 的 waiting 只剩这类 seq 时 `want` 恒 false，它们会永远滞留 → `has_unfinished` 恒真拖着全组空转，ALL 投票被永久压死。

即：**防止"死 seq 污染 DP 投票"**。

### 1.2 物理分配器（`block_manager.hpp`）——三个非典型决定

**① block 0 是哨兵，永不分配**

```cpp
Blocks(int _n) : _blocks(_n), _head(1), _tail(_n - 1), _size(_n - 1) {
  TORCH_CHECK(_n > 1, "GPU blocks 太少了: ", _n);
  for (int i = 1; i < _n; ++i) { _blocks[i].prev = i-1; _blocks[i].next = i+1; }
  _blocks[_n - 1].next = 0;
}
```

这一条**解释了全仓库多处魔数**：

| 位置 | 用法 | 语义 |
|---|---|---|
| `scheduler.hpp::_make_outputs` 的 `clone_blocks.emplace_back(src, 0)` | dst==0 | "只释放 src，不做拷贝" |
| `clone_and_zero_states_and_blocks` | `if (dst) { copy }` | 同上 |
| `model_runner._placeholder_seq_metadata` | `block_table=[0]` | padding/dummy 指向哨兵，永不读写 |
| `MtpState::info_slot == 0` | padding slot | 同一套"0 即无效"惯例 |
| `can_allocate_nolock`: `if (n >= num_blocks) return NEVER` | 保留 0 号 | 单请求永不独占全部块 |

**② 无 COW、无 CPU swap、无内容哈希前缀缓存**

- `Blocks` 只有 `ref_count + free_list`（intrusive 双向链表），**没有 copy-on-write 分叉**；
- `pin_block()` 只加 `ref_count`，用于 fork 时"共享父请求的块"；
- `SequenceStatus::SWAPPED` **存在但无人使用**（`_preempt` 只走"终止"或"重算"两条路）。

> 这是与 vLLM 最实质的差异之一：**vLLM 的 prefix caching 是内容哈希驱动的自动复用；DeepInfer 是请求驱动的显式 fork**（`fork_seq` / `branch_id`）。代价是必须由上层显式发起（`examples/benchmark.py` 就是靠 fork 构造共享前缀来测 prefill 的），收益是不用维护 block hash 表、也不用在 schedule 热路径上算哈希。

**③ 锁序固定：先 state 后 KV**

```cpp
BlockManagerMutexAcquire(block_manager, state_manager) {
  if (state_manager) state_manager->lock();
  block_manager->lock();
}
~...() { block_manager->unlock(); if (state_manager) state_manager->unlock(); }
```

RAII + 固定顺序 → 无死锁。`_schedule()` 全程持此锁，且**在 `py::gil_scoped_release` 之后取锁**。

### 1.3 关键架构切分：`_schedule()` 无 GIL，`schedule()` 有 GIL

```cpp
std::unique_ptr<SchedulerOutputs> _schedule() {
  py::gil_scoped_release release;              // ← 整个调度无 GIL
  BlockManagerMutexAcquire acquire{...};
  ...
}
py::tuple schedule(SequenceMetadataPool *pool) {
  auto p = _schedule();                        // 无 GIL
  ... // 这里恢复 GIL，因为要 allocate py::object 的 SequenceMetadata
  scheduler_outputs.seq_metadata_list[i] = std::move(p_metadata);
}
```

> **切分依据不是性能，而是所有权**：`SequenceMetadata` 里有 `py::object multi_modal_data / sampling_params / so_matcher`，必须在 GIL 下构造；而 `SchedulerOutputs` 只含 POD 与对内部 vector 的 const 引用，可以无 GIL 产出。**"能用 GIL-free 的绝不带 GIL"**——这是一条清晰的边界纪律。

### 1.4 Prefill 约束链（`_schedule_prefill`，逐条）

```text
①  队头 schedulable 为假 → break（单调性，队头不可调度则后面全不可调度）
②  finish_reason 已定（ABORTED/VE_FAILED/SO_FAILED）→ ignored_seqs，continue
③  首次分配：seqlen > prompt_limit → FINISHED_IGNORED
④  _can_allocate_with_state(get_num_blocks()) ：LATER→break，NEVER→FINISHED_IGNORED
⑤  fork 命中（branch_id 以 '_' 开头）：长度检查 → 剥标记 → _clone_state + 尾块 copy
⑥  num_curr_seqs + 1 > max_num_prompts → break
⑦  prefill_sum_seqlen_k 余量不足 → break
⑧  chunked 切分：seq_workload > (max_num_batched_tokens - used)/cp_align*cp_align
              → new_workload 向下对齐
⑨  chunk_checkpoints 精确切点（start+q 越过 checkpoint 则切到 checkpoint）
⑩  _adjust_chunk_size() 前后调整，避开图片内部
⑪  seqlen_q == 0 → TORCH_CHECK(!scheduled.empty()) 后 break（硬约束）
⑫  is_chunked && !chunked_prefill → break
⑬  seqlen_q < max(lookahead_tokens, 1) → break
⑭  —— 以下集中提交副作用 ——
    num_batched_tokens += ceil(seqlen_q/cp_align)*cp_align
    block_table 空 → acquire.allocate_seq(seq)（blocks + 一个 state slot）
    非 chunked → status=RUNNING + allocate_mtp_info_slot
⑮  is_chunked && engine_type==NORMAL → break（NORMAL 每步最多推进一个 chunked 请求）
```

**两个只在这里才看得到的细节**：

- ⑭ 的 token 预算按 `cp_align` **向上取整**计入 batch——即 `max_num_batched_tokens` 是"含 CP padding 的预算"，不是有效 token 数；
- ⑮ 让 NORMAL 引擎的 chunked prefill **一步只推进一个请求**（PROMPT 引擎则可连续多个），这是为了保证 NORMAL 引擎的 prompt step 能快速回到 decode（避免长 prefill 拖死 decode 时延）。

### 1.5 Decode 调度：我上一轮讲错了，这里纠正

```cpp
sort(running, [](a,b){ return a->get_output_len() > b->get_output_len(); });  // 降序：0 号输出最长
size_t l = 0, r = running.size();
while (l < r && scheduled.size() < max_seqs) {
    auto &seq = running[r - 1];                       // ← 从末尾取：输出最短的先调度
    while (!can_allocate_single_nolock() && r - l > 1) {
        auto victim = running[l];                     // ← 从前端杀：输出最长的先被抢占
        _preempt(victim, ignored_seqs); l += 1;
    }
    ...
}
```

**⚠️ 纠错**：我在上一轮报告中写成"保护长输出、牺牲短输出"，**方向反了**。真实策略是：

- **调度**：输出最短的优先（从 `r-1` 往前）；
- **抢占**：输出最长的优先被杀（从 `l` 往后）。

依据在 `sequence.hpp` 的注释里写得很直白：

> `FINISHED_PREEMPTED: "insufficient_system_resource" // 如果kvcache即将溢出，则占用kvcache最多的任务，会被我们强制终止。`

即这是**"一次释放最多块"的贪婪选择**（`output_len` 是块数的代理），不是公平性策略。副作用是长请求可能被反复抢占/终止（`stop_preempted_request=false` 时走重算）。这是个值得讨论的取舍：**用"块释放效率"换"长请求存活率"**。

### 1.6 抢占的资源闭环

```cpp
void _preempt(SequencePtr &seq, std::vector<SequencePtr> &ignored) {
  const bool so_blocks_recompute = seq->structured_output.status != 0;   // SO 无法重算
  if (stop_preempted_request || so_blocks_recompute) {
      seq->schedule.status = FINISHED_PREEMPTED; free_seq_nolock(*seq); ignored.push_back(seq);
  } else {
      seq->reset_for_recompute();          // 复位 cache/schedule/mtp（见 §2.3）
      waiting.push_front(seq);             // 队头：下一步优先重算
      seq->schedule.status = WAITING;
      free_seq_nolock(*seq);               // 注意：先 reset 再 free
  }
  num_preempted += 1;
}
```

`reset_for_recompute()` 是三层联动的：

```cpp
void Sequence::reset_for_recompute() { cache.reset_for_recompute(); schedule.reset_for_recompute(); mtp.reset_for_recompute(); }
```

---

## 二、序列 / KV 语义（`sequence.hpp` 逐字段）

### 2.1 `KvCacheState`：三寄存器的"乐观推进/回填"

```cpp
int previous_num_tokens{0};
int num_tokens{0};
int hit_tokens{0};          // 外部注入的前缀命中长度
int swa_valid_start{0};     // SWA 有效窗口起点（仅 prompt 侧传给 metadata）
std::vector<int> block_table;
int state_id{-1};

void set_num_tokens(int v)          { num_tokens = v; previous_num_tokens = v; }   // 建立增量基线
void advance_optimistically(int c)  { previous_num_tokens = num_tokens; num_tokens += c; }
void commit_processed_tokens(int c) { num_tokens = previous_num_tokens + c; }
```

语义精确表述：

$$n_{\text{prev}} \leftarrow n,\quad n \leftarrow n + q_{\text{optimistic}} \quad\text{（step 内）}$$
$$n \leftarrow n_{\text{prev}} + \text{token\_cnt} \quad\text{（step 末回填）}$$

- **为什么要三寄存器而非两**：`advance` 必须保留推进前的 `n` 作为回填基点；若只留 `n`，回填就得减，而"减多少"在收到 `token_cnt` 之前不可知。
- **`commit` 只对 decode 调用**（prompt 步的 `advance` 就是真值，因为 prefill 不产 draft）。
- **不变式**（`process_outputs` 里两条 `TORCH_CHECK`）：
  - decode 步：$n \in [\text{len},\ \text{len} + \text{nextn}]$；
  - prompt 步：$n \le \text{len}$，且 $n = \text{len}$ 表示该 seq 的 prefill 完成。
  - 综合：$n \le \text{len} + \text{nextn\_len}$。
- `hit_tokens` 与 `set_num_tokens` 配合：注入前缀命中时用 `set` 而不是 `advance`，以建立正确的回填基线。

### 2.2 `get_num_blocks()`：KV 需求的唯一公式

```cpp
int get_num_blocks() const {
  return (get_len() + schedule.total_inflight_tokens + get_nextn_len() + cache.block_size - 1) / cache.block_size;
}
```

三项含义：

| 项 | 含义 | 为什么必须计入 |
|---|---|---|
| `get_len()` | 已提交 token 数（prompt + output，**不含 draft**） | 实际已写入 KV 的量 |
| `total_inflight_tokens` | 在途步已下发但未记账的最大 query 数 | ZOS 下 GPU 可能正在写 |
| `get_nextn_len()` | `draft_tokens.size()` | 本轮将要写 draft 的 KV（DFlash main cache 复用主 KV 布局） |

**与另一处 bookkeeping 的一致性**（`zos.cpp`）：

```cpp
// _append_slot() reserves through num_cached_tokens + total_inflight_tokens + nextn + 1;
// GPU varlen correction may grow q to max_seqlen_q.
const int64_t max_k  = seq->seqlen_k - seq->seqlen_q + max_seqlen_q;   // 修正后最坏 K
const int64_t compressed_k = max_k / block_table_compress_ratio;
const int64_t n_blocks = (compressed_k + tokens_per_block - 1) / tokens_per_block;
TORCH_CHECK_LE(n_blocks, static_cast<int>(seq->block_table.size()));
```

> 两条独立公式必须相容：`get_num_blocks()` 用 `nextn_len`（本轮实际 draft 数），`zos.cpp` 用 `max_seqlen_q = nextn + 1`（最坏情况）。由于 `nextn_len ≤ nextn`，且 `_schedule_decode` 会先把 `block_table` 补齐到 `get_num_blocks()`，这里靠 `TORCH_CHECK_LE(n_blocks, block_table.size())` 兜底。**这是全仓库最脆弱的一致性点之一**——改任一侧都可能触发运行时断言。

### 2.3 `SequenceScheduleState::*_inflight_*`：抽屉深度 ≤ 2

```cpp
void start_inflight_step(int tc) { total += tc; last = tc; }
void finish_inflight_step() {
  if (total == last) { total = 0; last = 0; }   // 只有一步在途
  else               { total = last; }          // 有两步在途，丢掉更早那一步
}
```

**这只能用两个变量表示两种状态**：0 步在途、1 步在途、2 步在途。因此硬约束：**ZOS 流水深度至多 2**（当前步 + 上一步）。`schedule()` 里传的是 `max_seqlen_q = nextn + 1`（最坏情况），所以是**保守的**在途计数。

> 若将来把流水加深到 3 步，这段代码会静默给出错误结果（`total == last` 判定失效），而不是报错。**这是"用不变量换空间"的典型脆弱点**。

### 2.4 `MtpState`：五个向量 + 两个计数器的精密配合

| 字段 | 语义 | 谁写 | 谁读 |
|---|---|---|---|
| `golden_tokens` | 按输出长度索引的参考 token（teacher forcing） | 构造时 | `next_golden_tokens()` → metadata |
| `last_output_token` | 上轮输出，即本轮 golden | `process_outputs` | `refresh_for_sampling` |
| `accepted_tokens` | 上轮验证通过（SO matcher 更新用） | `process_outputs` | sampler（经 `MtpView`） |
| `draft_tokens` | 上轮新生成的 draft | `process_MTPInfo` | `get_nextn_len()` / `MTPVarlen` / `zos.cpp` |
| `accept_rates` / `prev_accept_rates` | 累乘后接受率（本轮 / 上轮） | `process_MTPInfo` 移位 | `MTPVarlen::schedule` |
| `confidence_raw` / `prev_confidence_raw` | 累乘前（后者仅供 metrics） | 同上 | `outputs.hpp` build_output |
| `info_slot` | MTP info pool 槽位（0=padding） | scheduler 分配/释放 | kernel 读写 |
| `has_draft_info` | 有未消费的 draft info | `process_MTPInfo` | `take_draft_info_slot()` |
| `process_cnt` / `schedule_cnt` | 已处理 / 已调度轮数 | `process_MTPInfo` / `schedule()` | `MTPVarlen`（判 accept_rate 属于第几轮） |

**`take_draft_info_slot()` 是破坏性读**：

```cpp
int take_draft_info_slot() { int read_slot = has_draft_info ? info_slot : 0; has_draft_info = false; return read_slot; }
```

即 **draft info 是"恰好消费一次"的单发消息**。它被 `SequenceMetadata` 构造函数和 `refresh_for_sampling()` 各调一次——注意 **ZOS 下两条路径都可能走**（先构造、后修正），所以 `refresh_for_sampling` 里有一句防守注释：

> `// Keep the slot captured for an in-flight sample after ZOS releases the request.`
> ```cpp
> if (seq.mtp.info_slot != 0) mtp_info_write_slot = seq.mtp.info_slot;
> ```

**`reset_for_recompute()` 的有意保留**：

```cpp
void reset_for_recompute() {
  draft_tokens.clear(); accept_rates.clear(); prev_accept_rates.clear(); confidence_raw.clear();
  has_draft_info = false; process_cnt = 0; schedule_cnt = 0;
}
// 保留 accepted_tokens（重算后更新 SO matcher 仍要用）与 prev_confidence_raw（仅供 metrics）
```

**`next_golden_tokens(output_len, nextn)` 取 $2n+1$ 窗口**：

```cpp
result.reserve(nextn * 2 + 1);
for (int i = output_len; i < output_len + nextn * 2 + 1; ++i)
    result.push_back(i < size ? (*golden_tokens)[i] : -1);
```

为什么是 $2n+1$（推断）：本轮最多接受 $n$ 个 draft，下一轮 golden 窗口起点会右移至多 $n$，预取 $2n+1$ 可保证"一次取、连用两轮"，避免每步重取元数据。**注意 reserve 与实际 push 数量不符（reserve 2n+1 但不一定 push 满）——无害，只是容量提示。**

### 2.5 `SequenceMetadata`：只读投影 + 双构造路径

```cpp
struct SequenceMetadata : public EnableObjectPool {
  // 借用型成员（指向宿主 Sequence）
  const std::string &request_id;
  SequenceData *seq_data, *compressed_seq_data;
  const std::vector<int> *prompt_tokens;        // 仅 is_prompt 非空
  const std::vector<int> &block_table;          // ← 对 vector 对象的引用，重分配安全
  py::object multi_modal_data, sampling_params;
  SamplingScalars sampling;                      // 值拷贝，不是引用
  int seqlen_q, seqlen_k;                        // 值拷贝：ZOS 下会被就地改写
  ...
  MtpView mtp;                                   // 两个 const vector 引用
  std::optional<std::vector<int>> &prompt_logprobs;
};
```

**两条构造路径的差异（精读要点）**：

| | 对象池路径（`Sequence*` 构造） | pybind 路径（Python 构造 dummy） |
|---|---|---|
| 用于 | 真实请求 | profile run / CUDA Graph padding / placeholder |
| `owned` | `nullptr` | `new OwnedStorage{...}`（自持存储） |
| 是否入池 | 是（`allocated_from_object_pool=true`） | 否（`pool.release` 静默跳过，靠 Python GC） |
| `is_placeholder` | `false` | `request_id == "placeholder"` |
| `mtp_info_read_slot` | `take_draft_info_slot()`（破坏性读） | 恒 0 |
| `so_*` | 从 Sequence 读 | 恒 0 / none |

**两个细节**：

```cpp
// ① 非最后 prefill chunk 不产 token → last_output_token = -1 让 sampler 跳过 SO 更新
if (is_prompt && !is_last_prompt_chunk()) last_output_token = -1;

// ② is_last_prompt_chunk 的判据
bool is_last_prompt_chunk() const { return is_prompt && seqlen_k >= seq_data->get_len(); }
```

`block_table` 用**对 vector 对象的引用**而非 `data()` 指针——这样 `_append_slot`/`erase` 导致的重分配不会悬垂。这个小选择支撑了整个"post_schedule 之后仍能安全读 block_table"的设计。

`SamplingScalars::capture()` 在 **Sequence 构造时一次性快照**（9 个标量），意味着**运行期改 Python 的 `SamplingParams` 不生效**；且 `eos_id` 必须在构造前由 `llm_engine.add_request` 写入（代码里确有 `sampling_params.eos_id = self.eos_id`），否则快照到的是默认值——**一个隐性的调用顺序契约**。

---

## 三、变长验证（`mtp_varlen.hpp` + `zos.cpp` 的修正回路）

### 3.1 决策闭环（4 个阶段）

```text
① 观测：worker 采到 forward_ms / sampler_ms（cuda_event 对），engine.post_process 调
        scheduler.update_mtp_runtime(len(seq_metadata_list), graph_size, fwd_ms, smp_ms)
② 估计：ForwardRuntimeEstimator（两层中位数，窗口 5000，每 graph ≥10 样本）
        SamplerRuntimeEstimator（9 样本滑动中位数，按 num_seqs 分桶，≥5 样本才给值）
        baseline 来自启动期 profile_decode_perf() 标定的 Pareto 曲线（磁盘缓存 TTL 1h）
③ 决策：MTPVarlen::schedule() 求 argmax 吞吐 → (best_bsz, 每请求 seqlen_q)
④ 修正：GPU 侧用"上一步真实接受数"重写 seqlen_q/seqlens_k（zos.cpp + tilelang kernel）
```

### 3.2 目标函数（逐符号）

```cpp
// 候选枚举：cur_bsz 从 num_seqs 开始，每加入一个 draft 候选 +1，同时 cur_expected += accept_rate
const double time_ms = baseline_ms + slope_ms * graph_size + sampler_ms;
const double cur_throughput = cur_expected / time_ms;
```

$$\text{throughput}(b)=\frac{\underbrace{n+\sum_{i\le b-n}a_i^{(\cdot)}}_{\texttt{cur\_expected}}}{\underbrace{T_{\text{base}}(g(b))+\lambda\,g(b)}_{\texttt{forward}}+\underbrace{T_{\text{smp}}(b)}_{\texttt{sampler}}}$$

三个非平凡点：

1. **分子是"期望生成 token 数"，不是 batch 大小**：$1 + \sum a_i$，其中 $a_i$ 是累乘接受率；
2. **分母含在线回归的斜率项** $\lambda g$：`slope` 由 $(\text{fwd\_ms} - \text{baseline})/g$ 的两层中位数给出，且 `max(0, slope)` 防负外推；
3. **候选 batch 被曲线"向上取整"到最近的捕获尺寸**：
```cpp
while (cur_bsz_idx < n && cur_bsz > curve[idx].first) ++idx;   // 选最小的 ≥ cur_bsz 的 graph
if (idx == n) return;                                          // 超出最大 graph → 该候选无效
```

**`best_bsz − num_seqs` 的分配规则**：

```cpp
seqlens_q.assign(num_seqs, 1);
for (int i = 0; i < best_bsz - num_seqs; ++i)
    seqlens_q[acc_queue_for_seqs_q[i].second] += 1;   // 按 accept_rate 降序
```

即：**额外 draft 全部给接受率最高的请求**（每请求至多 $+n_{\text{next}}$，因为队列里每个请求贡献 `nd` 项）。

### 3.3 两代 accept_rate 的严谨性（这是全文件最值得学的一点）

```cpp
bool post_process_paired = (mtp.process_cnt == mtp.schedule_cnt);
bool has_draft = mtp.has_draft_info || mtp.schedule_cnt == mtp.process_cnt + 1;

// 永远用 N-2 轮算 best_bsz
push_rates(acc_queue_for_bsz,  post_process_paired ? mtp.prev_accept_rates : mtp.accept_rates, ...);
// 永远用（当前的）accept_rates 分配 seqlen_q
push_rates(acc_queue_for_seqs_q, mtp.accept_rates, ...);

TORCH_CHECK(mtp.process_cnt <= mtp.schedule_cnt && mtp.schedule_cnt <= mtp.process_cnt + 1, ...);
```

**状态解释表**：

| `schedule_cnt − process_cnt` | 含义 | `accept_rates` 属于第几轮 | `prev_accept_rates` |
|---|---|---|---|
| 0 | forward 与后处理配对（非 ZOS / prompt 步） | $N-1$ | $N-2$ |
| 1 | 最新 forward 尚未产出（ZOS 典型） | $N-2$ | $N-3$ |

设计意图（注释原文）：

> 需要保证 `confidence_{i+1,...,nextn}` 不影响 `draft_token_i` 是否被验证。假设此时在第 $N$ 轮，我们期望使用 $N-2$ 轮的 accept_rate 去计算 `best_bsz`，并使用 $N-1$ 轮的 accept_rate 去分配每个请求的 `seqlen_q`（non-ZOS 在这里做，ZOS 在 GPU 上修正）。

即：**用越旧的统计做大粒度的批决策（稳），用较新的统计做细粒度的逐请求分配（准）**，且刻意避免"用尚未产生的信息"。

### 3.4 ZOS 修正回路（`zos.cpp` 三个函数逐条）

**`prepare_zos_inputs`（CPU，pinned 输出）**

```cpp
// 快路径：size 相同、顺序相同、无 placeholder → remap[i] = i
// 否则：用 prev 步的 seq_id → index 建哈希表，remap[i] = -1 表示 fresh
seqlens_k_ptr[i] = seq->seqlen_k - (nextn > 0 ? seq->seqlen_q : 0);
```

- `seqlen_k` **减去 seqlen_q**（当 `nextn>0`）：因为调度期的 `seqlen_k` 已含乐观的 q，kernel 将以"K 基数 + 实际 q"重建；
- 返回 `has_overlap=false` 时三个 tensor 全为 `nullopt` → **完全跳过修正**（快速路径）；
- padding 行：`remap=-1, seqlens_q=1, seqlens_k=1`（保证 kernel 不越界）。

**`prepare_decode_zos_inputs`（CPU → pinned，供 H2D）**

CSR 式打包（关键设计）：

```cpp
auto cu_seqlens_q         = torch::empty({n_seq},     pinned(kInt32));  // 前缀和
auto block_table_offsets  = torch::empty({n_seq + 1}, pinned(kInt32));  // CSR 偏移
auto block_tables         = torch::empty({total_blocks}, pinned(kInt32)); // 扁平打包
```

> **不做 `[n_seq, max_blocks]` 稠密填充**，而是把每请求的 `block_table` 前缀（长度 = 本次真正需要的块数 `n_blocks[i]`）紧凑拼成一条，用 offsets 索引。这既省 H2D 带宽（decode 期 block_table 可达数百项/请求），也让 kernel 侧用 `query_indices` 做间接寻址。

**`update_metadata_for_sampling_zos`（D2H 后回写）**

```cpp
metadata->seqlen_q = seqlens_q_ptr[i];
metadata->seqlen_k = metadata->seqlen_q + seq.cache.num_tokens;   // ← K 由已提交长度 + 本轮 q 得到
if (remap_ptr[i] >= 0) metadata->refresh_for_sampling(seq, nextn);  // 只在"上一步也在"时刷新
scheduler_outputs.seqlens_q[i] = metadata->seqlen_q;               // 同步回 SchedulerOutputs
```

三个要点：
1. **`seqlen_k` 不是从调度期继承，而是重算**（`num_tokens` 此刻已含上一轮 commit 的结果）；
2. `refresh_for_sampling` 只在 `remap >= 0`（有交集）时调用——fresh 请求的 metadata 本来就是准的；
3. 注释点出一个边界：*"EOS 后的额外 forward 可能未将 golden token 追加到 Sequence，采样侧以已缓存长度为准"*。

### 3.5 Engram 的 `ngram_token_ids` 双路径（`prepare_token_inputs`）

| 路径 | 产物 | 谁计算 |
|---|---|---|
| **无重叠** | 直接产 `input_ids[total_q]` + `ngram_token_ids[total_q, max_ngram]`（pinned），可一次 H2D | CPU 侧 `fill_engram_row`（含 lookup_table 压缩、image_sentinel 判定） |
| **有重叠** | 只算 fresh 行：`fresh_input_ids[n_seq, max_q]`、`win_compressed[n_seq, nextn+max_ngram-1]`、`curr_fresh[n_seq, max_q]` | CPU 算 fresh；**prev 行由 GPU kernel `gather_engram_tokens_zos` 从 `prev_sampled/prev_predict` + lookup_table 现算** |

虚拟压缩序列布局（注释原文，必须记住）：

```text
[0, L)            compressed_seq_data
[L, L+nextn)      last_next_n_token
[L+nextn, L+2n+1) curr tokens
|<------------ window_size = nextn + max_ngram_size - 1 ------------>|
```

`win_compressed` 的**左对齐搬运**：

```cpp
const int win_end = compressed.size() - int(fresh);
const int win_start = win_end - history;
const int src_start = std::max(0, win_start);
const int dst_start = src_start - win_start;     // 历史不足时右移补齐
std::copy(compressed.begin()+src_start, compressed.begin()+win_end, win_compressed + i*(history+nextn) + dst_start);
```

**这段的价值**：把"历史不足一个窗口"的边界情况用**下标算术**吃掉，而不是分支/kernel 变体 → 保证 kernel 侧只有一种布局。

---

## 四、EPLB 落地（从 policy 到 shuffle 到 kernel）

### 4.1 三段式流水

```text
[观测] eplb/__init__.py: expert_count[L, local_phy] 每 step 清零，滚入 window(默认100)
   │    每 rearrangement_interval 步触发（首次提前到 1/4 周期纠正冷启动）
   ▼
[决策] policy.rebalance_experts_global() : 两阶段
   │    ① 分配冗余副本（num_redundant 个）
   │    ② 若 num_local > 1，把副本均衡到各 rank
   ▼
[搬运] shuffle.expert_model_parallel_shuffle_inplace[_intranode]() : inplace 换权重
   │    单机 → symm_mem IPC 镜像直读；跨机 → bucketed batched P2P
   ▼
[生效] physical_to_logical_map / logical_to_physical_map / logical_count 三张表
        被 MoE dispatch kernel 与 sampler/kernel 直接读（同一 step 内原子可见）
```

### 4.2 `policy.py` 阶段①：冗余副本分配（逐行）

```python
tokens_per_expert_all_diff = tokens_per_expert + arange_logical * 1e-4   # 打破平局（保证 argmax 唯一）
for i in range(num_redundant_experts):
    score        = tokens / logical_count                    # [steps, L, E] 每副本平均负载
    score_added  = tokens / (logical_count + 1)              # 加一个副本后的平均负载
    values, indices = score.max(-1, keepdim=True)            # 当前最热点 expert 的负载
    values = values.expand_as(score)
    score.scatter_(-1, indices, score_added.gather(-1, indices))   # 局部替换：删掉热点项
    values.scatter_(-1, indices, score.max(-1).values)             # 重算 max（热点已被降级）
    redundancy_indices = values.sum(0).argmin(-1)            # 对 step 求和 × 对 L 独立 → [L]
```

**目标函数**（注释已写明）：选择使 $\sum_{\text{step}} \max_{e}(\text{load}_e / \text{count}_e)$ 最小的 expert 再加一个副本。

实现技巧是"**先假设给每个 expert 都加一个副本，再只用一次 argmin**"——用一次 scatter 把 argmax 位置的分数替换成 `score_added`，就地得到"给 $j$ 加副本后的 max"。**复杂度 $O(E)$ 而非 $O(E^2)$，且对全部 layer 向量化。**

**注意一个隐含前提**：`argmin` 沿 `-1` 维度独立作用于每个 layer，所以**每层独立决定给谁加副本**——不同 layer 的冗余副本可能落在不同 logical expert 上。

### 4.3 `policy.py` 阶段②：rank 级贪心（`_batch_greedy_indices`）

```python
weights       = tokens.sum(0, float) / logical_count          # 每副本窗口平均负载
physical_order= weights.gather(-1, physical_to_logical_map).argsort(-1, descending=True, stable=True)
                # ↑ 放置顺序：负载最高的 physical 专家先放（LPT 思想的变体）
vectors       = tokens.permute(1,2,0) / logical_count[..., None]   # [L, E, steps]

for job_index in range(num_physical):                          # 逐个"作业"
    vector = vectors[layers, expert_order[:, job_index]]
    torch.add(loads, vector.unsqueeze(1), out=candidates)      # 放到各 rank 后该 rank 的负载
    torch.maximum(candidates, peaks.unsqueeze(1), out=candidates)   # 与历史 peak 取 max
    torch.sum(candidates, dim=-1, out=scores)                  # Σ_step（目标值）
    scores.masked_fill_(occupancy == capacity, float("inf"))    # 已满的 rank 屏蔽
    rank = scores.argmin(dim=-1)                                # 同分返回最小 rank（确定性）
    indices[layers, rank*capacity + occupancy[layers,rank]] = physical_order[:, job_index]
```

**目标函数**（注释里写成了数学式）：

$$\min_{\text{assignment}} \sum_{s\in\text{steps}} \max_{r\in\text{ranks}} \Big(\sum_{e\to r} \text{load}_{e,s}\Big)$$

即**逐 step 峰值 rank 负载之和**——与 `__init__.py` 里 `moe_load_balance_ratio = (mean, max)` 的统计口径完全一致，保证了"优化的量"和"上报的量"是同一个。

**三个工程细节**：

| 细节 | 目的 |
|---|---|
| `argsort(..., stable=True)` | 同负载时保留原物理槽位顺序 → 重排结果稳定 → 减少无谓搬运 |
| `occupancy == capacity` 屏蔽 | 保证每个 rank 恰好 `capacity = num_local_physical_experts` 个 |
| `argmin` 同分取最小 rank | **确定性**：避免不同 rank 算出发散结果（虽然每 rank 都算同一份输入，但浮点同分处理需一致） |

**`indices` 的双重语义**（很容易读错）：

```python
physical_to_logical_map = physical_to_logical_map.gather(-1, indices)   # indices 是"新位置 ← 旧位置"
...
logical_to_physical_map = (indices.argsort(-1)
    .gather(-1, logical_to_physical_map.view(L,-1).long())              # 旧槽位 → 新槽位
    .view_as(logical_to_physical_map).int())
# 先把 -1 临时置 0 再还原（-1 不能参与 gather）
```

即 `indices[p] = 旧槽位号`，而反向映射需要 `argsort(indices)[old] = new`。

### 4.4 `shuffle.py`：无通信的配对 + 两阶段搬运

**① `_paired_sender`：双方算同一函数 → 无需协商**

```python
def _paired_sender(j, num_senders, num_receivers):
    n = num_receivers // num_senders
    return j // n if j < n * num_senders else j - n * num_senders
```

- 第 $j$ 个 receiver 由第 `_paired_sender(j)` 个 sender 供给；
- 前 $n \cdot n_s$ 个 receiver 均分（每 sender 连续 $n$ 个），余数轮转补齐；
- **sender 与 receiver 都调用它**，因为双方的 `(senders, receivers)` 列表都由**全局复制的 `origin/current` 数组**推导得出，所以结果必然一致 → **零额外通信的确定性配对**；
- `n == 0`（receiver 少于 sender）时条件 `j < 0` 恒假，走 else 分支，**不触发除零**。

**② 通用实现：两阶段 + 分桶 P2P**

```python
# 阶段 A：本卡 origin → recv_buffers（先落地，避免后续 P2P 覆写自己的源）
received, intra_copies = _plan_intra_copies(origin_local, current_local)
for src, dst in intra_copies:
    for buf, weight in zip(recv_buffers, weights): buf[dst].copy_(weight[src])

# 阶段 B：跨卡，按 (ep_rank - dst) % n_groups 分桶（DEEPINFER_REARRANGE_P2P_GROUPS）
p2p_buckets[(ep_rank - dst) % n] .extend(P2POp(isend, weight[src], dst_global, ep))
p2p_buckets[(src_rank - ep_rank) % n].extend(P2POp(irecv, buf[dst], src_global, ep))
for ops in p2p_buckets:
    for req in dist.batch_isend_irecv(ops): req.wait()

# 阶段 C：recv_buffers → 本卡槽位（同一 expert 的多个槽位只收一次再本地扇出）
experts_to_recv = _first_slot_per_expert(current_local, skip=received)
```

要点：
- **`recv_buffers` 跨层复用**（`torch.empty_like(experts_weight[0])`），注释明说"假设每层 `hidden_size_i` 相同"——**这是一个未校验的全局假设**（若某层形状不同会静默错位）；
- 分桶键 `(ep_rank - dst) % n` 让收发方向错开，缓解全网同时打到同一个对端；
- `_first_slot_per_expert(..., skip=received)` 保证"**每 expert 只收一份，其余本地扇出**"。

**③ 单机实现：`symm_mem` 镜像 + 两次 barrier**

```python
mirror = ipc_experts_weight[ep_rank]          # 本 rank 的对称内存镜像
ep.barrier().wait()                           # ① 等所有 rank 读完"上一层"的镜像
for m, weight in zip(mirror, weights): m.copy_(weight)   # 刷新本层权重到镜像
ep.barrier().wait()                           # ② 等所有 rank 镜像就绪
# 之后：卡内拷贝从 mirror 读（避免覆写源）；远端读取直接 copy_(peer_mirror[src_slot])
```

- **镜像只容纳一层**，逐层刷新复用 → 额外显存 $O(1\ \text{layer})$ 而非 $O(L)$；
- 两次 barrier/层 → 总 $2L$ 次 barrier（60 层 = 120 次），在"每 N 百步一次"的节奏下可接受；
- **卡内拷贝读 mirror 而不是 weight**，是为了避免"dst 覆写 src"的在位冲突；
- 前提：`PG.is_within_nvl_domain(ep_group) and PLATFORM.is_gpu`（`__init__.py` 里判定），NPU 走通用 P2P。

### 4.5 EPLB 与 kernel / 模型结构的接口

```python
# 每 step 由 DeepseekForCausalLM.sample() 调用（模型侧唯一钩子）
expert_lb.step(input_metadata.is_prompt)
# 三张表被 MoE dispatch kernel 直接读：
#   physical_to_logical_map [L, num_phy]        → 物理槽 → 逻辑专家
#   logical_to_physical_map [L, num_log, R+1]   → 逻辑 → 物理（-1 为无效位）
#   expert_count            [L, local_phy]      → 本步计数（kernel 侧原子累加）
```

**关键点：映射表是"活"的**——kernel 在同一 step 内以当前版本读取，所以**重排必须发生在 step 边界**（`_rearrange_experts` 前后 `torch.cuda.synchronize()` 并登记为 `@allow_cuda_sync()`），且 `NORMAL` 引擎可通过 `DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE` 把重排推迟到 prefill 之后。

---

## 五、与 worker 执行的耦合点（逐文件 → worker 函数）

| 文件/结构 | worker 侧读取点 | 时序位置 |
|---|---|---|
| `SchedulerOutputs.seqlens_q` | `Worker._correct_seqlens_qk` → `prepare_zos_inputs(...)` | forward 之前 |
| `SequenceMetadata.{seqlen_q,seqlen_k,block_table,state_id,mtp.*,golden_token}` | `prepare_decode_zos_inputs` → `DecodeInputsZOS` | forward 之前 |
| `update_metadata_for_sampling_zos` | `Worker.execute_model_step` 第 3 步（D2H 同步后） | sample 之前 |
| `block_table`（CSR 打包） | `prepare_common_decode_buffers_zos` → `block_tables[total_q, max_num_blocks]` | forward 之前 |
| `MtpState.accept_rates` | `MTPVarlen::schedule`（在 C++ 内，engine 侧） | schedule 内 |
| `SamplingScalars` | `Sampler.forward`（经 `sampler_inputs.driver`） | sample |
| `StructuredOutputState.{status,matcher}` | `Sampler._get_so_mask` → `update_so_status` | sample |
| `KvCacheState.num_tokens` | `update_metadata_for_sampling_zos`（`seqlen_k = seqlen_q + num_tokens`） | sample 之前 |
| `MtpState.{process_cnt,schedule_cnt}` | `process_MTPInfo` / `MTPVarlen` | post_process |

**worker 里的四个"契约点"**（顺序不可交换）：

```text
① forward(N)  ← 读 metadata（seqlen 可能不准，靠 ② 之后修正）
② post_process(N-1)  ← 写 cache.num_tokens（commit）、写 mtp.*（process_MTPInfo）
③ 修正 + D2H sync  ← 读 prev.sampler_output.num_accepted_tokens_gpu
④ sample(N)   ← 读修正后的 metadata
```

> ② 必须早于 ③/④：因为 `update_metadata_for_sampling_zos` 里 `seqlen_k = seqlen_q + cache.num_tokens` 依赖 ② 写好的 `num_tokens`。**这个顺序在 `worker.py` 里由注释与代码位置共同保证，没有断言保护**——是重构时的头号雷区。

---

## 六、与 kernel 的耦合点（每个字段喂给谁）

| 数据结构 | 消费 kernel | 形态要求 |
|---|---|---|
| `input_ids[total_q]` | `gather_input_ids_zos` / embed | pinned → H2D；overlap 路径只填 fresh 行 |
| `ngram_token_ids[total_q, max_ngram]` | `gather_engram_tokens_zos` / `engram_hash` | 虚拟窗口布局（§3.5） |
| `cu_seqlens_q[n_seq]` | 所有 varlen kernel | **inclusive 前缀和**（注意不是 exclusive） |
| `block_table_offsets[n_seq+1]` + `block_tables[total_blocks]` | `prepare_v4_decode_metadata` / attn | CSR 打包，非稠密 |
| `state_block_idx[n_seq]` | SWA/compress state kernel | state slot id |
| `seqlens_q/seqlens_k` | `compute_seqlens_q_by_confidence_rates` / `compute_seqlens_k_zos` | 被就地重写 |
| `remap[n_seq]` | `gather_*_zos` | `-1` = fresh（走 CPU 填值），`≥0` = 从 prev 输出取 |
| `kv_ca_index[klen]`（两阶段 top-k） | `decode_mla_mqa_sparse` | 逻辑下标 → `convert_kv_index` 转物理 slot |
| `valid_token_mask / gate_force_random` | MoE 路由 / 性能标定 | `profile_decode_perf` 里整体置真 |
| `physical_to_logical_map` | MoE dispatch | 重排时被 `copy_` 整体替换 |

**一个反复出现的形态约束**：几乎所有 ZOS 侧 buffer 都是 **pinned + 一次性 H2D**，且 **CUDA Graph 复用同一块地址**（`get_or_create(args.out, ...)`）。这就是为什么 `prepare_decode_zos_inputs` 必须"先算 total_q/total_blocks，再精确分配"——**形状必须是常数，图才能捕获**。

---

## 七、与模型结构的耦合点（哪些是"模型给的承诺"）

| 代码里的硬编码/断言 | 依赖的模型事实 |
|---|---|
| `block_size = 128`（`resolved_tokens_per_block`） | V4.1 的压缩率与 cache 布局 |
| `_kv_bytes_per_token` 断言 `kv_lora + rope == layout.num_elements` | LoRA rank + RoPE 维度 = 512 |
| `kv_compress_ratio ∈ {0,1,2}` | C4/C128 已从模型侧移除 |
| `derive_compress_ratios` 分 dense/sparse 两类 | 模型同时存在"有 indexer"与"无 indexer"的层 |
| `CP 下 mirror_cp_len` 相关断言 | mirror CP 的 `cp_align = cp_size*2` 模型侧约定 |
| `MtpState` 五向量 | 模型自带 MTP head + confidence head（DSpark/DFlash） |
| `ConfidenceHead` 温度可配为 $n$ 元组 | 每步 draft 的概率校准不同 |
| `KProject` 从主压缩 KV 投影 index K | V4.1 的 index K 复用主 KV（`use_kproj=False` 时才独立） |
| SWA `window_size` 必须等于 DFlash `sliding_window` | 二者共用同一 cache 布局 |
| `engram` 需 `enable_tokenizer_compression=True` | CompressedTokenizer 的 lookup table 契约 |
| `YoCo`：`n_prefill_layer < num_hidden_layers` | consumer 层的感受野有界 |
| `hierarchical`：`hc / mhc`（`hc_shifted` 决定 head 位置） | 超连接结构 |

**一句话**：这 7 个文件里所有 `TORCH_CHECK` 有一半以上其实在断言"**模型是 V4.1**"，而不是断言"代码是对的"。这正是前一轮讨论中"贴身定制"的具体形态。

---

## 八、精读发现清单（含对上一轮的纠正）

### 8.1 必须纠正的一处

> **上一轮我写"保护长输出、牺牲短输出"，方向错了。**
> 实际（`_schedule_decode` 排序 + 取端）是：**输出最短的优先调度，输出最长的优先被抢占**。依据是 `sequence.hpp` 里 `FINISHED_PREEMPTED` 的注释："占用 kvcache 最多的任务会被强制终止"。**这是"单次释放最多块"的贪婪选择，不是公平性策略**，代价是长请求可能被反复抢占/重算。

### 8.2 新发现的隐性约束（按风险排序）

| # | 约束 | 位置 | 违反后果 |
|---|---|---|---|
| 1 | **ZOS 流水深度 ≤ 2** | `start/finish_inflight_step` 只有三态 | 静默算错在途 token → 块不足 → 断言或越界 |
| 2 | **两套 block 需求公式必须相容** | `get_num_blocks()` vs `zos.cpp::max_k` 公式 | `TORCH_CHECK_LE` 触发（好情况）或 KV 越界（坏情况） |
| 3 | **`process_outputs` → `update_metadata_for_sampling_zos` 的顺序** | `worker.py` ④ 依赖 ②写好的 `num_tokens` | `seqlen_k` 偏差 → 采样错位 |
| 4 | **`shuffle.py` 假设每层权重形状相同** | `recv_buffers = empty_like(experts_weight[0])` | 静默错位拷贝 |
| 5 | **`MTPVarlen` 的 `lower_bound` 未处理 `graph_size` 超上界** | `curve_it->second` 可能解引用 `end()` | UB（实际因曲线覆盖全部 graph 尺寸而未触发） |
| 6 | **`first_token_latencies` / `scheduled` 是 `static std::vector`** | `process_outputs` | 非可重入；同进程多 scheduler 实例会串数据 |
| 7 | **`SamplingScalars` 构造时快照** | `Sequence` 构造函数 | 运行期改 `SamplingParams` 无效；`eos_id` 必须在构造前写入 |
| 8 | **`take_draft_info_slot()` 是破坏性读** | 两处调用（构造 + `refresh_for_sampling`） | 若将来新增第三条调用路径，draft info 会丢失且无报错 |
| 9 | **`SequenceStatus::SWAPPED` 是死枚举** | 无人使用、无 swap 实现 | 读代码时误以为支持 CPU offload |
| 10 | **`XGrammarCompiler.vocab_size` 未被使用** | `compiler.py` | 死字段（第 11 行赋值、全程不读） |

### 8.3 值得单独称道的三个实现

1. **`_schedule()` 的 GIL 切分**：按"是否含 `py::object`"决定 GIL 边界，而不是按"性能直觉"。这是 `schedule()` / `_schedule()` 拆成两层的**真实原因**。
2. **`policy.py` 的"假设性替换 + 一次 argmin"**：把 $O(E^2)$ 的副本分配选择压成 $O(E)$，且对全部 layer 向量化。
3. **`shuffle.py` 的 `_paired_sender`**：用"双方各自计算同一确定性函数"替代显式协商，把"谁给谁发"这个协商问题变成**纯函数**——零通信、无死锁、可单测。

### 8.4 `compiler.py` 的单独说明（最小、但有一处安全意识）

```python
if self.cache_limit_bytes <= 0:
    logger.warning("xgrammar cache is enabled WITHOUT a size limit ... may cause OOM "
                   "(DoS vector, see CVE-2025-32381). Set a positive byte limit for production.")
```

- 默认 `cache_limit_bytes = 10 GiB`，`disable_cache` 是 kill switch（注释："xgrammar ≤ 0.1.x had production incidents with cache enabled"）；
- `compile_json_schema(schema, strict_mode=True)` —— **strict 模式**意味着生成的 JSON 必须严格符合 schema（不接受额外字段），是服务端语义的正确默认；
- 错误统一归一化为 `RuntimeError`，配合 `matcher.py` 的 `aborted_matcher()`，形成"编译失败 → 请求级中止"而非"引擎崩溃"的隔离链。

---

## 九、一句话总结

> 这 7 个文件真正的"心脏"不是 `Scheduler::_schedule()`，而是 **`Sequence` 里那几个 int**：`num_tokens / previous_num_tokens / total_inflight_tokens / last_inflight_tokens / seqlen_q / schedule_cnt / process_cnt`。**整个 ZOS 架构就是为了让这几个整数在"CPU 先猜、GPU 后验、半路回填"的过程中始终保持一致**；`scheduler.hpp` 是它们的唯一写者，`zos.cpp` 是唯一的纠偏器，`mtp_varlen.hpp` 是唯一允许用它们做预测的人，而 `block_manager.hpp` 只负责为它们兑现物理块。**一旦这几个整数的语义被改动，全栈都会以"偶发数值错误"的方式坏掉**——这就是这套设计换来的性能所对应的账单。

---
## 🧑‍💻 User

继续阅读：
- deepinfer/engine/llm_engine.py —— 端到端编排，ZOS 的 Python 侧怎么串
- deepinfer/worker/model_runner.py + cuda_graph_runner.py —— graph 捕获策略
-  model_executor/layers/attention_backend.py —— MLA / SWA 的 kernel 接口
- model_executor/models/v4_modules/* —— YOCO / CED 的真实结构
- ve/multimodal_processor.py —— VE 侧异步预处理
- csrc/utils/object_pool.hpp —— 对象池实现
- llm_engine.py 和 attention_backend.py —— 如何把"调度→执行"的链条闭合

---
## 😀 默认助手 (7)

# 续读：ZOS 的 Python 侧编排 · Graph 捕获 · Kernel 接口 · YoCo/跨层真实结构 · VE 异步 · 对象池 · 闭环

> **命名澄清**：`v4_modules/` 下**没有名为 `CED` 的模块**。目录实际内容为 `attn / index_attn / compress / mla_attn / cross_layer / yoco / engram / engram_metadata / engram_parameters / hc / hc_blocks / hc_lazy / mhc_modeling / mhc_ops / heads / mlp / moe / linear / norm / rope / containers / builder / utils`。若"CED"指*跨层执行调度*，对应 `cross_layer.py` + `yoco.py`；若指*条件专家分发*，对应 `mlp.py/moe.py` 的 gate。下文按前者分析，并给出后者要点。

---

## 一、`llm_engine.py`：ZOS 的 Python 侧怎么串

### 1.1 核心发现：引擎被"回调"重入到 Worker 的 step 内部

```python
# llm_engine._init_workers()
driver_kwargs = {"engine_callbacks": EngineCallbacks(
    post_process_fn = self.post_process,        # ← 引擎方法
    pre_sampler_sync_fn = self.pre_sampler_sync # ← 引擎方法
)}
self.driver_worker = Worker(..., **driver_kwargs)
```

```python
# worker.execute_model_step() 内部
request_outputs = self._post_process_step(self.prev_step_state)   # → 调 engine.post_process
...
self.engine_callbacks.pre_sampler_sync_fn(scheduler_outputs)     # → 调 engine.pre_sampler_sync
```

> **这是整条 ZOS 链路真正的"接缝"**：`LLMEngine.step()` 并不在自己的栈帧里处理输出，而是在 `_run_workers("execute_model_step")` → `Worker.execute_model_step()` 的**中间**被回调进来。于是 Python 后处理与 GPU forward 天然处在同一个 step 的时间窗内，不需要线程池、不需要双缓冲队列——**靠"调用栈嵌套"实现重叠**。

代价是：**`LLMEngine` 与 `Worker` 之间存在代码级互相依赖（循环回调）**，`post_process` / `pre_sampler_sync` 既是引擎的公开方法也是 worker 的钩子，二者签名被硬绑定。

### 1.2 ZOS 在 Python 侧的"存在感"只有 4 处

| 位置 | 代码 | 作用 |
|---|---|---|
| `arg_utils.__post_init__` | `os.environ.setdefault("DEEPINFER_ZERO_OVERHEAD_SCHEDULING", "1" if engine_type.default_zero_overhead_scheduling else "0")` + `environ.rearm(...)` | 由 `EngineType` 推导默认值；`rearm` 让 `EnvVar` 描述符重新解析（否则早在此前已被缓存） |
| `_create_scheduler` | `PG.cpu_data_parallel_group() if envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING else PG.data_parallel_group()` | **用 gloo(CPU) 组投票**，避免在 CPU 调度路径上触发 CUDA 流同步 |
| `has_unfinished_requests` | `device = "cpu"/"cuda"` 分支 + `dist.all_reduce(MAX)` | 同上：ZOS 下用 CPU tensor 归约 |
| `worker.execute_model_step` | `if is_prompt or not envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING: 当场 post_process` | 决定是否把 post_process 推迟到下一步 |

> **架构含义**：ZOS 不是一个"模块"，而是一个**贯穿 C++ 调度器、Python 引擎、Python worker 的全局模式**，其开关是一个环境变量。这解释了为什么它到处以 `if envs.XXX` 的形式散落——**没有把它封装成策略对象，是这份代码在可读性上的主要欠债**。

### 1.3 `DeferredRelease`：四个通道的延迟释放

```python
class DeferredRelease:
    sequence_metadata_to_release: list[SequenceMetadata]   # ②
    request_outputs_to_release: np.ndarray[np.uint64]      # ③
    clone_unpins_to_release: (np.ndarray, np.ndarray)|None # ④
    clone_done_evt: torch.cuda.Event                       # ④ 的同步点
    def release(self):  # 只在 pre_sampler_sync 调用
        self.sequence_metadata_pool.release(...)     # 回对象池（§六）
        self.request_output_pool.release(...)        # 回 RequestOutput 池（C++）
        if self.clone_unpins_to_release:
            self.clone_done_evt.synchronize()        # 等 clone 拷贝完成
            self.state_manager.unpin_blocks(states_to_unpin)
            self.block_manager.unpin_blocks(blocks_to_unpin)
        free_deferred_pool_handles()                 # 多模态池句柄
```

**四个通道各自解决一个"过早释放"**：

| 通道 | 若立即释放的后果 |
|---|---|
| `SequenceMetadata` | entrypoint 还在读 `RequestOutput` 里指向 metadata 的字段 → 悬垂 |
| `RequestOutput` | `LLM._run_engine` 里对每个 output 调 `RequestOutput.clone(raw_output)`，raw 已被回收 |
| `clone_unpins` | fork 的块/state 还在被 CUDA copy 读 → use-after-free（所以挂 event） |
| 多模态句柄 | 共享池块被复用后图像数据被覆盖 |

**`clone_done_evt.record()` 的位置很讲究**：在 `collect_clone_unpins` 里 record（即 `clone_and_zero_states_and_blocks` 下发之后立刻），在 `release()` 里 synchronize（最多滞后一整步）→ **同步点被推迟到与下一步的 GPU 计算重叠**。

### 1.4 `step()` 的其它三个非显然点

```python
# ① clone/zero 只在真有 clone 时下发（避免空 kernel）
if clone_states.size > 0 or clone_blocks.size > 0 or zero_states.size > 0:
    to_release, to_release2 = self._run_workers("clone_and_zero_states_and_blocks_on_default_stream", ...)
    self.deferred_release.collect_clone_unpins(to_release, to_release2)

# ② SO 的"最后一块 prefill 必须已经开工"校验（判死只杀单请求）
for seq, md in zip(scheduler_outputs.scheduled_seqs, seq_metadata_list, strict=True):
    if md.is_last_prompt_chunk and md.sampling_params.so_args is not None:
        if md.so_status == 0 and md.so_matcher is not None: md.so_status = 1
        else: logger.error("[SO] ..."); md.so_status = -1; seq.status = FINISHED_SO_FAILED

# ③ PROMPT 引擎的硬断言
if not scheduler_outputs.is_empty() and self.engine_type == EngineType.PROMPT:
    assert scheduler_outputs.prompt_run, "Prompt engines should not perform decoding"
    assert seq_metadata_list[0].seqlen_q > 0
```

② 里的 `zip(..., strict=True)` 依赖一个**跨语言契约**（`scheduler.hpp` 注释："scheduled_seqs 与 seq_metadata_list 严格同序（scheduler.hpp schedule 同一循环按下标构建）"）。

### 1.5 `add_request` 的非显然点

```python
# ① 双 token 流：压缩 tokenizer（Engram 必需）
compressed_prompt_token_ids = CompressedTokenizer.encode(prompt_token_ids) if use_compressed_tokenizer else prompt_token_ids

# ② 多模态必须先改写 prompt（在 Sequence 构造前）
prompt_token_ids = self.multi_modal_processor.overwrite_prompt(prompt_token_ids, multi_modal_data)

# ③ token 值域校验放宽到 vocab + NUM_SPECIAL（图像哨兵在词表外）
max_valid_token_id = vocab_size + (ImageTokenOffset.NUM_SPECIAL if multi_modal_data else 0)

# ④ SO 异步构建：不入队时同步构建（调用方自管调度），入队时走 pump
if not enqueue: seq.so_matcher = self.so_matcher_builder.build(so_args).result()
else:           so_pending = True; self.so_matcher_builder.submit(seq, so_args, _enqueue)
```

④ 的 `_enqueue` 回调里再检查 `so_matcher.is_aborted()` ——**编译超时的请求在入队前就被判死**，而不是进了调度器再死。

---

## 二、`cuda_graph_runner.py`：捕获策略

### 2.1 batch size 集合的构造

```python
def build_graph_batch_sizes(max_size, interval):
    interval = min(interval, max_size)
    out, size = [], 1
    while size < interval: out.append(size); size *= 2       # 1,2,4,... 2^k < interval
    out.extend(range(interval, max_size, interval))          # interval 步长
    out.append(max_size)
```

以 `max=1536, interval=8` 为例：`[1,2,4, 8,16,24,...,1528, 1536]` → **约 192 张图**。小 batch 用 2 的幂（低延迟区间要密），大 batch 用固定步长（吞吐区间可稀）。

`max_batch_size = max_num_seqs × (1 + nextn)` —— 因为 varlen 投机下总 q 可达 `n_seq × (1+nextn)`。

### 2.2 倒序捕获 + 共享显存池（标准 trick，但细节值得记）

```python
self.graphs = {}
self.graph_memory_pool = None
for batch_size in reversed(self.batch_sizes):          # ← 从大到小
    input_metadata_source = input_metadata_source_full.recursive_fn(_cut)   # 前缀视图，零拷贝
    input_metadata_device = input_metadata_device_full.recursive_fn(_cut)
    output = output_full.narrow(batch_size)
    # warmup（在独立 stream 上，必须，否则首次 capture 会触发 cudaMalloc）
    with torch.cuda.graph(graph, pool=self.graph_memory_pool, stream=self.graph_stream,
                          capture_error_mode="thread_local"):
        model_output = _run_model(...); output.copy_(model_output)
    if self.graph_memory_pool is None: self.graph_memory_pool = graph.pool()   # 第一张图建立池
```

- **倒序**是必须的：先捕获最大图，其 pool 覆盖峰值需求，后续小图共享同一 pool；
- `_cut` 用 `x[:_bs]`——**前缀切片**，所有小图共用同一块显存的前缀，于是"填一次数据，所有图共用"（`model_runner.profile_decode_perf` 里就有 `input_metadata_source = graphs[-1].input_metadata_source; valid_token_mask.fill_(True)` 这种用法）；
- `capture_error_mode="thread_local"`：自定义捕获错误模式，避免污染全局 CUDA 状态。

### 2.3 `get_clean_list()` 清字段——为什么必须清

```python
# warmup 之后、capture 之前
for name in input_metadata_source.get_clean_list():
    setattr(input_metadata_source, name, None)
    setattr(input_metadata_device, name, None)
```

> **这些是被 forward 首次调用时"派生填充"的 tensor（如 `compute_attn_info` 展开的 `klen`/`ks`/`ke`/`slot_mapping_*`/`flash_mla_sched_meta_dict`）。如果不清，capture 时它们会被当作"已存在的图输入"直接从 warmup 的旧值读取 → 图里固化错误数据。清掉后，它们在 capture 期间会被重新分配，从而**成为图内部的节点而非图输入**。**

这是 Graph 捕获最典型的坑，`get_clean_list()` 把"哪些字段是派生的"显式编码在 `InputMetadata` 类上——**用类级声明代替运行时探测**，很干净。

### 2.4 ZOS 下的 `input_on_device`：省掉一次全量拷贝

```python
def _run_model(..., copy_inputs):
    if copy_inputs and cp_group.rank() == 0:
        input_metadata_device.recursive_copy_(input_metadata_source)   # CPU → GPU
    if cp_group.size() > 1: input_metadata_device.broadcast()
    return model(kv_caches=kv_caches, input_metadata=input_metadata_device)
```

| 模式 | `input_on_device` | source 位置 | 每步动作 |
|---|---|---|---|
| 非 ZOS | `False` | CPU（pinned） | `recursive_copy_`（H2D）+ broadcast |
| **ZOS** | `True` | **与 device 共享同一块显存**（`prepare_decode_v4_zos` 直接写 `args.out`） | **零拷贝**，只 broadcast |

> 这正是 `zos.cpp` 里 `args.out` 那套 `get_or_create(args.out, "positions", ...)` 存在的理由：**Python 侧元数据 kernel 直接写进"图的输入 buffer"**，于是 CPU 根本不参与数据生成（`prepare_v4_decode_metadata` 是个 GPU kernel），`copy_inputs` 自然为假。**"ZOS 省 CPU"的本质是"把元数据生成也搬到 GPU 上，并直接写进图输入"**。

### 2.5 一个必须记住的所有权契约

```python
def replay(self, cuda_graph_data):
    if cuda_graph_data.graph_handle is not None: cuda_graph_timer.flip_event_buffer(...)
    cuda_graph_data.graph.replay()
    return cuda_graph_data.output        # ← 同一块显存，被下一次 replay 覆写
```

`output` 是 `output_full.narrow(batch_size)`，**每张图独立但同一底层 storage**。因此返回值只在下一次 `replay()` 前有效——这正是 `worker.py` 里到处包 `Movable(...)` 的原因（`model_output = Movable(main_model_output.model_output)` 表示"这个 tensor 会被占用，尽快消费"）。

### 2.6 `select()` 与 CP broadcast 都在图内

```python
def select(self, total_q):
    index = bisect_left(self.batch_sizes, total_q)
    if index == len(...): raise ValueError(f"No captured graph for total_q={total_q}; maximum ...")
    return self.batch_sizes[index], self.graphs[self.batch_sizes[index]]
```

向上取整到最近的捕获尺寸；**超上界直接抛错**（而不是回退 eager）——意味着 `max_num_seqs × (1+nextn)` 是一个**硬上界**，调度器一旦排出更大的 `total_q` 就会崩。

`input_metadata_device.broadcast()` 被**捕获进图**（CP > 1 时）→ NCCL 集合通信成为图节点，这是 `destroy_cuda_graphs()` 必须显式调用的原因（NCCL 在图里分配了 dependency）。

---

## 三、`attention_backend.py`：MLA/SWA 的 kernel 接口（只有两个函数）

### 3.1 双路 KV 在**一次 kernel 调用**里拼起来

```python
def decode_mla_mqa_sparse(query, key_cache, head_dim_v, indices_in_kvcache, sched_meta, ...,
                          key_cache_extra=None, indices_in_kvcache_extra=None, index_length_extra=None, ...):
```

| 参数 | SWA 路径 | 压缩路径 |
|---|---|---|
| `key_cache` | `kv_cache.swa_cache` | `kv_cache.ca_cache` |
| `indices_in_kvcache` | `input_metadata.kv_swa_index` | `kv_ca_index` |
| `key_cache_extra` | —— | `kv_cache.ca_cache` |
| `indices_in_kvcache_extra` | —— | `kv_ca_index` |
| `index_length_extra` | —— | `dense_index_len`（dense 层） |

调用方（`attn.py::_decode_attention`）：

```python
if self.use_swa:
    attn_kv_args = dict(key_cache=kv_cache.swa_cache, indices_in_kvcache=kv_swa_index,
                        key_cache_extra=kv_cache.ca_cache, indices_in_kvcache_extra=kv_ca_index,
                        index_length_extra=dense_index_len)
else:
    attn_kv_args = dict(key_cache=kv_cache.ca_cache, indices_in_kvcache=kv_ca_index, index_length=dense_index_len)
```

> **关键设计**：`SWA ∪ top-k(压缩KV)` 的联合注意力不是"两次 attention 再合并"，而是**一个 kernel 两路 gather + 一次 softmax 归一化**。这就是为什么 `swa_index` 与 `kv_ca_index` 必须在同一坐标系（物理 slot）下——`index_attn._topk_decode` 末尾的 `convert_kv_index(...)` 正是为此存在。

### 3.2 两套实现 + head padding

```python
if fused_norm_rope_cast_kwargs is not None:
    output_fp8, output_sf, _, lse = fused_norm_rope_attn_rope_cast.prefill/decode(...)
    return (output_fp8, output_sf), lse      # ← 直接吐 FP8 + scale
num_heads = query.shape[1]
query, attn_sink, padded_heads = _pad_attention_heads(query, ...)   # FlashMLA 只支持 64 / 128
output, lse = flash_mla_sparse_fwd / flash_mla_with_kvcache(...)
if padded_heads is not None: output, lse = output[:, :num_heads], lse[:, :num_heads]
```

**融合路径的语义约束**（docstring 明确写出）：

> `query must be permuted by permute_q_b_proj and not yet normed/roped`

即：融合 kernel 要求 Q **已经按 `permute_q_b_proj` 重排过、且尚未做 RMSNorm/RoPE**——因为这三种变换被吞进 kernel 内部，并与 attention 融合。这解释了 `attn.py` 里 `_permute_q_b_proj_for_fused()` 的存在，以及为什么该路径**与 `DEEPINFER_BATCH_INVARIANT` 互斥**（融合后的归约顺序与分离实现不同）。

### 3.3 decode 侧的四个"关掉"的参数

```python
block_table=None,      # FlashMLA 保留形参但当前忽略
cache_seqlens=None,    # 同上
causal=False,          # sparse attention 只支持非 causal
is_fp8_kvcache=True,   # V4.1 的 KV cache 为 paged FP8
indices=indices_in_kvcache.unsqueeze(1),
```

- `causal=False`：**因果性由 index 集合本身保证**（top-k 只从 `klen` 范围内选，不可能看到未来）；
- `block_table/cache_seqlens` 被忽略 → 说明寻址完全由 `indices`（已转成物理 slot）+ `tile_scheduler_metadata` 决定，**FlashMLA 在这个模式下不走 block_table 间接寻址**；
- `tile_scheduler_metadata=sched_meta` 来自 `input_metadata.flash_mla_sched_meta_dict[self.sched_meta_key]`，而 key 是 `f"{kv_compress_ratio}_{has_packed_kv}_{use_swa}_{index_topk}"`（`attn.py` 里构造）——**按"层几何形状"分桶共享调度元数据**，避免每层重算。

---

## 四、`v4_modules/*`：YoCo 与跨层的真实结构

### 4.1 `yoco.py`：三个谓词 + 三个操作，全是"下标算术"

**谓词**：

```python
yoco_enabled(cfg)            = cfg.n_prefill_layer < cfg.num_hidden_layers
is_yoco_consumer_layer(cfg,l)= n_prefill_layer <= l < num_hidden_layers
yoco_shared_hidden_required(cfg, topo)   # 是否存在"真正消费 shared hidden"的 consumer
```

第三个谓词是最细的，有两个消费点：

| 消费点 | 条件 |
|---|---|
| ① SWA 的 `kv_swa` 由 shared hidden 投影 | `yoco_swa_share_hidden` 且该层 `use_swa` |
| ② packed KV source 需要 shared hidden 算新压缩 KV | `layer_id > n_prefill_layer` 且 `n_packed_heads > 0` 且 `is_kv_source_layer` |

注释特别指出：**第一个 consumer 层自己直接用本层的 attn-norm hidden，不需跨层传递；share-kv 下 consumer 可能全部复用更早的 KV，此时也没有消费者**——所以这是个"可能为空"的判定，不是简单 `>= n_prefill_layer`。

**操作 ①：`yoco_boundary_compact`（三步走）**

```python
# 情形 1：无 CP —— 直接按 query_indices 取
if input_metadata.yoco_cp_contrib_index is None:
    return gather(x, input_metadata.yoco_consumer_query_indices)

# 情形 2：有 CP —— 三步
contrib  = gather(x2d, input_metadata.yoco_cp_contrib_index)   # 2.a 取本 rank 保留的 token（-1 标记 pad）
gathered = userbuffer.cp.all_gather(contrib)()                  # 2.b 按 rank 序拼接
out      = gather(gathered, input_metadata.yoco_cp_boundary_index)  # 2.c 取 compact+CP 后属于本 rank 的
```

> **这是全仓库最难的 3 行**：CP 下"truncate + repartition"两个变换复合，用"contrib（本 rank 保留集）→ all_gather（全局）→ boundary_index（重分区后本 rank）"三步分解，避免一次不可分解的置换。`-1` 作为 pad 标记贯穿两跳。

**操作 ②：`gather_hc_lazy`** —— 把 `HCLazyPostData` 的四个字段（`residual / post_layer_mix / comb_res_mix / pre_layer_mix`）逐个做同样的重排；注意 **batch 维是 dim 0 且恒为 1**，token 维是 dim 1，所以写的是 `t.squeeze(0)...unsqueeze(0)`。

**操作 ③：`compact_yoco_boundary`** —— 重排 2 个主数据 + 2 个跨层缓存，并**主动失效两个派生缓存**：

```python
if cross_layer_runtime.topk_indices is not None:  cross_layer_runtime.topk_indices = yoco_boundary_compact(...)
if cross_layer_runtime.block_candidate_idxs is not None: cross_layer_runtime.block_candidate_idxs = yoco_boundary_compact(...)
cross_layer_runtime.candidate_mqa_lens = None                    # ← 行布局变了
cross_layer_runtime.candidate_mqa_schedule_metadata.clear()      # ← DeepGEMM schedule 依赖行布局
```

> **这是"派生状态必须随源失效"的教科书处理**：topk 与 candidate blocks 是"按 token 行"的数据，重排后必须跟着重排；而 `candidate_mqa_lens`（每 query 的候选长度）与 `DeepGEMM schedule metadata` 是**行布局的函数**，只能失效重建，不能重排。

### 4.2 `cross_layer.py`：拓扑的四级校验（fail-fast 的典范）

**`SharedLayerTopology` 构造时做四类校验，每类都对应一个真实故障模式**：

| 校验 | 代码 | 不校验的后果 |
|---|---|---|
| **配置副本一致性** | `assert shared_cfg == self._shared_cfg`（所有携带该 config 的层必须完全相同） | 不同层以为的共享关系不同 |
| **分段性** | `assert src == last_source`（consumer 只能指向"前一个最近的 source"） | 跳段共享 → `CrossLayerRuntime` 的单槽位被后来的 source 覆盖 |
| **精化链** | `coarse[fine[l]] == coarse[l]`，要求 `index-topk ⊆ index-K ⊆ KV` | index-K 从 A 层取、其 K 由 B 层的 KV 投影而来 → 跨段读未初始化显存 |
| **consumer 必带 config** | `assert layer_id in self._shared_cfg_layer_ids` | cache engine 按"是否携带 config"判共享 → 缺 config 的 consumer 会静默分配独立 cache 且永不写入 |

第三条的**等价变换**很漂亮：

> 在"分段"前提下，`coarse[fine[l]] == coarse[l]` **等价于**"fine 的每个段都落在 coarse 的单个段内"。

于是 $O(n^2)$ 的区间包含检查变成 $O(n)$ 的下标查表。

**`CrossLayerRuntime` 只有 8 个槽位**，靠"分段 + 层序递增"保证**不需要记录 source id**：

```python
shared_hidden, pre_rope_kv, topk_indices          # 三个"最近 source 写入、consumer 读"的槽
block_candidate_idxs, candidate_topk_block,
candidate_block_size_before_compress,
candidate_mqa_lens, candidate_mqa_schedule_metadata  # 候选块组（5 个字段 + 1 个 dict）
loaded_prefill_kv, loaded_prefill_kv_main_offset     # prefill KV 复用
loaded_prefill_index_kv                              # prefill index-K 复用
consumer_full_metadata, consumer_full_hidden         # YoCo compaction 的 full 副本
```

> **单槽位设计 = 用"层序"替代"作用域"**。这也是为什么分段校验必须 fail-fast：**一旦分段被破坏，单槽位会读到"更晚的 source 写的值"或"更早的 source 的残留"，没有任何运行时检查能发现。**

### 4.3 `decoder_layer.py`：boundary 那 15 行的完整语义

```python
if cross_layer_runtime.consumer_full_metadata is not None and cross_layer_runtime.consumer_full_hidden is None:
    assert isinstance(input_metadata, YocoMetadataMixin)
    assert self.engram is None, "yoco 层不能是 engram 层"
    compact_inputs, full_inputs = compact_yoco_boundary(hidden_states, residual, input_metadata,
                                                        cross_layer_runtime, hc_enabled=...)
    cross_layer_runtime.consumer_full_hidden = self.attn_norm_forward(*full_inputs)[1]   # ← full 版 attn-norm
    hidden_states, residual = compact_inputs
hidden_states, hidden_states_bf16, residual = self.attn_norm_forward(hidden_states, residual, hc_post_intervention)
...
if self.yoco_captures_shared_hidden and cross_layer_runtime.shared_hidden is None:
    cross_layer_runtime.shared_hidden = hidden_states_bf16.clone()   # TODO(YOCO): 去掉这个 clone
```

**执行顺序的三层含义**：

1. **先对 full 版做一次 `attn_norm_forward`**，只为拿 `consumer_full_hidden`（供 `_store_yoco_full_kv` 更新 full 长度 KV）；
2. **再对 compact 版做一次**，得到真正参与 attention 的 hidden；
3. `shared_hidden` 用 **compact 版**（`hidden_states_bf16.clone()`）。

> 即：**同一层要算两次 attn-norm**——full 版用于"写 KV"，compact 版用于"算 query"。这是一次**显式的算力买账**：省掉的是 consumer 层 attention 的 $O(n^2)$，付出的是每层一次 RMSNorm（$O(n)$）。注释里的 `TODO(YOCO): 去掉这个 clone` 说明这里还有一次可省的拷贝（防止 attention 内部 inplace 清空 bf16 备份）。

**`dflash_input_index` 的断言链也值得记**：

```python
if self.dflash_input_index is not None:
    assert cfg.dflash_cfg.dspark_target_attn_input, "HF DFlash only supports attention-input targets"
    assert self.hc_enabled, "DFlash attention 输入特征提取仅支持启用 HC 的模型"
```

即 DFlash 的 target 特征**取自 attention 的输入而非输出**（存进 `residual.residual`，见 `self._store_dflash_input(residual.residual, dflash_input)`），且必须 HC 开启。

### 4.4 `compress.py`：C1 与 C2 的真实差别

```python
def forward_pre_norm(self, projected_kv, state_cache, input_metadata):
    if self.kv_compress_ratio == 1:
        assert projected_kv.dtype == torch.bfloat16 and state_cache is None     # ← C1 没有 state！
        if input_metadata.mirror_cp_reverse_index is not None:
            projected_kv = gather(projected_kv.contiguous(), input_metadata.mirror_cp_reverse_index)
        return projected_kv
    assert projected_kv.dtype == torch.float32 and state_cache is not None      # ← C2 必须有 state
    score, latent = projected_kv.chunk(2, dim=-1)
    kv_compressed = compress_and_update_state_{prefill,decode}(score, latent, state_cache, ...)
```

| | C1 | C2 |
|---|---|---|
| Gate 模块 | `None`（`SoftmaxAPEGatedColumnParallelLora.Gate = None`） | 有 `Gate` |
| dtype | bf16 | **fp32**（score/latent 都要保精度） |
| state cache | **不需要**（`state_cache is None` 断言） | **必需**（ring buffer，长度 `ratio + backtracking - 1`） |
| 语义 | 直接投影（单 token softmax 恒为 1） | $\text{softmax}(\text{score})$ 对两个 token 的 latent 加权合并 |
| CP | 需要 gather 重排（mirror CP） | 由 kernel 内部处理 |

> `SoftmaxAPEGatedColumnParallelLora` 的 docstring 把 C1 的退化原因讲得很清楚：**"The single-token softmax of C1 is always 1, so only A is needed"**。这解释了为什么 `state_manager` 在纯 C1 模型上是 `None`（`llm_engine` 里 `BlockManager(cache_config.num_state_seqs) if bytes_of_state_per_seqs > 0 else None`），而 SWA 层仍有 state slot。

**`forward_store_kv_cache` 的三输出一次成型**：

```python
fused_norm_rope_hadamard_cast_store_kvcache(
    kv_compressed,
    norm_args=NormArgs(self.norm.weight, ...),
    rope_args=RopeArgs(positions_c, cos_sin_cache, rotary_dim, ...),
    cast_args=CastArgs.from_cache_format(cache_format),
    out_args=KVCacheOutArgs(kv_cache, slot_mapping, out=pre_rope_kv, out_stage="norm", layout=cache_format),
)
```

> 一趟 kernel 同时产出：① 写进 paged cache 的量化 KV；② 供 indexer 用的 **pre-RoPE** latent（`out_stage="norm"` 表示取 norm 之后、RoPE 之前的阶段）。**"顺手多产出一个中间态"是这份代码反复出现的优化范式**（对比 `attn.py` 里 `_q_b_norm_rope` 的 `fused_norm_rope_hadamard_cast_store_kvcache` 只产 Q）。

### 4.5 `mla_attn.py` / `builder.py`：参数容器与"物理共享"

**容器四类**（`mla_attn.py` docstring 自己列了）：

```text
LoraRopeCompressFlashMHAAutoRecomp   # 一层的全部参数
ColumnParallelLora                   # A → norm → B 的 MLA 低秩投影（Wq / Wkv）
SoftmaxAPEGatedColumnParallelLora    # 压缩 KV 的 A / Gate
MQAIndexer                           # 稀疏注意力的 indexer
```

**`ColumnParallelLora` 的巧思**：`out_features == 0` 时 `B = None`——于是 `Wkv` 只有 `A`（输出 `kv_lora_rank + rope_head_dim` 的 latent），**用"out_features=0"表达"这个投影不升维"**，而非另写一个类。

**`num_lora_rope_head` 的存在**：`A` 的输出维度是 `rank + num_lora_rope_head * rope_head_dim`——即**LoRA 潜向量与 RoPE 分量在同一次投影里拼出来**，这就是 MLA 的 "absorbed" 形态。

**`builder.py` 的两个关键点**：

```python
# ① 共享 module 是"物理共享"（同一个 Parameter 对象），不是复制
def _attach_shared_index_k(self, block, runtime_layer_id, shared_topology):
    if index_k_src == runtime_layer_id: self._index_k_modules[runtime_layer_id] = index_attn; return
    src = self._index_k_modules[index_k_src]
    index_attn.k_proj = src.k_proj                       # ← 直接赋同一个 nn.Module
    if hasattr(src, "k_norm"): index_attn.k_norm = src.k_norm
```

> 注释说明了目的："`named_parameters()` visits them once"——**避免重复加载、重复 allreduce、重复显存**。这也解释了为什么 `cache_engine` 的 `_get_index_mapped_layer_id` 能直接复用 `cache_list[src].ica_cache`：**模块对象本身就是同一个**。

```python
# ② MTP 层的 runtime_layer_id 要 -1
runtime_layer_id = layer_id if layer_id < gpt_cfg.num_hidden_layers else layer_id - 1
```

因为 `attn_module_list` 少了 lm_head 那一项，MTP 层在拓扑映射里必须偏移。**这是一个极易踩的 ±1 错误，只靠注释保护**。

### 4.6 `moe.py` / `engram.py` 的模型结构要点

| 结构 | 发现 |
|---|---|
| `TopKGate` | 打分函数 `sqrtsoftplus`、选路 `noaux_tc`、**两个 FP32 偏置**：`e_score_correction_bias`（文本）+ `e_score_correction_bias_vl`（**图像专用**）→ 多模态走了独立的专家路由偏置 |
| `FusedExperts` | `n_routed_experts = n_logical + n_redundant`；`n_local = n_routed // ep_size`（断言整除）→ **物理专家数与 EPLB 的 `num_physical_experts` 是同一个数** |
| `Engram` | n-gram 表由 `userbuffer.engram`（DeepEP `EngramBuffer`）持有；`fetch(hashes)` 返回**每层一个 hook**，延迟调用才真正取数据 |
| `EngramRuntime.initialize` | `cuda_view_for_write()`：把 CPU 上的表用 `_construct_storage_from_data_pointer` 造一个 **CUDA-tagged view** 交给 DeepEP 写——允许表常驻 CPU/pinned 而不复制（NPU 上强制 GPU 存储：`DEEPINFER_ENGRAM_FORCE_GPU_STORAGE` 默认 true） |
| `CompressedTokenizer.encode` | 走 C++ `encode_compressed`（单遍），**词表外 id 一律映射到 `IMAGE_SENTINEL=-1`** |

**`Engram` 的插入点**：`decoder_layer` 里作为 `hc_post_intervention` 回调，**在 HC 的 post 之后干预主干**：

```python
hc_post_intervention = lambda trunk: self.engram(trunk, embed_hook, images_seq_mask)
```

---

## 五、`ve/multimodal_processor.py`：VE 侧异步预处理

### 5.1 三级背压 + 池化零拷贝

```text
engine.add_request
  └─ processor.add(id, mm_data, num_cached_tokens)     # 立即返回
       ├─ overwrite_prompt(): 占位 token → IMAGE + N×IMAGE_PAD（同步，主进程）
       ├─ _compute_and_strip_cached_images(): 剥离已缓存图片的 binary
       └─ input_queue.put((id, mm_data))               # ① 进 worker
            ▼  fork 出的 worker 进程（每节点 num_workers 个）
       multi_modal_mapper(mm_data)  → 张量结果
       pool.alloc_tensor(...) + copy_ → 只写共享池，dict 里留 handle
       output_queue.put((id, dict))                    # ② maxsize=512 → 天然背压
            ▼
  fetch()（engine.step 每步调用）
       ├─ handle → get_multi_modal_pool().get_tensor(...)   # 零拷贝映射
       └─ scheduler.set_schedulable(id)                     # ③ 只有此时才可调度
```

**三级背压点**：`input_queue`（无界，靠 `output_queue` 反压）→ `output_queue(maxsize=DEEPINFER_MULTI_MODAL_BUFFER_SIZE=512)` → **`scheduler.set_schedulable(id)`**（调度器侧的 `schedule.schedulable` 标志）。

> `schedulable` 是**单调的**（`_schedule_prefill` 遇队头不可调度直接 `break`），所以多模态请求**必须排在队头并阻塞后续请求**——这是 `waiting` 队列 FIFO 语义的直接后果，也是"多模态请求会拖慢同批文本请求"的根因。`add_sequence` 里 `waiting.push_back` 且 `schedulable` 默认 false，就靠 `set_schedulable` 打开。

### 5.2 三个"工程硬伤处理"

| 处理 | 代码 | 为什么必须 |
|---|---|---|
| **`os.abort()` 防挂死** | `_check_worker_hang()`：`_pending` 非空且超过 `MULTI_MODAL_WORKER_TIMEOUT`（默认 1800s）无进展 → 直接 abort 进程 | worker 卡在 CUDA/解码库内部，`terminate()` 可能无效；**宁可整机崩也不静默挂住** |
| **NPU SIGTERM 修正** | `if PLATFORM.is_npu: signal.signal(SIGTERM, SIG_DFL)` | torch_npu 的 bad-fork 路径把 SIGTERM 改成 SIG_IGN，导致 `terminate()` 失效；恢复默认后内核直接杀，**不执行继承的 NPU cleanup**（否则影响父进程 context） |
| **SMT sibling 绑核** | `bind_cpus = range(visible_index*num_per_gpu + phy_cpus, ...)`（SMT 开启时） | 把 VE 进程绑到**超线程兄弟核**，避开主进程占用的物理核 |

### 5.3 `overwrite_prompt` 的核心不变量

```python
image_tokens = np.full(num_image_tokens, self.mm_image_pad_token, dtype=np.int32)
image_tokens[:1] = self.mm_image_token         # ← 首 token 必须是 IMAGE
```

注释给出理由：

> "首 token 必须是 IMAGE 才能支持 chunk prefill"

这与 `scheduler.hpp::_is_inside_image_pad_region` 的实现**严格耦合**——它从 `k_len` 往回扫，遇到 `IMAGE_PAD` 继续，遇到 `IMAGE` 判为"在图内"，遇到其他 token 判为"不在图内"。**如果首 token 不是 IMAGE，整个"不在图片中间切开"的保护会失效。**

`segment_input` 保留文本/图像交替结构，但文本段只存 `TextLengthInput(length=n)` ——**不传文本 token 内容**（引擎 input_ids 已有），避免无谓的跨进程拷贝。注释："推理侧不消费文本 token 内容"。

### 5.4 `_compute_and_strip_cached_images`：前缀缓存感知

```python
for start, count in image_spans:
    if start + count > num_cached_tokens: break     # 边界落在图中间 → 视为未缓存
    num_cached_images += 1
for seg in image_segments[:num_cached_images]: seg.binary = None
multi_modal_data["_num_cached_images"] = num_cached_images
```

只剥离"**整段落在 `[0, num_cached_tokens)` 内**"的图片 binary；下游按 `_num_cached_images` 跳过对应前缀。**这是把"KV 前缀命中"的信息一路透传到 VE 预处理**，避免重复解码已缓存的图片。

---

## 六、`object_pool.hpp`：对象池的全部实现（20 行）

```cpp
struct EnableObjectPool { bool allocated_from_object_pool_; };   // 标志位基类

template <class T> class ObjectPoolPyObject {
  static_assert(std::is_base_of_v<EnableObjectPool, T>);
  template <class... Args> py::object allocate(Args &&...args) {
    if (objects_.empty()) {
      auto obj = py::cast(new T(std::forward<Args>(args)..., true),   // ← 末尾追加 true
                          py::return_value_policy::reference);
      return obj;
    }
    auto p = std::move(objects_.back()); objects_.pop_back();
    auto ptr = p.cast<T *>();
    new (ptr) T(std::forward<Args>(args)..., true);                    // ← placement new 重建
    return p;                                                          // ← 复用同一个 PyObject
  }
  void release(py::iterable objs) {
    for (auto &&obj : objs) {
      auto p = obj.cast<T *>();
      if (p->allocated_from_object_pool_) { std::destroy_at(p); objects_.emplace_back(obj.cast<T *>()); }
    }
  }
  std::vector<T *> objects_;
};
```

**五个必须理解的点**：

| # | 点 | 含义 |
|---|---|---|
| 1 | **`py::cast(..., reference)`** | Python 侧拿到的是**不持有所有权**的 `py::object`；真实生命周期由池决定 |
| 2 | **`new (ptr) T(..., true)`** | 复用时**显式 `destroy_at` + placement new**，析构/构造都跑一遍（`std::optional`、`py::object`、`std::vector` 都能正确重置） |
| 3 | **`true` 是 `EnableObjectPool` 的构造参数** | 所有 `T` 的构造函数必须能接受末尾多一个 `bool`；`SequenceMetadata` 的两个构造函数都这样写了 |
| 4 | **`release` 只回收 `allocated_from_object_pool_` 为真的** | Python 自建的 dummy（`false`）不进池，靠 Python GC 回收 → **一个池同时服务两种来源** |
| 5 | **`objects_` 存 `T*` 而非 `py::object`** | 避免池持有强引用导致 Python 对象永不销毁（引用计数归零时 PyObject 被释放，池里的裸指针就悬垂） |

> ⚠️ 第 5 点是一个**隐含契约**：池里 `T*` 的有效性依赖"Python 侧仍持有该 `py::object`"。在 `DeferredRelease` 的设计下，`sequence_metadata_pool.release(list)` 的入参 list 由引擎持有到调用时——**如果引擎提前丢掉这个 list，池里就会出现悬垂指针**。这是零拷贝 + 对象池组合的固有风险。

**与 `DeferredRelease` 的配合**：`release()` 就是"**延迟回池**"。所以 ZOS 的"延迟一步"本质上也是"**池对象多占用一整个 step**"——池的峰值容量需求 = `max_num_seqs × 2`（当前步 + 上一步）。

---

## 七、闭环：`llm_engine.py` 与 `attention_backend.py` 如何把"调度→执行"接上

### 7.1 完整时序（标注跨界所有权转移）

```text
[Python/engine]  LLMEngine.step()
   │  scheduler.schedule(pool)  ───────────────► [C++] _schedule()（无 GIL）
   │  ◄── (seq_metadata_list, SchedulerOutputs)   借用内部 vector 存储
   │
   │  _run_workers("clone_and_zero_states_and_blocks...")  ──► [C++] cuda memcpy（可选）
   │
   │  _run_workers("execute_model_step")  ─────────────────► [Python/worker] ★重入点
   │                                                            │
   │      ┌──────────────── Worker.execute_model_step ──────────┤
   │      │ ① _correct_seqlens_qk()  → zos.cpp: remap/seqlens  │
   │      │    → cuda_graph_runner.select(total_q)              │
   │      │ ② execute_main_model()                              │
   │      │    ├ _prepare_input_tensors → prepare_v4_decode_metadata（GPU kernel）
   │      │    │    写进 args.out（= 图的输入 buffer，零 H2D）
   │      │    ├ cuda_graph_runner.replay() → attention_backend
   │      │    │    ├ prefill: prefill_attn_varlen_sparse（indices=[total_q,1,topk]）
   │      │    │    └ decode : decode_mla_mqa_sparse（SWA + 压缩 双路 extra）
   │      │    └ 返回 output（图的 output buffer，下次 replay 前有效）
   │      ├ ③ _post_process_step(prev)  ── 回调 ──► [Python/engine] post_process()
   │      │                                        └ _process_model_outputs：
   │      │                                           process_MTPInfo → scheduler.process_outputs
   │      │                                           （commit num_tokens / append tokens / SO 判死）
   │      │                                         └ deferred_release.collect_*（延迟释放）
   │      ├ ④ sample() → sampler._verification → 写 num_accepted_tokens_gpu
   │      └ ⑤ engine_callbacks.pre_sampler_sync_fn() ── 回调 ──► [Python/engine]
   │                                                           ├ deferred_release.release()
   │                                                           ├ do_save_kvcache / SO pump
   │                                                           └ finalize_seq（COMPLETION/NORMAL）
   │
   └─ 返回 request_outputs（np.uint64 指针数组）→ entrypoint 用 pool.clone() 复制
```

### 7.2 六个"闭合环"

| 环 | 正向 | 反向 | 闭合点（唯一） |
|---|---|---|---|
| **① 容量环** | `_schedule_*` 分配块 → `block_table` | `process_outputs` → `free_seq` → `free_blocks` | `BlockManagerMutexAcquire`（同一步内持锁做完） |
| **② 长度环** | `seqlen_q/seqlen_k` → `zos.cpp` → kernel | `sampler` → `num_accepted` → `commit_processed_tokens` | `update_metadata_for_sampling_zos` |
| **③ 并发环** | `start_inflight_step(max_q)` → `get_num_blocks()` | `finish_inflight_step()` | `process_outputs` 的第一行 |
| **④ 图环** | `input_metadata_source` → `_cut` → replay | `output.narrow` → `Movable` → sampler | `cuda_graph_runner.replay()` |
| **⑤ 对象环** | `pool.allocate()` | `DeferredRelease.release()` → `pool.release()` | `pre_sampler_sync` |
| **⑥ 可调度环** | `add_sequence`（schedulable=false） | `set_schedulable(id)` | 多模态 `fetch()` |

> **这六个环的"闭合点"全都在 `pre_sampler_sync` / `process_outputs` 这两个函数附近**——它们就是整个引擎的"心跳点"。任何 ZOS 相关的正确性论证，最终都会收敛到这两处。

### 7.3 `attention_backend.py` 在闭环中的位置

它是**唯一"不参与状态管理"的纯函数层**：

```text
输入：query(bf16/fp8+sf), key_cache(uint8 paged), indices(int32 物理 slot), sched_meta, attn_sink
输出：(output, lse)
约束：无副作用、无 cuda sync、不读 num_tokens、不写 cache
```

**它把闭环 ② 的"语义"完全剥离了**：

| 语义 | 由谁保证 | 在哪 |
|---|---|---|
| 因果性 | `indices` 只含 `[0, klen)` 内的 slot | `index_attn._topk_decode` 的 `end=topk_lens` |
| KV 有效性 | `block_table` 已 `_append_slot` | `_schedule_decode` |
| 数据新鲜度 | 上一步已 commit | `commit_processed_tokens` |
| 双路（SWA+压缩）一致 | 两组 indices 同坐标系 | `convert_kv_index` |

> **这个分层是干净的**：kernel 只知道"给我 slot 号和元数据，我算注意力"；所有"该看哪些 slot"的判断都在调度层与 `index_attn` 里完成。代价是**如果 `indices` 错了，kernel 不会报错，只会算出错误的数**——这是整条链路最不可观测的失效模式。

---

## 八、本轮新发现的风险清单

| # | 风险 | 位置 | 触发条件 / 后果 |
|---|---|---|---|
| 1 | **`replay()` 返回的 output 只在下一次 replay 前有效** | `cuda_graph_runner.replay` | 任何跨 step 持有 `CausalLMForwardOutput` 都会读到新数据；靠 `Movable` 纪律，无断言 |
| 2 | **`total_q > max_batch_size` 直接抛 ValueError** | `select()` | 调度器排出超额 batch → 崩（不是回退 eager） |
| 3 | **`get_clean_list()` 漏登一个派生字段** | `InputMetadata` 类声明 | 该字段被固化为"warmup 旧值"作为图输入 → **静默算错**，且只在特定 batch 组合下暴露 |
| 4 | **对象池 `T*` 依赖 Python 侧强引用** | `object_pool.hpp` | 引擎提前丢 list → 池里悬垂指针 |
| 5 | **`CrossLayerRuntime` 单槽位 + 分段校验** | `cross_layer.py` | 分段被破坏时，只能读到错的 source，**无运行时检查** |
| 6 | **`runtime_layer_id = layer_id - 1`（MTP 偏移）** | `builder.py` | 只靠注释；改 `attn_module_list` 结构即错位 |
| 7 | **`_attach_shared_index_k` 的物理赋值** | `builder.py` | `index_attn.k_proj = src.k_proj` 之后，两者**不可能再独立加载/更新** |
| 8 | **`os.abort()` 的 VE 熔断** | `multimodal_processor.py` | 单图解码卡死 → 整引擎进程被杀（设计上如此，但对多租户不友好） |
| 9 | **多模态请求会阻塞 waiting 队头** | `schedulable` 单调 + `_schedule_prefill` break | 一批请求里最前面的多模态请求未就绪 → 后续文本请求全部等待 |
| 10 | **`_is_inside_image_pad_region` 依赖"首 token 是 IMAGE"** | `scheduler.hpp` × `overwrite_prompt` | 跨语言隐式契约；改任一侧即失效 |
| 11 | **`causal=False` 的稀疏 attention** | `attention_backend.py` | 因果性完全由 indices 保证，kernel 不做保护 |
| 12 | **C1 模型 `state_cache is None`** | `compress.py` 断言 | 混合 C1/C2 模型（`derive_compress_ratios` 允许）时，C1 层不占 state，C2 层占 → StateBlock 布局必须支持"部分层无 state" |

---

## 九、一句话总结

> 本轮六个文件的真正发现不是"某个新算法"，而是**三种"用结构换性能"的手法**：
> ① **`llm_engine` ↔ `worker` 的回调重入**——用调用栈嵌套替代线程/队列，实现 ZOS 的 CPU/GPU 重叠；
> ② **`cuda_graph_runner` 的前缀视图 + 共享池 + 派生字段清零**——用"零拷贝 + 一次捕获 192 张图"替代动态 kernel 派发；
> ③ **`cross_layer` + `yoco` 的"单槽位 + 分段校验 + 下标算术"**——用"层序即作用域"替代显式的跨层通信协议，用 `coarse[fine[l]] == coarse[l]` 把区间包含降为 $O(n)$。
>
> 三者共同的代价是：**正确性几乎全部依赖"不变量 + 注释 + fail-fast 断言"，而不是类型系统或运行时检查**。`attention_backend.py` 只有两个函数、零状态，反倒是整个栈里**唯一能单独测试、单独替换**的部分——这或许正是它值得被单独拎出来看的原因。

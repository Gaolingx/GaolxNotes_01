# pytorch lightning liger kernel vllm不适合异步RL骨架

## 🧑‍💻 User

pytorch lightning+ligerkernel+vllm到底能不能担当大规模asyncrl 骨架？

---
## 😀 默认助手 (7)

I'll ground a few key claims before answering.

## 一句话结论

**不能当"骨架"，但 vLLM 已经比你想象中接管得多。** 这个组合在 2024–2025 年的判断是"三个好东西、缺一整个控制面"；到 2026 年，缺口位置已经从 vLLM 转移到了 **PyTorch Lightning 这一层**——vLLM 把 async RL 的**数据面 + 半个控制面**做进引擎了（pause/resume、per-token `weight_version`、六种 weight transfer 后端），而 PL 的同步、epoch/step 中心、单优化循环的假设与 async RL 的本质是直接冲突的。Liger 则**与骨架无关**，它是算子层加速。

| 场景 | 判定 |
|---|---|
| 单轮 GRPO/RLVR，≤ 数十卡，单集群，无弹性 | ✅ 够用（半异步 / one-off pipelining），PL 留着也无妨 |
| 解耦生成与训练、多版本并发、MoE EP、数百→数万卡、长程 agentic | ❌ 不够，PL 必须换成事件驱动/actor 化骨架，或直接用已验证系统 |

---

## 1. 先把三个东西的角色摆正

| | 它是什么 | 在 async RL 里对应哪个子系统 | 能否承担"骨架" |
|---|---|---|---|
| **vLLM** | 推理引擎（+2026 版内置 RL 生命周期） | rollout 引擎 + 权重同步数据面 | 单块最接近，但只是**一个子系统**（虽是大头） |
| **Liger Kernel** | Triton/CUTLASS 融合算子库 | 训练前反向的 FLOPS/显存优化 | ❌ 完全不是骨架，是性能插件 |
| **PyTorch Lightning** | 同步训练循环抽象（Trainer/Strategy/Callback） | learner 内部循环的样板 | ⚠️ 只在"同步 learner"这一段成立，且是最容易被剥离的一段 |

---

## 2. 关键更新：vLLM 在 2026 已经不是"纯推理引擎"了

这一块决定了整体答案的走向，值得单独看。

<details>
<summary><b>vLLM 当前实际的 async RL 能力（点开）</b></summary>

来自 vLLM 官方文档（`training/async_rl`、`training/weight_transfer`，文档日期 2026-10）：

**异步生命周期原语**（这正是过去要自己写的部分）：
- `pause_generation(mode="abort" | "wait" | "keep", clear_cache=...)`，其中 `mode="keep"` 会把在飞请求冻结在队列里，`resume_generation()` 后用**新权重**继续生成；
- **per-token 权重版本标签**：`finish_weight_update(weight_version=...)` 之后，每条输出的 `completion.weight_versions` 是一个 `[start, end) → version` 的 span 列表。

```python
for span in completion.weight_versions:
    # [span.start, span.end) 这段 token 由版本 span.version 的权重采样而来
    print(span.version, completion.token_ids[span.start:span.end])
```

这一条的分量很大：它把"哪些 token 是 off-policy 的"从**你猜**变成**引擎告诉你**。ENTR / DORA 那类"逐 token 信任域 + 有界滞后"的方法，正好需要这种 token 粒度的版本可观测性。

**完整的权重传输子系统**（可插拔，四次握手：init → start → update → finish）：

| 后端 | 传输 | 解决什么 |
|---|---|---|
| `nccl` | NCCL broadcast | 训练/推理分卡（默认方案） |
| `ipc` | CUDA IPC handle | 同卡 colocate |
| `sparse_nccl` | NCCL | checkpoint 坐标的稀疏权重补丁 |
| `sharded_rdt` | NIXL / Ray Direct Transport（pull） | MoE+EP 超大模型，每个 worker 只拉自己的切片 |
| `nccl_m2n` | NCCL M2N reshard | **训练与推理分片布局不同**（FSDP/EP ↔ TP） |
| `ModelExpress` | 版本化传输（含 delta） | 不可变权重版本、对象存储 checkpoint、增量权重 |

训练侧用法已经收敛成一行：多 rank trainer（FSDP/TP/PP/EP）**每个 rank 都建 engine 并调用 `send_weights()`**，rank 0 负责上线，其他 rank 负责参与 all-gather 以免死锁。这恰好是过去"自己写会写错"的细节。

**内存/共存**：sleep level 0/1/2、`wake_up(tags=["weights"])` 分阶段唤醒以避免 OOM、`sleep_preserve_parameter_names`（冻结参数不参与更新）、CUDA graph 卸载、`enable_nccl_comm_suspend`（NCCL 2.29.7+）、`release_kv_cache_memory`。

**周边**：`batch_invariance`（训练/推理数值一致性）、`sampling_mask`（distribution replay）、`weight_checker`、`layerwise (re)loading`、teacher scoring。这些全是 RL 专属需求，不是 serving 需求。

**同时它诚实标注了危险区**：`clear_cache=False` 时保留的 KV 可能由旧权重算出，即**同一条轨迹内部权重版本不一致**。DORA 把"intra-trajectory policy consistency"列为保持收敛的三大约束之一——vLLM 把问题暴露出来了，但**没有替你解决**。

</details>

所以"缺 weight sync / 缺 reshard / 缺 KV 复用"这些 2024 年的结论，现在基本过期了。

---

## 3. 真正的断点：PyTorch Lightning

这不是"PL 不够强"，而是它的**基本假设**与 async RL 的**基本事实**逐条对撞：

| PL 的假设 | async RL 的事实 | 后果 |
|---|---|---|
| 一步 = 对固定 batch 的一次前反向，所有 rank 锁步 | 数据**按到达时间**流入，形状/长度/版本都变，rank 可能空等 | `training_step` 契约失效 |
| 数据源是 DataLoader/IterableDataset，取值是纯 CPU 逻辑 | 数据源是**外部分布式系统 + 环境 + 工具**，取值可能阻塞数秒到数分钟 | 只能靠队列式 `IterableDataset` 硬接 |
| `Trainer` 拥有 `init_process_group` 和分布式策略 | 现场有**第二套分布式拓扑**（vLLM 的 TP/PP/DP/EP + NCCL/IPC/NIXL/RDT 通道） | 双 launcher、双 process group、env 冲突 |
| 权重只在"保存/加载 ckpt"时进出 | 权重同步是**每步的生命周期**（4 阶段协议 + 版本标签） | 只能塞进 callback，即"在与框架搏斗" |
| checkpoint 语义 = 恢复训练 | 需要的是"导出可传输参数" + **可恢复的 rollout buffer** | 语义不同，PL 的 ckpt 基本要被绕过 |
| 单 model、单优化器 | policy + reference + critic/value + reward 多模型、多并行度 | 要自己起多个 module |
| elastic restart 靠 torchelastic | restart 还要重建 buffer 版本、重 shard 推理侧权重 | 开箱不支持 |

两个具体化的痛点：

1. **手动优化逃生舱**：为了避免 buffer 饥饿时的空步推进 optimizer，你得设 `automatic_optimization = False`，然后你就在 PL 里面**手写训练循环**了——PL 的价值瞬间归零，只剩日志和 callback。
2. **两套 allocator 共处**：vLLM sleep/wake 用 `cumem`/managed allocation，训练侧用 PyTorch caching allocator，colocate 时的 OOM 抖动是常态。PL 对此没有任何建模。

经验证据也一致：以标题/摘要检索 arXiv，`all:"PyTorch Lightning" AND all:"RLHF"` 命中的论文数是 **0**；`abs:"PyTorch Lightning" AND abs:"reinforcement learning"` 的 10 条里只有 1 条相关（StarCraft II 数据集 API，与 LLM RL 无关）。而 async RL 这边的系统论文（AReaL、LlamaRL、PipelineRL、DORA、SAO、MARLaaS、ENTR）**没有一个是建在 PL 上的**——它们全是 Ray / 单控制器 / 自研事件循环。

> 需要说明：这是标题+摘要级检索，不等于全文统计；但方向性信号很明确。

---

## 4. Liger 的位置：与骨架无关，但已经 RL-native

<details>
<summary><b>Liger 的边界与它悄悄做对的 RL 适配（点开）</b></summary>

它是什么：HF `transformers` 模型的 monkey-patch 算子集（`AutoLigerKernelForCausalLM` / `apply_liger_kernel_to_*`），开箱与 FSDP、DeepSpeed、DDP、FlashAttention 共存，官方口径是吞吐 +20%、显存 −60%。它**不是** attention 后端（FA3/FA2 你还得自己上），也不是框架。

对 async RL 的价值主要在训练侧和"重算 logprob"侧，尤其是 **fused linear cross entropy**（把 LM head 与 CE 融合，省掉 `[B, L, V]` 的 logits 物化）。

有意思的是它最近的 RL 化动作：
- `LigerFusedLinearScaledCrossEntropyFunction` 输出 **per-token NLL + 可微的词表熵**——这正好是"熵归一化信任域"（ENTR 那类逐 token 门控）需要的量；
- 新增 **TP 前端** `LigerFusedLinearScaledCrossEntropyTPFunction(x, local_weight, target, tp_group, ...)`，从 rank 推导 `vocab_start`，在 TP 组内对输入梯度求和；
- 它的 fallback 实现**明确写了"adapted from Verl's fused PPO formulas"**；
- 还有 `LigerExpertParallelFusedMoe`（CUTLASS + NVSHMEM 原生路径）与 `LigerFusedLinearKLDivLoss`（对 reference 模型的 KL，即 GRPO/KL 惩罚项）。

也就是说，Liger 已经从"SFT 加速器"演进成"RL 训练算子层"，但它是**模型内部**的东西——换成 Megatron 风格的模型或自研 kernel 时就用不上了，而且它和 vLLM 的 kernel 不是同一套（logprob 数值差异要靠重算 + 门控来兜）。

**结论**：Liger 回答的是"训练侧便不便宜"，不回答"骨架成不成立"。加不加它，第 3 节的结论都不变。

</details>

---

## 5. 覆盖度矩阵：还缺什么

| 子系统 | PL | Liger | vLLM(2026) | 仍缺 |
|---|:--:|:--:|:--:|---|
| Rollout 引擎 | — | — | ✅ | — |
| 无中断权重同步 | ❌ | — | ✅（6 后端 + 版本标签） | — |
| KV / 显存共存 | ❌ | — | ✅（sleep/wake） | — |
| 训练引擎（FSDP/TP/EP） | ✅ | ✅ | — | 但需 step 级而非 epoch 级循环 |
| **异步编排与背压** | ❌ | — | ❌ | asyncio / Ray actor / 单控制器 |
| **Rollout buffer（带版本）** | ❌ | — | ⚠️ 只给 metadata | 版本化存储、过滤、去重、持久化 |
| **滞后策略 / off-policy 修正** | ❌ | ⚠️ 算子级 | ❌ | $k$ 上界、IS/TIS、逐 token 门控 |
| **轨迹内版本一致性** | — | — | ⚠️ 已标注风险 | `clear_cache` 取舍、stitching 策略 |
| **资源编排（训练/推理配比）** | ❌ | — | ❌ | colocate vs 分离、动态重分配、弹性 |
| **容错与可恢复 episode** | ⚠️ ckpt | — | ❌ | buffer 持久化、长程 agent 重试 |
| 环境/工具沙箱 | ❌ | — | ❌ | 你自己的 |
| 可观测性 | ⚠️ logger | — | ✅ metrics | 滞后分布、队列深度、KV 命中率 |

**大约 50–60% 的管道已被现成组件覆盖，但没覆盖的恰好是 async RL 里"新"和"难"的那部分。**

---

## 6. 先算清楚你值不值得异步

异步的收益**不是**简单的 overlap 比，而是**长尾比**：

$$ \text{同步每步代价} \propto \mathbb{E}\Big[\max_{i} L_i\Big] \quad\text{vs}\quad \mathbb{E}[L_i] $$

agentic / 长程推理下 $\mathbb{E}[\max_i L_i] / \mathbb{E}[L_i]$ 轻易到 2–3×，这才是 DORA/AReaL 报的 2–4× 的来源（DORA 另一处口径：rollout 占 step 时间的 50–80%）。**如果你的长度分布很窄，异步几乎白干，还会引入 off-policy 的不稳定。**

代价侧有两笔账：

1. **样本效率**：滞后 $k = v_\text{train} - v_\text{rollout}$。得用 $k$ 上界、截断重要性采样、或逐 token 熵归一化门控来兜。当前前沿能吃 30 个版本滞后还稳（ENTR）。
2. **权重同步成本**：

$$ C \approx \frac{P \cdot b}{B_\text{link}} $$

7B bf16 → 14 GB / 次同步。400 Gbit/s IB（≈50 GB/s）≈ 0.28 s；NVLink 900 GB/s ≈ 16 ms。乘上每小时同步次数就不能忽略。

> **实用建议**：想低成本验证异步，先上 **LoRA**——同步量从 GB 级掉到 MB 级，$C$ 直接消失，你会立刻看清"异步到底给你带来多少吞吐"。MARLaaS 的多租户方案正是靠 adapter 化的。

---

## 7. 如果你就是要保留 PL：最小可行配方

不是不行，但你要接受"只用上 PL 20% 的功能"：

```python
# 进程 A：rollout（vLLM），一个后台 asyncio 循环持续投喂 prompt
engine = AsyncLLMEngine(
    model=...,
    weight_transfer_config=WeightTransferConfig(backend="nccl"),  # 或 ipc（同卡）
    enable_sleep_mode=True,
)

# 进程 B：learner（PL）
class RolloutStream(IterableDataset):        # num_workers=0，数据留在 CPU
    def __iter__(self):
        while True:
            yield self.queue.get()           # {tokens, masks, old_logprobs, weight_version}

class WeightSync(pl.Callback):                # 版本同步塞进 callback
    def on_train_batch_end(self, trainer, pl_module, ...):
        if trainer.global_step % sync_every == 0:
            await engine.pause_generation(mode="keep", clear_cache=True)  # 关键：切干净
            self.wt_engine.send_weights()                                  # 4 阶段协议
            await engine.resume_generation()

Trainer(max_steps=..., strategy="fsdp2", callbacks=[WeightSync()], ...).fit(model)
```

要点与坑：`mode="keep"` 才是在飞请求不丢的异步原语；`clear_cache=True` 换来轨迹内权重一致、但丢掉 prefix cache 收益，两者必须做 A/B；队列不均会导致 DDP rank 间偏斜，需监控；buffer 饥饿时要么空转要么 `automatic_optimization=False` 自己控制。

---

## 8. 推荐路线

- **别从零造骨架**。已公开验证的异步系统：**AReaL**（完全解耦，2.77×，`inclusionAI/AReaL`）、**PipelineRL**（in-flight weight update，~2× / 128×H100，含开源实现）、**LlamaRL**（单控制器、8B/70B/405B、RDMA 权重同步）、**DORA**（多版本流式 rollout，万卡级，产出 LongCat-Flash-Thinking）、**SAO**（单 rollout 异步，GLM-5.2 750B-A40B）、**MARLaaS**（多租户 + LoRA + 分离式异步）。生态侧还有 verl / OpenRLHF / SkyRL / slime / NeMo-RL / TRL（小规模）。
- **如果要自研**，形如：
  **控制面** asyncio 或 Ray actor（`RolloutScheduler` / `BufferActor` / `LearnerActor`）
  → **rollout** vLLM `AsyncLLMEngine` + `pause/resume` + `weight_transfer_config`
  → **buffer** 版本化、带 token span 级 `weight_version`、支持部分轨迹
  → **learner** 裸 PyTorch 循环 + FSDP2/Megatron（**把 PL 去掉或降级**）+ Liger fused CE
  → **滞后策略** $k$ 上界 + 截断 IS / 逐 token 门控
  → **资源** 小集群 colocate（sleep/wake），大集群按 $T_\text{gen}:T_\text{train}$ 配比分离。
- **保留 Liger**，但只当优化项，别指望它解决结构问题；同时留一手"训练侧与 vLLM 的 logprob 不一致"的护栏（用训练侧重算的 logprob 算重要性比，vLLM 侧只用于过滤）。

---

## 9. Bottom line

- **vLLM**：✅ 该用的，而且 2026 版已经远超"推理引擎"——它是 async RL 的**数据面 + rollout 生命周期**，单块成熟度最高。
- **Liger Kernel**：🟡 有用，且已 RL-native（TP 感知 fused CE + 熵输出，fallback 抄自 verl 的 PPO 公式），但属于"更便宜"，不属于"能跑"。
- **PyTorch Lightning**：❌ **它才是瓶颈**。同步锁步、epoch 中心、单优化循环、自持 process group、ckpt 语义不匹配——async RL 每一条都踩它。它最多当 learner 内部的一个薄壳，而一旦你用 `automatic_optimization=False`，这层壳也就没意义了。

所以准确的表述是：**这三位顶不起 async RL 的骨架，但缺的那根梁不在 vLLM，在 PL。** 把 PL 换成事件驱动/actor 化的控制面（buffer + 滞后策略 + 资源编排 + 容错），剩下的部分 vLLM 已经替你做好了——而且这个判断在 2025 年还基本不成立。

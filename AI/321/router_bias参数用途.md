# router_bias参数用途

### 🧑‍💻 User

moe模型中，stacked_blocks.stacked.mlp.router_bias参数有什么用？
https://github.com/marin-community/marin

---

### 🤖 Assistant

## 结论

`stacked_blocks.stacked.mlp.router_bias` 是 Marin 的 **GrugMoE（QB-routed MoE）架构里 MoE 路由器的偏置向量**，形状为 `[num_experts]`（每个专家一个标量偏置，67B-A2B Snowball 配置里是 256）。它的作用只有一个核心目的：**在 top-k 选专家之前给 router logits 加一个可动态调节的偏置，用来做负载均衡（Quantile Balancing，简称 QB），而不是靠梯度训练出来的。**

代码位置：

- 训练版：`experiments/grug/moe/model.py` 的 `MoEMLP` 类（`router` + `router_bias`）
- 推理/导出快照：`lib/levanter/src/levanter/models/snowball.py` 的 `SnowballMoEMLP` 类

---

## 它在前向里具体做什么

`MoEMLP.__call__` 中的关键逻辑（`experiments/grug/moe/model.py`）：

```python
# 1. 无偏 logits（router 是 (hidden_dim, num_experts) 的线性权重）
router_logits = x_flat @ router

# 2. 加偏置后用于“选专家”，但偏置被 stop_gradient 隔离，不参与梯度
biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)

# 3. softmax 概率用【无偏】logits（只用于统计，不用于选专家）
router_probs = softmax(router_logits)

# 4. 在【有偏】logits 上取 top-(K+1)；第 (K+1) 个是 QB 阈值 alpha
_topk_logits, selected_experts = top_k(biased_logits, K + 1)
qb_alpha = _topk_logits[:, -1:]           # 阈值
selected_experts = selected_experts[:, :-1]  # 真正选中的 K 个专家

# 5. 组合权重用【无偏】logits 的 sigmoid，再重归一化到和 = 2.5
unbiased_topk = take(router_logits, selected_experts)
combine_weights = sigmoid(unbiased_topk) * (2.5 / sum(...))
```

所以关键点是：**`router_bias` 只影响“选哪些专家”，不影响“选中的专家按多大权重组合”**（权重和 softmax 概率都基于无偏 logits）。

---

## 为什么需要它：Quantile Balancing（QB）

Marin 的 MoE **没有 load-balancing auxiliary loss**（代码里只有 router z-loss：`router_z_loss_coef`，见 `next_token_loss`）。负载均衡完全靠 QB 这套在线调节 `router_bias` 的机制：

训练时每个 step 会算一个 `qb_beta`：

```python
# s_minus_alpha = logits - alpha，alpha 是 top-(K+1) 的第 (K+1) 个 logits（决策边际）
s_minus_alpha = router_logits - qb_alpha

# 每个设备本地：count = max(1, local_tokens * K / num_experts)
# beta = s_minus_alpha 转置后取 top-count 的第 count 大值，再跨设备 pmean
qb_beta = pmean(top_k(s_minus_alpha.T, count)[:, -1])
```

这个 `qb_beta` 会被累积进 `router_bias`（即 `router_bias += beta`），思路是：**如果某个专家的 logits 相对“被选中门槛 alpha”持续偏低（即它很少被选中），就持续抬高它的偏置，直到所有专家负载趋于均匀。** 这正是 README 里提到的那篇博客：

> [Mixture of Experts Quantile Balancing: Validated at 32B-A5B (1e22 FLOPs) Scale](https://openathena.ai/blog/quantile-balancing/)

因为 `router_bias` 用 `jax.lax.stop_gradient` 包裹，它**不参与反向传播**，是一个纯“调度/均衡”状态量，而非训练出来的参数（初始化是 `jnp.zeros((num_experts,))`）。

---

## 关于路径写法 `stacked_blocks.stacked.mlp.router_bias`

这个路径是序列化配置/checkpoint 里的嵌套地址：

- `stacked_blocks` → 模型里堆叠的 block 序列（Grug 的 `Transformer.blocks`，一个 `Block` 的 tuple）
- `stacked` → 每一个被堆叠的 block（`Block`）
- `mlp` → block 里的 MoE MLP（`MoEMLP` / `SnowballMoEMLP`）
- `router_bias` → 上面的偏置向量

在导出到 HuggingFace 格式时，它对应的 checkpoint key 是：

```
model.layers.{i}.mlp.router.bias
```

（见 `grugmoe_inference_state_dict` 里的 `f"{layer_prefix}.mlp.router.bias": block.mlp.router_bias`）

---

## 推理/导出时的处理

Snowball（67B-A2B 的生产快照）的 `SnowballMoEMLP` 注释说得很清楚：

> "The loaded `router_bias` already has the QB betas baked in (from the export), so we add it directly and never re-apply the update."

也就是说：**训练期间 `router_bias` 不断被 QB 更新；导出 checkpoint 时把最终累积好的偏置值固化进去；推理时直接 `logits + router_bias` 即可，不再做任何 QB 更新。**

顺带一提，README 里也印证了这点：

> "MoE runs also load `pending_qb_betas`"

即断点续训时不仅要恢复模型权重，还要恢复 `router_bias` / 未应用的 QB beta 状态，否则均衡状态会丢失。

---

**一句话总结**：`router_bias` 是 GrugMoE 每个专家一个的标量偏置，加在 router logits 上用于改变专家被选中的概率，由 Quantile Balancing 负载均衡算法在线维护（不参与梯度），目的是替代传统的 load-balancing aux loss、让各专家负载均衡。
# FSDP2架构全详解

## 🧑‍💻 User

fsdp2只支持zero2和zero3吗（将 reshard_after_forward 设为 True/False（或在根模块上做特定配置））

---
## 😀 默认助手 (7)

I'll verify the current FSDP2 API semantics against the official docs before answering.

Docs redirect to versioned URLs — following them.

Got the `main` source. Now let me check the same file at the stable release tag to see what's actually in the shipped version vs. nightly.

Confirmed the shipped v2.14.0 API. Two more checks: whether a no-shard/replicate mode exists in FSDP2 exports, and the ZeRO-1 (optimizer-state-only) option.

## 结论：基本对，但有三处需要修正

**(1) FSDP2 里已经没有"ZeRO stage"这个全局开关了** —— `reshard_after_forward` 是**每个 `fully_shard()` 调用（每个 param group）各自设置**的，不是整个模型的档位。

**(2) 参数/梯度/优化器状态的分片在 FSDP2 里是内建的**，所以 `True`/`False` 调的不是"要不要分片"，而只是*"forward 之后要不要把参数放回分片态、backward 要不要再 all-gather 一次"*。这决定了**下限就是 ZeRO-2 语义，没有 ZeRO-1**（也没有 ZeRO-0，那是 DDP 的活）。

**(3) 不止两档**：`reshard_after_forward` 还能取 `int`，加上 `reshard_after_backward()`、CPU offload、2D mesh(HSDP) 等，实际是一个连续谱。

最关键的一点：**默认配置本身就是"混合档"**——每层是 ZeRO-3 语义、root 模块是 ZeRO-2 语义，不存在"必须全局选一个 stage"。

---

## `reshard_after_forward` 的四种取值

| 取值 | forward 后参数状态 | backward 是否需要 all-gather | 显存/通信类比 |
|---|---|---|---|
| `True` | 释放，模块上注册的是分片 DTensor | 需要，全 mesh | **ZeRO-3** |
| `False` | 保留全量参数 | 不需要 | **ZeRO-2 语义** |
| `None`（本版默认） | 非 root → `True`；root → `False` | 混合 | **混合档**（按模块精度分层） |
| `int` = $k$ | reshard 到大小为 $k$ 的小 mesh（replicate 在外、shard 在内） | 需要，但只在 $k$ 卡内 | **ZeRO-3 与 ZeRO-2 之间** |

`int` 的约束（源码 `_get_post_forward_mesh_info`）：必须是 shard mesh size 的**非平凡因子**（排除 1 和自身）；传 `1` 会 warn 并等价于 `False`，传等于 mesh size 会等价于 `True`。典型用法是传"单机卡数"，让 backward 的 all-gather 只走 NVLink 范围。

<details>
<summary><b>各档的实际显存/通信语义（含与 DeepSpeed ZeRO-2 的细微差别）</b></summary>

- **`True`（ZeRO-3）**：参数静止态 $1/N$，前向 all-gather → 用完释放 → 反向再 all-gather → 算完再释放并 reduce-scatter 梯度。梯度、优化器状态始终 $1/N$。
- **`False`（"ZeRO-2"）**：forward 前仍然要 all-gather（因为静止态参数是分片的），但**保留到 backward 结束**，省掉反向那次 all-gather。峰值显存接近 ZeRO-2，但严格说与 DeepSpeed ZeRO-2 有两点不同：① 静止态（iteration 之间、backward 之后）参数是 $1/N$ 而非全量，更省；② ZeRO-2 参数本来就常驻全量，**不需要前向 all-gather**，而 FSDP2 `False` 仍然要做一次。所以它更像"ZeRO-2 ⇄ ZeRO-3 之间"。
- **`None`（现版本默认）**：非 root 模块 `True`、root 模块 `False`。原因是"root 的参数在 backward 一开始立刻就要用"，把它也 reshard 会立刻又 all-gather，纯亏一次通信。所以**给 root 传 `False` 是官方推荐的最佳实践**，不是"降级到 ZeRO-2"。
- **`False` 的副作用**：参数在前向/反向之间是全量的，**这段时间内不能改写参数**（backward 的 all-gather 不会与写入排序）。要改就先手动 `module.reshard()`。

</details>

---

## 标准写法

```python
from torch.distributed.fsdp import fully_shard

# bottom-up：先子模块，再 root
for blk in model.layers:
    fully_shard(blk, mesh=mesh)          # 非 root，默认 -> True (ZeRO-3)
fully_shard(model, mesh=mesh)            # root，默认 -> False (ZeRO-2 语义)

# ── 想强制"全局 ZeRO-3"（FSDP1 的 FULL_SHARD 等价）──
for blk in model.layers:
    fully_shard(blk, mesh=mesh, reshard_after_forward=True)
fully_shard(model, mesh=mesh, reshard_after_forward=True)

# ── 想强制"全局 ZeRO-2 风格"（FSDP1 的 SHARD_GRAD_OP 等价）──
for blk in model.layers:
    fully_shard(blk, mesh=mesh, reshard_after_forward=False)
fully_shard(model, mesh=mesh, reshard_after_forward=False)   # 显存换通信，峰值明显上升

# ── 中间档：forward 后先 reshard 到 8 卡，backward 只在 8 卡内 all-gather ──
fully_shard(blk, mesh=mesh, reshard_after_forward=8)
```

<details>
<summary><b>版本差异与 FSDP1 迁移对照</b></summary>

- 早期版本（如 2.4/2.5）里该参数是**纯 `bool` 且默认 `True`**，root 模块需要你手动传 `False`；而核对到的 2.14 版本默认值是 `None`，root 的特殊化变成了自动行为。如果你不确定自己的版本，`inspect.signature(fully_shard)` 看一下默认值即可。行为上"默认值是什么"只影响 root 那一层，显式传参永远安全。
- FSDP1 → FSDP2 对照：

| FSDP1 `ShardingStrategy` | FSDP2 写法 |
|---|---|
| `FULL_SHARD` | 全部 `reshard_after_forward=True` |
| `SHARD_GRAD_OP` | 全部 `reshard_after_forward=False` |
| `HYBRID_SHARD` | 2D mesh（shard 内维 / replicate 外维）+ `True` |
| `_HYBRID_SHARD_ZERO2` | 2D mesh + `False` |
| `NO_SHARD` | 不用 FSDP2；用 DDP（或新版无分片路径，见下） |

</details>

---

## 为什么"只有 ZeRO-2 / ZeRO-3"这个说法成立

因为 FSDP2 的**梯度一定是 reduce-scatter、优化器状态一定是分片的**（参数本身是 DTensor，优化器直接在分片参数上更新）。文档里写得很直接：

> FSDP shards module parameters, gradients, and optimizer states across data parallel workers... After gradient computation, FSDP frees the unsharded parameters and **reduce-scatters the unsharded gradients**.

所以：

- **ZeRO-1（只切优化器状态、梯度 all-reduce 全量）在 FSDP2 里做不出来** —— 你不能让梯度变成"全量、非分片"。真要 ZeRO-1，用 DDP + `torch.distributed.optim.ZeroRedundancyOptimizer`（该 API 在 2.14 仍存在）。
- **ZeRO-0（不分片）**也不在 FSDP2 语义里，用 DDP。2.14 源码里有一条**较新的无分片路径**：`fully_shard(..., dp_mesh_dims=DataParallelMeshDims(shard=None, replicate="dp"))` 会走 `DDPMeshInfo`（源码注释直接写 `# DDP`），配 SPMD/DTensor 模型用；但它只在内部 `replicate_with_fsdp` 相关代码里被引用，**未出现在 `torch.distributed.fsdp` 的公开导出**里，属于实验性 API，不建议生产依赖。
- `set_requires_gradient_sync(False)` **不是** ZeRO-1：它是 `no_sync` 的等价物，梯度完全不 reduce（连 reduce-scatter 都跳过），用于梯度累积省通信。

---

## 除 `reshard_after_forward` 之外的"调档"旋钮

| API / 参数 | 作用 | 类比 |
|---|---|---|
| `reshard_after_forward=int` | reshard 到更小的 shard mesh | ZeRO-2 ↔ ZeRO-3 之间 |
| `set_reshard_after_backward(False)` | backward 后**不** reshard，跨 iteration 保留全量参数 | 比 ZeRO-2 更激进（参数常驻全量），省掉下次前向的 all-gather，配合梯度累积用 |
| `set_reshard_after_forward(bool, recurse=True)` | 运行时切换；eval 设 `False`、train 设 `True` | — |
| `offload_policy=CPUOffloadPolicy()` | 参数/梯度/优化器状态 offload 到 CPU | ZeRO-Offload 风味 |
| 2D `DeviceMesh` | 节点内 replicate、节点间 shard | HSDP |
| `set_requires_gradient_sync(False)` | 跳过 reduce-scatter | `no_sync`（**不是** ZeRO 档位） |

一个容易踩的坑：**不要只在最顶层 root 上调用 `fully_shard`**。文档明确说应该 bottom-up 逐层调用——只分一个 group 时，参数会在整个 forward 期间全量驻留，等于把 ZeRO-3 退化成了比 ZeRO-2 还差的形态（因为既不释放又只有一次 all-gather 的机会，没有 overlap）。

---

## 一句话总结

FSDP2 不是"在 ZeRO-2 和 ZeRO-3 之间二选一"，而是**没有 stage 概念、只有 per-group 的 reshard 策略**：内建分片让它天然处于 ZeRO-2～ZeRO-3 区间，`True`/`False`/`int`/`None` 是这条区间上的点（外加 root 默认 `False` 的混合配置）；**ZeRO-1 及以下需要换到 DDP + `ZeroRedundancyOptimizer`，ZeRO-0 需要 DDP**。

<details>
<summary><b>本回答的核对范围</b></summary>

依据 PyTorch `v2.14.0` tag 的 `torch/distributed/fsdp/_fully_shard/{_fully_shard.py,_fsdp_init.py,__init__.py}`、`torch/distributed/fsdp/__init__.py`（导出列表）、`torch/distributed/optim/zero_redundancy_optimizer.py`，并与 `main` 分支的 `_fully_shard.py` 做了比对（关键点一致）。v2.14.0 与 main 的主要差异在 `_apply()` 实现和 `finalize_backward`/`reset_iter_state` 等更细的 API，不影响上文的档位结论。

</details>

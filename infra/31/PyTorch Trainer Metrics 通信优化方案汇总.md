# PyTorch Trainer Metrics 通信优化方案汇总

## 🧑‍💻 User

我发现我的torch trainer里边metrics拖性能主要是通信频繁了（要跨 rank reduce求sum max mean啥的
虽然1个step总共几十mb但是有几十次
有什么好的解决办法吗

---
## 😀 默认助手 (7)

## 先判断瓶颈性质

“几十次 × 每次 ~1MB” 属于最不划算的情况：**比带宽延迟更受 launch/sync 开销主导**。单个 NCCL collective 的固定开销约 10–30 µs，40 次就是 0.4–1.2 ms；而 1MB 在 NVLink 上纯传输其实只要几 µs。所以你的收益主要来自**减少 collective 的数量**，减少总字节数带来的收益相对次要。

先花 5 分钟确认数字，别凭感觉优化：

```python
import torch.distributed as dist

_stats = {"n": 0, "bytes": 0}
_orig = dist.all_reduce

def _counted(t, *a, **k):
    _stats["n"] += 1
    _stats["bytes"] += t.numel() * t.element_size()
    return _orig(t, *a, **k)

dist.all_reduce = _counted   # 只抓 python 层调用，DDP 内部 bucket 抓不到

# 每 step 打印一次 _stats，然后清零
```

同时用 profiler 看 `nccl:all_reduce` 的累计 CUDA 时间和**step 之间的空泡（gap）**。如果 metrics 之后 GPU 出现 idle gap，那问题更严重：`.item()` 触发了全流水线同步，并非通信本身慢；这会让整个训练失去重叠。

---

## 按收益排序的解法

### ① 合并成 1–3 次 collective，这是收益最大的改动

40 次 `all_reduce(1元素)` → **1 次 `all_reduce(40元素)`**，延迟几乎变成原来的 1/40。做法是把所有要 reduce 的标量拼进一个预分配 buffer。

不同 reduce op 不能混在一起，所以按 op 分桶，最多 3 次：

- `sum` 桶（绝大多数 metric：loss、correct count、token count、TP/FP/FN……）
- `max` / `min` 桶（grad_norm、max logit 等，通常只有寥寥几个）

```python
import torch
import torch.distributed as dist

class MetricAccumulator:
    """本地 GPU 累加，只在 flush 时做 1~3 次 collective。"""
    OPS = ("sum", "amax", "amin")

    def __init__(self, specs: dict[str, str], device, pg=None):
        # specs: {"loss": "sum", "grad_norm": "amax", ...}
        self.device, self.pg = device, pg
        self.names = list(specs)
        self.ops = [specs[n] for n in self.names]
        self.pos = {n: i for i, n in enumerate(self.names)}
        n = len(self.names)

        # float64 累加，避免几万 step 后 fp32 掉精度
        self.acc = torch.zeros(n, dtype=torch.float64, device=device)
        # 分组索引
        self.groups = {
            op: torch.tensor([i for i, o in enumerate(self.ops) if o == op],
                             dtype=torch.long, device=device)
            for op in self.OPS
        }
        self.reset()

    def reset(self):
        self.acc.zero_()
        for op, neg in (("amax", float("-inf")), ("amin", float("inf"))):
            idx = self.groups[op]
            if idx.numel():
                self.acc.index_fill_(0, idx, neg)

    @torch.no_grad()
    def update(self, **values):          # 传 tensor，别传 float
        idx, vals = [], []
        for name, v in values.items():
            idx.append(self.pos[name])
            vals.append(v.detach().reshape(1).to(torch.float64))
        idx = torch.tensor(idx, device=self.device, dtype=torch.long)
        vals = torch.cat(vals)
        for op in self.OPS:
            sel = torch.isin(idx, self.groups[op])
            if sel.any():
                # include_self=True 等价于原地累加/取极值，且不产生 D2H 同步
                self.acc.scatter_reduce_(0, idx[sel], vals[sel],
                                         reduce=op, include_self=True)

    @torch.no_grad()
    def flush(self, dst=0):
        """1~3 次 collective。dst=0 表示只有 rank0 要日志 → 用 reduce 而非 all_reduce。"""
        out, rank = {}, dist.get_rank()
        for op, idx in self.groups.items():
            if idx.numel() == 0:
                continue
            part = self.acc.index_select(0, idx).contiguous()
            if self.pg is None:
                dist.reduce(part, dst=dst, op=getattr(dist.ReduceOp, op.upper()))
            else:
                dist.reduce(part, dst=dst, op=getattr(dist.ReduceOp, op.upper()), group=self.pg)
            if rank == dst:
                names = [self.names[i] for i in idx.tolist()]
                out.update(zip(names, part.tolist()))   # 一次 .tolist()，一次同步
        self.reset()
        return out
```

如果你需要全 rank 都拿到值（比如 rank 0 之外的 rank 也要做自适应控制），把 `reduce` 换成 `all_reduce`。

> 关键点：`update()` 里**一次 `.item()` 都没有**，所以整个累加过程完全异步，`loss.detach()` 直接塞进 buffer 就行，不会打断 GPU 流水线。

### ② 降低频率：按 logging 周期 reduce，而不是每 step reduce

这条通常能单独带来 **10–100 倍的提升**，而且几乎零成本：

```python
# 每 step：只 update（纯 GPU 运算，无通信）
acc.update(loss=loss, correct=correct, n_tokens=n_tokens)

# 每 log_every 步：才 flush 一次
if step % log_every == 0:
    m = acc.flush()
    if dist.get_rank() == 0:
        logger.info({k: v / (world_size * n_tokens_local) for k, v in m.items()})
```

注意均值别在各 rank 上求完再平均：$\text{mean} = \dfrac{\sum_r s_r}{\sum_r n_r}$。因此要 reduce **sum 和 count 两个量**，而不是各 rank 的 mean 取平均。

### ③ 减少 payload：只传充分统计量，别传原始数据

“1 step 几十 MB” 说明你在 reduce 的很可能不是标量。典型的大 payload 元凶：

| 反模式 | 改成 |
|---|---|
| `all_gather` 各 rank 的预测结果再算 accuracy/F1 | 本地算 `correct/total`、`TP/FP/FN` 计数 |
| 每 rank 传完整 confusion matrix（1000 类 = 4MB × N） | 本地累加 partial CM，**只在 eval 结束**时 reduce 一次 |
| 传 per-sample loss 列表 | 只传 sum + count |
| 传 per-token logits / embeddings | 本地降维成标量统计量 |
| `all_gather_object` 传 python 对象 | 张量化 + 拼接 → 一次 collective |
| fp32 传大张量 | bf16 传（仅限极值类；求和仍用 fp32/fp64） |

`all_gather_object` 尤其麻烦：它走 pickle + python 对象，每次都是同步且延迟地极高，通常不是性能瓶颈，而是纯粹的浪费。

### ④ 只需 rank 0 日志 → 用 `reduce(dst=0)` 而不是 `all_reduce`

`all_reduce` 的 ring 流量是 $\frac{2(N-1)}{N} \times \text{payload}$，`reduce` 是 $\frac{N-1}{N} \times \text{payload}$。**带宽直接省一半**，延迟也更低（省掉 broadcast 反向那一跳）。大 payload 场景收益明显。

### ⑤ 异步 + 跨 step 流水线，彻底隐藏延迟

最理想是 **step $t$ 发起，step $t+K$ 才消费**：这样通信被 $K$ 个 step 的计算完全覆盖。

```python
self._pending = deque()          # (Work, buffer)

# step t 结束时
work = dist.all_reduce(buf, op=dist.ReduceOp.SUM, async_op=True)
self._pending.append((work, buf))   # buf 必须是独立 buffer，不能复用！

# 到达 log 点时才 wait（K 个 step 之后）
if len(self._pending) >= K:
    work, b = self._pending.popleft()
    work.wait()
    values = b.tolist()
```

两个坑：

1. **buffer 双/多缓冲**：`async_op=True` 时 buffer 在 wait 之前不能改，必须用 ring buffer，每个 pending 条目独立。
2. `Work.wait()` 对 NCCL 只是插入流同步，不会 CPU 阻塞，但要读值前得保证 `tolist()` 触发的那次 D2H 也排在正确位置（PyTorch 会处理，自己用多 stream 才需手动 `record_event`）。

### ⑥ 清理隐式同步点

这些经常是真正的性能瓶颈，而且 profiler 里看起来像“通信慢”，实际是**流水线被打断**：

```python
if loss > 0.5:                      # tensor.__bool__ → 同步！
    ...
print(f"loss={loss}")               # 每 step 打印 → 同步
logger.info(f"{metric:.4f}")        # %-格式化 float(tensor) → 同步
x = torch.cat(...).cpu()            # 内层循环里 D2H
dist.barrier()                      # 每 step 一次 = 自杀
sorted(tensors) / max(tensor_list)  # python 比较触发 __bool__
t.cpu().numpy()  # 反复来回
```

改成：**整个 logging 窗口内只在 flush 那一次读值，其余全程留在 GPU。**

---

<details>
<summary><b>进阶：独立 process group / stream，别拖累 DDP 的梯度 bucket</b></summary>

metric 的 collective 和 DDP 的梯度 all_reduce 走同一个 comm 会**互相抢占延迟**。梯度 reduce 是关键路径，被 metric 插队会直接拉长 step time。

```python
# 给 metrics 单开一个 process group（独立的 NCCL comm）
metric_pg = dist.new_group(backend="nccl")

# 或者放在单独 stream 上，让它和计算重叠而不是串行
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    s.wait_stream(torch.cuda.current_stream())
    acc.flush()          # 在 s 上发 collective
torch.cuda.current_stream().wait_stream(s)   # 只在真要读值时同步
```

现代 NCCL（2.26+）有 implicit launch ordering，独立 comm 的安全性没问题。

</details>

<details>
<summary><b>进阶：`all_reduce_coalesced` / `ncclGroupStart` 的适用边界</b></summary>

`dist.all_reduce_coalesced([t1, t2, ...])` 走 `ncclGroupStart/End`，把 N 次 launch 合并成一次 group launch，**省的是 host 侧 launch 开销**（每次 5–20 µs → 分摊到接近 0），但**底层仍是 N 个独立 kernel，带宽不省、kernel 数不省**。

所以：

- 纯延迟瓶颈（很多 1 元素 tensor）→ `_coalesced` 有效，但手动拼成一个大 tensor 更好（kernel 数也降了）
- 想省带宽 → 必须手动拼 buffer

手动拼接一定不亏，因为它同时解决 launch 数和 kernel 数。`_coalesced` 只在“各 tensor 大小不一、拼接有拷贝成本”时才值得。

</details>

<details>
<summary><b>进阶：torch.compile / functional collectives 自动 bucket</b></summary>

`torch.distributed._functional_collectives` 配合 `torch.compile` 可以减少 python 层 dispatch 开销（40 次调用 × ~10 µs 的 python + dispatcher 成本 = 0.4ms，如果 CPU-bound 这很可观）。但对你“几十 MB”的场景，payload 优化收益更大，这是次要项。

</details>

<details>
<summary><b>进阶：确实需要每 step 都 reduce（自适应控制类 metric）</b></summary>

如果 metric 用于动态控制（clamp、skip、MoE aux loss 平衡、动态 loss scale），就不能降频。那就：

1. **只 reduce 必需的极少数标量**，其余全部延后
2. 拼成一个极小 tensor（几个元素），一次 `all_reduce`
3. 用 `async_op=True` 让它的延迟和下一步 forward 的首几个 kernel 重叠
4. 确认它没在 DDP 的 bucket 之后排同一流上（用独立 pg）

即使如此，一两次 collective 的延迟也应该 < 0.05 ms，不会是瓶颈。

</details>

---

## 一个判断标准

做完 ①+② 之后，每个 step 的通信应该是 **每 $K$ 步 1–3 次 × 几十 KB**。对于 8 卡 NVLink，摊到每 step 是**微秒级**，就可以认为 metrics 不再是瓶颈了。

如果优化后仍然慢，那大概率是：
- payload 没真正降下来（还在 reduce 大张量）→ 回到 ③
- 还有藏起来的 `.item()` / `print` / `barrier` → 回到 ⑥
- 单步 step time 本身就短（<20ms），此时任何同步开销都占比大 → 优先做 ⑤ 的跨 step 流水线

要不要把你 metrics 里具体在 reduce 哪些量贴出来（或者贴一段 trainer 代码）？我可以直接指出哪几个是主要消耗，以及哪些能改成充分统计量。

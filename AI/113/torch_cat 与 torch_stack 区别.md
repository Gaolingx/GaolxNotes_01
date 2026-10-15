# torch.cat 与 torch.stack 区别

## 🧑‍💻 User

torch.stack和torch.cat有什么区别

---
## 😀 默认助手 (7)

## 核心区别一句话

- **`torch.cat`（拼接）**：在**已有的维度**上连接，**维度数不变**。
- **`torch.stack`（堆叠）**：在**新建的维度**上堆叠，**维度数 +1**。

两者都要求除连接/堆叠维度外的其余维度形状一致（`cat` 甚至要求该维度可以不同，`stack` 要求**所有**形状完全相同）。

---

## 形状变化对照

设两个张量形状都是 $(3, 4)$：

| 操作 | 结果形状 | 说明 |
|---|---|---|
| `torch.cat([x,y], dim=0)` | $(6, 4)$ | 沿第 0 维延长，维度数不变 |
| `torch.cat([x,y], dim=1)` | $(3, 8)$ | 沿第 1 维延长，维度数不变 |
| `torch.stack([x,y], dim=0)` | $(2, 3, 4)$ | 新建第 0 维，维度数 +1 |
| `torch.stack([x,y], dim=1)` | $(3, 2, 4)$ | 新建第 1 维，维度数 +1 |
| `torch.stack([x,y], dim=-1)` | $(3, 4, 2)$ | 新建最后一维 |

对 `stack`，`dim` 取值范围是 $[-(n+1),\ n]$（$n$ 为输入维度数）；对 `cat`，范围是 $[-n,\ n-1]$。

---

## 代码示例

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])   # shape (2, 2)
y = torch.tensor([[5, 6], [7, 8]])   # shape (2, 2)

>>> torch.cat([x, y], dim=0)
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])          # shape (4, 2)

>>> torch.cat([x, y], dim=1)
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])    # shape (2, 4)

>>> torch.stack([x, y], dim=0).shape
torch.Size([2, 2, 2])

>>> torch.stack([x, y], dim=0)
tensor([[[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]]])
```

---

## 什么时候用哪个

- 想把一批**同形状**的张量组成一个 batch → 用 **`stack`**（比如 `[img1, img2, ...]` → `(B, C, H, W)`）。
- 想沿某个维度**接长**序列/特征 → 用 **`cat`**（比如把两个卷积特征图在通道维拼起来 `(N, C1, H, W)` + `(N, C2, H, W)` → `(N, C1+C2, H, W)`）。

<details>
<summary>进阶：两者的数学关系与常见坑</summary>

**1. `stack` 可以用 `cat` 表达**

$$\text{stack}(xs, \text{dim}) = \text{cat}\big([x.\text{unsqueeze}(\text{dim}) \text{ for } x \in xs],\ \text{dim}\big)$$

例如 `torch.stack([x, y], 0)` 等价于 `torch.cat([x.unsqueeze(0), y.unsqueeze(0)], 0)`。

**2. 报错差异**

```python
a = torch.randn(3, 4)
b = torch.randn(3, 5)

torch.cat([a, b], dim=1)   # ✅ (3, 9) —— 非连接维相同即可
torch.stack([a, b], dim=0) # ❌ RuntimeError: stack expects each tensor to be equal size
```

**3. `cat` 的坑**：如果某个张量是空张量（如 $(0, 4)$），`cat` 仍可用；但 `stack` 里所有张量必须严格同形，且都不为空（`dim` 插入处不能为 0 尺寸之外的问题另说）。

**4. 性能**：二者都会分配新内存（不是 view），`cat` 在需要拼接大量小张量时建议先 `torch.stack` 再一个 `cat`，减少 kernel 调用次数。

**5. 与之混淆的 `torch.concat`**：`torch.concat` 只是 `torch.cat` 的别名，行为完全一致。

</details>

**记忆口诀**：`cat` = 延展（变长），`stack` = 垒高（加维）。

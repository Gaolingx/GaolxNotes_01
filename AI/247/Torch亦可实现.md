# Torch亦可实现

## 🧑‍💻 User

话说有哪些事情是torch很难做到只能jax才能跑起来的

---
## 😀 默认助手 (7)

先说结论，避免误导：**严格意义上"只有 JAX 能跑、PyTorch 完全跑不起来"的场景现在已经很少了**。原因是 PyTorch 这几年用 `torch.func`(functorch)、`torch.compile`、`torch.export`、`DTensor` 把很多老差距补上了。

所以更准确的说法是两类：

- **JAX 天然、优雅、几乎零摩擦**，而 PyTorch 要么做不到、要么要靠一堆 workaround、要么性能/内存上很吃亏；
- **真正硬性做不到**的少数场景（主要是硬件后端和某些 SPMD 语义）。

下面按"差距从硬到软"排列。

---

## 1. 任意组合的函数变换（JAX 的看家本领）

JAX 的 `jit / grad / vmap / pmap / checkpoint` 都是**作用于纯函数的正交变换**，可以无限嵌套：

```python
per_sample_grads = jax.vmap(jax.grad(loss))          # 每样本梯度
grad_of_grad     = jax.grad(lambda x: jnp.sum(jax.grad(f)(x)))  # AD 套 AD
outer = jax.jit(jax.vmap(jax.grad(jax.jit(train_step))))         # 随便套
```

`torch.func` 确实提供了 `vmap / grad / jacrev / hessian`，也支持一定程度的组合，但：

- 对**含控制流、含随机、含原地操作**的函数做 `vmap`，PyTorch 覆盖度和稳定性明显弱于 JAX；
- `vmap` 里再套 `grad` 再套 `vmap` 这种深层嵌套，PyTorch 容易报错或踩坑；
- JAX 的 `jax.lax.scan` 让"在变换内部写循环"变得自然，PyTorch 的 vmap 对 `for` 循环/数据依赖分支处理更脆弱。

## 2. 任意阶 & 任意模式自动微分

- **高阶导数**：MAML、WGAN-GP 的梯度惩罚、hypergradient、LOLA、对优化器步求导——JAX 里就是多套一层 `grad`。PyTorch 要用 `create_graph=True` 或 `torch.func`，在"AD 穿过 AD 穿过控制流"时经常有坑。
- **前向模式 AD**：`jax.jvp` 是一等公民，可和反向模式任意混合。PyTorch 的前向模式（`torch.autograd.forward_ad`、`torch.func.jvp`）存在但支持面窄很多。
- **复数 AD**：JAX 对复值函数的微分（holomorphic / non-holomorphic）处理得很完整，物理/信号/量子类计算常用。

<details>
<summary>为什么这里 JAX 结构性更优（点开看）</summary>

因为 JAX 把 AD、vmap 都实现成了对**纯函数**的变换，语义上就是"函数的函数"，组合不产生额外约束。PyTorch 的核心是带状态、带别名的 Tensor + tape，functorch 是在其上叠了一层，遇到状态/别名/动态形状时就要打补丁。
</details>

## 3. 贯穿长循环/长时程的可微 + 重算（remat）

`jax.lax.scan / while_loop / cond` + `jax.checkpoint`（=remat）是**原生配套**的，使"对整个时间积分/迭代过程求导"变得可控内存：

- 微分方程求解器（**Diffrax**）、可微渲染、可微物理仿真、可微 CFD（**JAX-Fluids**）、分子动力学（**jax-md**）；
- 对比 PyTorch：`torchdiffeq` 能做，但反向穿过几百上千步时内存/实现难度高，且和 `torch.compile`、vmap 的组合没那么顺。

## 4. "不只是神经网络"的纯函数式数值计算

JAX 本质是 **NumPy + autodiff + XLA**，所以可以对**任意数值算法整体**做 `jit/vmap/pmap`，而不只是 NN 层。典型：

- **mctx**：整个蒙特卡洛树搜索（MCTS）用 `lax.while_loop` 写成纯函数，可 JIT + vmap 批量跑；
- Optax、Equinox、Flax 等把优化器/训练循环也做成可变换的纯函数。

在 PyTorch 里把这些"算法整体"包进一次编译/批量变换，通常要自己重写、很难复用。

## 5. 硬件后端：TPU / XLA / OpenXLA（较硬性差距）

- **JAX 是 Google TPU 的一等公民**，PyTorch 走 `torch_xla`，覆盖率、性能调优、新特性都落后。
- 想移植到**非主流加速器**时，走 OpenXLA 后端比走 PyTorch 的自定义后端生态通常更现实。
- 同一个程序从 CPU/GPU/TPU 切换，JAX 基本是改个设备，语义一致。

## 6. SPMD / 自动分片

`jax.experimental.shard_map`、`pjit`(现 `jax.jit` 的 sharding)、GSPMD + `jax.Array` 的 **sharding 抽象**，让"数组怎么切"成为类型的一部分，编译器自动做通信推导。

PyTorch 有 `DTensor`、FSDP、张量并行，但整体模型不如 JAX 统一、自动，尤其是"任意 sharding 声明的自动 partition"。

## 7. 可复现性 / 显式 PRNG / 确定性

- `jax.random.PRNGKey` 是**显式值**，可当参数传、可 fork、天然并行可复现；PyTorch 用全局 RNG 状态，分布式/并行下复现要额外费力。
- XLA 下同一程序跨设备数值一致性更好。

## 8. 编译期元编程

形状多态（shape polymorphism）、`jax.export` 到多后端、注册自定义 primitive、`custom_jvp / custom_vjp`、`pure_callback` 等，做"编译期可组合的抽象"比 PyTorch 的编译器栈更顺手。

---

## 反过来：别被"只能 JAX"带偏

<details>
<summary>PyTorch 反而更好 or 差距已抹平的场景（点开）</summary>

- **动态形状、动态控制流、eager 调试**：PyTorch 体验更好，JAX 的"函数得是纯的/静态形状"经常反噬。
- **生态与工程化**：HuggingFace、各种 CV/NLP 库、部署（TorchServe、ONNX、TensorRT、移动端）远强于 JAX。
- **`torch.func` / `torch.compile` / `torch.export`**：已经把 vmap、高阶导、图编译这些原本 JAX 独有的东西补了很多。
- 于是"只有 JAX 能跑"更多变成"JAX 天然、少踩坑、写起来更优雅，PyTorch 能做但更痛"。

</details>

---

### 一句话总结

真正硬性偏向 JAX 的是 **TPU/XLA 后端**和**某些 SPMD/分片语义**；而对 **任意嵌套的函数变换、任意阶/多模式 AD、把纯数值算法（MCTS、ODE、物理仿真）整体 JIT+vmap+微分** 这类需求，JAX 是"设计上就该这么用"，PyTorch 通常能做但代价明显更大。

如果你有具体场景（比如"我想对某个训练循环求二阶导"或"要上 TPU"），可以告诉我，我帮你判断到底选哪边、或者 PyTorch 具体要怎么写才勉强行。

---
## 🧑‍💻 User

那jvp/vjp（sdpa fwd/bwd）、gan的grad reg这种torch都不好弄吧

---
## 😀 默认助手 (7)

你这个判断我基本要反过来讲：**这两件事恰恰是 PyTorch 的主场，不是短板。** 会容易搞混是因为术语把"能力"和"使用体验"混在一起了。先把概念对齐再下结论。

## 先对齐术语

- **VJP** = reverse-mode 的 vector-Jacobian product，也就是**反向传播**。你写一个自定义 `backward`，本质就是在写一个自定义 VJP。
- **JVP** = forward-mode 的 Jacobian-vector product，即**前向模式求导**。

所以"给 SDPA 手写 fwd/bwd""给 WGAN-GP 写双重反向"——这俩本质上就是**自定义 VJP / 让梯度穿过梯度**，而 PyTorch 的 `autograd` 生态就是靠这个起家的。（FlashAttention、xformers、 apex 全是这个模式写的。）

---

## 1. WGAN-GP 的 gradient penalty：PyTorch 原生就支持

这是**教科书级的 double backprop**，从 PyTorch 0.x 就能做，靠 `create_graph=True`：

```python
alpha = torch.rand(B, 1, 1, 1, device=x.device)
xg = (alpha * x + (1 - alpha) * y).requires_grad_(True)

d_xg = D(xg)
grad = torch.autograd.grad(
    outputs=d_xg.sum(), inputs=xg,
    create_graph=True, retain_graph=True   # ← 关键：把梯度图也建出来
)[0]
grad_penalty = ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

loss = -d_x.mean() + d_y.mean() + 10.0 * grad_penalty
loss.backward()   # 反传会再穿过 grad 的计算图
```

官方 WGAN-GP 实现就是这么写的。所以"torch 不好弄"这个印象不成立——**它只是经常被人写错**（忘了 `create_graph`、忘了 `requires_grad_`）。另外可以用 `torch.autograd.gradgradcheck` 验证二阶梯度正确性，这也是 torch 独有的成熟工具链。

## 2. SDPA 的 fwd/bwd：自定义 VJP = 写个 `backward`

`F.scaled_dot_product_attention` 直接 dispatches 到 Flash / memory-efficient 后端，手写的话就是标准 `torch.autograd.Function`：

```python
class FlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, lse = flash_fwd(q, k, v)      # 自己/内核算，保存 lse 等残差
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(ctx, dout):                # ← 这就是自定义 VJP
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = flash_bwd(dout, q, k, v, out, lse)
        return dq, dk, dv

attn = FlashAttnFn.apply
```

新版更现代的写法是 dispatcher 层的注册，PyTorch 2.x 有专门的 VJP 注册入口：

```python
@torch.library.custom_op("mylib::flash_attn", mutates_args=())
def flash_attn(q, k, v) -> torch.Tensor: ...

# 注册自定义的反向（+可选 jvp）
flash_attn.register_autograd(backward_impl, setup_context=...)
```

所以**自定义 VJP 在 PyTorch 里是一等公民**，比 JAX 的 `custom_vjp` 历史还久、生态还大。

---

## 那 JAX 到底强在哪？——强在"组合"和"前向模式"，不在"能不能自定义"

真正 JAX 更顺的是这三点，而不是"能不能手写 backward"：

<details>
<summary><b>① 自定义规则和 vmap/jvp/vjp/jit 的任意组合（JAX 的核心优势）</b></summary>

```python
@jax.custom_vjp
def attn(q, k, v):
    return _attn(q, k, v)

def fwd(q, k, v):
    out, lse = _attn_fwd(q, k, v)
    return out, (q, k, v, lse)

def bwd(res, g):                       # 自定义 VJP 规则
    q, k, v, lse = res
    return _attn_bwd(g, q, k, v, lse)

attn.defvjp(fwd, bwd)

# 然后可以随便叠：
jax.jit(jax.vmap(jax.grad(attn)))      # 自定义规则 + vmap + grad + jit 干净组合
```

JAX 里你注册的 `custom_vjp` 会被 `vmap`、`jvp`、`jit` 无缝穿过。PyTorch 里 `torch.autograd.Function` 遇到 `torch.vmap`（`torch.func`）通常要额外写 `vmap` 静态方法，覆盖不到位就报错。

</details>

<details>
<summary><b>② 前向模式（JVP）成熟度</b></summary>

- JAX：`jax.jvp` 一等公民，`custom_jvp` 和反向模式可任意混合。
- PyTorch：前向模式有 `torch.autograd.forward_ad`（dual tensor）、`torch.func.jvp`、`Function.jvp`，但覆盖面窄、很多算子和库没实现 JVP，容易中断。

做雅可比、隐式微分、数值验证、或者"前向模式更省内存"的场景，JAX 体验明显好。

</details>

<details>
<summary><b>③ 对自定义规则再做高阶 / 再嵌套</b></summary>

- JAX：`grad(grad(f))`、`vjp` 套 `jvp` 等，只要自定义规则写对，高阶自动成立。
- PyTorch：普通算子 `create_graph=True` 能拿二阶；但一旦走 `autograd.Function`，**二阶导需要额外实现 `backward` 的二阶（`make_double_backward` / 再写 `backward` 的 backward）**，或用 `torch.func` 重写一遍，很繁琐且容易错。

这也是为什么"hypergradient、MAML 内层步求导、LOLA"在 JAX 里优雅，在 PyTorch 里要小心实现。

</details>

---

## 一张对照表

| 能力 | PyTorch | JAX |
|---|---|---|
| 自定义反向 (VJP) | ✅ 一等公民（`autograd.Function` / `register_autograd`） | ✅ `custom_vjp` |
| 双重/高阶反传 (WGAN-GP) | ✅ 原生 `create_graph=True` | ✅ `grad(grad(f))` |
| 自定义 VJP 与 vmap 组合 | ⚠️ 要额外写 `vmap` 静态方法，易崩 | ✅ 无缝 |
| 前向模式 (JVP) | ⚠️ 有但覆盖窄 | ✅ 成熟 |
| 自定义规则 + 高阶导数 | ⚠️ 要手写二阶 | ✅ 自动成立 |
| 生态 / 已实现的算子库 | ✅✅ 海量 | ✅ 增长中 |

---

## 结论

- **"torch 弄不了 WGAN-GP 的 grad reg / SDPA 的 fwd-bwd"** ——不对，这两样是 PyTorch 最标准的能力，FlashAttention 整个生态都建在它上面。
- **真正 JAX 更舒服的**是：自定义规则之后的**可组合性**、**前向模式**、以及**对自定义规则再做高阶导数**。是"更优雅、更少踩坑"，而不是"PyTorch 跑不起来"。

如果你手上有具体那段代码卡住了（比如 vmap 穿 `autograd.Function` 报错，或者要写二阶），贴出来我可以直接给你 PyTorch / JAX 两版对照写法。

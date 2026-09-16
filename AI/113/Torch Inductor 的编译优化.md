# Torch Inductor 的编译优化

### 🧑‍💻 User

torch inductor那套是怎么做的，为什么可以hook模型算子然后编译优化？

---

### 🤖 Assistant

# Torch Inductor 是怎么"hook"算子并编译优化的

这个问题其实牵涉到 `torch.compile` 的整套机制。Inductor 本身只负责"后端编译",而"hook 算子"这件事是 **TorchDynamo** 干的。先把这个三段式流水线摆出来,后面再逐层拆。

## 1. 整体流水线

```
model(x)
   │  torch.compile 包裹
   ▼
┌─────────────────────────────────────────────┐
│ ① TorchDynamo   捕获 Python 字节码 → FX Graph │  ← "hook" 发生在这里
└─────────────────────────────────────────────┘
   ▼
┌─────────────────────────────────────────────┐
│ ② AOTAutograd   展开 autograd，得到前向+反向联合图 │
└─────────────────────────────────────────────┘
   ▼
┌─────────────────────────────────────────────┐
│ ③ TorchInductor lower 成 Triton/C++ kernel   │  ← 真正的编译优化
└─────────────────────────────────────────────┘
   ▼
运行生成好的 kernel（没有逐算子 Python 开销）
```

---

## 2. 核心机制:Dynamo 靠什么"hook"

很多人以为 hook 是去 monkey patch 每一个 `torch.*` 算子,其实不是。Dynamo 用的是一把更底层的钥匙:**CPython 的 frame evaluation API(PEP 523)**。

### 2.1 接管字节码解释器

CPython 允许你替换"帧求值函数"(也就是负责解释字节码的函数)。PyTorch 在 C++ 层把它暴露出来:

```python
# torch/_dynamo/eval_frame.py 内部本质上是:
import torch._C._dynamo.eval_frame as eval_frame

eval_frame.set_eval_frame(my_callback)   # 装一个回调
```

装上之后,**每一个 Python 函数调用(每个 frame)在被 CPython 执行之前,都会先经过这个回调**。回调可以返回:

- `None` → 照常执行
- 一个新的 code object / 编译产物 → 改执行它

这就是"hook"的根:Dynamo 不是在算子层面打补丁,而是**接管了整段 Python 代码的执行权**。

### 2.2 符号执行,而不是真的跑

Dynamo 拿到一个 frame 后,用一个符号解释器(`InstructionTranslator` / `symbolic_convert.py`)逐条"模拟"执行字节码,维护一个**符号栈**。栈上的每个值都被包成 `VariableTracker` 子类:

```python
TensorVariable / ConstantVariable / LocalVariable / UserDefinedObjectVariable ...
```

关键区别是:执行 `torch.sin(x)` 时,**不是真的去 launch CUDA kernel**,而是往一张图里追加一个节点:

```python
def f(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    c = a + b
    return c * 2

# Dynamo 符号执行后得到(概念上):
#   %sin : f32[2,3] = aten.sin.default(%x)
#   %cos : f32[2,3] = aten.cos.default(%y)
#   %add : f32[2,3] = aten.add.Tensor(%sin, %cos)
#   %mul : f32[2,3] = aten.mul.Scalar(%add, 2)
```

它只是"记录你想做什么",而不是"现在做"。最后得到一张 **FX Graph**(`torch.fx.Graph`),节点都是 `call_function(aten::xxx)` 这种数据流 DAG。

### 2.3 为什么算子能被识别出来

这里有个前提让这件事变得可行:**PyTorch 的所有算子都注册在同一个 c10 dispatcher 里**(`torch._ops` / `aten`),有统一的调用约定。所以当符号解释器看到 `CALL_FUNCTION` 字节码时,它检查被调用的对象:

- 是注册过的 torch op → 生成图节点
- 是普通 Python 函数 / `nn.Module` → **内联**进去继续 trace(默认 `inline` 策略)
- 是控制流/不支持的动态行为 → **graph break**,退回 eager 执行

正因为它统一注册、统一 dispatch,才能"认出一个调用是算子并把它图形化"。

---

## 3. AOTAutograd:把反向也变成图

Dynamo 默认只捕获前向。但编译优化(融合、内存复用)往往要打通前向+反向。AOTAutograd 做的事是:

- 用 `grad_fn` 机制 trace 出反向,得到 **前向+反向的联合图(primal / tangent)**;
- 把一些复合算子(如 `aten::cross_entropy`)**decompose** 成基础算子;
- 对需要梯度的部分做 functionalization。

这样 Inductor 拿到的就不是"一个模型的前向",而是"一次训练迭代所需的完整计算图"。

---

## 4. Inductor 拿到图之后做什么

这才是 Inductor 的本职。它的编译过程大致是:

### 4.1 FX Graph → Inductor IR

把 ATen 算子 lower 成 Inductor 自己的中间表示(`torch._inductor.ir` 里约一百个左右的核心算子,如 `ops.mm / ops.add / ops.reduction`),并做大量 **fx pass** 去重排、融合、化简。

### 4.2 关键优化

| 优化 | 说明 |
|------|------|
| **算子融合** | 把一串 elementwise(如 `sin → cos → add → mul`)融进一个 kernel,避免多次读写显存 |
| **布局选择** | 决定用 NCHW 还是 NHWC/channels_last,减少 transpose |
| **内存规划** | 复用 buffer,做 activation checkpointing |
| **tiling / 分块** | 把矩阵乘、卷积切成适合 GPU 的 block |
| **autotune** | 对 tile size、num_warps、vectorization 等参数跑一遍实测选最优 |
| **向量化 / 软件流水** | 生成更贴近硬件的代码 |

### 4.3 代码生成

- **GPU** → 生成 **Triton** kernel(再交给 Triton 编译成 PTX/CUDA)
- **CPU** → 生成 **C++/OpenMP** 代码,用 clang 编译成 `.so`

所以 Inductor 本质是一个"**从计算图到 Triton/C++ 的编译器后端**",用的大多是经典编译器技术(IR lowering、调度、tiling、codegen)。

---

## 5. 正确性靠什么保证

"hook 之后乱改"不会出错,是因为有这层保险:

- **Guards(守卫)**:Dynamo 每次 trace 都会记录假设(shape、dtype、device、stride、全局状态等),编译产物带 guards。每次调用先验 guards,变了就**重新 trace/编译**。
- **Graph break**:遇到无法符号化的 Python 结构(如 data-dependent 控制流、某些自定义对象),就在那里切一刀,前后两段图 + 中间退回 eager,保证语义一致。
- **Fallback**:不支持的算子直接退回 eager,不硬来。

---

## 6. 一句话总结

> **能 hook,是因为 Dynamo 用 PEP 523 的 frame eval hook 接管了 Python 字节码解释,做符号执行;能编译优化,是因为 torch 算子统一注册在 dispatcher 里,可以被识别并图形化成一张静态数据流图(FX Graph),之后 Inductor 把它当成普通编译器输入,做融合、tiling、内存规划,再生成 Triton/C++ kernel。**

换句话说,这套设计把"动态的 Python 代码"在运行时变成了"静态的可分析计算图",而一旦有了图,剩下就是标准的编译器工作了——Python 层面逐算子的 dispatch 开销也在这个过程中被彻底消除。

---

<details>
<summary>想深入的话,可以看的源码入口</summary>

- 帧接管:`torch/_dynamo/eval_frame.py`(C++ 实现在 `torch/csrc/dynamo/eval_frame.cpp`)
- 符号执行:`torch/_dynamo/symbolic_convert.py`(`InstructionTranslator`)
- 变量追踪:`torch/_dynamo/variables/`(各种 `VariableTracker`)
- Guards:`torch/_dynamo/guards.py`
- Inductor 融合 pass:`torch/_inductor/fx_passes/`
- Triton 代码生成:`torch/_inductor/codegen/triton.py`
- 调度器:`torch/_inductor/scheduler.py`

</details>
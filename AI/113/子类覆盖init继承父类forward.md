# 子类覆盖init继承父类forward

### 🧑‍💻 User

pytorch中，nnmodule子类如何覆盖init但继承父类forward方法？

---

### 🤖 Assistant

在 PyTorch 中，`nn.Module` 的继承完全遵循 Python 标准的面向对象编程（OOP）规则。

如果你想在子类中覆盖（重写）`__init__` 方法，但保留并使用父类的 `forward` 方法，你只需要**在子类中定义 `__init__` 方法，并且不定义 `forward` 方法**即可。

这里有一个关键点：**在子类的 `__init__` 中，必须调用 `super().__init__()`**，并且你需要确保子类的初始化逻辑依然创建了父类 `forward` 方法中所依赖的所有层或属性。

### 代码示例

下面是一个典型的应用场景：我们继承一个已有的模型（父类），在子类的 `__init__` 中修改某个层（比如把普通的 ReLU 换成 LeakyReLU），然后直接复用父类的 `forward` 前向传播逻辑。

```python
import torch
import torch.nn as nn

# 1. 定义父类
class ParentModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU() # 父类默认使用 ReLU

    def forward(self, x):
        # 父类的 forward 逻辑
        print("=> 正在执行父类的 forward 方法")
        x = self.fc(x)
        x = self.activation(x)
        return x

# 2. 定义子类
class ChildModel(ParentModel):
    def __init__(self, in_features, out_features, use_leaky=True):
        # 第一步：必须调用父类的 __init__，这会初始化 self.fc 和 self.activation
        super().__init__(in_features, out_features)
      
        # 第二步：覆盖或增加子类特有的逻辑
        # 这里我们把父类创建的 ReLU 替换为 LeakyReLU
        if use_leaky:
            self.activation = nn.LeakyReLU(0.1)
          
        # 注意：这里我们完全不写 forward() 方法。
        # Python 会自动向上查找并继承 ParentModel 的 forward 方法。
```

### 运行测试

<details>
<summary><b>点击展开查看测试代码与输出</b></summary>

```python
# 创建输入张量
dummy_input = torch.randn(2, 5)

print("--- 测试父类 ---")
parent = ParentModel(5, 3)
output_parent = parent(dummy_input)

print("\n--- 测试子类 ---")
# 实例化子类，它会执行自己的 __init__，但调用父类的 forward
child = ChildModel(5, 3, use_leaky=True)
output_child = child(dummy_input)

print("\n子类的激活函数类型:", type(child.activation))
```

**输出结果：**
```text
--- 测试父类 ---
=> 正在执行父类的 forward 方法

--- 测试子类 ---
=> 正在执行父类的 forward 方法

子类的激活函数类型: <class 'torch.nn.modules.activation.LeakyReLU'>
```
</details>

### 开发注意事项

1. **不要忘记 `super().__init__()`**：如果你继承的是另一个 `nn.Module`，在子类的 `__init__` 开头调用 `super().__init__(...)` 是强制性的。如果不调用，PyTorch 内部的状态（如参数注册机制）将无法正常工作，会报 `AttributeError`。
2. **属性名称必须对齐**：因为你继承了父类的 `forward`，而父类的 `forward` 会调用特定的类属性（如 `self.fc`）。你在子类的 `__init__` 中无论怎么魔改，都必须保证父类 `forward` 里用到的属性名是存在且可调用的。

---

### 🧑‍💻 User

什么情况下会用到`nn.Module.__init__()`？

---

### 🤖 Assistant

在 PyTorch 中，**只要你自定义了一个继承自 `nn.Module` 的类，你就必定（且必须）会用到对 `nn.Module` 初始化方法的调用**。

通常我们推荐写成 `super().__init__()`，但它的本质就是调用了 `nn.Module.__init__(self)`。

如果你问的是**“为什么要调用它（它的底层作用是什么）”**以及**“什么时候会显式地写成 `nn.Module.__init__(self)` 而不是 `super()`”**，我们可以从以下几个方面来深入理解：

### 1. 为什么必须调用它？（底层机制）

`nn.Module` 的强大之处在于它能自动追踪和管理模型中的所有参数（Parameters）、子模块（Modules）和缓冲区（Buffers）。为了实现这一点，`nn.Module.__init__()` 在底层会初始化一系列极其重要的内部字典。

当你调用 `nn.Module.__init__()`（或 `super().__init__()`）时，PyTorch 会在你的对象实例中创建以下隐藏属性：
*   `self._parameters = OrderedDict()`：用于存储可学习的参数（如权重和偏置）。
*   `self._modules = OrderedDict()`：用于存储子网络层（如 `nn.Linear`, `nn.Conv2d`）。
*   `self._buffers = OrderedDict()`：用于存储不需要梯度更新的状态（如 BatchNorm 中的 running mean）。
*   `self._backward_hooks` / `self._forward_hooks`：用于存储各种钩子函数。

**如果不调用会怎样？**
如果你在自定义的 `__init__` 中忘记调用它，直接写 `self.fc = nn.Linear(10, 10)`，PyTorch 的 `__setattr__` 魔法方法在试图将 `nn.Linear` 注册到 `_modules` 字典时，会发现字典根本不存在，从而直接报错：
`AttributeError: cannot assign module before Module.__init__() call`

### 2. `super().__init__()` vs `nn.Module.__init__(self)`

在 99% 的情况下，我们使用 Python 3 推荐的 `super()` 语法：

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # 推荐写法
        self.fc = nn.Linear(10, 2)
```

但是，在极少数特定场景下，你可能会看到或需要直接写 `nn.Module.__init__(self)`：

<details>
<summary><b>场景一：多重继承 (Multiple Inheritance)</b></summary>

当你的类同时继承自 `nn.Module` 和另一个不相关的类时，Python 的 `super()` 会严格按照 MRO（方法解析顺序）链式调用。如果多重继承的另一个父类没有正确处理 `super()`，链条就会断裂。

为了绕过复杂的 MRO 并确保 `nn.Module` 被绝对初始化，有些开发者会显式调用：

```python
class MixinLogger:
    def __init__(self, log_name):
        self.log_name = log_name

# 多重继承
class ComplexModel(nn.Module, MixinLogger):
    def __init__(self):
        # 显式分别调用两个父类的初始化方法
        nn.Module.__init__(self)
        MixinLogger.__init__(self, log_name="complex_model")
      
        self.fc = nn.Linear(10, 10)
```
</details>

<details>
<summary><b>场景二：老旧的 Python 2 代码库</b></summary>

在 Python 2 中，`super()` 的语法很繁琐，必须写成 `super(MyModel, self).__init__()`。一些从老版本迁移过来的代码，或者习惯了旧式类的开发者，可能会直接写：

```python
class OldSchoolModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self) # Python 2 时代的旧式类常见写法
        self.fc = nn.Linear(10, 10)
```
但在现代 PyTorch 开发中，这已经不再推荐。
</details>

### 总结

*   **什么时候用到？** 每次创建自定义神经网络层或模型时都必须用到。
*   **作用是什么？** 帮你在底层建立好“户口本”（各种 `OrderedDict`），以便 PyTorch 能够接管和追踪你定义的网络参数、层级和状态。
*   **该怎么写？** 坚持使用 `super().__init__()`，除非你遇到了极其复杂的多重继承问题需要手动干预 MRO。
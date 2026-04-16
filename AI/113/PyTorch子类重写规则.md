# PyTorch子类重写规则

### 🧑‍💻 User

在继承一个nn.module类时，我不了解pytroch的子类重写规则是什么，为什么可以直接pass，可以重写forward但不写init方法，但不能只重写init不重写forward+不调用super().__init__()？

---

### 🤖 Assistant

要理解这些现象，我们需要区分**Python的面向对象编程（OOP）继承规则**和**PyTorch `nn.Module`的底层设计**。

简单来说：你遇到的情况，一部分是Python语言特性决定的，另一部分是PyTorch为了管理神经网络权重而强制要求的。

下面为你逐一拆解这三种情况：

### 1. 为什么可以直接 `pass`？

```python
import torch.nn as nn
class MyModel(nn.Module):
    pass
```

**原因（Python继承规则）：**
在Python中，如果子类内部没有任何代码（直接 `pass`），它会**完全继承**父类（`nn.Module`）的所有属性和方法，包括父类的 `__init__` 和 `forward`。

**结果：**
* 语法上完全合法，不会报错。
* 但是**没有任何实用价值**。当你实例化它并传入数据时（例如 `model(x)`），它会调用父类 `nn.Module` 默认的 `forward` 方法，而 PyTorch 源码中 `nn.Module` 的 `forward` 方法是这样写的：
  ```python
  def forward(self, *input):
      raise NotImplementedError
  ```
  所以你一运行就会报 `NotImplementedError` 错误。

---

### 2. 为什么可以重写 `forward` 但不写 `__init__` 方法？

```python
class MyModel(nn.Module):
    def forward(self, x):
        return x * 2
```

**原因（Python继承规则）：**
在Python中，**如果你在子类中没有定义 `__init__` 方法，Python在实例化子类时，会自动去调用父类的 `__init__` 方法。**

**结果：**
因为父类 `nn.Module` 的 `__init__` 被自动调用了，所以PyTorch底层需要的那些内部状态（比如参数字典、模块字典等）都被正确初始化了。同时你又重写了 `forward`，所以模型既有合法的内部结构，又有前向传播逻辑。
* **适用场景：** 这种写法通常用于**无状态（Stateless）**的操作，比如自定义一个简单的激活函数或形状变换逻辑，因为它不需要包含任何可学习的参数（如 `nn.Linear`）。

---

### 3. 为什么不能只重写 `__init__`，不调用 `super().__init__()` 且不重写 `forward`？

```python
class MyModel(nn.Module):
    def __init__(self):
        # 致命错误：忘记写 super().__init__()
        self.fc = nn.Linear(10, 10)
  
    # 忘记写 forward
```
这里其实包含了两个致命错误，我们分开看：

#### 致命错误 A：重写了 `__init__` 但没有调用 `super().__init__()`
**原因（Python继承规则 + PyTorch底层要求）：**
在Python中，**如果你在子类中定义了 `__init__`，Python就只会执行你写的 `__init__`，绝不会再自动帮你调用父类的 `__init__`。** 除非你手动用 `super().__init__()` 去调用。

如果你不调用父类的初始化方法，会发生什么？
<details>
<summary><b>点击展开：PyTorch 底层在 __init__ 里做了什么？（深入理解）</b></summary>
`nn.Module` 的 `__init__` 方法内部初始化了8个非常重要的有序字典（OrderedDict），用来追踪你的模型组件：

```python
# PyTorch nn.Module 源码中的 __init__ 片段
def __init__(self):
    self._parameters = OrderedDict()  # 追踪 nn.Parameter
    self._buffers = OrderedDict()     # 追踪不需要梯度的张量（如BN层的均值）
    self._non_persistent_buffers_set = set()
    self._backward_hooks = OrderedDict()
    self._is_full_backward_hook = None
    self._forward_hooks = OrderedDict()
    self._forward_pre_hooks = OrderedDict()
    self._state_dict_hooks = OrderedDict()
    self._load_state_dict_pre_hooks = OrderedDict()
    self._modules = OrderedDict()     # 追踪子模块 (如 nn.Linear)
```
如果你不调用 `super().__init__()`，你的类就不会有 `self._modules` 和 `self._parameters`。
当你在 `__init__` 里写 `self.fc = nn.Linear(10, 10)` 时，PyTorch 底层的 `__setattr__` 方法会试图把 `fc` 注册进 `self._modules` 字典里。但因为这个字典根本不存在，程序就会直接崩溃，抛出经典错误：
`AttributeError: cannot assign module before Module.__init__() call`
</details>

#### 致命错误 B：没有重写 `forward`
就像第1点提到的，如果你不重写 `forward`，它就会调用父类的 `forward`，从而抛出 `NotImplementedError`。

---

### 总结：标准继承模板

为了避免所有的坑，继承 `nn.Module` 的标准、也是最安全的模板永远是这样的：

```python
import torch.nn as nn

class StandardModel(nn.Module):
    def __init__(self):
        super().__init__() # 第一步：必须先初始化父类，建立PyTorch底层组件字典
      
        # 第二步：定义你自己的组件（现在PyTorch可以成功把它们注册进字典了）
        self.layer1 = nn.Linear(10, 10) 
      
    def forward(self, x):
        # 第三步：必须告诉模型数据是怎么流动的
        x = self.layer1(x)
        return x
```
# Python常见语法概览

### 🧑‍💻 User

讲讲常见python语法

---

### 🤖 Assistant

好的，我们来系统地梳理一下 Python 中非常常见的语法和基础概念。这些是日常编程中使用频率最高的部分。

---

## 1. 变量与基本数据类型

Python 是**动态类型**语言，变量不需要声明类型。

```python
name = "Alice"         # 字符串 str
age = 25               # 整数 int
height = 1.68          # 浮点数 float
is_student = True      # 布尔值 bool
nothing = None         # 空值 NoneType
```

### 常用类型转换

```python
int("42")      # 42
str(2024)     # "2024"
float("3.14") # 3.14
bool(0)       # False
```

---

## 2. 字符串操作

```python
s = "Hello, Python"

# 拼接与重复
print(s + "!")          # Hello, Python!
print(s * 2)            # Hello, PythonHello, Python

# 索引与切片（左闭右开）
print(s[0])            # H
print(s[7:13])         # Python
print(s[-6:])          # Python
print(s[::-1])         # 反转字符串

# 常用方法
s.lower()              # 'hello, python'
s.upper()              # 'HELLO, PYTHON'
s.replace("Hello", "Hi")
s.split(", ")          # ['Hello', 'Python']
", ".join(["A", "B"]) # 'A, B'
len(s)                 # 14
```

### f-string（格式化字符串）

```python
name = "Bob"; age = 30
print(f"{name} is {age} years old.")   # Bob is 30 years old.
print(f"Next year: {age + 1}")         # 可放入表达式
```

---

## 3. 基本容器类型

### 列表 `list` ——可变、有序

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
fruits.insert(1, "mango")
fruits.remove("banana")
last = fruits.pop()       # 弹出末尾元素
print(fruits[0])          # 'apple'
print(fruits[-1])         # 最后一个
print(len(fruits))
```

### 元组 `tuple` ——不可变

```python
point = (3, 5)
x, y = point             # 元组解包
print(x, y)              # 3 5
```

### 字典 `dict` ——键值对

```python
person = {"name": "Alice", "age": 25}
print(person["name"])
person["city"] = "Shanghai"
person.get("gender", "unknown")   # 安全取值
for key, value in person.items():
    print(f"{key}: {value}")
```

### 集合 `set` ——无序、不重复

```python
a = {1, 2, 3}
b = {3, 4, 5}
print(a & b)   # {3}
print(a | b)   # {1, 2, 3, 4, 5}
print(a - b)   # {1, 2}
```

---

## 4. 控制流

### 条件判断

```python
score = 85
if score >= 90:
    print("A")
elif score >= 80:
    print("B")
else:
    print("C")
```

Python 没有 `switch-case`，一般用 `if-elif-else` 或更现代的模式（match case，3.10+）。

### 循环

```python
# for 遍历可迭代对象
for fruit in ["apple", "banana"]:
    print(fruit)

# range() 生成整数序列
for i in range(5):          # 0,1,2,3,4
    print(i)

# while 循环
count = 0
while count < 3:
    print(count)
    count += 1

# break (退出)/ continue (跳过本次)
for i in range(10):
    if i == 3:
        break
    if i % 2 == 0:
        continue
    print(i)
```

---

## 5. 列表推导式 / 字典推导式

非常 Pythonic 的快速构建方式。

```python
# 列表推导式
squares = [x**2 for x in range(5)]               # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0]     # [0, 2, 4, 6, 8]

# 字典推导式
word_length = {word: len(word) for word in ["hi", "hello"]}  # {'hi': 2, 'hello': 5}

# 集合推导式
unique_lengths = {len(word) for word in ["hi", "hello", "hi"]}  # {2, 5}
```

---

## 6. 函数

```python
def greet(name, greeting="Hello"):
    """简单的问候函数（docstring）"""
    return f"{greeting}, {name}!"

print(greet("Alice"))              # Hello, Alice!
print(greet("Bob", "Hi"))          # Hi, Bob!
print(greet(greeting="Hey", name="C")) # 关键字参数
```

- 可以不按顺序传参，使用关键字参数。
- 默认参数在定义时只计算一次，要避免使用可变对象作为默认值。

### 不定数量参数

```python
def add(*args):              # 接收任意数量的位置参数
    return sum(args)

def info(**kwargs):          # 接收任意数量的关键字参数
    for k, v in kwargs.items():
        print(f"{k}: {v}")

add(1, 2, 3)                 # 6
info(name="Alice", age=25)   # name: Alice  age: 25
```

### lambda 表达式（匿名函数）

```python
square = lambda x: x * x
print(square(5))             # 25

# 常用在 sorted、map、filter 等
pairs = [(1, 3), (2, 1), (4, 2)]
pairs.sort(key=lambda x: x[1])   # 按第二个元素排序
print(pairs)                      # [(2, 1), (4, 2), (1, 3)]
```

---

## 7. 类与面向对象

```python
class Animal:
    species = "Unknown"         # 类属性

    def __init__(self, name):   # 构造函数
        self.name = name         # 实例属性

    def speak(self):            # 方法
        raise NotImplementedError

class Dog(Animal):
    species = "Canine"

    def speak(self):
        return f"{self.name} says Woof!"

d = Dog("Buddy")
print(d.speak())                # Buddy says Woof!
print(isinstance(d, Animal))    # True
```

一些魔法方法（dunder methods）常见：

```python
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __str__(self):
        return f"Point({self.x}, {self.y})"
    def __repr__(self):
        return f"Point({self.x!r}, {self.y!r})"
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
```

---

## 8. 异常处理

```python
try:
    num = int(input("Enter a number: "))
    result = 10 / num
except ValueError:
    print("That's not a number!")
except ZeroDivisionError:
    print("Can't divide by zero.")
else:
    print("Result:", result)       # 无异常时执行
finally:
    print("Execution finished.")   # 总会执行
```

---

## 9. 模块与导入

```python
import math
print(math.sqrt(16))           # 4.0

from datetime import datetime
print(datetime.now())

import numpy as np             # 常用别名
```

---

## 10. 文件读写

```python
# 写文件
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("Hello\nWorld")

# 读文件
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()          # 一次读取全部
    # 或逐行读取：
    # for line in f:
    #     print(line.strip())
```

使用 `with` 语句能自动关闭文件，安全可靠。

---

## 11. 一些非常实用的内置函数与方法

- `enumerate`：带索引的遍历
  ```python
  for idx, val in enumerate(["a", "b", "c"]):
      print(idx, val)
  ```
- `zip`：并行遍历
  ```python
  for a, b in zip([1,2,3], ['a','b','c']):
      print(a, b)
  ```
- `map` / `filter`（可用推导式替代，但有时简洁）
  ```python
  list(map(str, [1,2,3]))           # ['1','2','3']
  list(filter(lambda x: x>0, [-1,0,1])) # [1]
  ```
- `any` / `all`：任意/全部为真
  ```python
  any([False, True, False])   # True
  all([True, 1, "non-empty"]) # True
  ```
- `sorted` / `reversed`
- `sum`, `max`, `min`, `abs`, `round`

---

## 总结

以上覆盖了 Python 中你在大多数日常编程中会用到的语法结构：

- **基础类型与转换**
- **字符串与容器操作**
- **控制流**
- **推导式**
- **函数与 lambda**
- **面向对象**
- **异常处理**
- **文件IO**
- **常用内置函数**

掌握这些后，阅读和编写 Python 代码都会非常流畅。
# Python函数可空参数声明方法

### 🧑‍💻 User

python函数多个参数如何声明可空参数

---

### 🤖 Assistant

在 Python 中声明可空参数（即允许传入空值或不传值的参数），最标准的做法是**为参数设置默认值 `None`**，并结合**类型提示（Type Hints）**来明确指明该参数可以为 `None`。

以下是具体的实现方法：

### 1. 现代 Python 语法（Python 3.10 及以上）推荐做法

从 Python 3.10 开始，你可以直接使用 `|` 运算符来表示联合类型。使用 `类型 | None = None` 是最简洁直观的方式。

```python
def create_user(username: str, age: int | None = None, email: str | None = None):
    print(f"用户名: {username}")
    if age is not None:
        print(f"年龄: {age}")
    if email is not None:
        print(f"邮箱: {email}")

# 调用示例
create_user("Alice")                  # age 和 email 默认为 None
create_user("Bob", 25)                # email 默认为 None
create_user("Charlie", email="c@c.com") # 明确指定 email，age 为 None
```

---

### 2. ⚠️ 必须遵守的核心规则：参数顺序

在 Python 中，声明多个参数时，**带有默认值的参数（即可空参数）必须放在没有默认值的参数（必填参数）之后**。

```python
# ❌ 错误示范：必填参数放在了默认参数后面
def my_function(param1: str | None = None, param2: int):
    pass # 会引发 SyntaxError: non-default argument follows default argument

# ✅ 正确示范：必填参数在前，可空/默认参数在后
def my_function(param2: int, param1: str | None = None):
    pass
```

如果你的函数有非常多的可空参数，建议强制使用**关键字参数**调用，以避免顺序混乱：

```python
# 使用 * 号强制后面的参数必须通过关键字传递
def update_profile(user_id: int, *, bio: str | None = None, avatar: str | None = None):
    pass

# 只能这样调用:
update_profile(101, bio="Hello", avatar="image.png")
```

---

<details>
<summary><b>展开查看：旧版本 Python (3.9 及以下) 的写法</b></summary>

如果你使用的 Python 版本低于 3.10，你需要从 `typing` 模块导入 `Optional`。

`Optional[类型]` 的本质就是 `Union[类型, None]`。

```python
from typing import Optional

def create_user(username: str, age: Optional[int] = None, email: Optional[str] = None):
    pass
```
*注意：即使写了 `Optional[int]`，你依然需要在后面加上 `= None`，否则它在调用时依然是一个必须传入参数的必填项（虽然允许你显式传入 `None`）。*

</details>

<details>
<summary><b>展开查看：在函数内部如何处理可空参数</b></summary>

在函数体内部处理这些可空参数时，最规范的写法是使用 `is not None` 进行判断，而不是直接用 `if param:`，因为当参数值为 `0`、`""` (空字符串) 或 `False` 时，`if param:` 也会判断为假，从而导致逻辑错误。

```python
def process_data(count: int | None = None):
    # ❌ 错误做法 (如果传入 count=0，也会进不去 if)
    # if count:
    #     print(count)
  
    # ✅ 正确做法
    if count is not None:
        print(f"传入的数量是: {count}")
    else:
        print("未传入数量")
```
</details>
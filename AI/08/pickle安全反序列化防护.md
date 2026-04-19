# pickle安全反序列化防护

### 🧑‍💻 User

针对 pickle 格式如何进行安全检查？

---

### 🤖 Assistant

Python 的 `pickle` 模块由于其设计机制，**天生是不安全的**。恶意构造的 pickle 数据可以在反序列化（`pickle.loads()` 或 `pickle.load()`）时执行任意代码（通常通过滥用 `__reduce__` 魔术方法）。

**安全防御的第一原则是：绝对不要反序列化来自不受信任来源的 `pickle` 数据。**

如果你确实需要在系统中使用 `pickle`，可以通过以下几种方式进行安全检查和防护。为了方便阅读，我将核心防御策略和具体实现细节折叠，你可以展开查看重点信息。

<details open>
<summary><h3>1. 密码学签名验证（最推荐的安全方案）</h3></summary>

如果你必须在网络上传输 pickle 数据，或者将其保存在可能被篡改的存储中，**唯一的安全方式是使用密码学签名（如 HMAC）**。

在加载数据**之前**，先验证签名。如果签名不匹配，说明数据被篡改或并非来自受信任的源，直接丢弃，不要将其传入 `pickle.loads()`。

```python
import pickle
import hmac
import hashlib

SECRET_KEY = b'super_secret_key_only_known_to_backend'

def secure_pickle_dumps(obj):
    """序列化并附加 HMAC 签名"""
    pickled_data = pickle.dumps(obj)
    # 计算签名
    mac = hmac.new(SECRET_KEY, pickled_data, hashlib.sha256).digest()
    return mac + pickled_data

def secure_pickle_loads(data):
    """验证签名并反序列化"""
    if len(data) < 32: # SHA-256 签名长度为 32 字节
        raise ValueError("数据长度异常")
      
    received_mac = data[:32]
    pickled_data = data[32:]
  
    # 重新计算签名
    expected_mac = hmac.new(SECRET_KEY, pickled_data, hashlib.sha256).digest()
  
    # 使用 compare_digest 防止时序攻击
    if not hmac.compare_digest(expected_mac, received_mac):
        raise ValueError("签名验证失败，拒绝反序列化！")
      
    return pickle.loads(pickled_data)

# 使用示例
data = {"user": "admin", "role": "superuser"}
payload = secure_pickle_dumps(data)

# 安全加载
safe_data = secure_pickle_loads(payload)
print(safe_data)
```
</details>

<details>
<summary><h3>2. 白名单机制：重写 `Unpickler.find_class`</h3></summary>

如果你只需要反序列化特定的内建类型或自定义类，你可以通过继承 `pickle.Unpickler` 并重写 `find_class` 方法来实现**严格的白名单检查**。

这种方法可以阻止恶意数据调用 `os.system` 或 `subprocess.Popen` 等危险函数。

```python
import pickle
import builtins
import io

# 定义允许反序列化的白名单
ALLOWED_CLASSES = {
    ('builtins', 'int'),
    ('builtins', 'str'),
    ('builtins', 'dict'),
    ('builtins', 'list'),
    # 如果有自定义类，也需要在这里注册，例如：
    # ('my_module', 'MySafeClass')
}

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 检查模块和类名是否在白名单中
        if (module, name) in ALLOWED_CLASSES:
            return super().find_class(module, name)
        # 否则抛出异常，拒绝加载
        raise pickle.UnpicklingError(f"全局对象 {module}.{name} 被禁止！")

def safe_loads(pickled_bytes):
    return RestrictedUnpickler(io.BytesIO(pickled_bytes)).load()

# --- 测试 ---
# 1. 正常数据
good_data = pickle.dumps({"name": "Alice", "age": 30})
print("正常加载:", safe_loads(good_data))

# 2. 恶意数据模拟 (尝试执行危险代码)
class Malicious:
    def __reduce__(self):
        import os
        return (os.system, ('echo "你被黑了"',))

bad_data = pickle.dumps(Malicious())

try:
    safe_loads(bad_data)
except pickle.UnpicklingError as e:
    print("成功拦截恶意数据:", e)
```
</details>

<details>
<summary><h3>3. 使用第三方安全审查工具</h3></summary>

如果你需要对现有的未知 `.pkl` 文件进行安全静态分析，或者在 CI/CD 管道中进行检查，可以使用专门的工具：

*   **Fickling**：由 Trail of Bits 开发的反编译器和安全分析工具，专门针对 Python 的 pickle 格式。它可以静态分析 pickle 数据并检测其中是否包含恶意代码（如注入的 RCE）。
    *   安装：`pip install fickling`
    *   使用：可以直接在命令行中检查 `.pkl` 文件，它会尝试还原并展示 pickle 文件实际要执行的 Python 代码逻辑，而不会真正执行它。
*   **Bandit**：Python 的静态代码分析工具。它可以扫描你的 Python 源代码，并警告你所有使用了 `pickle` 模块的地方（标记为中/高危漏洞），强制开发者进行人工安全审查。
</details>

<details>
<summary><h3>4. 替代方案（治本之法）</h3></summary>

如果不需要序列化复杂的 Python 对象（如自定义类的实例、函数等），**强烈建议放弃使用 `pickle`**，改用以下格式：

*   **JSON (`json`)**: 最安全、跨语言支持最好。适合基础数据结构（字典、列表、字符串、数字）。
*   **MessagePack (`msgpack`)**: 类似于 JSON，但采用二进制格式，体积更小，速度更快，且同样安全。
*   **Protocol Buffers / Thrift**: 适用于强类型、高性能、跨语言的微服务通信。
*   **Safetensors**: 如果你在做机器学习（如 PyTorch 模型权重保存），不要用传统的 `torch.save`（基于 pickle），请换用 HuggingFace 开发的 `safetensors`，它不仅绝对安全，而且加载速度更快。
</details>

### 总结建议
如果你作为安全工程师或后端开发者在审查系统设计：
1. **能不用 `pickle` 就不用**，这是最根本的解决办法。
2. 必须用时，**只使用 HMAC 签名验证**过的 `pickle` 数据。
3. 绝对不要尝试编写正则去“过滤”或“清洗”恶意的二进制 pickle 数据，这是徒劳且极其危险的。
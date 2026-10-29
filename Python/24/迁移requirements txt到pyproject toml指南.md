# 迁移requirements txt到pyproject toml指南

## 🧑‍💻 User

requirements.txt如何迁移到pyproject.toml

---
## 😀 默认助手 (7)

# requirements.txt → pyproject.toml 迁移指南

一句话结论：**这两个文件语义不同**。`requirements.txt` 通常是“完整锁定清单”（含传递依赖、hash），而 `pyproject.toml` 的 `[project.dependencies]` 是“直接依赖声明”。因此迁移的正确目标是：把**声明**写进 `pyproject.toml`，把**锁定**交给 lock 文件，而不是把几十行 `==` 原样搬进 `dependencies`。

---

## 一、先做决策：你要的是哪种迁移

| 你的场景 | 推荐做法 |
|---|---|
| 应用/服务，想统一管理 | `pyproject.toml` 声明依赖 + 生成 lock 文件（`uv.lock` / `poetry.lock` / `pdm.lock`） |
| 要发布到 PyPI 的库 | 必须 `pyproject.toml`，只写直接依赖、**放宽版本**（`>=x,<y`），不要 `==` |
| 暂时不想换工具链 | 保留 `requirements.in` + `pip-compile`，或让 `pyproject.toml` 作为 `pip-compile` 的输入 |
| 有 hash 锁定的严格环境 | hash 无法写进 `pyproject.toml`，必须保留 lock 文件或 `requirements.txt` 产物 |

---

## 二、用工具自动迁移（最省事）

**uv（推荐，最快）**

```bash
uv init                      # 若还没有 pyproject.toml
uv add -r requirements.txt   # 直接把 requirement 行转成 [project.dependencies]
uv add --dev -r requirements-dev.txt
uv lock                      # 生成 uv.lock
```

**PDM**

```bash
pdm init                      # 生成 pyproject.toml
pdm import requirements.txt   # 官方 import 子命令，直接写入 dependencies
```

**Poetry（无官方 import）**

```bash
poetry init            # 交互式，会询问依赖
poetry add $(grep -vE '^\s*(#|$|-r |-c )' requirements.txt | sed 's/--hash=.*//')
```
Poetry 2.x 也支持 `[dependency-groups]`（PEP 735），dev 依赖建议放那里而不是 `[tool.poetry.group.dev]`。

**pip-tools（不改变现有习惯）**

```bash
# pyproject.toml 里写好 [project].dependencies 后
pip-compile pyproject.toml -o requirements.txt
```

> `dephell deps convert` 也能转换，但项目 2020 年后基本停更，不建议在新项目用。

⚠️ 自动迁移的通病：**会把传递依赖也塞进 `dependencies`**。迁完一定要人工删掉间接依赖（只留你 `import` 的那个包）。

---

## 三、手工映射对照表

| requirements.txt 写法 | pyproject.toml 对应位置 |
|---|---|
| `requests==2.31.0` | `dependencies = ["requests>=2.31,<3"]`（库）或保留 `==2.31.0`（应用） |
| `requests[socks]>=2.31` | `"requests[socks]>=2.31"` —— extras 语法完全一样 |
| `uvicorn==0.27.1 ; sys_platform != "win32"` | 原样保留，marker 语法一致 |
| `foo @ git+https://...@v1.0` | `"foo @ git+https://...@v1.0"`（PEP 508 直接引用） |
| `-e ./pkg` / `-e git+...` | `[tool.uv.sources] pkg = { path = "./pkg", editable = true }` |
| `-r other.txt` | 拆平后合并进 `dependencies`（pyproject 不支持 include） |
| `-c constraints.txt` | 无对应项，用 lock 文件表达 |
| `--index-url` / `--extra-index-url` | `[[tool.uv.index]]` 或 `[tool.poetry.source]`，**不是** `[project]` |
| `--find-links` | 工具特定配置（`[tool.uv]`） |
| `pytest`、`ruff`（dev） | `[dependency-groups].dev = [...]`（PEP 735） |
| `docs`、`test` 等分组 | `[project.optional-dependencies]` 或 `[dependency-groups]` |
| `--hash=sha256:...` | **无法表达**，交给 lock 文件 |
| `python_version` 约束 | `requires-python = ">=3.10"` |

`dependencies` 只接受 **PEP 508 字符串**，所以 `--hash`、`-r`、`--index-url` 这类 pip 命令行选项一律不能写进去。

---

## 四、完整示例

**迁移前 `requirements.txt`**

```text
Django==4.2.11
requests[socks]>=2.31,<3
uvicorn[standard]==0.27.1 ; sys_platform != "win32"
bar @ git+https://github.com/foo/bar.git@v1.0
numpy==1.26.4
-e ./local_pkg
```

**迁移后 `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "myapp"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "Django>=4.2.11,<5",
    "requests[socks]>=2.31,<3",
    "uvicorn[standard]==0.27.1; sys_platform != 'win32'",
    "bar @ git+https://github.com/foo/bar.git@v1.0",
    "numpy==1.26.4",
    "local-pkg",
]

# dev 依赖（PEP 735，uv/PDM/Poetry 2.x 都支持）
[dependency-groups]
dev = ["pytest>=8", "ruff>=0.4"]

[tool.uv.sources]
local-pkg = { path = "local_pkg", editable = true }
```

`numpy` 若只是被间接依赖，就该删掉；`Django==` 在应用场景可保留精确版本，在库场景要放宽。

---

## 五、几个容易踩的坑

1. **`[project]` 不是装饰性的** —— 一旦有 `[project]`，就必须有 `[build-system]`，否则 `pip install .` 会失败。
2. **不要把所有 `==` 搬进去** —— 库项目会与用户环境冲突（dependency hell），应用项目则应该改用 lock 文件。
3. **传递依赖要清理** —— 迁移后跑一次 `uv tree` / `pipdeptree`，确认没有把 `certifi`、`urllib3` 这类间接包写成直接依赖。
4. **`-e` 和本地路径** —— 标准 `[project]` 只支持 `file://` URL 形式，editable 语义是工具扩展（`[tool.uv.sources]`），不同工具写法不通用。
5. **私有源** —— 写工具段而非 `[project]`；`uv` 用 `[[tool.uv.index]]`，Poetry 用 `[[tool.poetry.source]]`。
6. **hash 锁定丢失** —— 若合规要求 hash，迁移后必须用 `uv export --format requirements-txt --hash` 或 `pip-compile --generate-hashes` 重新生成。
7. **marker 引号** —— TOML 字符串里 marker 用单引号更安全：`"pkg; python_version < '3.11'"`。
8. **`requirements.txt` 不必删除** —— 很多项目保留它作为部署产物（由 lock 导出），`pyproject.toml` 只作为唯一事实来源。

---

## 六、迁移后校验清单

```bash
uv sync                      # 或 pdm install / poetry install
uv tree                      # 确认依赖树未被改坏
pip install -e .             # 确认项目本身可安装
```

<details>
<summary>两个推荐的工具链组合（点开看）</summary>

**组合 A：uv 全家桶**
```bash
uv init && uv add -r requirements.txt && uv lock
# 需要给老 CI 用 → 导出
uv export --format requirements-txt --no-dev > requirements.txt
```

**组合 B：pip-tools 渐进式**（改动最小，`requirements.in` 保留）
```bash
# pyproject.toml 写 [project].dependencies
pip-compile pyproject.toml -o requirements.txt
pip-compile pyproject.toml --extra dev -o requirements-dev.txt
```

</details>

需要我针对你实际的 `requirements.txt` 内容做一次具体转换吗？把文件贴出来即可。

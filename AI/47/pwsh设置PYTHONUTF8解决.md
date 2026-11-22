# pwsh设置PYTHONUTF8解决

## 🧑‍💻 User

python 报错UnicodeDecodeError: 'gbk' codec can't decode byte 0x9c in position 665: illegal multibyte sequence
怎么设置pwsh环境变量解决

---
## 😀 默认助手 (7)

# 原因定位

报错发生在 **position 665、byte 0x9c**。在 GBK 里，`0x9C` 只能当**尾字节**使用，即前一个字节必须是合法的 GBK 首字节（`0x81–0xFE`）；这里它被当成非法多字节序列，说明这个文件**其实是 UTF-8 编码**，却被 Python 用系统默认的 GBK 去解码了。

触发点通常是这类代码：

```python
with open("xxx.txt") as f:      # ← 没写 encoding，Windows 上默认用 GBK
    text = f.read()
```

所以在 Windows 上有两条路：**改环境变量（治标，全局生效）** 和 **改代码显式指定 encoding（治本）**。下面先讲环境变量。

---

## 一、临时生效（仅当前 pwsh 窗口）

```powershell
$env:PYTHONUTF8 = "1"
python your_script.py
```

`PYTHONUTF8=1` 打开 **UTF-8 Mode**，会让 `open()` 的默认编码、`sys.stdout/stderr/stdin`、filesystem encoding 全部变成 UTF-8，基本能直接解决这个报错。

更轻量的等价写法（不用环境变量）：

```powershell
python -X utf8 your_script.py
```

---

## 二、永久生效（推荐 `[Environment]::SetEnvironmentVariable`）

**当前用户级**（不需要管理员，安全）：

```powershell
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
```

**系统级**（需要管理员权限的 pwsh）：

```powershell
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "Machine")
```

**验证是否写入成功：**

```powershell
[Environment]::GetEnvironmentVariable("PYTHONUTF8", "User")   # 应输出 1
```

**注意**：永久环境变量**不会影响已经打开的终端**，必须新开一个 pwsh 窗口（或重启 IDE / VS Code / PyCharm）才生效。验证：

```powershell
python -c "import sys; print(sys.flags.utf8_mode)"   # 应为 1
```

> `setx PYTHONUTF8 1` 也能达到同样效果（写用户级），但 `setx` 不会刷新当前会话，且有 1024 字符长度限制，`[Environment]::SetEnvironmentVariable` 更稳妥。

---

## 三、顺带把控制台输出编码也修掉

如果同时还出现中文乱码或 `UnicodeEncodeError`（打印到终端时），再补这两条：

```powershell
# 永久写用户级
[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")

# 当前窗口的代码页和 .NET 输出编码
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

<details>
<summary><b>四、写进 pwsh 配置文件（每次启动自动生效）</b></summary>

```powershell
# 查看配置文件路径
$PROFILE

# 若文件不存在则创建，然后追加
Add-Content -Path $PROFILE -Value '$env:PYTHONUTF8 = "1"'
Add-Content -Path $PROFILE -Value '$env:PYTHONIOENCODING = "utf-8"'
```

缺点：只对交互式 pwsh 生效，任务计划、CI、双击运行的脚本不受影响——所以长期方案仍是第二节的 User 级变量。

</details>

---

## 五、⚠️ 关键提醒：`PYTHONUTF8=1` 是全局开关，会"反向"弄坏 GBK 文件

`PYTHONUTF8=1` 把默认编码改成了 UTF-8。如果你项目里**确实有 GBK 编码的文件**（老的中文 CSV、Windows 导出的日志），它们反而会开始报 `UnicodeDecodeError`。所以**真正稳的做法是在代码里显式声明编码**，环境变量只做兜底：

```python
# 明确知道是 UTF-8
with open("xxx.txt", encoding="utf-8") as f:
    text = f.read()

# 编码不确定时：绕过默认编码探测
with open("xxx.txt", "rb") as f:      # 二进制读，自己解码
    raw = f.read()
text = raw.decode("utf-8")            # 或 raw.decode("gbk", errors="replace")
```

排查文件到底是什么编码：

```powershell
python -c "import sys;d=open(r'C:\path\to\file.txt','rb').read(700);print(d[640:680])"
```

再配合 `chardet` / `charset-normalizer` 自动检测：

```powershell
python -m pip install charset-normalizer
python -c "from charset_normalizer import from_path;print(from_path(r'C:\path\to\file.txt').best())"
```

---

## 小结

| 目标 | 命令 |
|---|---|
| 立刻修好当前窗口 | `$env:PYTHONUTF8 = "1"` |
| 单次运行 | `python -X utf8 script.py` |
| 一劳永逸（用户级） | `[Environment]::SetEnvironmentVariable("PYTHONUTF8","1","User")` |
| 控制台中文乱码 | `chcp 65001` + `PYTHONIOENCODING=utf-8` |
| 治本 | `open(..., encoding="utf-8")` 显式指定 |

改完记得**新开终端 / 重启 IDE**，再用 `python -c "import sys;print(sys.flags.utf8_mode)"` 确认返回 `1`。

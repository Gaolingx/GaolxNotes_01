# PowerShell拉取远端脚本执行

### 🧑‍💻 User

如何在powershell中使用命令实现拉取远端ps1代码执行？

---

### 🤖 Assistant

在 PowerShell 中“拉取远端 ps1 脚本并执行”本质上是**从 URL 下载脚本内容，再将其作为代码运行**。常见手段是利用 `Invoke-WebRequest` / `Invoke-RestMethod` 获取内容，然后通过 `Invoke-Expression` 或脚本块调用来执行。

⚠️ **强烈警告**：直接执行远程脚本极其危险，只应用于受信任的端点，且仅在开发/测试环境中酌情使用。生产环境应要求脚本签名或使用安全部署方式。

下面按推荐程度列出几种典型写法。

---

### 方法一：Invoke-RestMethod + Invoke-Expression（最简洁）
`Invoke-RestMethod` 会把远程文本直接按字符串返回，无需额外处理。

```powershell
# 基础用法
Invoke-RestMethod -Uri "https://example.com/script.ps1" | Invoke-Expression
```

或使用别名：

```powershell
irm https://example.com/script.ps1 | iex
```

> **说明**：`irm` 是 `Invoke-RestMethod` 的别名，`iex` 是 `Invoke-Expression` 的别名。管道传递的脚本内容会在当前作用域执行。

---

### 方法二：Invoke-WebRequest + 脚本块调用
将内容转为脚本块再执行，可避免 `Invoke-Expression` 的某些隐患（但风险依然很高）。

```powershell
# 下载内容并转为脚本块
$scriptBlock = [scriptblock]::Create((Invoke-WebRequest -Uri "https://example.com/setup.ps1").Content)

# 执行脚本块
& $scriptBlock
```

在 Windows PowerShell 5.1 中，如果不加 `-UseBasicParsing`，`Invoke-WebRequest` 会尝试解析 DOM，可能造成开销或错误，建议加上：

```powershell
$response = Invoke-WebRequest -Uri "https://..." -UseBasicParsing
$scriptBlock = [scriptblock]::Create($response.Content)
& $scriptBlock
```

PowerShell 7+ 已默认使用基本解析，无需该参数。

---

### 方法三：先下载到临时文件再执行
适合需要保留脚本副本或审计的场景。

```powershell
$tempFile = Join-Path $env:TEMP "temp_script.ps1"
Invoke-WebRequest -Uri "https://example.com/tool.ps1" -OutFile $tempFile
& $tempFile
# 用完后删除
Remove-Item $tempFile
```

---

### 方法四：传统 .NET WebClient（向后兼容）
在极老的 PowerShell 环境中也可以使用。

```powershell
(New-Object Net.WebClient).DownloadString("https://example.com/script.ps1") | Invoke-Expression
```

---

### 注意事项

1. **执行策略**
   如果 PowerShell 会话的 `ExecutionPolicy` 为 `Restricted`，直接运行下载的脚本可能会被阻止。
   解决方案：
   - 临时放开（不推荐）：`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   - 使用脚本块或 `Invoke-Expression` 通常不受执行策略限制（因为它们运行的是字符串，不是文件）。
   - 若一定要执行 `.ps1` 文件，可绕过：`powershell -ExecutionPolicy Bypass -File $tempFile`

2. **HTTPS 与 TLS**
   旧系统可能不支持 TLS 1.2，需强制启用：
   ```powershell
   [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
   ```

3. **权限作用域**
   远程脚本将在**当前用户权限**下运行，能访问一切当前上下文可用的资源（环境变量、凭据、文件系统等）。

---

### 典型“拉取即执行”单行命令
不少 DevOps 工具（如 Chocolatey 安装）会采用这种形式：

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://example.com/install.ps1'))
```

这条命令做了三件事：
- 临时放宽执行策略
- 确保 TLS 1.2 可用
- 下载字符串并交给 `iex` 执行

请根据你的环境选择最合适的方法，并务必在可信网络和可信源下操作。

---

### 🧑‍💻 User

linux bash如何curl完成类似操作？

---

### 🤖 Assistant

在 Linux bash 中，用 `curl` 拉取远程脚本再执行的思路与 PowerShell 完全一致：**先下载内容，再交给解释器运行**。最常见的写法是管道直接执行：

---

### 方法一：管道直接执行（最常用）

```bash
curl -sSL https://example.com/script.sh | bash
```

这也是一行式的经典写法。`curl` 会把脚本内容输出到 stdout，`bash` 接收并执行。

如果脚本是 **PowerShell 脚本**（.ps1），在 Linux 上要交给 `pwsh`（PowerShell 7）执行：

```bash
curl -sSL https://example.com/script.ps1 | pwsh -Command -
```

> 注意末尾的 `-` 表示从 stdin 读取脚本内容。

---

### 方法二：先下载到临时文件再执行

```bash
curl -sSL -o /tmp/script.sh https://example.com/script.sh
bash /tmp/script.sh
rm -f /tmp/script.sh   # 用后清理
```

保留脚本副本便于审计。`.ps1` 文件则用 `pwsh /tmp/script.ps1` 执行。

---

### 方法三：通过 `source` 导入（在当前 shell 环境生效）

如果需要脚本设置的变量、函数对当前终端生效，用 `source`：

```bash
source <(curl -sSL https://example.com/script.sh)
```

或简写：

```bash
. <(curl -sSL https://example.com/script.sh)
```

> 这种方法运行在当前 shell 中，不会新建子进程，副作用（环境变量、`cd`、`export`）会保留。而 `bash script.sh` 是在子 shell 中执行的，对当前环境无影响。

---

### 方法四：条件执行（带健康检查）

```bash
curl -fsSL https://example.com/script.sh && bash
```

`-f` 让 HTTP 返回 4xx/5xx 时 curl 直接报错退出，避免把错误页当脚本执行。

---

### curl 常用参数说明

| 参数 | 作用 |
|------|------|
| `-s` | 静默模式，不显示进度条 |
| `-S` | 出错时仍显示错误信息（配合 `-s` 使用） |
| `-L` | 自动跟随 3xx 重定向 |
| `-f` | HTTP 错误时失败并退出（不输出错误页） |
| `-o 文件` | 将输出写入文件而非 stdout |
| `-I` / `-X HEAD` | 只获取响应头，可用于检查脚本是否可访问 |

完整写法常组合为 `curl -fsSL`，即“静默、跟随重定向、遇错即止”。

---

### 典型“拉取即执行”单行命令

很多安装器（如 `nvm`、`rustup`、`docker` 安装脚本）都采用这种模式：

```bash
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
```

---

### 与 PowerShell 方案的对照

| 功能 | PowerShell | Linux bash |
|------|-----------|------------|
| 下载内容并执行 | `irm url \| iex` | `curl -fsSL url \| bash` |
| 下载到文件 | `iwr -OutFile f` | `curl -o f url` |
| 文件执行 | `& ./script.ps1` | `bash script.sh` |
| 在当前环境生效 | `. ./script.ps1` | `source <(curl -fsSL url)` |

---

### ⚠️ 安全提醒

- `curl | bash` 是**公认的高危操作**：脚本内容对用户完全不可见，却直接以当前用户权限运行。
- 务必确认：URL 是 HTTPS、来源可信、域名未被劫持。
- 建议先下载到本地检查内容，再决定是否执行：
  ```bash
  curl -fsSL -o inspect.sh https://example.com/script.sh
  less inspect.sh    # 先看内容
  bash inspect.sh    # 确认无误后再跑
  ```
- 生产环境优先使用带签名的包（如 GPG 校验）、版本锁定或官方仓库，而不是裸管道执行。
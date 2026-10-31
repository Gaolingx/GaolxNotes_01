# Sunshine配对支持SSH远程脚本调用API

## 🧑‍💻 User

sunshine远程桌面支持powershell cli中实现pair device吗

---
## 😀 默认助手 (7)

## 结论

**Sunshine 没有内置的 `pair device`、`--pair` 这类命令行配对命令**，但可以借助它 Web UI 后面的 REST 接口 `/api/pin`，用 PowerShell 脚本实现“自动批准配对”。

依据是源码：`src/main.cpp` 的命令分发表 `cmd_to_func` 里只有 `creds`、`help`、`version`、`restore-nvprefs-undo` 四个子命令，没有任何 pin/pair 相关命令。

还有一个容易被误解的前提：

> 配对是 Moonlight 客户端发起的双向密钥交换，主机端**只能“批准/拒绝”**一个已经存在的配对请求，不能凭脚本单方面把一台设备“加进去”。所以 PIN 必须是客户端屏幕上显示的那 4 位数字，脚本只能省掉“打开浏览器点提交”这一步。

---

## 接口长什么样（两个版本差异很大）

| 版本 | 端点 | 请求体 | 是否需要 pairing_id |
|---|---|---|---|
| v0.23.1 及更早 | `POST /api/pin` | `{"pin":"1234"}` | 否 |
| 最新代码（master） | `GET /api/pin` | — 返回 `{"pairings":[{"id","name","address"}]}` | — |
| 最新代码 | `POST /api/pin` | `{"pairing_id":"32位hex","pin":"1234","name":"My PC"}` → `{"status":true/false}` | **是** |
| 最新代码 | `DELETE /api/pin` | `{"pairing_id":"32位hex"}` 取消请求 | 是 |

认证与 CSRF（`src/confighttp.cpp`）：

- 认证方式是 **HTTP Basic**，即 Web UI 的用户名/密码，默认端口 `47990`，使用 HTTPS 和自签名证书。
- CSRF：`validate_csrf_token()` 里明确写了 *“If neither Origin nor Referer is present … non-browser clients (curl, scripts) never send these headers”* → **纯脚本调用不需要 CSRF token**。只有当请求带了不匹配的 `Origin`/`Referer` 时才需要 `X-CSRF-Token`（可以先去 `GET /api/csrf-token` 取）。
- 别忘了配置项 `origin_web_ui_allowed` 要允许来源 IP（`lan` 或 `localhost`），否则返回 403。
- `POST /api/pin` 要求 `Content-Type: application/json`；它会**阻塞等待握手**，最长到 `ping_timeout`，所以 `-TimeoutSec` 要给够。

<details>
<summary>可直接用的 PowerShell 脚本（自适应新旧两套 API）</summary>

```powershell
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Pin,          # Moonlight 客户端上显示的 4 位 PIN
    [string]$Name     = 'My Moonlight',
    [string]$BaseUrl  = 'https://localhost:47990',
    [string]$UserName = 'sunshine',
    [string]$Password = $env:SUNSHINE_PASS,      # 不要把密码硬编码在脚本里
    [string]$PairingId = ''                      # 留空 = 自动选第一个待批准请求
)

if ($Pin -notmatch '^\d{4}$') { throw 'PIN 必须是 4 位数字' }

# 自签名证书处理
if ($PSVersionTable.PSVersion.Major -lt 6) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    if (-not ('TrustAllCerts' -as [type])) {
        Add-Type @"
using System.Net; using System.Security.Cryptography.X509Certificates;
public class TrustAllCerts : ICertificatePolicy {
    public bool CheckValidationResult(ServicePoint sp, X509Certificate c, WebRequest r, int p) { return true; }
}
"@
    }
    [Net.ServicePointManager]::CertificatePolicy = New-Object TrustAllCerts
}

$basic = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("${UserName}:${Password}"))
$script:headers = @{ Authorization = "Basic $basic" }

function Invoke-SunshineApi {
    param([string]$Method, [string]$Path, [string]$Json)
    $splat = @{
        Uri         = "$BaseUrl$Path"
        Method      = $Method
        Headers     = $script:headers
        TimeoutSec  = 120
        ErrorAction = 'Stop'
    }
    if ($PSVersionTable.PSVersion.Major -ge 6) { $splat.SkipCertificateCheck = $true }
    if ($Json) { $splat.Body = $Json; $splat.ContentType = 'application/json' }
    Invoke-RestMethod @splat
}

# 1) 先探测是否为“新版”接口（有 pending pairings 列表）
$pending = $null
try { $pending = Invoke-SunshineApi -Method GET -Path '/api/pin' } catch { $pending = $null }

if ($null -ne $pending -and $pending.PSObject.Properties.Name -contains 'pairings') {
    $target = $pending.pairings |
        Where-Object { $PairingId -eq '' -or $_.id -eq $PairingId } |
        Select-Object -First 1
    if (-not $target) {
        throw '没有待批准的配对请求。请先在那台客户端上点“配对”，让它显示 PIN。'
    }
    Write-Host "批准配对：$($target.name) 来自 $($target.address)"
    $body = @{ pairing_id = $target.id; pin = $Pin; name = $Name } | ConvertTo-Json -Compress
    $result = Invoke-SunshineApi -Method POST -Path '/api/pin' -Json $body
}
else {
    # 2) 旧版接口，只认 pin（多一个 name 字段无害）
    $body   = @{ pin = $Pin; name = $Name } | ConvertTo-Json -Compress
    $result = Invoke-SunshineApi -Method POST -Path '/api/pin' -Json $body
}

if ($result.status -eq $true) { Write-Host '配对成功 ✅' } else { Write-Warning '配对失败：PIN 错误 / 超时 / 客户端已取消' }
```

等价的 `curl.exe` 一行（新版）：

```powershell
curl.exe -k -u sunshine:密码 -H "Content-Type: application/json" `
  -d '{\"pairing_id\":\"0123456789abcdef0123456789abcdef\",\"pin\":\"1234\",\"name\":\"My PC\"}' `
  https://localhost:47990/api/pin
```

</details>

---

## 几个实战注意点

1. **必须先有 PIN**：脚本无法“凭空配对”。如果要做无人值守部署，只能让客户端侧在配对时把 PIN 传出来（例如截图识别、客户端日志解析，或自己写一个实现了相同 `pair?uniqueid=...&phrase=getservercert...` 流程的客户端）。
2. **服务方式运行不影响**：Sunshine 以 Windows 服务（`sunshineservice`）运行时，`/api/pin` 一样可用，PowerShell 不需要管理员权限。
3. **设置 Web UI 密码**：`sunshine.exe --creds <用户名> <密码>` 可以设置或重置 Web UI 凭据（写入凭据文件，重启服务后生效），这是为脚本准备认证条件的官方途径，但它**只改账号密码，不涉及配对**。
4. **安全**：一个能带凭据调 `/api/pin` 的脚本等于“可随时批准任意设备接入”。密码请走 `Get-Credential`、环境变量或凭据管理器，不要落在脚本里。
5. **版本兼容**：如果你的版本既没有 `GET /api/pin`，`POST /api/pin` 也只返回 `{"status":false}` 且日志里报 `pairing_id must contain exactly 32 hexadecimal characters`，说明是“中间版本”，直接用上面的自适应写法即可（脚本会自动落到旧版分支）。

如果你的目标其实是 **“在主机上一次性批量清除/重建可信客户端”**，那确实有纯 CLI 能力的接口：`GET /api/clients/list`、`POST /api/clients/unpair`、`POST /api/clients/unpair-all`——这些可以完全脚本化，不受 PIN 限制。需要的话我可以再写一份针对它们的 PS 脚本。

---
## 🧑‍💻 User

这是否意味着可以通过ssh+bash/pwsh配对？

---
## 😀 默认助手 (7)

## 简短回答

**可以。但需要说明：SSH 只是为你提供主机上的 shell，并不是配对方式。真正的配对通道是 Sunshine 监听 `47990` 的 HTTPS API；无论通过本地 PowerShell、SSH 后的 bash，还是直接从另一台机器远程调用，操作都由这个 API 完成。**

SSH 的实际价值只有两个：

1. **网络可达性**：Web UI 端口被防火墙挡住，或 Sunshine 只绑在 `127.0.0.1`/内网时，SSH 是唯一的进入方式。
2. **不需要额外开端口**：运维习惯、跳板机、仅允许 22 端口的环境。

它**不会**改变最关键的限制：你仍然需要那 4 位 PIN，而 PIN 只显示在 **Moonlight 客户端**上，主机无法通过任何方法推算出它，因为 PIN 参与的是 `hash(pin + salt + challenges)` 挑战应答，缺了它握手就进行不下去。

---

## 三种调用方式（都能跑通，区别只在网络）

| 做法 | 命令形态 | 说明 |
|---|---|---|
| 本地 shell | `pwsh -File pair.ps1 -Pin 1234` | 最直接，`localhost:47990` 默认一定被 `origin_web_ui_allowed` 放行 |
| SSH 进去跑 | `ssh nas 'bash ~/pair.sh 1234'` | 等价于上面，只是 shell 在远端 |
| 不进主机直接远程调 | `curl -k -u u:p https://host:47990/api/pin` | 不需要 SSH；要求端口可达且 `origin_web_ui_allowed` 允许你的网段 |
| SSH 端口转发（推荐折中） | `ssh -N -L 47990:127.0.0.1:47990 user@host` | 之后本机调 `https://localhost:47990`，既不开端口也不用自签证书例外（证书仍然是自签，`-k` 还是要的） |

```bash
# 1) 打隧道
ssh -N -L 47990:127.0.0.1:47990 user@host &

# 2) 在本机调
curl -k -u sunshine:密码 https://localhost:47990/api/pin
```

---

## bash 版脚本（curl + jq，与 pwsh 版行为一致）

```bash
#!/usr/bin/env bash
# usage: pair.sh <PIN> [client_name] [pairing_id]
set -euo pipefail

PIN="${1:?usage: pair.sh 1234 [client_name] [pairing_id]}"
NAME="${2:-headless-client}"
BASE="${SUNSHINE_URL:-https://localhost:47990}"
SUSER="${SUNSHINE_USER:-sunshine}"
SPASS="${SUNSHINE_PASS:?请先 export SUNSHINE_PASS}"

[[ "$PIN" =~ ^[0-9]{4}$ ]] || { echo "PIN 必须是 4 位数字" >&2; exit 1; }

api() { curl -sS -k --max-time 120 -u "${SUSER}:${SPASS}" \
             -H 'Content-Type: application/json' "$@"; }

pending="$(api "${BASE}/api/pin" || true)"

if grep -q '"pairings"' <<<"$pending"; then
    # 新版 API：需要 pairing_id
    id="${3:-$(jq -r '.pairings[0].id // empty' <<<"$pending")}"
    [[ -n "$id" ]] || { echo "没有待批准的配对请求：先让客户端发起配对" >&2; exit 1; }
    api -X POST "${BASE}/api/pin" \
        -d "$(jq -cn --arg i "$id" --arg p "$PIN" --arg n "$NAME" \
              '{pairing_id:$i,pin:$p,name:$n}')"
else
    # 旧版 API：只认 pin
    api -X POST "${BASE}/api/pin" \
        -d "{\"pin\":\"${PIN}\",\"name\":\"${NAME}\"}"
fi
```

注意几个在 SSH + bash 场景下容易出错的地方：

- **`curl -u` 会先收 401 再重发**（Sunshine 返回带 `WWW-Authenticate: Basic` 的 401），POST 的 body 会被重放，不影响结果；想一次性带 `Authorization: Basic` 头也行，但 `base64 -w0` 是 GNU 专有，macOS 上要用 `openssl base64 -A`。
- **JSON 别用字符串拼接 pin/name**，用 `jq -cn --arg` 转义（客户端名字里有引号/中文会炸）。
- **`--max-time` 要给够**：`POST /api/pin` 会阻塞等握手，上限是配置里的 `ping_timeout`，默认几十秒。
- **Linux 上的宿主机**：Sunshine 常以 systemd user service 跑，`SSH` 用户与之无关，HTTP 调用不受影响；`sudo` 也不需要。
- **Windows OpenSSH 默认 shell 是 `cmd`/PS 5.1**：PS 5.1 没有 `-SkipCertificateCheck`，要用我上一条给的 `CertificatePolicy` 回调；想省事就 `ssh host pwsh -File ...`（先装 PowerShell 7）。

---

## 什么时候才算“真正全自动配对”

只有一种情况能摆脱人工输入 PIN：**客户端和主机你都能控制，且客户端会把 PIN 打印出来**。

- `moonlight-embedded`（树莓派、部分电视盒）：`moonlight pair <host>` 会在客户端终端直接打印 4 位 PIN。这时可以：

```bash
PIN=$(ssh client 'moonlight pair 192.168.1.10 2>&1 | grep -oE "[0-9]{4}" | head -1') &
sleep 2
ssh host "bash ~/pair.sh $PIN 'Living Room'"
```

- 官方 `moonlight-qt`（Windows/Android/iOS）：PIN 只在 GUI 对话框里，没有 CLI，只能截图 OCR 或人工读出来。

至于“绕过配对、直接往 `sunshine_state.json` 里塞一条设备记录”这类做法是**行不通**的：那条记录里的客户端证书和共享密钥是配对过程中现场协商出来的，手写进去握手一定失败。

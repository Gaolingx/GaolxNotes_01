# VSCode远程SSH硬件密钥登录方案概述

## 🧑‍💻 User

vscode ssh如何使用硬件密钥登录

---
## 🤖 Assistant

这是一个纯配置类问题，我直接给出完整方案。用硬件密钥（YubiKey / FIDO2 安全密钥 / PIV 智能卡）登录 VSCode Remote-SSH，本质是**配好 SSH 客户端 + 告诉 VSCode 用哪个 SSH**。下面分两种主流方式。

## 前提：VSCode 必须用「完整版 OpenSSH」

VSCode Remote-SSH 默认可能调用 Git 自带的旧版 SSH，或 Windows 内置版本，二者对硬件密钥支持不同：

- **FIDO2/U2F（`sk-` 密钥）**：需要 OpenSSH **≥ 8.2**
- **PKCS#11 / PIV**：任意支持 PKCS11 的 OpenSSH 均可

在 `settings.json` 中显式指定：

```json
{
  "remote.SSH.path": "C:\\Windows\\System32\\OpenSSH\\ssh.exe",
  "remote.SSH.configFile": "C:\\Users\\你的用户名\\.ssh\\config"
}
```

> Windows 上推荐用系统自带的 `C:\Windows\System32\OpenSSH\ssh.exe`（可用 `ssh -V` 确认版本）；不要用 Git for Windows 的旧版 `ssh.exe`。Linux/macOS 用 `/usr/bin/ssh`。

---

<details>
<summary><strong>方案一：FIDO2 / U2F 硬件密钥（推荐，最贴合「硬件密钥」）</strong></summary>

### 1. 生成 sk 密钥（插入密钥）

```bash
ssh-keygen -t ed25519-sk -O resident -O verify-required -C "yubikey-$(hostname)"
```

参数说明：

| 参数 | 作用 |
|------|------|
| `-t ed25519-sk` | 生成 FIDO2/U2F 类型的密钥对（私钥在密钥硬件内，不可导出） |
| `-O resident` | 密钥可驻留在硬件中，换电脑用 `ssh-keygen -K` 即可恢复 |
| `-O verify-required` | 每次认证都要 **PIN + 触摸**；不加则只需触摸 |
| `-C` | 注释，方便识别 |

生成后得到 `~/.ssh/id_ed25519_sk` 和 `id_ed25519_sk.pub`。

### 2. 把公钥传到服务器

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_sk.pub user@your-server
```

没有 `ssh-copy-id` 时手动追加到服务器 `~/.ssh/authorized_keys` 即可。

### 3. 配置 `~/.ssh/config`

```sshconfig
Host myserver
    HostName 1.2.3.4
    User myuser
    Port 22
    IdentityFile ~/.ssh/id_ed25519_sk
    IdentitiesOnly yes
```

### 4. 在 VSCode 中连接

1. `F1` → `Remote-SSH: Connect to Host` → 选择 `myserver`
2. 首次会弹出 **PIN 输入框**，然后**触摸密钥**完成认证

### 5. 减少反复弹窗（关键技巧）

VSCode Remote-SSH 可能建立多条 SSH 连接，导致多次 PIN/触摸提示。用**连接复用**把多条连接合并成一条：

```sshconfig
Host myserver
    HostName 1.2.3.4
    User myuser
    IdentityFile ~/.ssh/id_ed25519_sk
    IdentitiesOnly yes
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

> Windows 上 OpenSSH 对 `ControlPath` 支持有限，若报错可改用 `\\.\pipe\...` 命名管道，或干脆接受多次触摸。

</details>

---

<details>
<summary><strong>方案二：PKCS#11 / PIV 智能卡（YubiKey PIV 等）</strong></summary>

这种方式把密钥放进硬件槽位，SSH 通过 PKCS#11 动态库读取，无需 `sk-` 类型私钥文件。

### 1. 安装中间件

- YubiKey：安装 **YubiKey PIV Manager / ykcs11**
- 其他智能卡：OpenSC（`opensc-pkcs11.so`）

常见库路径：

| 平台 | 库路径 |
|------|--------|
| Linux | `/usr/local/lib/libykcs11.so` 或 `/usr/lib/x86_64-linux-gnu/opensc-pkcs11.so` |
| macOS | `/usr/local/lib/libykcs11.dylib` |
| Windows | `C:\Program Files\Yubico\Yubikey PIV\ykcs11.dll` |

### 2. 用 `ssh-keygen` 查看/导出公钥

```bash
# 列出硬件中的公钥
ssh-keygen -D /usr/local/lib/libykcs11.so

# 导出公钥文件（用于上传服务器）
ssh-keygen -D /usr/local/lib/libykcs11.so > ~/.ssh/id_piv.pub
```

把导出的公钥追加到服务器 `authorized_keys`。

### 3. 配置 `~/.ssh/config`

```sshconfig
Host myserver
    HostName 1.2.3.4
    User myuser
    PKCS11Provider /usr/local/lib/libykcs11.so
    IdentitiesOnly yes
```

Windows 示例：

```sshconfig
Host myserver
    HostName 1.2.3.4
    User myuser
    PKCS11Provider "C:\\Program Files\\Yubico\\Yubikey PIV\\ykcs11.dll"
    IdentitiesOnly yes
```

之后在 VSCode 中正常连接即可，认证时触摸密钥（PIV 通常还会要求 PIN）。

</details>

---

<details>
<summary><strong>方案三：通过 ssh-agent 统一管理（多机/免配置文件）</strong></summary>

适合不想每次改 config 的场景，把硬件密钥交给 agent：

**Linux/macOS：**
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_sk     # 提示触摸
```

**FIDO2 常驻密钥从硬件加载：**
```bash
ssh-add -K    # 从 token 加载 resident key
```

**Windows：**
1. 打开「服务」→ 启动 `OpenSSH Authentication Agent`
2. `ssh-add` 加载密钥（VSCode 会自动继承 `SSH_AUTH_SOCK`）

之后 `~/.ssh/config` 无需写 `IdentityFile`，agent 会自动提供密钥。

</details>

---

<details>
<summary><strong>常见问题与排查</strong></summary>

| 现象 | 原因 / 解决 |
|------|-------------|
| 连接时报 `no such identity` 或找不到密钥 | VSCode 用了旧版 SSH，设置 `remote.SSH.path` 指向新版 |
| 每次连接弹多次 PIN/触摸 | 开启 `ControlMaster` 连接复用（见方案一第 5 步） |
| `key type ssh-ed25519-sk not supported` | 客户端或**服务器**OpenSSH 低于 8.2，升级两端 |
| Windows 下 WSL 里用了密钥但连不上 | WSL 无法直接访问 USB；用系统 Windows OpenSSH，或用 `usbipd` 透传 |
| PKCS11 报库找不到 | 路径写错或未安装中间件；用绝对路径并加引号 |
| 认证成功但 VSCode 仍显示连不上 | 多为防火墙/监听方式，试 `"remote.SSH.remoteServerListenOnSocket": true` |
| 想调试细节 | 终端执行 `ssh -v user@host`，看密钥交换与认证流程 |

**调试小贴士**：先用命令行 `ssh myserver` 验证硬件密钥能登录，能登上再交给 VSCode，可快速定位是 SSH 层还是 VSCode 层的问题。

</details>

---

**一句话总结**：生成 `sk-` 密钥（或配置 PKCS#11）→ 公钥上传服务器 → 在 `~/.ssh/config` 里 `IdentityFile` / `PKCS11Provider` → VSCode 设置 `remote.SSH.path` 指向新版 SSH → 用 `ControlMaster` 减少重复触摸。

如果你告诉我**操作系统**和**密钥型号**（如 YubiKey 5 NFC / 某品牌 FIDO2），我可以给出更精确的逐条命令。

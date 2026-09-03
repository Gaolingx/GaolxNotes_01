# SSH公钥免密登录

### 🧑‍💻 User

linux如何注册public key免密登录ssh？

---

### 🤖 Assistant

# Linux SSH 公钥免密登录配置指南

## 三步完成免密登录

### 1. 生成密钥对（如果没有）

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

一路回车即可（可设密码也可留空）。生成的文件位于：

- **私钥**: `~/.ssh/id_ed25519` ⚠️ 自己保留，切勿外传
- **公钥**: `~/.ssh/id_ed25519.pub` → 放到远程服务器上

> 💡 如果远程服务器较老不支持 Ed25519，可改用 `-t rsa -b 4096`。

---

### 2. 把公钥注册到远程服务器

#### 方法 A：`ssh-copy-id`（最简单，推荐）

```bash
ssh-copy-id user@remote_host
```

它会自动把公钥追加到远程服务器的 `~/.ssh/authorized_keys` 中。

#### 方法 B：手动复制

```bash
cat ~/.ssh/id_ed25519.pub | ssh user@remote_host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

---

### 3. 设置正确的权限（远程服务器端）

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

> ⚠️ 权限不对会导致免密登录失败，SSH 对权限非常严格。

---

### 4. 测试登录

```bash
ssh user@remote_host
```

无需输入密码即表示配置成功 ✅

---

## 故障排查

| 问题 | 解决 |
|------|------|
| 仍然要求密码 | `ssh -v user@remote_host` 查看详细日志 |
| `Permission denied (publickey)` | 检查远程 `~/.ssh/authorized_keys` 权限是否为 `600` |
| `Bad owner or permissions` | 远程 `~/.ssh` 权限必须是 `700`，家目录不能是 `777` |
| 公钥已添加但无效 | 检查远程 `/etc/ssh/sshd_config` 中 `PubkeyAuthentication yes` 是否开启 |

### 查看详细调试信息

```bash
ssh -vvv user@remote_host
```

---

## 附加优化：配置 `~/.ssh/config`

编辑本地的 `~/.ssh/config`：

```
Host myserver
    HostName 192.168.1.100
    User root
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

之后只需 `ssh myserver` 就能免密登录，简洁方便。
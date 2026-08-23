# SSH公钥创建教程

### 🧑‍💻 User

安装了openssh，Windows ssh publickey创建怎么做？

---

### 🤖 Assistant

在 Windows 上安装好 OpenSSH（通常是 OpenSSH 客户端）后，创建 SSH 公钥/私钥对主要使用 `ssh-keygen` 命令。以下是完整步骤：

---

## 1. 打开终端
- 按 `Win + R` 输入 `powershell` 或 `cmd` 打开命令行。
- 推荐使用 **PowerShell**（支持更多便利命令）。

---

## 2. 生成密钥对（推荐 ed25519，兼容性优先可选 RSA）
```powershell
# 默认在当前用户的 .ssh 目录生成 id_ed25519 和 id_ed25519.pub
ssh-keygen -t ed25519 -C "your_email@example.com"
```
> `-C` 是注释，常填邮箱，方便识别。

如果你是老旧系统或需要兼容，可用 RSA 4096 位：
```powershell
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

**交互提示：**
- `Enter file in which to save the key` → 直接回车使用默认路径（`C:\Users\你的用户名\.ssh\id_ed25519`）。
- `Enter passphrase` → 可直接回车留空（无密码），或输入密码保护私钥（推荐设置）。
- 确认密码后生成完毕。

---

## 3. 查看并复制公钥内容

**查看公钥文件内容：**
```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```
或
```powershell
type %USERPROFILE%\.ssh\id_ed25519.pub
```

**直接复制到剪贴板（二选一）：**
- **PowerShell：**
  ```powershell
  Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard
  ```
- **CMD：**
  ```cmd
  type %USERPROFILE%\.ssh\id_ed25519.pub | clip
  ```

---

## 4. 将公钥部署到远程服务器（实现免密登录）

1. 登录远程服务器（初次可能需要密码）。
2. 将**公钥内容**追加到服务器用户目录下的 `~/.ssh/authorized_keys` 文件中。
   - 如果服务器是 Linux / macOS，可直接用 `ssh-copy-id`（需在 Windows 另行安装或手动操作）。
   - 手动做法（在远程终端上执行）：
     ```bash
     mkdir -p ~/.ssh
     echo "这里粘贴你的公钥内容" >> ~/.ssh/authorized_keys
     chmod 700 ~/.ssh
     chmod 600 ~/.ssh/authorized_keys
     ```
3. 也可在 **Windows 端** 使用 `scp` 直接上传（略高级）：
   ```powershell
   type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh user@remote_host "cat >> ~/.ssh/authorized_keys"
   ```

---

## 5. 测试连接
```powershell
ssh user@remote_host
```
若无需输入密码即登录成功，说明配置完成。

---

### 常见问题
- **找不到 ssh-keygen？**
  确保 OpenSSH 客户端已安装：
  `设置 → 应用 → 可选功能 → 添加功能 → OpenSSH 客户端`，或直接 `Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Client*'` 检查状态。
- **权限问题？**
  Windows 下 `.ssh` 目录通常不会被过分限制，但若连接报权限错，可尝试禁用当前用户对该私钥文件的继承权限，并仅保留当前用户的完全控制（右键 → 属性 → 安全 → 高级）。

如果还需要设置 Windows 作为 SSH 服务器接受公钥登录，请补充说明，我会提供对应配置。
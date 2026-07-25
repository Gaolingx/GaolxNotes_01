# HuggingFace Git认证错误解决

### 🧑‍💻 User

huggingface使用git的时候出现认证错误怎么办？

---

### 🤖 Assistant

在 Hugging Face 中使用 Git 时遇到认证错误，通常是因为 Hugging Face **不支持使用账户密码进行 Git 操作**，而需要使用 **Access Token** 或 **SSH 密钥**。

以下是解决此问题的几种常见方法：

### 方法一：使用 Access Token (HTTPS 方式 - 最简单)

如果你是通过 `https://huggingface.co/...` 克隆或推送代码，你需要使用具有读写权限的 Access Token 来替代密码。

1.  **生成 Access Token**：
    *   登录 Hugging Face 官网。
    *   点击右上角头像 -> `Settings`。
    *   在左侧菜单选择 `Access Tokens`。
    *   点击 `Create new token`，建议选择 `Write` 权限（如果只需读取，选 `Read` 即可）。
    *   复制生成的 Token（格式通常为 `hf_...`）。

2.  **在 Git 中使用 Token**：
    当你执行 `git push` 或 `git clone` 时，Git 会提示输入用户名和密码。
    *   **Username**: 输入你的 Hugging Face 用户名。
    *   **Password**: 粘贴刚才复制的 Access Token（**不要**输入登录密码）。

3.  **避免每次输入（可选）**：
    如果你想将 Token 保存到远程仓库的 URL 中，可以使用以下命令修改 `.git/config`：
    ```bash
    git remote set-url origin https://<你的用户名>:<你的Token>@huggingface.co/<你的用户名>/<仓库名>
    ```
    *(注意：这种方法会将 Token 明文保存在本地 git 配置文件中，请注意安全)*

---

### 方法二：使用 SSH 密钥 (推荐，更安全)

如果你希望避免频繁输入 Token，配置 SSH 密钥是最佳方案。

1.  **生成 SSH 密钥**（如果已有可跳过）：
    ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ```
    一路回车即可（默认保存在 `~/.ssh/id_ed25519`）。

2.  **将公钥添加到 Hugging Face**：
    *   复制公钥内容：
        ```bash
        cat ~/.ssh/id_ed25519.pub
        ```
    *   登录 Hugging Face -> `Settings` -> `SSH and GPG keys`。
    *   点击 `Add SSH key`，粘贴公钥内容并保存。

3.  **测试 SSH 连接**：
    ```bash
    ssh -T git@hf.co
    ```
    如果提示 `Hi <username>, successfully authenticated` 则表示成功。

4.  **将仓库远程地址改为 SSH 格式**：
    进入你的本地仓库目录，修改 remote URL：
    ```bash
    git remote set-url origin git@hf.co:<你的用户名>/<仓库名>
    ```
    之后即可直接 `git push` 无需输入密码。

---

<details>
<summary>🔧 其他常见问题排查 (点击展开)</summary>

### 1. 清除 Windows/Mac 缓存的错误凭据
如果你的系统记住了旧的错误密码，即使输入了正确的 Token 也可能报错。你需要清除凭据管理器中的记录。

*   **Windows**:
    打开控制面板 -> 凭据管理器 -> Windows 凭据 -> 找到 `git:https://huggingface.co` -> 删除。
*   **Mac**:
    打开“钥匙串访问” -> 搜索 `huggingface.co` -> 删除相关条目。
*   **Linux / Git 命令行**:
    ```bash
    git credential-cache exit
    # 或者如果使用 store
    git config --global --unset credential.helper
    ```

### 2. Token 权限不足
如果你可以 `clone`（拉取）但无法 `push`（推送），请检查你的 Access Token 是否具有 `Write` 权限。

### 3. 使用 Hugging Face CLI (可选)
你也可以使用官方的 CLI 工具来配置认证，它会自动处理 Token 缓存：
```bash
pip install huggingface_hub
huggingface-cli login
```
执行后输入你的 Access Token 即可。

</details>
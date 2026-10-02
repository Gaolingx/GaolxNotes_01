# gh授权访问私仓

## 🧑‍💻 User

linux gh如何授权访问github私有git仓库？

---
## 🤖 Assistant

# 使用 `gh` (GitHub CLI) 授权访问 GitHub 私有仓库

在 Linux 上，`gh` CLI 可以帮你完成认证，并把它配置成 Git 的凭据助手，从而克隆/推送私有仓库。下面分几种场景说明。

---

## 一、安装 gh

<details>
<summary>各发行版安装命令（点开查看）</summary>

**Debian / Ubuntu**（官方源）：
```bash
(type -p wget >/dev/null || sudo apt install wget -y) \
  && sudo mkdir -p -m 755 /etc/apt/keyrings \
  && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && sudo apt update && sudo apt install gh -y
```

**Fedora / RHEL / CentOS**：
```bash
sudo dnf install gh -y
```

**Arch**：
```bash
sudo pacman -S github-cli
```

</details>

---

## 二、交互式登录（推荐新手）

```bash
gh auth login
```

按提示选择：

| 问题 | 选择 |
|------|------|
| What account do you want to log into? | **GitHub.com** |
| What is your preferred protocol for Git operations? | **HTTPS** 或 **SSH** |
| Authenticate Git with your GitHub credentials? | **Yes** ← 关键，让 gh 配置 git 凭据助手 |
| How would you like to authenticate? | **Login with a web browser**（或粘贴 Token） |

浏览器方式会显示一个 **one-time code**，在任意设备打开 `https://github.com/login/device` 输入即可。

---

## 三、非交互式 / 服务器无浏览器环境

适用于纯命令行服务器。先创建一个 **Personal Access Token (PAT)**（`Settings → Developer settings → Personal access tokens`），**必须包含 `repo` scope** 才能访问私有仓库。

**HTTPS 方式：**
```bash
echo "ghp_xxxxxxxxxxxxxxxx" | gh auth login --with-token
# 或者：
gh auth login --git-protocol https --hostname github.com
```

**通过环境变量（适合 CI，不落盘）：**
```bash
export GH_TOKEN=ghp_xxxxxxxxxxxxxxxx
gh repo clone myorg/private-repo
```

> 优先级：`GH_TOKEN` > `GITHUB_TOKEN`。设置了环境变量后，`gh` 和配置好的 git 凭据助手都会用它。

---

## 四、让 git 使用 gh 的凭据（核心步骤）

即使登录成功，普通 `git` 命令也可能不生效，需要让 gh 配置 Git 凭据助手：

```bash
gh auth setup-git
```

这会在你的 `~/.gitconfig` 写入类似内容：

```ini
[credential "https://github.com"]
    helper =
    helper = !/usr/bin/gh auth git-credential
```

之后即可正常操作私有仓库：

```bash
gh repo clone myorg/private-repo
# 或者普通 git 命令：
git clone https://github.com/myorg/private-repo.git
```

---

## 五、常用验证与排查命令

```bash
gh auth status                    # 查看当前登录状态、协议、scope
gh auth token                     # 打印当前 token（可喂给脚本）
gh auth refresh -s repo           # 追加/刷新 scope（如原来缺 repo）
gh auth logout                    # 退出登录
```

**检查私有仓库访问：**
```bash
gh repo view myorg/private-repo   # 能打开说明权限正常
```

---

## 六、SSH 方式（可选）

若在 `gh auth login` 时选了 SSH，gh 会：
1. 生成/上传 SSH 公钥到 GitHub；
2. 在 `~/.ssh/config` 写入别名 `github.com`。

测试：
```bash
ssh -T git@github.com
# Hi <username>! You've successfully authenticated...
```

克隆时使用 SSH URL：
```bash
git clone git@github.com:myorg/private-repo.git
```

---

## 七、CI / 脚本最佳实践

```bash
# 临时注入，不污染全局配置
export GH_TOKEN="$MY_PAT"
git config --global credential.helper '!gh auth git-credential'
git clone https://github.com/myorg/private-repo.git
```

- **不要把 PAT 写进代码仓库**；用环境变量或 secret 管理。
- 服务器场景建议使用 **fine-grained PAT**，只授予目标仓库的 `Contents: Read`（按需 `Write`）权限，最小化授权。
- 若公司使用 **GitHub Enterprise Server**，加 `--hostname git.your-company.com`。

---

## 常见问题速查

| 现象 | 原因 / 解决 |
|------|------|
| `git clone` 提示 403 / 要求输密码 | 忘了 `gh auth setup-git`，或 token 缺 `repo` scope |
| `gh auth status` 显示缺 repo scope | `gh auth refresh -s repo` |
| CI 里 gh 不认登录 | 用 `GH_TOKEN` 环境变量，而非 `gh auth login` |
| 想同时登录多个账号 | `gh auth login --hostname ...`，用 `gh auth switch` 切换 |

---

一句话总结：**`gh auth login` 完成认证 → 选择 “Authenticate Git” = Yes → 必要时手动跑 `gh auth setup-git`**，之后私有仓库的 clone/push 就能无缝走 gh 的凭据了。

需要我针对某个具体发行版或 CI 平台（如 GitHub Actions、Jenkins、GitLab Runner）展开吗？

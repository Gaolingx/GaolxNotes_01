# linux如何以sudo方式执行标准用户下的uv run》

## 🧑‍💻 User

linux如何以sudo方式执行标准用户下的uv run？

---
## 🧑‍💻 User

linux如何以sudo方式执行标准用户下的uv run》

---
## 😀 默认助手 (7)

# Linux 下让 `sudo uv run` 正常工作

## 先说结论

`sudo uv run ...` 失败的根因**几乎总是这三个**，而**不是** uv 本身：

| 坑 | 原因 |
|---|---|
| `sudo: uv: command not found` | uv 装在 `~/.local/bin` 或 `~/.cargo/bin`，root 的 `secure_path` 里没有 |
| 缓存/Python 跑到 `/root` 下 | `sudo` 默认（`env_reset`）把 `HOME` 重设为 `/root`，`UV_CACHE_DIR`、`uv python` 也跟着跑偏 |
| 项目目录被 root 污染 | root 身份写 `.venv/`、`uv.lock`、`__pycache__`，之后普通用户跑就 `Permission denied` |

最通用的写法（PATH 走用户环境，缓存隔离到 root 自己的目录）：

```bash
sudo env PATH="$PATH" UV_CACHE_DIR=/var/cache/uv \
  uv run --frozen --no-sync --no-python-downloads <你的命令>
```

`env PATH="$PATH"` 由**你的 shell** 先把 `$PATH` 展开，再交给 root，所以不受 `SETENV` 限制，也不需要改 sudoers。

---

## 方案速查

| 场景 | 推荐命令 |
|---|---|
| 临时跑一次，最省事 | `sudo env PATH="$PATH" uv run ...` |
| uv 不在用户 PATH 但你知道位置 | `sudo "$(command -v uv)" run ...` |
| 长期用，想让 `sudo uv` 直接可用 | `sudo install -m 0755 "$(command -v uv)" /usr/local/bin/uv` |
| 只有**被启动的程序**需要 root，uv 本身不需要 | `sudo -v; uv run sudo -n ...` |
| 以另一个普通用户身份运行 | `sudo -u alice -H bash -lc 'cd ~/proj && uv run ...'` |
| 需要 `sudo` 保留完整环境 | `sudo -E ...`（需 sudoers 允许 `SETENV`，常被拒） |

---

## 方案 1：`sudo env PATH=...`（首选）

```bash
# 最小可用
sudo env PATH="$PATH" uv run python -c "import sys; print(sys.executable)"

# 生产级：不碰用户的缓存、不动 .venv / uv.lock、不下载 Python
sudo env PATH="$PATH" \
  UV_CACHE_DIR=/var/cache/uv-root \
  uv run --frozen --no-sync --no-python-downloads \
  ./scripts/need_root.sh
```

关键参数说明：

- `--frozen --no-sync`：**不修改** `.venv` 和 `uv.lock`，直接用已存在的环境。这是避免「跑完 sudo 后普通用户再跑就报权限错」最有效的一招。
- `--no-python-downloads`（或 `UV_PYTHON_DOWNLOADS=never`）：防止 root 往 `/root/.local/share/uv` 里下 Python。
- `UV_CACHE_DIR=...`：**别**指向 `~/.cache/uv`，否则 root 会在用户缓存里创建 root 属主文件，把 uv 缓存搞脏。

如果 uv 压根不在 PATH 里：

```bash
sudo "$(command -v uv)" run ...
# 或
sudo env PATH="$HOME/.local/bin:$PATH" uv run ...
```

## 方案 2：把 uv 装到系统路径

`sudo` 的 `secure_path`（Debian/Ubuntu 默认含 `/usr/local/bin`）会优先于用户 PATH，所以放进 `/usr/local/bin` 就一劳永逸：

```bash
sudo install -m 0755 "$(command -v uv)" /usr/local/bin/uv
sudo uv run ...          # 现在能找到 uv 了
```

注意：这只是解决「找不到 uv」，`HOME=/root` 和 `.venv` 污染问题**依然存在**，仍需 `UV_CACHE_DIR` / `--frozen --no-sync` 配合。

<details>
<summary><b>方案 3：改 sudoers（让 <code>sudo uv</code> 自然可用）</b></summary>

用 `sudo visudo -f /etc/sudoers.d/uv` 新建：

```
# 让用户 PATH 生效（等价于每次自带 env PATH=...）
Defaults  env_keep += "PATH"

# 或者把你的 bin 目录塞进 secure_path
Defaults  secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/you/.local/bin"
```

只想给某个用户开恩（更细粒度）：

```
Defaults:you  env_keep += "PATH"
you ALL=(root) NOPASSWD: /usr/local/bin/uv run
```

最后一条只允许 `uv run`，不允许 `uv python install` 之类的其他子命令。改完务必 `sudo visudo -c` 校验，写错会锁死 sudo。

</details>

## 方案 4：反过来做——不要 `sudo uv`，只提权必要的子进程

大多数「需要 root」的需求其实是「uv 启动的那个程序需要 root」。让 uv 保持普通用户身份，缓存、`.venv`、`uv.lock` 全都归你，只在最后一步提权：

```bash
# 先预热 sudo 凭证（默认 5~15 分钟有效）
sudo -v
uv run sudo -n ./scripts/bind_port_80.sh     # -n 非交互，走已缓存凭证

# 没有 tty 的场景（CI/脚本）用 SUDO_ASKPASS
export SUDO_ASKPASS=/usr/bin/ssh-askpass
uv run sudo -A ./script.sh
```

这是**最干净的方案**：uv 的一切副作用都留在标准用户身份下。

---

## 权限善后与排查

<details>
<summary><b>已经把项目搞成 root 属主了？</b></summary>

```bash
sudo chown -R "${SUDO_USER:-$USER}:${SUDO_USER:-$USER}" .venv uv.lock .pytest_cache
```

预防手段是显式指定环境目录，让 root 的 venv 落到别处：

```bash
sudo env PATH="$PATH" UV_PROJECT_ENVIRONMENT=/tmp/uv-root-venv uv run ...
```

</details>

<details>
<summary><b>常见报错对照表</b></summary>

| 报错 | 原因 | 解法 |
|---|---|---|
| `sudo: uv: command not found` | root 的 PATH 里没有 uv | `sudo env PATH="$PATH" uv ...` 或装到 `/usr/local/bin` |
| `Failed to create virtualenv` / `Permission denied` | root 要写用户的 `.venv` | `--frozen --no-sync`，或 `UV_PROJECT_ENVIRONMENT=/tmp/...` |
| `No interpreter found for Python 3.x` | root 的 uv 看不到用户的 Python | 加 `--python-preference only-system`，或先 `sudo uv python install 3.x` |
| uv 缓存诡异损坏 / hash 不匹配 | 用户与 root 混用同一 `UV_CACHE_DIR` | 给 root 单独设 `UV_CACHE_DIR=/var/cache/uv-root` |
| `error: Project not found` | 用了 `sudo -i`（切到 `/root`） | 用普通 `sudo`（保留 cwd），或加 `--directory /path/to/proj` |
| `sorry, you are not allowed to preserve the environment` | `sudo -E` 需要 `SETENV` 权限 | 改用 `sudo env VAR=...` |

</details>

---

## 一句话建议

**能用方案 4（`uv run sudo -n`）就别用 `sudo uv run`。** 如果确实必须整体提权，记住三件套：`env PATH="$PATH"` + 独立 `UV_CACHE_DIR` + `--frozen --no-sync`，跑完再 `chown` 收尾。

---
## 🧑‍💻 User

```
sudo env PATH="$PATH" UV_CACHE_DIR=/var/cache/uv \
  uv run --frozen --no-sync --no-python-downloads <你的命令>
```
这句话做了什么，有没有动系统级环境变量

---
## 🧑‍💻 User

sudo env PATH="$PATH" UV_CACHE_DIR=/var/cache/uv这句话做了什么

---
## 🧑‍💻 User

sudo env PATH="$PATH" UV_CACHE_DIR=/var/cache/uv这句话做了什么，有没有动系统级环境变量

---
## 😀 默认助手 (7)

# 一句话回答

**没有动任何系统级环境变量。** 三个赋值（`PATH`、`UV_CACHE_DIR`）都只注入到这一次命令的进程树里，命令退出即消失。你的登录 shell、`/etc/environment`、`/etc/profile`、`~/.bashrc`、`/etc/sudoers`、systemd 一概没碰。

唯一「持久化」的副作用是**文件系统**上多出一个 `/var/cache/uv` 目录（uv 的缓存内容），与「环境变量」无关，`rm -rf` 就能清掉。

---

## 逐段拆解

```bash
sudo env PATH="$PATH" UV_CACHE_DIR=/var/cache/uv \
  uv run --frozen --no-sync --no-python-downloads <你的命令>
```

| 片段 | 由谁处理 | 作用 |
|---|---|---|
| `sudo` | shell | 以 root 启动下一条命令，默认 `env_reset` 会清空环境、把 `HOME` 设成 `/root` |
| `env` | `/usr/bin/env`（sudo 经 `secure_path` 找到） | **关键技巧**：它就是一个普通可执行文件，所以可以带 `NAME=VALUE` 参数。它把这些赋值加进自己的 environ，再 `execvp` 下一个命令 |
| `PATH="$PATH"` | **你的 shell 先展开**，再把字面量 `PATH=/home/you/.local/bin:...` 当参数传给 sudo | root 拿到你的 PATH，于是能找到 `~/.local/bin/uv`；`env` 用的是改过的 PATH 去查找 `uv`，所以查找也生效 |
| `UV_CACHE_DIR=/var/cache/uv` | uv 读取 | 缓存写到 `/var/cache`，不污染 `~/.cache/uv`（否则 root 会在用户缓存里丢一堆 root 属主文件） |
| `uv run` | uv | 运行项目环境里的命令 |
| `--frozen` | uv | 不更新 `uv.lock`，锁文件与 `pyproject.toml` 不一致就直接报错 |
| `--no-sync` | uv | 不同步/不改动虚拟环境，只复用已有的 `.venv`（不存在会报错而不是新建） |
| `--no-python-downloads` | uv | 禁止自动下载 Python，避免 root 往 `/root/.local/share/uv` 里塞解释器 |

### 为什么不用 `sudo -E`

`sudo -E` 需要 sudoers 里的 `SETENV` 权限，Debian/Ubuntu 默认的 `%sudo ALL=(ALL:ALL) ALL` 常常直接拒绝：`sorry, you are not allowed to preserve the environment`。而 `sudo env PATH=...` 只是「让 `env` 这个普通程序收到参数」，**任何配置下都合法**，这是它存在的理由。

---

## 作用域边界：三层

<details>
<summary><b>第 1 层：完全没有写任何配置文件</b></summary>

以下文件一律未被读取也未被修改：

- `/etc/environment`、`/etc/profile`、`/etc/profile.d/*`
- `~/.bashrc`、`~/.profile`、`~/.zshenv`
- `/etc/sudoers`、`/etc/sudoers.d/*`
- systemd 的 `Environment=` / `environment.d/`
- 你当前 shell 自己的 `$PATH`（只是被**读**了一次用于展开，没有改）

验证：关掉终端重开，`echo $PATH` 完全不变。

</details>

<details>
<summary><b>第 2 层：只在这次进程树内有效</b></summary>

```
你的 shell
  └─ sudo            ← env_reset 生效，环境已被清空
       └─ env        ← 在这里注入 PATH / UV_CACHE_DIR
            └─ uv    ← 看得见这两个变量
                 └─ python / 你的脚本  ← 继承，也看得见
```

命令一结束，这棵树消失，变量随之消失。兄弟进程、父 shell 都感知不到。

自检：

```bash
sudo printenv UV_CACHE_DIR            # 空 → 说明没注入
sudo env PATH="$PATH" UV_CACHE_DIR=/x printenv UV_CACHE_DIR   # 输出 /x
sh -c 'sudo env PATH="$PATH" sh -c "echo \$PATH"' ; echo "$PATH"  # 对比：外层没变
```

</details>

<details>
<summary><b>第 3 层：真正会留下痕迹的东西</b></summary>

| 痕迹 | 位置 | 是否由这条命令造成 | 清理 |
|---|---|---|---|
| uv 缓存 | `/var/cache/uv` | ✅ 是（首跑自动创建） | `sudo rm -rf /var/cache/uv` |
| sudo 凭证时间戳 | `/run/sudo/ts/<user>` | ✅ 是（任何 sudo 都有） | 自动过期，或 `sudo -k` |
| 运行时产物（`__pycache__`、日志、输出文件） | 项目目录 / cwd | ⚠️ 你的**被启动程序**写的，属主是 root | `sudo chown -R "$USER:$USER" ...` |
| `.venv` / `uv.lock` | 项目目录 | ❌ 被 `--frozen --no-sync` 挡住了 | — |

也就是说：`--frozen --no-sync` 防的是 **uv 自己**改环境，防不了**你跑的那个程序**在项目里留下 root 属主文件。这是这条命令唯一的实际风险点。

</details>

---

## 想更干净的两个小改进

**① 预建缓存目录**（避免首跑时 root 顺手建一个，也便于事后按用户身份删除）：

```bash
sudo install -d -o "$USER" -g "$USER" -m 0755 /var/cache/uv
```

目录属主给普通用户后，root 写进去的文件虽然仍是 root 属主，但**你作为目录 owner 可以直接删**，不用再 sudo 收尾。

**② 注意跨文件系统的性能代价**：uv 缓存到 `.venv` 的复用依赖硬链接/reflink。如果 `/var/cache` 与 `/home` 不在同一分区（`df /var/cache /home` 看挂载点），uv 只能退化成复制，磁盘占用和安装时间都会变差。这种场景改用同分区路径更合适，例如 `UV_CACHE_DIR=/home/you/.cache/uv-root`——代价是 root 会写进你的家目录，需要接受或事后 chown 取舍。

---

## 一句话总结

这条命令的本质是：**用 `env` 绕过 sudo 的环境重置，把两个变量一次性塞给 root 进程**——纯进程级、零配置文件改动、退出即失效；真正需要留意的只有 `/var/cache/uv` 这个磁盘目录和你的程序可能写出的 root 属主文件。

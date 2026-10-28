# tmux默认sock在tmp的原因与处理

## 🧑‍💻 User

```
newuser005@jk01:~$ tmux ls
no server running on /tmp/tmux-1007/default
newuser005@jk01:~$ 
```
为什么我的tmux指向了tmp目录

---
## 😀 默认助手 (7)

## 一句话回答

这不是"你的 tmux 被配置错了"，而是 **tmux 的默认行为**：它把 server 的 Unix socket 放在 `$TMUX_TMPDIR`，没设置时用 `$TMPDIR`，再没有就用 `/tmp`，然后在后面拼上 `/tmux-<UID>`。

```
/tmp / tmux-1007 / default
 │      │          └─ socket 名字（默认 default，对应 tmux -L default）
 │      └─ 你的数字 UID（newuser005 = 1007）
 └─ 基准目录（TMUX_TMPDIR → TMPDIR → /tmp）
```

所以 `no server running on /tmp/tmux-1007/default` 只说明"这个 socket 上当前没有 tmux server 在跑"（你可能根本没启动过，或者你在另一个环境/另一个 socket 名下启动过）。它**不是错误**，`tmux ls` 在这种情况下退出码为 1。

---

## 为什么偏偏是 `/tmp`？

| 原因 | 说明 |
|---|---|
| Unix socket 路径长度硬限制 | Linux 上 `sockaddr_un.sun_path` 只有 108 字节，`/tmp/tmux-1007/default` 足够短，而长 `$HOME` 路径有溢出风险 |
| `/tmp` 人人可写，需要隔离 | 所以 tmux 给每个 UID 建一个 `tmux-<uid>` 子目录，权限 `0700`，属主必须是该用户；属主/权限不对时 tmux 会**拒绝启动**，防止别人抢注 socket（hijack） |
| 历史设计早于 XDG 规范 | tmux 不用 `~/.config` / `$XDG_RUNTIME_DIR`，只认 `TMUX_TMPDIR` 和 `TMPDIR` |

---

## 先确认你现在到底在用哪个 socket

<details>
<summary>诊断命令（点开）</summary>

```bash
# 1. 看环境变量——这是 90% 的问题根源
echo "TMUX_TMPDIR=[$TMUX_TMPDIR]  TMPDIR=[$TMPDIR]  UID=$(id -u)"
env | grep -i tmp

# 2. 看 socket 目录
ls -ld /tmp/tmux-* ; ls -l /tmp/tmux-$(id -u)/

# 3. 有 server 在跑吗（即使 tmux ls 说没有）
ps -ef | grep -v grep | grep tmux
ss -x | grep tmux          # 或 lsof -U | grep tmux

# 4. 如果有 server，直接问它
tmux display-message -p '#{socket_path}'      # tmux 3.x
```

</details>

**最常见的坑**：服务器第一次启动时用的是**第一个 client 的环境变量**。如果两个登录会话里 `TMPDIR` 不一样（比如某个 `/etc/profile.d/*.sh` 或 `pam_systemd` 改了它），你就在找 `/tmp/tmux-1007/default`，而 server 其实躺在 `/run/user/1007/tmux-1007/default` —— 表现就是"明明有 tmux 在跑，`tmux ls` 却说 no server"。用 `sudo tmux` 更明显：那会去 `/tmp/tmux-0/default`。

---

## 想换掉它，三种做法

### 1. 最省事：设置 `TMUX_TMPDIR`（推荐给长期驻留的 session）

注意 tmux 会**自己拼 `/tmux-<uid>`**，所以你要给的是**父目录**，且该父目录必须已存在：

```bash
mkdir -p ~/.local/run
chmod 700 ~/.local/run
export TMUX_TMPDIR="$HOME/.local/run"     # 实际 socket: ~/.local/run/tmux-1007/default
```

写进 `~/.bashrc` / `~/.profile`（要保证 SSH 登录、cron、systemd 各路径都生效，最好放 `~/.profile` 或 `environment.d`）。

### 2. 用 `$XDG_RUNTIME_DIR`（干净，但**登出即消失**）

```bash
export TMUX_TMPDIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
```

`/run/user/1007`（即 tmpfs、`0700`、systemd 管理）很合适，但 systemd 会在**该用户最后一个会话结束时清空它**，tmux session 随之被干掉。想让它撑过登出就别用这个。

### 3. 完全绕开：`-S` 显式指定 socket

```bash
tmux -S ~/.tmux.sock new -s work     # 启动
tmux -S ~/.tmux.sock attach          # 连接（注意：-S 时路径必须完全一致）
```

区别：`-L 名字` 是"改 socket 文件名"（仍在 `<base>/tmux-<uid>/` 下），`-S 路径` 是"直接指定完整路径"，`-S` 比 `TMUX_TMPDIR` 优先级更高。

---

## 顺带提醒：`/tmp` 里的 session 可能会被"清理"掉

- 重启后 `/tmp`（如果是 tmpfs）被清空 → session 全没，这跟 tmux 无关。
- 部分发行版/清理工具（`systemd-tmpfiles --clean`、`tmpwatch`、`tmpreaper`）会动 `/tmp` 里的老旧内容，有可能连带删掉 `/tmp/tmux-*`。可检查：

```bash
cat /usr/lib/tmpfiles.d/tmp.conf /etc/tmpfiles.d/*.conf 2>/dev/null
systemctl list-timers | grep -i tmp
```

（具体哪些发行版默认会删，请以你机器上的实际配置为准，别照搬结论。）

---

## 一个稳妥的落地配置

```bash
# ~/.profile
export TMUX_TMPDIR="$HOME/.local/run"
mkdir -p "$TMUX_TMPDIR"
```

想开机自动拉起（systemd user 服务，需在 `~/.config/systemd/user/` 下）：

```ini
# ~/.config/systemd/user/tmux.service
[Unit]
Description=tmux server
[Service]
Environment=TMUX_TMPDIR=%h/.local/run
ExecStart=/usr/bin/tmux new-session -d -s main
ExecStop=/usr/bin/tmux kill-server
[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now tmux.service
systemctl --user show-environment | grep TMUX_TMPDIR   # 确认环境一致
```

---

**结论**：`/tmp/tmux-1007/default` 是 tmux 的正常默认值；它出现只是因为没设 `TMUX_TMPDIR`/`TMPDIR`。想搬家就设 `TMUX_TMPDIR=$HOME/.local/run`（记住给父目录、别给最终目录），并保证**所有启动 tmux 的入口环境变量一致**——不一致正是"找不到 server"的头号原因。

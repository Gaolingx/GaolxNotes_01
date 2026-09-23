# 进程移入tmux

## 🧑‍💻 User

如何把运行中的进程转移到tmux

---
## 🤖 Assistant

# 把运行中的进程转移到 tmux

核心难点在于：进程启动时已经绑定了一个**控制终端（controlling terminal）**和一组文件描述符（stdin/stdout/stderr）。要把它"搬"进 tmux，本质上是**重新挂接这些文件描述符**并改变它的控制终端。下面按推荐程度排列。

---

## 方法一：`reptyr`（最推荐）

`reptyr` 专门干这件事：用 `ptrace` 把目标进程的 stdin/stdout/stderr 重定向到当前终端。

### 1. 安装

```bash
# Debian / Ubuntu
sudo apt install reptyr

# Fedora
sudo dnf install reptyr

# Arch
sudo pacman -S reptyr

# macOS 用户请改用其他方案（见文末）
```

### 2. 放开 ptrace 限制

从内核 3.5 起，默认不允许非父进程 `ptrace`：

```bash
cat /proc/sys/kernel/yama/ptrace_scope   # 通常是 1
sudo sysctl -w kernel.yama.ptrace_scope=0
# 永久生效
echo 'kernel.yama.ptrace_scope = 0' | sudo tee /etc/sysctl.d/10-ptrace.conf
```

### 3. 操作步骤（关键）

假设某个进程正在当前普通终端里跑：

```bash
# ① 挂起它
Ctrl+Z

# ② 从当前 shell 的作业表中"剥离"，避免它被 SIGHUP 杀掉
disown -h %1          # bash
# 或
disown %1

# ③ 记下 PID
jobs -l               # 例如 [1]+ 12345 Stopped   python train.py
```

然后**进入 tmux**（如果你还在原终端，直接 `tmux` 即可）：

```bash
tmux new -s rescue
# 在新窗格里：
reptyr 12345
```

成功后会看到进程的输出重新出现在 tmux 窗格中。按 `Ctrl+B` `D` 分离，进程继续在 tmux 里活着。

> 如果进程不是当前 shell 的子进程（比如已经 `nohup` 或属于别的会话），第②步略过，直接 `reptyr <PID>`，但需要 root 或满足 ptrace 权限。

---

## 常见坑

| 现象 | 原因 / 解决 |
|------|------------|
| `Unable to ptrace the process` | `ptrace_scope` 未放开，见上方 sysctl |
| `Target process is not a normal process` | 目标已是 session leader，`reptyr` 无法接管 |
| 进程在 `reptyr` 后立刻退出 | 可能收到 `SIGHUP`；务必先 `disown`，或用 `screen`/`tmux` 拉起的 shell 里操作 |
| 输出正常但按键无响应 | 部分程序缓存了 `isatty()` 结果，重启该程序更稳妥 |
| `reptyr: ... permission denied` | 用 `sudo reptyr <PID>`，或先 `sudo -i` |

**不能转移的情况**：进程已经是会话首进程（session leader），或它 fork 出了独立终端组，这类只能重启。

---

<details>
<summary><b>方法二：自己用 <code>gdb</code> 重定向 fd（无需 reptyr，但繁琐）</b></summary>

原理：在 tmux 里用新 shell 拿到一个伪终端（如 `/dev/pts/3`），再用 `gdb` 对目标进程调用 `dup2`。

```bash
# 在 tmux 窗格里
tty            # 得到 /dev/pts/3
```

另开终端：

```bash
PID=<目标PID>
sudo gdb -p $PID
```

```gdb
(gdb) call (int) close(0)
(gdb) call (int) close(1)
(gdb) call (int) close(2)
(gdb) call (int) open("/dev/pts/3", 0)   # stdin
(gdb) call (int) open("/dev/pts/3", 1)   # stdout
(gdb) call (int) open("/dev/pts/3", 2)   # stderr
(gdb) detach
(gdb) quit
```

然后切回 tmux，对新终端执行 `Ctrl+Z` / `bg` 或 `disown` 让进程重新成为前台。此法对复杂程序容易失败，仅作了解。

</details>

<details>
<summary><b>方法三：<code>retty</code> / <code>neercs</code>（老旧或小众工具）</b></summary>

- **retty**：`reptyr` 的前身，用 Linux 的 `TIOCSTI` 注入按键，已多年未维护。
```bash
  sudo ./retty <PID>
  ```
- **neercs**：一个自带"接管进程"功能的终端复用器，可 `neercs -a <PID>`，但生态小、兼容性差。

一般直接用 `reptyr` 即可，这两个仅在特殊场景考虑。

</details>

---

## 一句话总结

```bash
# 推荐流程
sudo sysctl -w kernel.yama.ptrace_scope=0 # 一次性放开
disown -h %1 # 在 tmux 里做
reptyr <PID>
```

macOS 没有 Linux 的 `ptrace` 语义，`reptyr` 不可用；只能靠程序自身支持（如 `tmux` 内置、或用 `dtach`/`nohup` 事先规划）来间接达到目的。**最好的做法仍是：一开始就在 tmux 里启动进程。**

需要我针对你的具体进程（比如某个 `python`、`node` 或 `ssh` 会话）给出精确命令吗？告诉我 `ps -ef` 的输出即可。

# VSCode集群节点限制

## 🧑‍💻 User

如何解决集群登录节点不让用vscode remote问题（是指那一类并发ssh都不行），表现是你拿vscode连就直接卡死
就他连上去不是会自动下ssh server装上吗，但是装完之后只能在ide里边列出文件，无法再新建终端了
反正你新建终端就一直黑屏卡住，cpu和ram没有炸（日志反复出现IPC "Pty Host" errored with Error: spawn .../node EAGAIN、No ptyHost response to createProcess after 5 seconds）
只能用xshell这种传统sshtool
合理怀疑估计是机器配置确实不行或者限制了cgroups和ssh连接数

---
## 🧑‍💻 User

如何解决集群登录节点不让用vscode remote问题（是指那一类并发ssh都不行），表现是你拿vscode连就直接卡死
就他连上去不是会自动下ssh server装上吗，但是装完之后只能在ide里边列出文件，无法再新建终端了
反正你新建终端就一直黑屏卡住，cpu和ram没有炸（
只能用xshell这种传统sshtool
合理怀疑估计是机器配置确实不行或者限制了cgroups和ssh连接数

---
## 😀 默认助手 (7)

# 一句话结论

`spawn …/node EAGAIN` 跟 CPU/内存无关，是**进程/线程配额（task 配额）撞墙**。而且在 Linux 上**线程也算 task**，所以 VS Code 那堆 node 进程是第一梯队受害者，Xshell 不是。

## 为什么能这么确定

`fork()/clone()` 返回 `EAGAIN`，内核里只有这几条路径：

| 触发点 | 表现 |
| --- | --- |
| `RLIMIT_NPROC`（`ulimit -u`）超限 | `fork`/`clone` → `EAGAIN`，**顺带 `setuid()` 也返回 `EAGAIN`** |
| cgroup `pids` 控制器（`pids.max`）超限 | `fork`/`clone` → `EAGAIN` |
| `kernel.pid_max` / `kernel.threads-max` 满 | `EAGAIN` |
| `vm.max_map_count`、`RLIMIT_AS`、memcg | `ENOMEM`（不是你这个） |
| inotify 配额 | `ENOSPC`（不是你这个） |
| `noexec` 挂载 | `EACCES`/`EPERM`（不是你这个） |

你那两条日志是同一条因果链：

- `IPC "Pty Host" errored with Error: spawn …/node EAGAIN` —— 主进程想 fork 出 pty host，**内核直接拒绝**。
- `No ptyHost response to createProcess after 5 seconds` —— 下游症状，pty host 压根没起来。

### 三个现象串起来看，基本就实锤了

1. **文件列表能用**：它由已经在跑的 server / extension host 直接服务，**不需要新进程**。
2. **终端黑屏**：每个终端都要新 fork 一个 `node`(pty host) + 一个 shell，全是要新 task 的操作。
3. **你说"并发 ssh 也不行"** —— 这条最有价值。sshd 的 session 子进程在 `setuid()` 时也会撞 `RLIMIT_NPROC`/ucount，失败同样是 `EAGAIN`。所以"登录卡住"和"pty host 起不来"**是同一个根因**，只是撞在不同的 fork 上。

再加一个反直觉点：**登录节点越强，越容易踩**。高核数机器上 node 的线程池、libuv pool、watcher 都会更肥，一个 server 进程几十个线程很正常；而 HPC 登录节点的 `TasksMax`/`pids.max` 往往设得很保守（就是怕用户把登录节点跑挂）。CPU/内存没炸，完全符合预期。

---

<details>
<summary><b>定位真凶：三条只读命令（点开）</b></summary>

```bash
# 1) 用户级 rlimit（-1 / unlimited 才算没事）
echo "ulimit -u = $(ulimit -u)"
echo "pid_max   = $(cat /proc/sys/kernel/pid_max)"
echo "threads-max = $(cat /proc/sys/kernel/threads-max)"

# 2) cgroup 链条上哪一层定了数字（v2）
cg=$(cut -d: -f3 /proc/self/cgroup); echo "my cgroup = $cg"
p="/sys/fs/cgroup$cg"
[ -d "$p" ] || p="/sys/fs/cgroup"
while [ "$p" != "/sys" ] && [ -n "$p" ]; do
  printf '%-64s pids.max=%s current=%s\n' "$p" \
    "$(cat "$p/pids.max" 2>/dev/null)" "$(cat "$p/pids.current" 2>/dev/null)"
  p=$(dirname "$p")
done

# 3) 你现在已经吃掉多少（线程才是大头）
echo "我占用的 task 数 = $(ps -u "$USER" -L --no-headers | wc -l)"
pgrep -u "$USER" -a node | wc -l
```

判读：

- `pids.max` 是个数字（512 / 1024 / 4096…）→ **cgroup 掐的**。
- `ulimit -u` 是个小数字 → **rlimit 掐的**（多半来自 `/etc/security/limits.d/` 的 `nproc`）。
- 两边都是 `unlimited` 才去看 `pid_max` / `threads-max`。
- 连 VS Code 时盯一眼 `pids.current` 是否贴着 `pids.max`（$\text{current}/\text{max} \to 1$），不用再猜。

顺手用一下：`systemctl --user show -p TasksMax -p TasksCurrent -p TasksAccounting`（在会话里能跑的话，这是最直白的读数）。

**别在共享登录节点上做 fork 炸弹压测**——数现有 task + 查限额已经足够定性，真炸了先把整台登录节点搞挂，锅是你的。

</details>

<details>
<summary><b>顺手排掉几个"长得像"的坑（点开）</b></summary>

```bash
cat /proc/sys/fs/inotify/max_user_watches /proc/sys/fs/inotify/max_user_instances
df -h "$HOME" /tmp ; mount | grep -E " ($HOME|/tmp) "
grep -E 'Seccomp|Threads' /proc/self/status
```

- 家目录若是 `noexec`，node 执行会 `EACCES`，日志长得完全不同。
- inotify 配额小会让 watcher 崩，报 `ENOSPC`，也常被误认为"卡死"。
- 家目录配额满 → server 装不上 / 反复重装。

</details>

<details>
<summary><b>残留进程要先清，否则越连越死（点开）</b></summary>

每次失败连接都会留下挂着的 `vscode-server` 进程，继续吃你的 task 配额，下次更连不上：

```bash
pkill -u "$USER" -f 'vscode-server'
pkill -u "$USER" -f '.vscode-server/bin'
rm -rf ~/.vscode-server/data/logs
```

然后再重连一次，观察 `pids.current` 的变化曲线。

</details>

---

# 解法

## 路线 A：别在登录节点跑 VS Code（唯一"正确"答案，也不违反集群政策）

这是 HPC 的标准姿势：拿计算节点，让 server 跑在计算节点上。

```bash
# 登录节点
salloc -N1 -c4 --mem=8G --time=08:00:00
squeue --me          # 拿到节点名 cn123
```

然后本地 `~/.ssh/config`：

```sshconfig
Host hpc-login
  HostName login.hpc.example.edu
  User alice

# 计算节点跑 sshd 且登录节点能转发，就直连
Host hpc-cn-*
  User alice
  ProxyJump hpc-login
```

计算节点不开 sshd 时，两条替代路：

```bash
# 路1：在计算节点起 server，本地经登录节点做端口转发
ssh -L 8080:cn123:8080 alice@login.hpc.example.edu

# 路2：code tunnel（纯出站 443，不需要任何入站 ssh）
code tunnel --name hpc-cn123     # 计算节点上跑
```

本地 VS Code 用 `Remote-SSH` 连 `hpc-cn-*`，或用 "Connect to Tunnel"。另外先看集群有没有 **Open OnDemand / JupyterHub**，它们自带的 VS Code 应用已经把这些配额问题处理好了。

## 路线 B：留在登录节点，把配额省着用

先在 `settings.json` 里降开销（远程侧）：

```jsonc
{
  "remote.SSH.useExecServer": true,          // 新版本用 Rust exec server 替掉一部分 node 进程
  "remote.SSH.remoteServerListenOnSocket": true,
  "remote.SSH.connectTimeout": 60,

  "files.watcherExclude": {
    "**/.git/objects/**": true, "**/scratch/**": true, "**/data/**": true
  },
  "search.followSymlinks": false,
  "telemetry.telemetryLevel": "off",
  "extensions.autoUpdate": false,
  "update.mode": "none",
  "terminal.integrated.enablePersistentSessions": false
}
```

再降单进程线程数（注入给 server 进程）：

```bash
# 优先试这个（VS Code 会 source 它给 server 设环境变量；版本不认就走下面的 SetEnv）
cat >> ~/.vscode-server/server-env-setup <<'EOF'
export UV_THREADPOOL_SIZE=2
export NODE_OPTIONS=--v8-pool-size=2
EOF
```

或者 SSH 侧（需要服务端 `PermitUserEnvironment yes`）：

```sshconfig
Host hpc-login
  SetEnv UV_THREADPOOL_SIZE=2 NODE_OPTIONS=--v8-pool-size=2
```

想先量收益值不值得折腾：`ps -L -p $(pgrep -u "$USER" -f vscode-server | head -1) | wc -l`，看一个 node 进程到底占多少线程。**实操上更省事的是混合模式**：Remote-SSH 只用来编辑/跑扩展，就地跑交互命令继续用 Xshell——因为它不需要新进程，天然躲过这道墙。

## 路线 C：找管理员（把话说对，才有人理你）

```text
请在 <登录节点> 上放宽普通用户的 task 配额。VS Code Remote 的 pty host 起不来：

  IPC "Pty Host" errored with Error: spawn .../node EAGAIN
  No ptyHost response to createProcess after 5 seconds

同一用户并发 ssh 也会 setuid/fork EAGAIN，属于同一根因（task 数超限，CPU/内存均正常）：

- systemd: DefaultTasksMax / user-<uid>.slice 的 TasksMax，当前 = <X>
- cgroup v2: /sys/fs/cgroup/user.slice/user-<uid>.slice/pids.max，当前 = <Y>
- PAM: /etc/security/limits.d/*.conf 里 @users 的 nproc，当前 = <Z>

建议 >= 8192。同时建议对登录节点用 user-<uid>.slice 的 CPUQuota/MemoryMax 限速，
单纯卡 task 数只会让用户撞死墙，而不是被平滑限速。
```

## 关于"是不是 ssh 连接数限制"

不是。`MaxSessions` / `MaxStartups` / `MaxLogins` / `pam_slurm_adopt` 这类只会让你**连不上或开不了 shell**，报的是 `Connection closed`、`Too many sessions`、`Connection refused`。你这个 `spawn … EAGAIN` 明确来自 fork 路径，和它们无关——但它们可以顺手排掉，省得日后混着出现时再猜一遍。

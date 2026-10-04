# 训练被SIGTERM终止

## 🧑‍💻 User

训练遇到RECEIVED SIGNAL 15: SIG SIGTERM是有人掐了吗

---
## 😀 默认助手 (7)

# SIGTERM (信号 15) 到底是谁掐的？

**不一定有人在掐你**，但也不排除。SIGTERM 是"礼貌地请求进程退出"的信号，触发来源有很多种，需要具体排查。

## 常见触发来源

| 来源 | 典型场景 |
|------|---------|
| **集群调度器** | Slurm/K8s/Volcano 超时、被抢占（preempted）、节点维护、作业被更高优先级任务挤掉 |
| **内存超限** | cgroup OOM 监控到内存超标时，发 SIGTERM 让你优雅退出（注意：真正 OOM 通常是 SIGKILL/信号 9） |
| **时间/配额限制** | 作业运行超过 walltime，调度器到期发信号 |
| **手动操作** | 有人 `kill`、`scancel`、删 Pod、或者父进程（如 `timeout`）退出 |
| **容器/云平台** | Spot 实例被回收、节点缩容、K8s Pod eviction |
| **上游脚本** | 训练脚本自己捕获到了外部信号，或因父进程死亡触发级联 |

## 关键区分：SIGTERM(15) vs SIGKILL(9)

- **SIGTERM (15)**：可被捕获，进程有机会做清理。**大多来自调度系统或人为 kill**。
- **SIGKILL (9)**：不可捕获，**典型是 OOM Killer 或强制 kill**。

你遇到的是 15，说明大多是**调度器/平台按规则发的**，而不是内核直接杀。

## 怎么定位真凶

```bash
# 1. 看集群侧记录（最有效）
scontrol show job <jobid>        # Slurm，看 Reason / ExitCode
sacct -j <jobid> --format=JobID,State,ExitCode,DerivedExitCode,End
kubectl describe pod <pod>       # K8s，看 Events 里的 eviction/OOMKilled

# 2. 看系统日志
dmesg -T | grep -i -E "killed process|oom"
journalctl -k | grep -i oom

# 3. 看进程退出上下文
grep -i sigterm train.log
# 确认是不是 timeout 命令或 shell 脚本发的
```

## 防护建议

1. **注册信号处理器**，在收到 SIGTERM 时保存 checkpoint，避免白跑：
```python
import signal, sys
def handler(signum, frame):
    save_checkpoint()   # 紧急存档
    sys.exit(0)
signal.signal(signal.SIGTERM, handler)
```
2. 让训练**定期 checkpoint**，别只在结束时存。
3. 检查作业的 **walltime / 内存申请** 是否够，K8s 里设好 `resources.requests/limits`。
4. 如果是被抢占，考虑用 **可重入/断点续训** 或提高作业优先级。
5. 用 `nohup`/`tmux` 避免 SSH 断开级联导致父进程退出。

---

**一句话结论**：SIGTERM 更多是**调度器或平台规则**（超时、抢占、配额、节点回收）造成的，"有人手动掐"只是其中一种可能。优先去查 `sacct` / `kubectl describe` / `dmesg` 这类集群和内核日志，就能确定到底是谁发的信号。

如果你能告诉我用的是 **Slurm、K8s 还是裸机 + 云平台**，我可以给你更具体的排查命令。

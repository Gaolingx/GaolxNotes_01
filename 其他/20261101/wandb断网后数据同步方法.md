# wandb断网后数据同步方法

## 🧑‍💻 User

说起来wandb跑到一半网断了后面又能access了怎么把数据给同步上去

---
## 😀 默认助手 (7)

先说结论：**九成情况下不用做额外操作**。中途断网时，wandb 的本地 service 进程会一直把数据写入 `./wandb/run-*/` 下的 `.wandb` 文件（本地队列），网络一恢复就会自动补传。只有进程已经退出、或者 service 进程卡死时，才需要手动 `wandb sync`。

## 先做判断：跑的是哪一类 run

```bash
# 1) 训练进程还活着吗？
pgrep -af "python.*train"    # 你自己的启动命令
pgrep -af "wandb"

# 2) 本地目录是哪类？
ls -d wandb/offline-run-* wandb/run-* 2>/dev/null

# 3) service 在干嘛（有没有在重试）
tail -n 50 wandb/run-*/logs/debug.log
```

- 训练进程**还在运行**，且 debug.log 里能看到 retrying/upload 相关日志，那么**别动它**，等它自己补完；最多确认一下服务器端历史是否还在增长。
- 训练进程**已经退出或被杀**，就按下面三种情况处理。

## 情况 A：目录是 `offline-run-*`（本来就离线跑）

```bash
wandb login                      # 确认有凭证
wandb sync wandb/offline-run-2024xxxx_xxxxxx-abc12345
```

批量全部补传（在 `wandb/` 的**父目录**执行）：

```bash
wandb sync --sync-all
# 补完顺手删本地目录，避免占盘：
wandb sync --sync-all --clean
# 网络抖动时忽略单条失败继续：
wandb sync --sync-all --ignore
```

## 情况 B：目录是 `run-*`（在线跑，但进程挂了）

这类 run 在断网前那部分**已经传上去了**，缺的只是缓冲区里的尾部数据。有两种做法：

```bash
# 推荐：以 append 方式把本地尾部并回同一个 run，不会新建一个空 run
wandb sync --append wandb/run-20240101_120000-abc12345

# 备选：如果服务端那个 run 已经被写坏/重复，用 replace 整体覆盖
wandb sync --replace wandb/run-20240101_120000-abc12345
```

`--append` 依赖目录名里的 run id，所以务必**原样传入目录路径**，不要传 `.wandb` 文件。

## 情况 C：想接着训练（不是补传，是续跑）

```python
wandb.init(id="abc12345", resume="allow")   # 或 resume="must" 强制续上
```

wandb 会复用同一个 run，断开那段末尾的 history 可能丢失，但曲线是连续的。

<details>
<summary>常见问题（补传失败时看这里）</summary>

| 症状 | 原因 / 处理 |
|---|---|
| `wandb: ERROR api_key not configured` | 先执行 `wandb login`，或检查 `~/.netrc` 里的 api.wandb.ai |
| 卡在 `wandb: Network error (ConnectionError)` | 代理问题；`export HTTPS_PROXY=...`，或临时设 `NO_PROXY` |
| 数据传了但 run 显示 crashed | 正常，页面上的状态不会自动改；数据本身已经在 |
| 补传完 run 里曲线出现重复点 | `--append` 按 `_step` 对齐，一般不会重复；真重复了用 `--replace` |
| 找不到 `wandb sync` 子命令 | 版本太老，`pip install -U wandb` |
| 服务器上有两个同名 run | 说明误用了普通 `wandb sync`（不带 `--append`），会新建 run；删掉多余的那个即可 |

</details>

<details>
<summary>预防：以后想让断网更稳</summary>

- 训练脚本启动前设 `WANDB_MODE=offline`，跑完再 `wandb sync --sync-all`，最稳。
- 或者保持 online，但把 `WANDB_DIR` 指到大盘（`export WANDB_DIR=/data/wandb`），避免磁盘满导致缓冲写失败。
- 关键实验加 `WANDB__SERVICE_WAIT=300`，给 service 更长的重连窗口。

</details>

要不要把你实际的报错贴一下（`debug.log` 尾部几行 + 目录结构），我帮你确认是 A/B/C 哪一种？

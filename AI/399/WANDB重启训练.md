# WANDB重启训练

## 🧑‍💻 User

pytorch lightning跑到一半wandb挂了，主训练进程活着怎么重启wandb？

---
## 🤖 Assistant

下面按"为什么会这样 → 手动救活 → 自动化 → 坑"来讲，代码都是可以直接贴的。核心结论先说：

> Lightning 的 `WandbLogger` 把 wandb run 缓存在 `logger._experiment` 里，只要它非 `None`，`experiment` 这个 property 就**永远不会**再帮你重新 `wandb.init()`。所以主进程活着的情况下，你要做的就是：**自己重新 `wandb.init()` 出一个 run，然后把它塞回 `logger._experiment`**。

```python
# 关键两行（细节见下文）
new_run = wandb.init(..., reinit="finish_previous", id=old_id, resume="allow")
logger._experiment = new_run
```

---

## 1. 先判断挂的是哪一层

W&B 0.13+ 是**独立进程**（wandb-core / service）通过 socket 跟训练进程通信，所以有三种情况，处理方式完全不同：

| 现象 | 原因 | 要不要救 |
|---|---|---|
| 只是网络断了，控制台刷 `Network error (ConnectionError), retrying request`，训练照跑 | SDK 自己在**缓冲 + 重试**，本地 `.wandb/` 有事务日志 | **不用管**，恢复网络后自动补传；实在不行收工 `wandb sync` |
| 抛 `CommError: The wandb backend process has shutdown` / `BrokenPipeError` | wandb-core 进程死了 | 按本文重启 |
| `wandb.log` 卡住不动 | backend 僵尸 / socket 阻塞 | 线程 + timeout 兜底（见下） |

排查：看 `<run_dir>/files/logs/debug-internal.log`，`ps aux | grep wandb` 看服务进程还在不在。

---

## 2. 手动救活（pdb / 回调 / 信号处理里两段代码）

### 2.1 先存下 run id（续跑才需要）

```python
logger = trainer.logger          # 或 [l for l in trainer.loggers if isinstance(l, WandbLogger)][0]
old_id = getattr(logger._experiment, "id", None) or logger._id
print("old run id:", old_id)     # 续跑靠这个；一开始没给 id 就是 None
```

> ✅ **最好的预防**：训练开头就固定 run id，如 `WandbLogger(project="p", id="my-run-001")`（等价于 `version=`）。Lightning 默认已经带 `resume="allow"`，只要 id 在就能续上同一条曲线。

### 2.2 重建 run 并接管

```python
import wandb

cfg = dict(logger._wandb_init)         # Lightning 存的 init 参数：project/dir/name/resume/id... 以及你传的 entity 等 kwargs
cfg.update(
    id=old_id,                         # ← 想续同一条曲线就留；想开新 run 就删掉这行
    resume="allow",                    # ← 同上
    reinit="finish_previous",          # 结束旧 run 并把新 run 设为全局 wandb.run
    settings=wandb.Settings(init_timeout=30, finish_timeout=30),  # 别让 finish 无限等
)

try:
    wandb.finish()                     # 旧 backend 死了可能抛/卡，包一下
except Exception:
    pass

new_run = wandb.init(**cfg)

# 把新 run 塞回 logger，Lightning 之后所有 self.log() 就走它了
logger._experiment = new_run
logger._id = old_id
logger._wandb_init["id"] = old_id

# ⚠️ 新 run 是一张白纸，原来 init 时做的这两件事要补回来：
new_run.define_metric("trainer/global_step")
new_run.define_metric("*", step_metric="trainer/global_step", step_sync=True)
new_run.config.update(trainer.model.hparams, allow_val_change=True)   # 视需要
```

`trainer` / logger 对象都不变，训练循环**不用碰**。

**`reinit` 取值怎么选**（来自 wandb `Settings`）：

- `"finish_previous"`（或 `True`）：先 `finish_all_active_runs()` 再建新 run，**会**把 `wandb.run` 指向新的。旧 run 还响应时用它。
- `"create_new"`：直接建新 run，**不**碰旧 run、**不**更新全局 `wandb.run`。旧 backend 已经僵死、`finish` 会卡时最省事——因为 Lightning 走 `logger.experiment.log`，不依赖全局。
- `"return_previous"`（默认行为）：如果旧 run 还在 active 列表里，你 `init` 会**原样返回那个死 run**，等于没救——这就是为什么必须显式指定 `reinit`。
- 另外注意：`resume="must"` + 有 active run 会直接报错 `Cannot resume a run while another run is active`。

---

<details>
<summary><b>3. 自动化：包一个会自愈的 WandbLogger（展开看代码）</b></summary>

最稳的检测方式是**在真正写数据的地方 try/except**，因为这正是 backend 死掉时第一个报错的位置。

```python
import threading
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class ResilientWandbLogger(WandbLogger):
    """wandb-core 挂掉后自动重建 run，继续训练。"""

    def __init__(self, *args, max_restarts: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self._restarts = 0
        self._max_restarts = max_restarts
        self._new_run()

    # ---- 首次 init 后补上 Lightning 会做的 define_metric ----
    def _new_run(self):
        run = self.experiment
        try:
            run.define_metric("trainer/global_step")
            run.define_metric("*", step_metric="trainer/global_step", step_sync=True)
        except Exception:
            pass

    @staticmethod
    def _safe_finish():
        try:
            wandb.finish()
        except Exception:
            pass

    def _finish_bounded(self, timeout: float = 20.0):
        """后台线程里 finish，超时就不管了，避免死 backend 把训练卡住。"""
        t = threading.Thread(target=self._safe_finish, daemon=True)
        t.start()
        t.join(timeout)

    @rank_zero_only
    def _rebuild(self):
        if self._restarts >= self._max_restarts:
            raise RuntimeError("wandb 重启次数超限，放弃自愈")
        self._restarts += 1

        run_id = getattr(self._experiment, "id", None) or self._id
        cfg = dict(self._wandb_init)
        cfg.update(
            id=run_id, resume="allow", reinit="finish_previous",
            settings=wandb.Settings(init_timeout=30, finish_timeout=30),
        )

        self._finish_bounded()          # 尽力而为，不无限等
        new_run = wandb.init(**cfg)     # 起新的 wandb-core，连回同一个 run id

        self._experiment = new_run
        self._id = run_id
        self._wandb_init["id"] = run_id
        self._new_run()
        print(f"[wandb-watchdog] re-attached to run {run_id} (#{self._restarts})")
        return new_run

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        try:
            return super().log_metrics(metrics, step)
        except Exception as e:                       # CommError / BrokenPipeError / ...
            print(f"[wandb-watchdog] log failed: {type(e).__name__}: {e}")
            self._rebuild()
            try:                                     # 把丢掉这一批补上
                return super().log_metrics(metrics, step)
            except Exception:
                return None
```

用法就是原来 `WandbLogger(...)` 换成 `ResilientWandbLogger(...)`。

**logger 已经建好了、不想改类**，就 monkey-patch：

```python
import types
trainer.logger.log_metrics = types.MethodType(
    ResilientWandbLogger.log_metrics, trainer.logger
)
```

**如果是"卡死"而不是"抛异常"**，再加一个后台心跳线程：每隔 N 秒用 `run.status()` 之类的调用配 timeout 探活，超时就触发 `_rebuild()`（注意别和主线程的 `log` 并发写同一 run，加锁）。

</details>

---

## 4. 必踩的坑

| # | 坑 | 说明 |
|---|---|---|
| 1 | 只写 `logger._experiment = None` | **没用**。`experiment` property 会先检查全局 `wandb.run`，它若还指向死 run 就直接复用 → 你还是在往死 run 写。必须让全局 run 换掉（`reinit="finish_previous"`）或无视全局（`create_new` + Lightning 走 `self.experiment`） |
| 2 | `wandb.finish()` 卡住 | 默认会等数据上传完才返回。用 `Settings(finish_timeout=...)` 或线程 + `join(timeout)` 兜底 |
| 3 | 新 run 是白纸 | `define_metric`、`config`、以及你之前 `logger.experiment.config.update()` 的东西都得**重新灌一遍**，否则曲线 x 轴 / 超参全丢 |
| 4 | 续跑要先有 id | 一开始没固定 `id`/`version` 就续不上；本地 `.wandb/run-*` 事务日志还在，旧数据可以 `wandb sync` 补 |
| 5 | DDP | `WandbLogger` 只在 **global rank 0** 建 run（`@rank_zero_experiment` / `@rank_zero_only`）。重启逻辑也必须只在 rank 0 执行，否则多进程互踩 |
| 6 | 私有属性 | `_experiment` / `_wandb_init` / `_id` 都是私有，跨 Lightning 小版本可能改；**锁死版本**，升级后回归测一下 |
| 7 | 兜底方案 | 实在救不回来：`os.environ["WANDB_MODE"]="offline"` 重新 init，落盘到本地，收工 `wandb sync <run_dir>` |

---

<details>
<summary><b>5. 预防：让 wandb 别那么容易挂（展开）</b></summary>

通过环境变量设置（wandb 会把 `WANDB_xxx` 映射到对应 setting）：

```bash
export WANDB_INIT_TIMEOUT=60        # wandb.init 最长等多久（默认 90s）
export WANDB__SERVICE_WAIT=60       # 等 wandb-core 服务起来的超时（双下划线！内部 x_service_wait）
export WANDB_FINISH_TIMEOUT=60      # finish 上传超时，防止收工卡死
export WANDB_STOP_ON_FATAL_ERROR=false
export WANDB_MODE=online
# 固定 run id，方便断点续跑
export WANDB_RUN_ID=my-run-001
export WANDB_RESUME=allow
```

代码里对应 `wandb.Settings(init_timeout=60, finish_timeout=60, ...)`。

另外几个降低崩溃面的点：

- 别把 `wandb.log` 打得过密（官方建议 ≤ 数次/秒），高频 `self.log` 会把 service 打爆；
- 大对象（大图/表格）慢一点、小一点；
- `WandbLogger(log_model="all")` 在崩溃时容易出问题，可改成只存本地再手动上传。

</details>

---

**一句话总结**：只要训练进程活着，`wandb.init(..., reinit="finish_previous", id=旧id, resume="allow")` 重建一个 run，然后 `logger._experiment = new_run` 就接上了；想一劳永逸就把这段 try/except 逻辑塞进 `WandbLogger.log_metrics` 的 override 里。

需要的话我可以把上面 `ResilientWandbLogger` 补成完整可跑的例子（含 DDP 判断和心跳线程）。

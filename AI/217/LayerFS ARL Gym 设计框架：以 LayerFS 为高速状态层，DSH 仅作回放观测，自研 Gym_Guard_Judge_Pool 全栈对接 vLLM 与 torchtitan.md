# LayerFS ARL Gym 设计框架：以 LayerFS 为高速状态层，DSH 仅作回放观测，自研 Gym/Guard/Judge/Pool 全栈对接 vLLM 与 torchtitan

## 🧑‍💻 User

https://x.com/yifanxu_ephai/status/2112876555907064008
介绍下dsh-layerfs，这个是rl sandbox用的
1. 我想知道对于普通用户非rl用途有什么用，对于普通用户不是很heavy吗，感觉还不如起个git？
2. 怎么接入agentic rl pipeline（我已经有了验证器、推理后端（vllm）、训练后端（torchtitan）需要改进gym和相关warpper（rl env））

---
## 😀 默认助手 (7)

# LayerFS / dsh-layerfs：它是什么，以及怎么塞进你的 agentic RL 训练栈

下面先给结论，再分开回答两个问题。为了避免把“文件系统引擎”和“dsh 插件”混为一谈，先说明这三者实际上是三个层次的东西。

---

## 0. 先把三层关系理顺（这一步很重要）

从你贴的那条推文到实际能用来跑 RL，中间隔着三层：

| 层 | 仓库 / 包 | 是什么 | 你需要它吗 |
|---|---|---|---|
| **引擎** | `Ephemeral-AI-Lab/layerfs`（Rust + FUSE + SQLite，MIT） | 真正做文件系统：Store（单文件 SQLite，无 DB server）、Branch、Commit、Workspace、Docker Sandbox | ✅ RL 用这个 |
| **SDK** | `@layerfs/agent` / `@layerfs/backend` / `@layerfs/runtime` + `layerfs` CLI | 给 agent 和宿主调用的 JS API（`fork / mount / exec / commit / merge / publish`） | ✅ RL 用这个 |
| **插件** | `Ephemeral-AI-Lab/dsh-layerfs` | **DeepSeek Harness（dsh）web 的 UI 插件**：给 dsh 里的 agent 挂 `layerfs` 命令，加 `Work` / `History` 两个可视化 tab。它“changes no dsh code”，纯粹是 UI + 集成层 | ❌ RL 基本不用 |
| 兄弟项目 | `Ephemeral-AI-Lab/ephemeral-sandbox` | 无头 Rust 沙箱（gateway/CLI/MCP），也是 workspace-per-agent 的思路 | 可选，见 §2.6 |

> **关键判断：`dsh-layerfs` 对你来说是“人类观看 agent 干活的界面”，不是 RL 基础设施。** 你要接入 pipeline 的是 `@layerfs/agent`。推文里那句 "A filesystem for RSI agents and **recursive exploration for RL training**" 指的是引擎的 fork 语义，不是插件。

它的 10 个设计点，对 RL 真正有意义的只有 4 个（其余是工程糖）：

| 设计点 | 本质机制 | 对 RL 的价值 |
|---|---|---|
| **04 递归文件系统** | `branch.fork(name, fromRef)` 是纯元数据操作，**零文件拷贝，可任意深度** | ⭐⭐⭐ GRPO 的 $G$ 条 rollout、tree search、从第 $k$ 步重试，全都变成一次 fork |
| **06 瞬时 bootstrap** | 被 `.gitignore` 的 `node_modules/.venv/target` 也留在 commit 里 | ⭐⭐⭐ 每条 rollout 省掉一次 `pip install` / `npm ci`，mount 5–20ms |
| **07 自动 checkpoint** | 每次工具调用 = 一个 Commit（无变更则不提交） | ⭐⭐ 步骤级可回滚、可 diff、`fs.blame` 给行级归因 |
| **05 存储去重** | 内容分块 + 去重 + 压缩，10 份拷贝 → 0.31 | ⭐⭐ 上万条近似 rollout 的树存得下（157 snapshots: 100MB vs 2GB） |
| 01/02/03/08/09/10 | ephemeral 容器语义、单 Store 多机提交、冲突拒绝、append-only、`.lww`、大文件小改 | 单机 RL 基本用不到，或你已经有了 |

---

## 1. 普通用户（非 RL）用它值不值？会不会太重、不如起个 git？

**你的直觉基本是对的，而且官方文档自己也这么写。** 原话是：

> "Use something else if you need crash or power-loss durability..., full POSIX node types, or top sequential I/O bandwidth. **For a few long-lived checkouts on one machine, Btrfs snapshots or Git worktrees may be all you need.**"

### 1.1 成本清单（这些是硬约束，不是配置问题）

<details>
<summary><b>展开：普通用户升级到 LayerFS 需要支付的成本</b></summary>

| 项 | 现实 |
|---|---|
| **平台** | 所有打开同一个 Store 的进程必须**共享同一个 Linux kernel**；Store 必须在**本地 ext4/xfs/btrfs** 上。macOS / Windows 只能靠 `@layerfs/runtime` 起一个长驻 Linux worker 做转发（WSL2 还要自己跑 Docker Engine，不能用 Docker Desktop 的 pipe）。 |
| **安装** | **没发布到 npm**。要自己装 Rust 1.85.1、编译 `layerfs-daemon` + 原生 addon（冷编译 1–2.5 分钟）、自己 build runtime 镜像、macOS/Windows 还要打 sealed bundle（`manifest.json` 校验）。官方把"让 coding agent 按 prompt 帮你装"当成主要安装方式——这本身就说明了门槛。 |
| **Docker + FUSE** | 必须有 Docker Engine（≥ API v1.48 / Docker 28.0），Sandbox 是容器，Workspace 是容器里的 FUSE 挂载。 |
| **持久性** | **只跑 Disposable profile**：WAL + `synchronous=OFF`。OS crash / 断电可能丢历史或损坏 Store。官方明说"别当作唯一副本，建议在 Workspace 里 `git push` 出去"。 |
| **GC** | **没有**。`历史永不删除，there is no garbage collection yet`。Store 只涨不消。 |
| **POSIX 完整性** | pjdfstest 876 条断言 0 个非预期失败，但只保留了 overlayfs 76% 的通过项：不支持 FIFO、socket、device node、setuid/setgid、独立 atime/ctime。 |
| **性能** | 相对 libfuse passthrough：随机读写 1.49–2.23×、小文件 1.27–10.71× 更快；但**顺序写 0.31×、顺序读 0.81×、随机写 0.40×**。跑大文件构建产物（`cargo build`、大型镜像 I/O）会疼。 |
| **并发** | 一个 Store 一个 writer，短持有；`sandbox.create` / `sandbox.close` 要抢**项目锁**（等 60s 后 `Busy`）。 |

</details>

### 1.2 它真正解决、而 git 解决不了的那一个痛点

git 的 `worktree` 给你一棵**干净的树**：没有 `node_modules`、没有 `.venv`、没有 `target/`、没有本地 `.env` 和数据库文件。所以并行 agent 的真实成本不是 `git worktree add`（毫秒级），而是**每个 worktree 一次的 `npm ci` / `uv sync`**（几分钟 × agent 数 × 尝试次数）。

LayerFS 的 `mount` 直接把 commit 里的**整棵树，包含 ignored 文件**挂出来，5–20ms。官方对比的就是 "≈5 min install vs 0.05 s mount"。

顺带的好处：每次工具调用一个 commit → agent 执行 `rm -rf src` 你不用心慌，`fork` 回上一步就行；Bash 跑在容器里，`--dangerously-skip-permissions` 的爆炸半径被限制。

### 1.3 直接对比

| | git + worktrees | LayerFS |
|---|---|---|
| 源码版本管理 | ⭐ 完整、生态成熟 | 能用（Commit/Branch/Merge/publish），但没有远程、没有 PR/CI 生态 |
| 并行 checkout | 便宜 | 更便宜（且可递归 fork） |
| **依赖/构建产物** | ❌ 每个 worktree 重装 | ⭐ 随 commit 走，秒级就绪 |
| 持久性 | ⭐ 强（这是它的本职工作） | ❌ Disposable，非断电安全 |
| 跨机器协作 | ⭐ push/pull | ❌ 单 kernel，无网络协议 |
| 顺序 I/O | ⭐ 原生速度 | ❌ 0.31× |
| 空间效率（多快照） | 差（每个 worktree 一份） | ⭐ 19.9× 优于 Btrfs+zstd |
| 安装成本 | 0 | 半天到一天 |
| 每次改动自动留痕 | ❌ 要自己 commit | ⭐ 工具调用即 commit |

### 1.4 我的建议（诚实版）

- **一个人、一个长期 checkout、要 PR/CI/远程协作** → **别上 LayerFS**，`git worktree` + devcontainer 就够了。你判断"不如起个 git"在这类场景下是对的。
- **下面这三种情况下它突然划算**：
  1. 你在一台机器上同时跑 **≥3 个 agent / ≥5 次并行尝试**，且每次都要重装依赖或重编译（Node/Python/Rust 项目尤甚）；
  2. 你想要 **"尝试 → 秒回滚"** 的探索式工作流，而不是 git 的 stash/branch 手工体操；
  3. 你想让 agent **自由跑 bash** 而不炸你的主力机器（容器 + 每命令挂载）。
- **仍然保留 git 作为真相源**：在 Workspace 里 `git push`，LayerFS 只作为"工作台 + 历史沙盘"。这是官方推荐的用法。

---

## 2. 接入 agentic RL pipeline（你已有 verifier + vLLM + torchtitan）

前提对齐一下你的现状：**verifier ✅、推理 vLLM ✅、训练 torchtitan ✅，缺的是 gym env + wrapper**。那 LayerFS 要嵌的正是 gym 这一层。

### 2.1 概念映射（先想清楚这个表，代码是次要的）

| RL 概念 | LayerFS 原语 | 说明 |
|---|---|---|
| **task family / repo** | `Project`（一个 Store 文件） | 一个 Store = 一个 kernel = 一台机器。按任务族分 Store |
| **task instance 初始状态** | `main` 上的一个 Commit / Layer（含 `.venv`、`node_modules`、`target/`） | ⭐ 关键：**初始状态里必须包含已装好的依赖**，这是全部收益来源 |
| **group（GRPO 的 $G$ 条同 prompt rollout）** | 从同一 base commit `fork` $G$ 个 Branch | 零拷贝，$G$ 可以开到 8/16/32 |
| **rollout / episode** | 一个 Branch | commit 链就是 trajectory 的物化 |
| **turn \(t\)** | `workspace.commit` 一个 Commit（或 `fs.apply` 一次提交多个文件改动） | 每 turn 一个 commit，不要每个 fs 操作一个 |
| **action** | 模型生成的 tool-call tokens / bash | 由你的 harness 解析 |
| **observation** | `workspace.exec` 返回的 stdout/stderr（+ exitCode） | 注意截断到 token 预算 |
| **环境隔离** | `sandbox.create` → `workspace.mount(sandbox, branch)` | 1 Sandbox 最多 16 Workspace；空闲 2 分钟自关 |
| **checkpoint / 断点续跑** | commit id | 从第 \(k\) 步重新采样 = `fork(branch, commit_k)` |
| **tree search / best-of-N / beam** | 递归 fork + 分支打分 + 剪枝 | 推文里那些 `explorers ×3 ... ★ best 0.91` 就是这个 |
| **行级 credit assignment** | `fs.blame(ref, path)` | 哪一行是第几步写的 |
| **step 级 diff 特征** | `branch.diff_commits(from, to)` | 可以直接拿来做 reward shaping 或 auxiliary loss |
| **verifier 输入契约** | commit id（或未提交的 Workspace） | `verifier(commit_ref) → score` 可确定性复现 |
| **多 agent 协作 rollout** | 同 Store 多 Branch + `publish`/`merge`，同文件冲突直接 `Conflict` 拒绝 | 天然的"不许静默覆盖" |

### 2.2 参考架构（Python 栈 + Node 桥）

**这是最容易踩的坑：LayerFS 只有 JS/Node SDK，没有 Python 绑定，而且 `layerfs` CLI 是"每条命令一个新 Node 进程"**（`LayerFS.open` + 原生 addon + 连 daemon 都在进程内重新做一遍）。在 RL 里每步 spawn 一个 Node 进程是灾难。所以：

```text
┌──────────────── Python 侧（你的 gym / rollout worker） ────────────────┐
│  GymEnv.reset()  → 取 task_id, base_commit                            │
│  GymEnv.step()   → 把 tool calls 发给 env-server                     │
│  轨迹缓冲 (tokens, logprobs, loss_mask, commit_ids, reward)           │
└──────────────────────────┬────────────────────────────────────────────┘
                           │  JSON-RPC over UDS / HTTP / ZMQ
┌──────────────────────────▼────────────────────────────────────────────┐
│  Node env-server 池（每进程 = 1 个 LayerFS client = 1 个 Project）      │
│  · long-lived open client（@layerfs/agent）                           │
│  · Sandbox pool（预热，避免 create 抢项目锁）                          │
│  · fork / mount / exec / commit / unmount / close 批量执行            │
└──────────────────────────┬────────────────────────────────────────────┘
                           │  @layerfs/agent → layerfs.node (addon)
┌──────────────────────────▼────────────────────────────────────────────┐
│  Docker Engine + layerfs-daemon + FUSE，Store 在本地 ext4/xfs/btrfs     │
└───────────────────────────────────────────────────────────────────────┘
        ▲                                                      
        │ reward
┌───────┴────────┐        ┌──────────────┐        ┌────────────────────┐
│ 你的 verifier   │◄───────│  vLLM 推理后端 │───────►│  torchtitan 训练   │
│ (在 sandbox 内跑)│        │ (组 batch 生成) │        │ (吃 token+reward)  │
└────────────────┘        └──────────────┘        └────────────────────┘
```

**为什么必须分两个进程/语言**：`@layerfs/agent` 一个进程只能加载一份、只能开一个 Store（"A second copy at another path refuses to load"）；而 Python 的 env 需要成千上万个并发协程。用 N 个 Node 进程 × N 个 Store 做**分片**，Python 侧按 `task_id % N` 路由。

### 2.3 Node 侧：env-server（长驻 client + Sandbox 池）

```js
// env-server.mjs —— 一个进程 = 一个 Project = 一个 shard
// 启动: LAYERFS_HOME=... LAYERFS_STORE_PATH=... LAYERFS_IMAGE=sha256:... LAYERFS_DAEMON=... node env-server.mjs
import { LayerFS } from "@layerfs/agent";

const lf = await LayerFS.open(process.env.LAYERFS_PROJECT); // 只开一次，永不 close

// ---- Sandbox 池：sandbox.create 要抢项目锁，绝对不能在 rollout 热路径上现建 ----
const POOL = [];
async function warmPool(n = 8) {
  for (let i = 0; i < n; i++) POOL.push(await lf.sandbox.create());
}
// 每个 Sandbox 最多 16 个 Workspace，所以 n = ceil(并发rollout数 / 16) + 余量

class Rollout {
  constructor(sandbox, branch, ws) { Object.assign(this, { sandbox, branch, ws }); this.commits = []; }

  static async start(baseRef, tag) {
    const branch = `${tag}-${Math.random().toString(36).slice(2, 8)}`;
    await lf.branch.fork(branch, baseRef);          // ← 零拷贝，这就是 G 条 rollout 的成本
    const { sandbox } = POOL.pop() ?? (await lf.sandbox.create());
    const ws = await lf.workspace.mount(sandbox, branch); // ~5-20ms
    return new Rollout(sandbox, branch, ws);
  }

  async exec(cmd, msg) {
    const r = await lf.workspace.exec(this.ws, `timeout 300 ${cmd}`, msg); // exec 自己没有超时
    return { stdout: r.stdout, stderr: r.stderr, code: r.exitCode };
  }

  // 每 turn 一次 commit（不要每次 fs.write 一次）
  async commit(msg) {
    const r = await lf.workspace.commit(this.ws, msg);
    if (r.kind === "Committed") this.commits.push(r.commit ?? msg);
    return r;
  }

  async score() {                     // verifier 跑在同一个容器里，环境一致
    const r = await this.exec(process.env.VERIFY_CMD);
    return { score: parseFloat(r.stdout), commit: this.commits.at(-1) ?? null };
  }

  async end(keepBranch = false) {
    await lf.workspace.unmount(this.ws, /* discard */ !keepBranch);
    POOL.push({ sandbox: this.sandbox });
  }
}
```

**RL 热路径上要避开的三个 API**：`sandbox.create` / `sandbox.close`（抢项目锁）、`project.list_*`（读放大）、以及 `layerfs` CLI（每次起新 Node 进程）。`mount` / `exec` / `commit` / `unmount` / `fork` 都**不**碰项目锁，可以并行。

### 2.4 Python 侧：Gym 接口

```python
# layerfs_env.py
import json, uuid
from dataclasses import dataclass, field

class LayerFSEnv:
    """不是标准 gym.Env —— tool 执行是 IO 密集异步的，建议用 async 版本
       或把 G 个 env 打包成 vectorized step。"""
    def __init__(self, rpc, task_spec):
        self.rpc = rpc                 # → Node env-server (UDS)
        self.task = task_spec          # {task_id, base_commit, image, max_turns}

    async def reset(self):
        self.handle = await self.rpc("rollout.start",
                                     base=self.task["base_commit"],
                                     tag=self.task["task_id"])
        self.turns = 0
        self.obs = await self.rpc("rollout.exec", self.handle, self.task["setup_cmd"])
        return self.obs

    async def step(self, tool_calls: list[dict]):
        """tool_calls 由 vLLM 的生成结果解析而来（文件操作走 fs.*，bash 走 exec）。"""
        obs = []
        for tc in tool_calls:
            if tc["name"] == "bash":
                obs.append(await self.rpc("rollout.exec", self.handle, tc["cmd"]))
            else:                       # 一批文件改动用 fs.apply，只产生一个 commit
                obs.append(await self.rpc("rollout.fs_apply", self.handle,
                                          tc["changes"], tc["message"]))
        await self.rpc("rollout.commit", self.handle, f"turn {self.turns}")
        self.turns += 1
        done = self.turns >= self.task["max_turns"]
        reward = 0.0
        if done:
            reward = (await self.rpc("rollout.score", self.handle))["score"]
            await self.rpc("rollout.end", self.handle)
        return obs, reward, done

    def state_fingerprint(self):
        """训练用：当前 turn 对应的 commit id，可复现、可 fork、可 blame。"""
        return self.handle["commits"][-1]
```

### 2.5 和 vLLM / torchtitan 的接口契约（这是最容易出 bug 的地方）

<details>
<summary><b>展开：token 对齐、loss mask、GRPO 分组</b></summary>

1. **action = tokens，不是文本**。env 侧不要重新 tokenize 模型输出，直接保存 vLLM 返回的 token ids + logprobs（`prompt_logprobs` / `echo` 模式或 `return_tokens_as_token_ids`）。observation 也以 token 形式拼回上下文，**`loss_mask=0`**。
2. **GRPO 的分组天生对齐 fork**：同一 prompt 的 $G$ 条 rollout 必须从**同一个 base commit** fork。用 `branch.fork(name, baseRef)` 一次，$G$ 个 Branch 共享全部依赖文件（去重后存储成本 ≈ 1 份），于是：
   $$A_i = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^{G})}{\mathrm{std}(\{r_j\}) + \epsilon}$$
   分母的方差来自采样温度，不来自环境差异 —— 这正是你想要的（环境差异被 fork 消除了，credit 归给策略）。
3. **每个 rollout 的 reward 必须绑定 commit id**。`verifier(commit_ref)` 才可复现；否则你没法解释"这条 rollout 为什么 0.7"。生产上把 `(task_id, branch, commit_ids[], score)` 写进 JSONL，和 token 轨迹用 `(rollout_id, turn)` join。
4. **torchtitan 侧基本不用改**：它只吃 packed tokens + labels + advantages。你的 rollout 产出 `(input_ids, loss_mask, logprobs, advantage)` 的迭代器即可。注意 turn 粒度的 advantage 广播到 token 时，只广播到**模型生成的 token**上。
5. **exec 输出必须截断**：`workspace.exec` 无大小限制地返回全部输出（>1MiB 会 spool 到临时文件），而 token 预算是有限的。建议 `| tail -c 8000` 或在 Node 侧传 `{stdout: sink}` 的 Writable 开关。

</details>

### 2.6 吞吐与容量：把数字先按住

| 指标 | 现实约束 | 你该做的事 |
|---|---|---|
| **并发 rollout 上限** | Sandbox = 一个容器；1 Sandbox ≤ 16 Workspace；单进程最多 6 个并发调用到一个 Sandbox（daemon 限 8） | 池化 Sandbox；并发 = 沙箱数 × 16，按 CPU/RAM 定 |
| **Sandbox 生命周期** | 空闲 2 分钟自关；`create`/`close` 抢项目锁 | 预热池 + 复用，别在 step 里建 |
| **Store 只增不减** | **没有 GC**，历史永不删 | 每跑 $N$ 条 rollout 就整个 Store 重建：删 project + volume，从 golden folder 重新 `LayerFS.init` |
| **写是单 writer** | 读取不等待，写排队（短持有） | 每 turn 一次 commit，别每 fs op 一次 |
| **持久性** | Disposable，断电可丢 | 任务语料放 git / 对象存储；rollout 产物本来就是一次性的，丢得起 |
| **失败语义** | **从不重试**；`Unknown` 必须人工检查，不能回放 | wrapper 把 `Unknown/Failed` 标成 invalid rollout，`loss_mask=0` 丢弃 |
| **单 kernel** | 一个 Store 只能一台机器 | 按 GPU 节点分片，每节点一个 Project，base commit 用同一个 golden folder `init` |

**存储量估算**：假设仓库树 50k 文件 / 300MB（含依赖），每 turn 改 5 个文件、每个 2KB 编辑、20 turn：

$$
\text{每 rollout 新增} \approx 20 \times 5 \times 2\text{KB} = 200\text{KB} \;(\text{去重后}) + \text{元数据}
$$

一万条 rollout ≈ 2GB 量级。这个数字可以接受；但如果你的编辑是"重写整个大文件"，收益会掉——除非命中 §10 "Big file, small edit"。

### 2.7 分阶段落地（建议按这个顺序，每步都能单独止损）

| 阶段 | 做什么 | 止损判据 |
|---|---|---|
| **P0 基线测量（0.5 天）** | 在你**真实任务仓**上 `LayerFS.init`，测：`mount` p50/p99、`exec` 开销、`commit` p50/p99（随树大小）、每 turn Store 增量 | 如果 `commit` p99 ≫ 你一个 turn 的生成时间，或某 turn 新增 > 10MB，**放弃**，改用预热容器池 |
| **P1 单机打通（2–3 天）** | Node env-server（单 Project、Sandbox 池）+ Python env 的 `reset/step`；跑通一条 SWE 风格 rollout，reward 走 commit id | 端到端成功、reward 可复现 |
| **P2 GRPO 分组** | `fork` × $G$、并发 $K$ 条 rollout、token/loss_mask 轨迹落盘 | 训练能跑 100 step 不崩、不 OOM、Store 不爆盘 |
| **P3 树搜索 / 断点续跑** | 从 `commit_k` 重 fork 做 best-of-N / beam；用 `fs.blame`、`branch.diff_commits` 做辅助信号 | 开启后 reward 曲线上升或相同 reward 下采样成本下降 |
| **P4 观测** | 用 `@layerfs/backend` 的 call trace / 存储统计做 dashboard；`LAYERFS_TRACE=1` 记录每次调用 | 定位"哪一步慢、哪个 rollout 吃盘" |

### 2.8 三个容易出问题的细节

1. **不要让 agent 看到整个 Store**。Store 是所有 Branch 共享的，如果 agent 手里有 `layerfs` CLI，它就能 `fs.read` 别人的 branch（信息泄漏 + reward hacking）。RL 里只给它 sandbox 内的普通 bash，**不给 CLI**。
2. **base commit 必须是"装好依赖"的树**。这是全部收益的源头。做法：本地起容器 → `npm ci` / `uv sync` → `docker commit` 或直接导出一个目录 → `LayerFS.init(name, folder, timeout)`（默认 300s，大仓要传更大值）。之后所有 fork 都从 `main` 走。
3. **`exec` 不带 timeout**，而且"命令退出 ≠ 它启动的后台进程结束"，`commit` 会把这半截状态一起存下来。所以在 server 侧无条件包 `timeout`，并在 turn 结束时杀掉残留后台进程（否则上一 turn 的 dev server 会污染下一 turn 的 reward）。

### 2.9 什么时候**不**该用 LayerFS（替代方案）

如果你的任务满足任意两条，直接把 `tmpdir + overlayfs + 预热容器池` 或 `git worktree + 共享 `node_modules` 缓存目录` 做完就行，别背 Rust 构建和 FUSE：

- 任务仓很小（< 5k 文件），依赖安装 < 10s；
- **不需要**从中间步骤重采样 / 树搜索（只有单条 trajectory）；
- 不需要按工具调用粒度回滚；
- 你的训练节点不是 Linux / Store 不在本地盘。

在 RL 语境下，LayerFS 唯一不可替代的价值就一句话：**"把一条 episode 的中间状态变成零成本可拷贝、且带着依赖和构建产物的对象"**。如果这个能力对你的算法（tree search / resample-from-step-k / 大规模 best-of-N）不是必需的，那就是在为一个用不到的特性承担全部安装成本。

---

## 3. 一句话收尾

- **普通用户**：你的直觉对，`git worktree` 更划算；只有当你在一台 Linux 机器上并行跑多个 agent、且每次都要花几分钟装依赖/编译时，"5 分钟 vs 0.05 秒的 mount"才开始值回票价。而且**永远让 git 当真相源**（文档自己也这么说，因为 Store 是 Disposable）。
- **RL 用户**：你要的是 **`@layerfs/agent` SDK**，不是 `dsh-layerfs` 插件。用一个长驻 Node env-server 池把 `fork / mount / exec / commit` 包成 RPC，Python 侧写 Gym 风格的 `reset/step`，reward 以 `commit_id` 为契约，GRPO 的 $G$ 条分组直接映射到从同一 base commit 的 $G$ 次零拷贝 fork。**先做 P0 的三个数字测量（mount、commit p99、每 turn Store 增量），再决定要不要把整个 gym 焊在它上面。**

参考：`github.com/Ephemeral-AI-Lab/layerfs`（引擎 + `api/docs/agent.md` 是 SDK 权威参考）、`github.com/Ephemeral-AI-Lab/dsh-layerfs`（UI 插件）、`layerfs.ai/docs/platform/`（平台与限制，上面所有硬约束都出自这里）。

---
## 🧑‍💻 User

Windows SmartVersion是什么东西 和上文提到的插件以及git/svn有什么异同？

---
## 😀 默认助手 (7)

# SmartVersion：它不是 VCS，而是一个“单文件差分归档器”

先给结论，后面展开，并给出与 git、svn、LayerFS 的逐维度对照。

> **一句话**：SmartVersion 把**同一个文件（或一组文件）的多个历史版本**全部装进一个 `.svf` 文件里，只存版本之间的**二进制差分 + 压缩**，然后可以按版本号把任意一版“解出来”，或者切出一个只含某几版的“极小补丁包”发给别人。
> 它更接近 `zip + bsdiff + 版本目录`，而不是 `git / svn`。

顺带纠正一个容易产生的错觉：**它跟上文那个插件没有关系**。`dsh-layerfs` 是给 DeepSeek Harness 的 agent 挂 CLI 和两个 UI tab 的插件；它是**文件系统 + 挂载 + 执行**。SmartVersion 只负责**存储和还原某个文件的历史**，既不挂载、不执行，也没有工作树的概念。二者唯一的共同点是“单文件、追加式、按差分存多份近似快照”。

---

## 1. 它到底是什么

<details>
<summary><b>展开：来源、现状、形态、数据模型</b></summary>

**来源与合规形态**

- 作者：**Gilles Vollant**（法国，WinImage 的作者，也是 UnZIP / Info-ZIP 相关工作的老人），2002 年起，2023 年仍在更新（4.00）。
- 形态：长期以 **shareware** 发布（Windows 上有 GUI `SmartVS.exe` + 控制台 `smv.exe`），**4.00（2023）起源码在 GitHub 以 MIT 发布**（`gvollant/smartversion`）。
- 平台：Windows 是主战场（GUI）；4.x 起有 Linux / macOS / Windows ARM64 / FreeBSD / AIX 的**纯命令行版**。注意它已经**不只是 Windows 工具**了，虽然你搜到的多半还是 Win32 时代的页面。

**数据模型（关键，决定了它能干什么、不能干什么）**

一个 `.svf` 文件 = 一个**线性编号的版本列表** `v1, v2, v3 … vN`，每个版本是一组文件的快照：

$$
|\text{SVF}| \;\approx\; |v_1|_{\text{compressed}} \;+\; \sum_{i=2}^{N} |\Delta(v_{i-1} \to v_i)|
$$

- 存的是**块级二进制差分**（内部 blocksize 8–8192，可选 zlib / LZMA / zlib+lzma 混合，4.x 换成 **zstd**，3.70 还支持过 LZHam）。
- **`-br base_directory`（reference / base 目录）是这个工具的核心概念**：`cr`（create reference）只存第一版的**文件名 + 大小 + 校验和**，不存内容；之后要 `x`（解出）任意版本，你必须本地有那个 base 目录的文件，工具拿它当“第 0 版”来套差分。所以它是**前向差分链**模型，不是 Git 那种“每版都是完整快照、随时能独立还原”。

**它的 CLI 动词（就这几个，很能说明定位）**

| 命令 | 作用 |
|---|---|
| `smv cr` / `cz` | 建 SVF，第一版为 reference（只存名/大小/校验和）或直接压缩存入 |
| `smv i` / `iz` | **加一个新版本**（从目录或 zip 导入） |
| `smv a` / `d` | 在**已有版本**里增/删文件（`-v 1-3`） |
| `smv x` / `xz` | **解出某一版**（`-v #`，默认最新）到目录 / zip |
| `smv l` / `lv` | 列出内容 |
| `smv t` | 用 base 目录测试完整性 |
| `smv av` | 把另一个 SVF 的版本**追加**进来，**要求边界版本完全一致**，否则拒绝 |

看到没有：**没有 branch、没有 merge、没有 commit message/作者、没有 diff 到“行”、没有远程协议**。`av` 那个“必须严格匹配才能追加”，本质上就是**只能 fast-forward 的、毫无三方合并能力的拼接**。

**典型用法**（官方给的两个例子就是全部场景）

1. 作者 ↔ 编辑改书稿/文档：第一次发全量，之后每次只发“极小补丁 SVF”，对方用本地旧版当 base 解出新版。
2. 站长每天存档一版网站（HTML + JPG/GIF）：10 个 ZIP 共 10MB → 一个 SVF 只占 2.5MB；事后还能“只保留我要的那 10 个版本”切出一个归档 SVF（partial patch）。

</details>

---

## 2. 和 git / svn / LayerFS 的异同

### 2.1 一张总表

| 维度 | **SmartVersion** | **git** | **svn** | **LayerFS（`dsh-layerfs` 底下那层）** |
|---|---|---|---|---|
| 定位 | 版本**归档** + 差分补丁**分发** | 分布式源码版本控制 | 集中式版本控制 | 给 agent 用的**可挂载文件系统**（Workspace/Commit） |
| 数据结构 | **线性**版本链 $v_1..v_N$ | **DAG**（commit 图） | 线性 revision + 廉价 copy（分支=目录拷贝） | **Branch/Commit/Layer**，可按 commit 递归 fork |
| 单次提交的粒度 | 整个文件集合的一个“版本号” | tree snapshot + parent(s) | tree 变更集 | 一棵树（且带 ignored 文件、依赖、构建产物） |
| 存储方式 | **前向块级差分 + 压缩** | 内容寻址对象 + packfile(delta+zlib) | 服务端 svndiff + 全文/增量 | 内容分块去重 + 压缩（157 snapshots 100MB vs 2GB） |
| 需要 base 才能还原 | ✅ `-br` 必须给第 0 版 | ❌ 每个 commit 自足 | 半（本地 working copy） | 半（Store 就是全部历史，本地必须能打开同一个 Store） |
| 分支 / 合并 | ❌ 无分支、无三方合并；`av` 仅 fast-forward | ✅ 核心能力 | ✅（merge tracking，比较痛） | ✅ merge / publish / `Conflict` 拒绝 / `.lww` |
| 冲突处理 | 无（单写者顺序追加） | 三方合并 + 冲突标记 | update 冲突 + 锁 | 同文件双改直接 `Conflict`，不乱合 |
| 元数据 | 版本号、名字、注释、校验和 | 作者/时间/message/parent | 同上 + revprop | 无作者，commit/exec message ≤4096B |
| 工作副本 | ❌ 只有“解到某个目录” | ✅ 完整 working tree + index | ✅ working copy | ✅ **FUSE 挂载的 Workspace**，可 `exec` 命令 |
| 能跑命令 / 隔离 | ❌ | ❌ | ❌ | ✅ Docker Sandbox 内挂载执行 |
| “每一小步自动留痕” | 手动 `smv i` | 手动 `git commit` | 手动 commit | ✅ **每次工具调用自动 Commit** |
| 传输 | 把 `.svf` 文件发出去（邮件/U 盘） | push/pull（智能协议，可 file://） | 中心服务器 | 单机单 kernel，**没有跨机协议** |
| 跨平台 | Windows GUI + 各平台 CLI | 全平台 | 全平台 | **Linux only**（Store 必须本地 ext4/xfs/btrfs，macOS/Windows 靠 worker 转发） |
| 成熟度 / 生态 | 单人维护 20 余年，小而专 | 事实标准 | 企业中仍有大量存量 | 2025/26 新项目，Disposable、无 GC |

### 2.2 像在哪（这是你会把它联想到一起的原因）

1. **都是“单文件 + 追加式 + 只存差异”。**
   `store.sqlite`（LayerFS）和 `xxx.svf`（SmartVersion）都是**一个文件装全部历史、只增不改**；都靠“先分块/差分，再去重/压缩”把 N 份近似快照压到接近 1 份。
2. **都靠“base + 差分”重建旧状态。**
   SmartVersion 的 `-br base_directory` 和 LayerFS 的 “Workspace 在已提交的 Store 上挂一层 overlay” 是同一类思路。
3. **都没有“远程仓库”这个中间层。**
   SmartVersion 直接发文件；LayerFS 干脆规定“必须同一个 Linux kernel”，连网络协议都不做。
4. **都被“当成 VCS”误会过。**
   SmartVersion 常年出现在各种“版本控制软件列表”里；LayerFS 的 README 也得专门写一段 “Use something else if you need durability / POSIX / sequential bandwidth”。

### 2.3 不像在哪（这才是重点）

| | SmartVersion | LayerFS |
|---|---|---|
| 你操作的对象 | 一个**档案文件**：`smv i` / `smv x` | 一个**活的工作树**：`mount` / `exec` / `commit` |
| 分支能力 | 完全没有 | 核心卖点：零拷贝递归 fork |
| 冲突语义 | 没有（`av` 边界不匹配就整个拒绝） | `Conflict` 拒绝 + `.lww` 最后写入者胜 + merge 标记 |
| 谁能用 | 人（发补丁的厂商、存档的作者） | **agent（每条工具调用一个 commit）** |
| 有没有“执行” | 没有 | 有（Docker + FUSE + daemon） |
| 主要价值 | **补丁体积最小化** | **每条 rollout / 每个 agent 一颗秒级就绪的树** |

和 git 的区别就更直接了：**SmartVersion 没有“提交图”，所以它无法回答 git 能回答的大部分问题**：谁改的、为什么改、这个改动和那个改动是不是同一个祖先、两边都改了同一行怎么办、我要把这两个分支合起来。它只会回答一件事：**“第 7 版长什么样，以及从第 6 版到第 7 版的最小补丁有多大。”**

和 svn 的区别：svn 至少在服务端有**变更集、作者、原子提交、锁、mergeinfo**，是个真正的协作系统；SmartVersion 连“并发写”这个概念都没有，`smv i` 就是一个人顺序往后追加版本号。

---

## 3. 那它今天还有用吗？该用什么替代？

<details>
<summary><b>展开：适用场景与现代替代品</b></summary>

**仍然成立的场景（很少但真实）**

- **Windows 桌面软件/游戏发增量更新包**：你有 `app_v1.exe`，想给已装 v1 的用户发一个只含差异的小包。这活 `bsdiff/bspatch`、`zstd --patch-from`、微软的 `msdelta`/`PatchAPI`、ClickOnce 也能干，但 SmartVersion 的好处是**一个 SVF 里塞多个版本 + 能给不同用户切不同版本子集 + GUI**。
- **大二进制资产的“版本档案馆”**：设计稿、CAD、视频工程、扫描件，diff 在行级别无意义，但块级差分很有效。
- **离线/窄带分发**（邮件、U 盘），没有服务器可用。

**不该用的场景（大多数）**

- **管源码** → git。行级 diff、merging、code review、CI，SmartVersion 一样都没有，用它管代码是纯亏。
- **团队协作 + 需要审计/权限** → svn / git + 托管平台。SmartVersion 没有身份、没有原子提交、没有权限模型。
- **只是想给大二进制做版本管理** → 现代做法一般是 **Git LFS / DVC**（把大文件外置，git 管指针）、**restic / borg / zpaq**（去重快照备份）、或 `zstd --patch-from=old new`（一条命令生成差分）。它们的共同点是**融入现有工作流**，而不是让你回到“发文件”的时代。

**一句话选型**

| 你要什么 | 用什么 |
|---|---|
| 源码的版本、分支、合并、协作 | **git**（集中式合规场景 → svn） |
| 给已安装用户发最小增量补丁 | `zstd --patch-from` / `bsdiff` / `msdelta`，需要多版本档案时再考虑 SmartVersion |
| 大二进制进版本库 | **Git LFS / DVC** |
| 多份近似快照要省空间地存起来 | restic/borg/zpaq（备份）或 LayerFS（要挂载执行时） |
| 给 agent 一颗 5ms 就绪、带依赖、可 fork/回滚的树 | **LayerFS**（Linux + Docker + FUSE） |

</details>

---

## 4. 收尾：三句话记住区别

1. **SmartVersion = 档案柜**：一个文件里按序号放很多版，取哪版拿哪版，取的时候要你手里有“第一版”当底。它的 KPI 是**补丁字节数**。
2. **git / svn = 账本**：记录**谁在什么时候、基于谁的改动**做了什么，能分支、能合并、能追责。它们的 KPI 是**协作正确性**。
3. **LayerFS（`dsh-layerfs` 的底座）= 工作台**：一颗能秒挂、能跑命令、能 fork 出无限副本的活工作树，历史是副产品。它的 KPI 是**每次尝试的启动延迟**。

三者唯一重叠的地方是“**用差分 + 压缩把近似快照存小**”——这是存储层技巧，不代表它们属于同一类工具。你把 SmartVersion 和 git 并列、把 LayerFS 和 dsh 插件并列，其实是把**存储技巧**、**协作系统**、**执行环境**三件事放在了一张比较表里。

---
## 🧑‍💻 User

这样，看来LayerFS 不是 gym，别指望直接当 env 用
它给的是「状态层」——而 agentic RL 里最贵、最容易出问题的一环恰恰是 reset 到干净基线，这正是它的长处
但 Gym wrapper 必须你自己写，这是绕不掉的
我们建议的架构是这样
reset：从 baseline commit 里 fork 一条 Branch（这一步 LayerFS 极快，也是它真正的价值）
step：在这一支的 Sandbox 里跑一条命令，只收 stdout / exit code / 变更的文件列表
observation：命令输出 + 当前 diff 摘要，别把整个文件系统塞进 token
reward：交给你的 verifier 算，这一步完全独立
done：verifier 判定通过 / 超过最大步数 / 连续无进展
这样你的 env 就是纯 Python 薄壳，只调 layerfs CLI，不碰 dsh 那套
下面几个工程坑都得提前想清楚
①隔离形态：文档里提到 Sandbox 在部分平台靠 dsh Live + WSL2/Docker 实现，如果你要跑不可信代码，得自己加一层容器或 cgroup，不能只依赖它
②吞吐：每次 reset 都会挂一份 Workspace，并发多了会吃 inode 和挂载点，得设上限并复用
③训练侧对接：vLLM 出 rollout、torchtitan 吃 token，中间缺的是「动作空间的定义」——你需要把 agent 的动作限定成有限命令集，否则 RL 信号会极稀疏
④最要命的是奖励作弊：agent 完全可以直接改测试文件、改 verifier 配置来「通过」
所以在 Sandbox 里必须把 verifier 和测试用例挂成只读（或者在宿主机侧跑 verifier），并且用 git status 比对，禁止它改动判定相关目录
一句话总结：LayerFS 解决的是「reset 又慢又贵」这个瓶颈，不是替你解决 RL 环境本身；把它当高速快照层，Gym 接口和奖励防护都得你自己写

现在为我搭建基于DSH+LayerFS的arl gym design 框架

---
## 😀 默认助手 (7)

# ARL-Gym on DSH + LayerFS：设计框架 v0.1

先接住你的定位，并把它固化成三条**架构铁律**——后面所有代码都是这三条的推论。

> **LayerFS 是状态层（快照/fork/mount），不是 env，也不是 sandbox jail。**
> Gym 接口、动作空间、奖励防护、隔离加固，全部是我们自己写的代码。

**铁律（任何实现都必须满足）**

| # | 铁律 | 推论 |
|---|---|---|
| **L1** | **可信通道 / 不可信通道必须分离** | observation 可以来自不可信通道（agent 的 stdout、sandbox 里的 `git status`）；**判定只能来自可信通道**（宿主机侧 LayerFS Store 的 `diff_commits` + judge 容器里的 verifier） |
| **L2** | **LayerFS 从不重试，我们也不重试** | 所有 `Unknown` / `Failed` 一律 → `valid=false`、`loss_mask=0`，禁止 replay。否则噪声进训练 |
| **L3** | **训练热路径零依赖 dsh** | `import arlgym` 不 import 任何 dsh 包。dsh 只出现在两处：① 人类看回放（把同一个 Project 挂进 `dsh-layerfs` 的 Work/History tab，只读）② tool schema 对齐（保证训练分布 = 部署分布） |

---

## 1. 责任边界（一页纸）

| 组件 | 拥有什么 | **不**拥有什么 |
|---|---|---|
| **LayerFS** | Store（单文件 SQLite）、Branch/Commit/Layer、`fork`（零拷贝）、`sandbox`（容器）、`workspace`（FUSE 挂载 + exec + commit）、容器内的资源隔离**下限** | ❌ 不是 gym ❌ 不是 verifier ❌ 不是安全边界（Docker container ≠ microVM）❌ 不是 GC ❌ 不是持久存储 |
| **arlgym（我们）** | `reset / step / done`、动作空间、观测渲染、Sandbox 池与背压、guard、judge 编排、轨迹/token 记录、Store 轮转 | ❌ 不算奖励（交给你的 verifier）❌ 不训模型 |
| **你的 verifier** | 判定逻辑、测试集、部分奖励 | ❌ 不碰文件系统状态（只读 judge 环境） |
| **vLLM** | 采样 + logprobs（**必须原样保存，禁止重新 tokenize**） | ❌ 不定 done ❌ 不解析测试输出 |
| **torchtitan** | 吃 `(input_ids, loss_mask, logprobs, advantage)` | ❌ 不感知 LayerFS 存在 |
| **DSH** | 人类调试/回放、tool 语义参考实现 | ❌ 不在训练环里 |

---

## 2. 部署拓扑

```
┌─ 训练节点（Linux, 本地 ext4/xfs/btrfs） ────────────────────────────────────┐
│                                                                            │
│  Python: arlgym ShardRunner (asyncio)                                      │
│    ├─ G 个 ArlEnv ───────────┐                                             │
│    ├─ SandboxPool (cap=S)    │  ← 唯一接触 LayerFS 的入口                    │
│    └─ JudgeRunner ───────────┘                                             │
│           │ subprocess: `layerfs ...`（thin shell，JSON out）                │
│           ▼                                                                 │
│  ┌─ 外层边界：一台专用 VM / Firecracker / 独占云机（无凭证、出网白名单）──┐   │
│  │  dockerd（硬化：default-runtime=runsc、seccomp、ulimit、独立 bridge） │   │
│  │    ├─ Sandbox #1..S  ← LayerFS 创建的容器，每个 ≤16 Workspace(FUSE)   │   │
│  │    └─ Judge Sandbox  ← 同一机制，但由 judge 独占、agent 不可达         │   │
│  │  layerfs-daemon + Store: /var/lib/docker/volumes/<vol>/_data/         │   │
│  │                          global/store.sqlite  ← 单文件、单写者        │   │
│  └────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
        ▲ reward / tokens                                    │ rollout 请求
        │                                                    ▼
   torchtitan ◄── EpisodeRecord(JSONL/parquet) ◄── vLLM (batch 生成 + logprobs)
```

**分片规则**：一个 Store = 一个 kernel = 一台机器。$N$ 个 shard = $N$ 个 `(LAYERFS_HOME, LAYERFS_STORE_PATH, LAYERFS_CONTAINER_PREFIX)` 三元组，`task_id % N` 路由。

---

## 3. 数据契约（先定 schema，再写代码）

<details>
<summary><b>展开：TaskSpec / Action / Observation / Episode 的完整定义</b></summary>

```python
# arlgym/env/spec.py
from dataclasses import dataclass, field
from typing import Literal, Any

@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    base_commit: str              # LayerFS Commit / Branch name，如 "main" 或 "a3f1c9d"
    image: str                    # "sha256:..." 不可变镜像（含依赖 + /opt/judge）
    prompt: str                   # 任务描述（进 system+user）
    max_turns: int = 24
    wall_clock_s: int = 900
    exec_timeout_s: int = 300
    verifier: str = "judge:v1"    # 镜像内 /opt/judge/run.sh 的版本标识
    protected: list[str] = field(default_factory=lambda: ["tests/", "conftest.py", "pyproject.toml"])
    store_shard: int = 0

@dataclass
class Action:
    """动作空间是封闭文法（见 §6③）。由 vLLM constrained decoding 保证合法性。"""
    tool: Literal["bash", "read", "write", "patch", "finish"]
    arg: str                      # bash: 单条命令；write/patch: 目标路径
    body: str | None = None       # write/patch 的内容
    thought: str | None = None    # 可选 CoT，不进 loss 也可进

@dataclass
class Change:
    path: str
    status: Literal["A", "M", "D", "R"]
    src: Literal["store", "sandbox"]   # ← L1: store=可信, sandbox=仅供参考

@dataclass
class TurnObs:
    exit: int | None
    stdout: str          # 已按 token 预算截断（头+尾）
    stderr: str
    changed: list[Change]         # store 通道 → 可信；sandbox 通道 → 仅展示
    diffstat: str | None          # "3 files changed, +12 -4"
    diff_excerpt: str | None      # 仅前 N 个文件、每个 M 行的 hunk
    commits: list[str]            # 本 turn 之后的 commit ids
    guard_hits: list[str]         # 命中 protected 的路径（可空）
    budget: dict                  # turns_left / tokens_left / wall_left
```

**Episode 记录（训练侧唯一真相）**

```python
@dataclass
class Turn:
    idx: int
    action: str                 # 原始生成文本（含 json 外壳）
    action_ids: list[int]       # vLLM 返回的 token ids（原样）
    action_logprobs: list[float]
    obs_text: str               # observation 渲染后的纯文本
    obs_ids: list[int]          # 用 env 侧 tokenizer 编的（loss_mask=0）
    commit: str | None
    changed: list[str]

@dataclass
class EpisodeRecord:
    episode_id: str
    task_id: str
    group_id: str               # = f"{task_id}@{base_commit}"  ← GRPO 分组键
    base_commit: str
    final_commit: str | None
    turns: list[Turn]
    reward: float
    reward_audit: dict          # verifier 的原始输出（可信）
    valid: bool
    fault: str | None           # Unknown / NoSuchSandbox / Backpressure / GuardTrip ...
    guard: dict
    store_bytes_delta: int      # 用于成本核算
```

</details>

---

## 4. 代码骨架

### 4.1 唯一接触 LayerFS 的模块（thin CLI shell）

<details>
<summary><b>展开：<code>arlgym/layerfs/cli.py</code></b></summary>

```python
"""薄壳：subprocess 调 layerfs CLI，JSON 进出。
   绝不重试（L2）。exit 2 = 命令行错误（我们的 bug），125 = LayerFS 拒绝/失败。"""
import json, os, shlex, subprocess
from dataclasses import dataclass

CLI = os.environ.get("ARLG EX_CLI", "layerfs")   # 实际用 ARLG_EX_CLI
PREFIX = os.environ.get("LAYERFS_CONTAINER_PREFIX", "arl-")

class LayerFSError(Exception):
    def __init__(self, code, message, detail=None):
        super().__init__(f"{code}: {message}")
        self.code, self.detail = code, detail
class ConfigError(Exception): ...
class Transient(Exception): ...      # 仍然不重试，只是打标给上层做 invalid

@dataclass
class Res:
    data: dict | None
    exit: int

def call(*args, project: str, timeout: float | None = 120.0) -> dict:
    cmd = [CLI, "--project", project, *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, timeout=timeout,
                       env={**os.environ, "LAYERFS_CONTAINER_PREFIX": PREFIX})
    out = p.stdout.decode("utf-8", "replace").strip()
    if p.returncode == 0:
        return json.loads(out) if out else {}
    if p.returncode == 2:
        raise ConfigError((p.stderr or b"").decode("utf-8", "replace"))
    if p.returncode == 125:                       # {"code","message","detail"}
        e = json.loads((p.stderr or b"{}").decode("utf-8", "replace") or "{}")
        raise LayerFSError(e.get("code", "Failed"), e.get("message", ""), e.get("detail"))
    raise Transient(f"unexpected exit {p.returncode}: {p.stderr[:400]!r}")

# ---- 语义封装（每个都对应一条 CLI，动词严格照 CLI 表）----
def fork(project, name, from_ref):        return call("branch","fork", name, from_ref, project=project)
def sandbox_create(project, image):       return call("sandbox","create","--image",image, project=project)
def mount(project, sandbox, branch):      return call("workspace","mount",sandbox,branch, project=project)
def commit(project, ws, msg):             return call("workspace","commit",ws,"-m",msg, project=project)
def unmount(project, ws, discard=True):   return call("workspace","unmount",ws,"--discard" if discard else "", project=project)
def sandbox_close(project, sb):           return call("sandbox","close",sb, project=project)
def diff_paths(project, a, b, cursor=None, size=1000):
    args = ["branch","diff_commits",a,b,"--page-size",size,"--no-hunks"]
    if cursor: args += ["--cursor", cursor]
    return call(*args, project=project)
def list_commits(project, branch, size=50): return call("branch","list_commits",branch,"--page-size",size, project=project)
def ls_sandboxes(project):                return call("project","list_sandboxes", project=project)
```

**step 命令的包装器**（这是整个框架里最不显眼、最容易出错的一段）

```python
# 为什么不用 --stream：--stream 直接把退出码透传，而 125 也代表 LayerFS 失败 →
# 命令自己退出 125 时无法区分。所以用非 stream 模式（永远 exit 0），
# 把真实退出码用 marker 打出来；同时把输出钳制在固定字节数，天然对上 token 预算。
STEP = r"""cd {ws} || exit 97
timeout -k 5 {t} bash -lc {cmd} >/tmp/.arl.out 2>/tmp/.arl.err
rc=$?
echo __ARL_EXIT__$rc
echo __ARL_STDOUT__
tail -c {so} /tmp/.arl.out
echo __ARL_STDERR__
tail -c {se} /tmp/.arl.err"""

def run_step(project, ws, user_cmd, *, timeout_s=300, stdout_b=6000, stderr_b=2000):
    ws_path = ws.split(":", 1)[1]          # "1:/workspaces/3" -> "/workspaces/3"
    script = STEP.format(ws=ws_path, t=timeout_s, cmd=shlex.quote(user_cmd),
                         so=stdout_b, se=stderr_b)
    r = call("workspace","exec", ws, "-m", "step", "--", script, project=project,
             timeout=timeout_s + 60)
    return _parse_markers(r)
```

</details>

### 4.2 Sandbox 池（对付 ② 吞吐）

<details>
<summary><b>展开：<code>arlgym/layerfs/pool.py</code></b></summary>

```python
"""S 个 Sandbox × 16 Workspace = 容量上限。
   sandbox.create/close 抢项目锁 → 必须预热 + 只用一次（绝不在 step 热路径）。
   mount/exec/commit/unmount 不抢项目锁 → 可并行（单 Sandbox 最多 6 并发调用）。"""
import asyncio, collections

WS_PER_SANDBOX = 16

class SandboxPool:
    def __init__(self, project, image, size=8, queue_timeout=120.0):
        self.project, self.image, self.size = project, image, size
        self._sb: list[dict] = []              # {"n": "1", "free": 16}
        self._lock = asyncio.Lock()            # 保护 create（项目锁本身就慢）
        self._slots = asyncio.Semaphore(size * WS_PER_SANDBOX)
        self._queue_timeout = queue_timeout
        self._reaper: asyncio.Task | None = None

    async def warm(self):
        # 预热到 size；create 串行（LayerFS 项目锁），一次到位，之后不再 create
        for _ in range(self.size):
            async with self._lock:
                r = await asyncio.to_thread(cli.sandbox_create, self.project, self.image)
            self._sb.append({"n": r["sandbox"], "free": WS_PER_SANDBOX - 1, "image": r["image"]})

    async def lease(self, branch) -> str:
        try:
            await asyncio.wait_for(self._slots.acquire(), self._queue_timeout)
        except asyncio.TimeoutError:
            raise Backpressure("sandbox pool saturated")      # → episode invalid
        sb = min((s for s in self._sb if s["free"] > 0), key=lambda s: -s["free"], default=None)
        if sb is None:                                          # 全部满 → 扩一个
            async with self._lock:
                r = await asyncio.to_thread(cli.sandbox_create, self.project, self.image)
            sb = {"n": r["sandbox"], "free": WS_PER_SANDBOX - 1, "image": r["image"]}
            self._sb.append(sb)
        sb["free"] -= 1
        try:
            m = await asyncio.to_thread(cli.mount, self.project, sb["n"], branch)
            return m["ws"]            # "1:/workspaces/7"
        except Exception:
            sb["free"] += 1; self._slots.release(); raise

    async def release(self, ws, *, discard=True):
        try:
            await asyncio.to_thread(cli.unmount, self.project, ws, discard)
        finally:
            n = ws.split(":", 1)[0]
            for s in self._sb:
                if s["n"] == n: s["free"] += 1
            self._slots.release()
```

**容量公式**（把它写进 config，别凭感觉设）

$$
C_{\max} = S \times 16, \qquad
S \ge \max\!\left(\left\lceil \frac{C_{\max}}{16}\right\rceil,\;
\frac{Q_{\text{commit}} \cdot t_{\text{publish}}}{0.7}\right), \qquad
N_{\text{shard}} \ge \left\lceil \frac{Q_{\text{commit}} \cdot t_{\text{publish}}^{\text{p99}}}{0.7} \right\rceil
$$

- $Q_{\text{commit}}$：全节点每秒 commit 数；$t_{\text{publish}}$：**单 Store 单写者**的发布延迟（含 merge + 树重建）。
- 分母 0.7 = 70% 利用率上限（留出 judge 与 reaper 的余量）。
- **这条公式就是"LayerFS 无 GC、单写者"这两条硬约束在容量上的投影**：并发拉不上去时，先看是不是 publish 串行化了，答案通常是**加 shard**，不是加 sandbox。

</details>

### 4.3 Gym 本体（`reset / step / done`）

<details>
<summary><b>展开：<code>arlgym/env/env.py</code></b></summary>

```python
class ArlEnv:
    def __init__(self, task: TaskSpec, pool, judge, obsr, cfg):
        self.task, self.pool, self.judge, self.obsr, self.cfg = task, pool, judge, obsr, cfg
        self.history: list[Turn] = []
        self.valid = True; self.fault = None

    # ---------------- reset：唯一"贵"的一步，也是 LayerFS 的全部价值 ----------------
    async def reset(self, episode_id: str):
        self.branch = f"{self.task.task_id}-{episode_id[:8]}"
        await asyncio.to_thread(cli.fork, self.pool.project, self.branch, self.task.base_commit)
        # ↑ 零文件拷贝；依赖 / .venv / node_modules / target 随 base commit 走
        self.ws = await self.pool.lease(self.branch)          # ~5–20 ms 挂载
        self.prev_commit = self.task.base_commit
        self.t0 = time.monotonic()
        self.no_progress = 0
        return self._initial_obs()

    # ---------------- step ----------------
    async def step(self, a: Action):
        if a.tool != "bash":
            r = await self._apply_structured(a)               # read/write/patch：走 fs.*
        else:
            r = await asyncio.to_thread(cli.run_step, self.pool.project, self.ws, a.arg,
                                        timeout_s=self.task.exec_timeout_s)
        # ① 每 turn 一次 commit（checkpoint；tree search / 断点续跑靠它）
        if self.cfg.checkpoint_every_turn:
            c = await asyncio.to_thread(cli.commit, self.pool.project, self.ws, f"turn {len(self.history)}")
            self.cur_commit = c.get("commit", self.prev_commit)
        # ② 权威变更集：只信 Store 的 diff（L1）
        changed = await self._trusted_changes(self.prev_commit, self.cur_commit)
        # ③ guard：protected 路径命中 → 记账（不阻断，reward 时判）
        hits = guard.check(changed, self.task.protected)
        self.no_progress = self.no_progress + 1 if not changed else 0

        obs = self.obsr.build(r, changed, hits, self._budget(), self.prev_commit, self.cur_commit)
        self.history.append(Turn(idx=len(self.history), action=render_action(a), **obs.raw))
        self.prev_commit = self.cur_commit

        done, why = self._done(a, obs)
        reward = 0.0
        if done:
            reward, audit = await self.judge.score(self.branch, self.cur_commit, self.task, hits)
        return StepResult(obs=obs, reward=awaitator,
                          done=done, done_reason=why, reward_audit=audit if done else {})

    def _done(self, a, obs):
        if a.tool == "finish":                       return True, "finish"
        if len(self.history) >= self.task.max_turns: return True, "max_turns"
        if time.monotonic() - self.t0 > self.task.wall_clock_s: return True, "wall_clock"
        if self.no_progress >= self.cfg.no_progress_k:          return True, "no_progress"
        if not self.valid:                           return True, self.fault
        return False, None

    async def close(self, keep_branch=False):
        await self.pool.release(self.ws, discard=not keep_branch)
```

**done 的判定顺序**（写死，别让模型决定）：

```
hard fault (Unknown/NoSuchSandbox/Backpressure) → invalid, 不跑 verifier
finish 动作                                       → 跑 verifier
max_turns / wall_clock / 连续 K 次空 diff          → 跑 verifier（通常是 0 分，但保留信号）
```

</details>

### 4.4 Judge（对付 ④ 奖励作弊）——**必须是独立容器**

<details>
<summary><b>展开：<code>arlgym/reward/judge.py</code> 与可信/不可信矩阵</b></summary>

**核心思路：把"判定资源"放进镜像层，把"agent 的产物"从 Store 读出来，两个来源在 judge 容器里汇合。**
LayerFS 不提供子目录只读挂载（`workspace.mount` 没有 ro 参数），所以你的"挂成只读"要靠**镜像烘焙 + 覆盖恢复**，不是 `mount -o ro`。

```bash
# 任务镜像（pin by digest）里预置：
/opt/judge/run.sh          # 唯一的官方入口，输出 __ARL_VERDICT__{json}
/opt/judge/tests/          # 测试用例的权威副本（镜像层，agent 改不到）
/opt/judge/toolchain/      # runner 自己的解释器/pytest（不吃树里的 .venv）
```

```python
class JudgeRunner:
    async def score(self, branch, commit, task, guard_hits) -> tuple[float, dict]:
        if guard_hits:                       # 触碰受保护路径：直接 0，但仍留证据
            return 0.0, {"verdict": "guard_trip", "hits": guard_hits}
        jsb = self.judge_pool.acquire_sandbox()          # judge 独占，agent 不可达
        jws = await asyncio.to_thread(cli.mount, project, jsb, branch)   # 挂 agent 的 branch
        try:
            # ① 用镜像里的权威测试覆盖树里的同名路径（等价于"只读"，但不需要 ro mount）
            # ② 用镜像里的 toolchain 跑；PYTHONPATH / conftest 全部重置
            out = await asyncio.to_thread(cli.run_step, project, jws, f"""
                cd {jws_path} && set -e
                rm -rf tests && cp -a /opt/judge/tests ./tests
                rm -f conftest.py sitecustomize.py usercustomize.py
                find . -name '*.pth' -newer /opt/judge/stamp -delete
                env -i PATH=/opt/judge/toolchain/bin:/usr/bin:/bin HOME=/tmp \
                  /opt/judge/run.sh {" ".join(shlex.quote(x) for x in task.protected)}
            """, timeout_s=task.exec_timeout_s)
            return self._parse(out)      # __ARL_VERDICT__{"passed":..,"score":..,"tests":..}
        finally:
            await self.pool.release(jws, discard=True)
```

**为什么 judge 重跑不贵**：因为 base commit 里**带了 `.venv`/`target`/`node_modules`**（LayerFS 的 instant bootstrap），judge 挂载后是"零安装就绪"。这正是 instant bootstrap 在 RL 里唯一真正值钱的地方——不是为了 agent，是为了**每条 rollout 的独立的、干净的判定环境**。

**可信 / 不可信矩阵（写进 code review checklist）**

| 资源 | 来源 | 可信 | 用途 |
|---|---|---|---|
| judge 镜像（runner、pytest、权威 tests） | `sha256:…` digest | ✅ | 判定 |
| agent 的 solution 文件 | Store commit | ❌ 被测对象 | 判定输入 |
| **变更路径列表** | `branch.diff_commits` | ✅ | guard / diff 摘要 / 成本 |
| agent stdout / exit code | `workspace.exec` | ❌ | **仅 observation** |
| sandbox 内 `git status` | 树内 `.git`（agent 可写！） | ❌ | 仅 observation（好看/便宜），绝不用于判定 |
| 树里的 `.venv` / `site-packages` | Store commit | ⚠️ | 可用，但 **runner 不用它**启动 |

</details>

### 4.5 训练侧对接

<details>
<summary><b>展开：<code>arlgym/train/records.py</code> —— 拼 token 序列</b></summary>

```python
def assemble(ep: EpisodeRecord):
    """把多轮 episode 拼成一条训练序列：obs 段 loss_mask=0，action 段 =1。"""
    ids, mask, lp, owner = [], [], [], []
    for t in ep.turns:
        ids  += t.obs_ids   + t.action_ids
        mask += [0]*len(t.obs_ids) + [1]*len(t.action_ids)
        lp   += [0.0]*len(t.obs_ids) + t.action_logprobs
        owner+= [t.idx]*(len(t.obs_ids)+len(t.action_ids))
    return dict(input_ids=ids, loss_mask=mask, logprobs_ref=lp, turn_idx=owner,
                group_id=ep.group_id, reward=ep.reward, valid=ep.valid)

# GRPO：同一 group_id 的 G 条 rollout 从同一 base_commit fork 出来，
# 所以方差只来自采样温度，不来自环境差异 —— 这就是 fork 在算法上的意义：
#   A_i = (r_i - mean(r)) / (std(r) + eps)
```

**与 vLLM 的契约（三条，缺一条就会静默错）**

1. 保存 **vLLM 返回的 token ids 与 logprobs 原样**，禁止用自己的 tokenizer 重编码 action；observation 段才用我们自己编。
2. 每 turn 一次采样 → 天然支持 prefix caching（前序 turn 的 obs+action 是稳定前缀）。别把多 turn 塞进一次生成。
3. constrained decoding 的 schema 是 §6③ 的 `Action`，**只约束外壳**（`{"tool":"bash","arg":"..."}`），`arg` 内部仍是自由文本。

</details>

---

## 5. 一次 episode 的时序（文字版）

```
reset()
 ├─ layerfs branch fork <task>-<ep> <base_commit>            ~ms，零拷贝
 ├─ SandboxPool.lease() → layerfs workspace mount <sb> <branch>   ~5–20ms
 └─ 初始 obs = task.prompt + repo 概览（ls / 最近 commit）

loop:
 ├─ vLLM 采样 → Action（constrained decoding 保证合法）
 ├─ bash → workspace exec（bounded output wrapper，cmd 自带 timeout）
 │  read/write/patch → fs read / fs write / fs apply（一个 apply = 一个 commit）
 ├─ workspace commit -m "turn k"                 ← checkpoint（可关，见 §7 成本门）
 ├─ branch diff_commits prev cur --no-hunks      ← 权威变更集（可信通道）
 ├─ guard.check(changed, protected)              ← 只记账，不阻断
 ├─ obs = exec 结果（不可信） + diff 摘要（可信） + 预算
 └─ done? ──否──> loop
            └是─> JudgeRunner.score()
                    ├─ 独立 judge sandbox（agent 不可达）mount 同一 branch
                    ├─ 覆盖 /opt/judge/tests 权威副本、清 conftest/sitecustomize/*.pth
                    └─ env -i + 镜像 toolchain 跑 /opt/judge/run.sh → __ARL_VERDICT__{...}

close()
 ├─ workspace unmount --discard（未提交的脏改动一并丢弃）
 └─ Sandbox 归还池；每 N 条 episode 做一次 Store 轮转（LayerFS 无 GC）
```

---

## 6. 四个坑的对策（你的 ①–④ + 我补的 ⑤–⑧）

### ① 隔离形态：LayerFS 的 Sandbox 是"容器"，不是安全边界

<details>
<summary><b>展开：三层边界与 dockerd 硬化清单</b></summary>

**事实**：LayerFS 通过 Docker Engine 创建容器（`layerfs.role=sandbox`，只挂 Store volume），默认 runtime。官方自己也只承诺"agent 工作隔离"，同族的 `ephemeral-sandbox` 更是直说 "not a hardened microVM boundary for mutually untrusted tenants"。

**结论：把它当"资源隔离 + 状态隔离"，安全边界要自己在外面套。**

| 层 | 手段 | 防什么 |
|---|---|---|
| **L0 外层 VM** | 训练节点跑在专用 VM / Firecracker / 独占云机；无云凭证、无 metadata 访问、出网只允许 pip/npm 镜像（或全断） | 容器逃逸、内核 CVE 的爆炸半径 |
| **L1 容器运行时** | 专用 dockerd：`--default-runtime=runsc`（gVisor）；seccomp=默认；AppArmor 默认；`--default-ulimit nofile=4096,nproc=1024` | 系统调用面、fork bomb |
| **L2 LayerFS Sandbox** | 每任务族一个 `image`（不带 pip token、不带 ssh key）；`LAYERFS_CONTAINER_PREFIX` 每 shard 独立 | 误用、串味 |
| **L3 判定隔离** | judge sandbox 独立、`env -i`、`--network` 由镜像/daemon 决定（最稳是让 runsc 默认无网） | 作弊 |

⚠️ **不要用 `dsh-layerfs` 的 macOS/WSL2 路径做训练环境**（它靠 `@layerfs/runtime` 起 worker 转发 Store 调用）。训练节点必须原生 Linux + 本地 ext4/xfs/btrfs。

</details>

### ② 吞吐：挂载点 / inode / 项目锁

- **别在 step 里 mount/unmount**：一次 lease 覆盖整个 episode（`reset` 挂、`close` 卸）。
- **上限写进 config**：`max_workspaces = S*16`，超出走背压（`Backpressure` → invalid），不要无限排队把 p99 拖死。
- **每 Sandbox ≤16 Workspace 是硬限**，第 17 个 mount 会 `Capacity`；单 Sandbox 单进程最多 6 并发调用。
- **`sandbox.create/close` 抢项目锁**：只在池预热和扩池时调用，串行化。
- **Store 无 GC** → 必须轮转：每 $N$ 条 episode 或每增长 $B$ GB 就新开一个 Project（`layerfs init <epoch>-<shard> <golden_dir>`），旧 Store 整目录删掉。**别指望删分支省空间。**

### ③ 动作空间：把"自由 bash"关进一个封闭文法

<details>
<summary><b>展开：action schema + 为什么 turn = 单命令</b></summary>

```json
{
  "type": "object",
  "required": ["tool"],
  "properties": {
    "tool":  {"enum": ["bash", "read", "write", "patch", "finish"]},
    "arg":   {"type": "string", "maxLength": 2000},
    "body":  {"type": "string", "maxLength": 16000},
    "thought": {"type": "string", "maxLength": 800}
  },
  "additionalProperties": false
}
```

- 用 vLLM 的 **structured outputs / guided_json**（xgrammar / outlines）把这份 schema 变成硬约束：**模型不可能吐出非法动作**，于是"格式错误"这一类最廉价的失败从分布里消失，你只剩"策略差"这一种失败要学。
- **`turn = 一条命令`**（不是"一次生成 10 条命令"）。理由：
  - LayerFS 的 commit / `diff_commits` 天然按 turn 切分 → **credit assignment 的行是对齐的**；
  - bash 里 `&&` 串联 10 步，中间失败你无法定位是哪一步；
  - 代价是 turn 数↑ → 用 `max_turns` + `no_progress_k` 早停，别用"多命令打包"省 turn。
- `read/write/patch` 三个结构化动作存在的意义：让"改一行"和"跑一次测试"的 token 成本差 100 倍时，动作空间不至于退化成只有 bash（否则模型会用 `sed -i` 做所有事，diff 爆炸）。
- `finish` 必须是**唯一**的结束信号之一（另一个是预算耗尽），否则模型学不会"何时停"。

</details>

### ④ 奖励作弊：威胁模型 → 对策 → 检测

| 攻击 | 对策 | 检测信号 |
|---|---|---|
| 改 `tests/`、`conftest.py`、verifier 配置 | **judge 从镜像 `/opt/judge/tests` 覆盖后再跑**；`protected` 前缀命中即 0 分 | `diff_commits`（可信通道）命中 protected |
| 打印假 "PASSED" | **判定在 judge 容器内由 `/opt/judge/run.sh` 产生**；agent stdout 只进 observation | verifier 输出必须是 `__ARL_VERDICT__{json}`，缺失即 invalid |
| `sitecustomize.py` / `usercustomize.py` / `*.pth` / 根 `conftest.py` 注入 | 跑之前删除 + `env -i` + 用镜像 toolchain（不用树里的 `.venv` 启动 runner） | 新增文件出现在"可疑清单" → 记账、可加重惩罚 |
| 改 `.git` 掩盖改动 | **权威变更集来自 Store，不来自 git** | 两边不一致 → 打标 `tamper_suspect` |
| 留后台进程/守护进程伪造结果 | judge 是**新容器 + 新 workspace**；只有 commit 的内容进得去 | `env -i`、无监听端口 |
| 用超时/跳过蒙混（无限循环 → timeout 被算通过；`-k` 反选） | timeout = fail；要求 **collected test 数 == 期望值**，缺一即 fail | verdict 里的 `collected/skipped` 计数 |
| 缓存污染（LayerFS 保留 ignored 文件 = `__pycache__`、`.pytest_cache` 跨 episode 泄漏） | `.lww` + 每 episode 清 `__pycache__`/`.pytest_cache`；**但保留** `.venv`/`node_modules`（这是性能来源） | 每 turn 记录树里 `__pycache__` 数量 |

**一句话**：你的方案里"把 verifier 挂成只读"在 LayerFS 上**没有对应 API**，可行的等价实现是 **"镜像烘培 + 覆盖恢复 + 独立容器"**；"在宿主机侧跑 verifier"是同一件事的更彻底版本。

### ⑤（补充）不可信观测污染

`git status` 在 workspace 里跑，`.git` 是可写的 —— agent 可以 `git add . && git commit` 甚至 `git checkout` 来伪造"我改了什么"。所以：**diff 摘要两套**，Store 的进 `Change(src="store")`，git 的只进 `src="sandbox"` 且默认不渲染给模型（省 token 又防污染）。

### ⑥（补充）Store 无 GC + Disposable

- 每 $N$ 条 episode 轮转 Project。轮转成本 = 一次 `init`（大仓可能几十秒，用 `timeout` 参数放大）→ **异步做，不阻塞 rollout**。
- 断电/OS crash 可能丢 Store：跑崩了就整 shard 重建，别修。**这是可接受的，因为 rollout 本来就是一次性的**；任务语料（golden dir、tests）在 git/对象存储里。

### ⑦（补充）失败语义：不重试

```python
RETRYABLE = False        # 常量，写死
INVALID_ON = {"Unknown", "Failed", "Busy", "NoSuchSandbox", "Capacity", "Conflict", "Backpressure"}
```

`Unknown` 尤其危险：它意味着"发出了但回复丢了"，**重放会造出重复 commit / 重复扣分**。一律 `valid=False`。

### ⑧（补充）exec 输出无上限

LayerFS 的 exec 不截断（>1MiB spool 到临时文件），而 token 预算是硬的。所以 §4.1 那个 `tail -c` 包装器不是优化，是**正确性要求**。

---

## 7. 阶段门（每阶段都有止损判据）

| 阶段 | 内容 | 通过条件 / 止损 |
|---|---|---|
| **P0 量三个数**（半天） | 在真实任务仓上：`mount` p50/p99、`commit` p50/p99（随树大小）、每条 episode Store 增量 | 若 `commit p99` ≫ 单 turn 生成时间，或单 episode 增量 > 10MB → **关掉 checkpoint_every_turn**，只保留 finish 时的 commit |
| **P1 单 shard 打通** | `ArlEnv` + Pool + Judge，1 条 episode 端到端；同 commit 重跑 reward 一致 | reward 可复现（±0） |
| **P2 GRPO 分组** | $G$=8/16 fork、$K$ 并发、轨迹落盘、invalid masking | 100 step 不崩、盘不涨、p99 延迟可接受 |
| **P3 树搜索** | 从 `commit_k` 重 fork 做 best-of-N；`fs.blame` + `diff_commits` 做辅助信号 | 同 reward 下采样成本下降，或 reward 上升 |
| **P4 观测** | `@layerfs/backend` 的 call trace + `LAYERFS_TRACE=1`；dashboard | 能回答"哪一步慢 / 哪条 rollout 吃盘 / 谁触发了 guard" |

**P0 的测量脚本骨架**（别跳过）

```python
# tools/bench.py —— 就是为了让 §4.2 的公式有真实数字
async def bench(task, n=30):
    t = []
    for i in range(n):
        b = f"bench-{i}"
        await at(cli.fork, P, b, task.base_commit)
        s = time.perf_counter(); ws = await at(cli.mount, P, "1", b); t.append(("mount", time.perf_counter()-s))
        s = time.perf_counter(); await at(cli.run_step, P, ws, "true"); t.append(("exec_true", time.perf_counter()-s))
        s = time.perf_counter(); await at(cli.commit, P, ws, "bench"); t.append(("commit_noop", time.perf_counter()-s))
        await at(cli.unmount, P, ws, True)
    # 再跑一次"改 5 个文件 + commit"的版本，量 publish 的真实延迟
```

---

## 8. 配置（可直接抄）

```yaml
# arlgym/config/shard-0.yaml
layerfs:
  project: swe-epoch7-shard0
  home: /var/lib/arl/lf/home
  store_path: /var/lib/docker/volumes/arl-vol-0/_data/global/store.sqlite
  docker: /var/run/docker.sock
  container_prefix: arl-s0-
  trace: /var/lib/arl/lf/traces/s0.jsonl
pool:
  size: 8                 # → 128 并发 workspace
  queue_timeout_s: 120
  workspaces_per_sandbox: 16
episode:
  checkpoint_every_turn: true     # P0 测完再决定是否关
  no_progress_k: 3
  max_turns: 24
  exec_timeout_s: 300
obs:
  stdout_bytes: 6000
  stderr_bytes: 2000
  diff_excerpt_files: 5
  diff_excerpt_lines: 40
guard:
  protected: ["tests/", "conftest.py", "pyproject.toml", ".github/"]
  suspicion: ["sitecustomize.py", "usercustomize.py", "*.pth"]
judge:
  pool_size: 2
  restore_from: /opt/judge/tests
  runner: /opt/judge/run.sh
  require_collected_tests: true
rotation:
  every_episodes: 2000
  every_gb: 40
```

---

## 9. 不做的事（明确写下来，防止范围蔓延）

1. **不做 Python 原生绑定**：CLI 薄壳 + 必要时一个长驻 Node sidecar（只在 P2 之后、$Q_{\text{commit}}$ 顶到天花板时才考虑）。
2. **不把 LayerFS 当 verifier 沙箱**：judge 用自己的容器逻辑，即使它就是 LayerFS 的 Sandbox——**语义上归 judge 拥有**。
3. **不让 dsh 进训练环**，也不在训练节点上跑 dsh web。
4. **不实现"从任意 episode 恢复"的跨机迁移**：单 kernel，Store 不搬家。
5. **不做 reward 归一化以外的 reward hacking 博弈**：判定侧只做"结构上不可达"，不做"检测模型在想什么"。

---

## 10. 与你原方案的差异（三条，都是硬事实驱动）

| 你的设想 | 落地为 | 为什么 |
|---|---|---|
| "verifier 和测试用例挂成只读" | **镜像烘焙 + judge 覆盖恢复** | LayerFS 的 `workspace.mount` 没有 ro 参数，子目录只读无从实现；等效且更强的是"判定资源进镜像层" |
| "只调 layerfs CLI" | CLI 薄壳，但 **step 用非 stream 模式 + marker 解析** | `--stream` 与 125 退出码冲突（命令自己 exit 125 时无法区分） |
| "变更的文件列表" | **两套**：Store `diff_commits`（可信，进 guard/权重）+ sandbox `git status`（不可信，仅展示） | 树里 `.git` 可写，agent 能伪造 status；Store 的 commit 图它碰不到 |

---

**如果只做一件事**：先把 §7 的 P0 三个数字测出来。这三个数字决定了两件事——`checkpoint_every_turn` 开不开、以及**你到底是在为"reset 快"付钱，还是在为"每条 rollout 一个干净判定环境"付钱**。后者才是 LayerFS 在 RL 里真正难以替代的地方，也是你这套架构成立的前提。

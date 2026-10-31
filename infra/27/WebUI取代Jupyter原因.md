# WebUI取代Jupyter原因

## 🧑‍💻 User

很多针对下沉用户的训练框架比如llamafactory、unsloth、gptsovits都是有webui/desktopapp的，为什么不用jupyter notebook？

---
## 😀 默认助手 (7)

## 先修正一个前提

这三个项目并非不用 notebook，实际上大多同时维护 notebook 和 WebUI。但 notebook 与 WebUI 的地位完全不同。三家的形态差异值得细看，基本勾勒出一条用户分层线：

| 项目 | 主形态 | notebook 的角色 | 隐含用户画像 |
|---|---|---|---|
| LLaMA-Factory | WebUI (LLaMA Board) 为主 | Colab 一键 demo，用于引流和试用 | 有 GPU 但不想装环境 |
| Unsloth | 本质上就是 notebook 库 | 主分发形式 | 会改代码、用免费 Colab T4 的学生或研究者 |
| GPT-SoVITS | WebUI 加 Windows 整合包为主 | Colab 备选 | 有 4090 的 Windows 用户、UP 主、配音爱好者 |

所以更准确的说法是：**notebook 覆盖的是采用代码心智模型的那批人，WebUI 与桌面版覆盖的是采用产品心智模型的那批人。** 后者才是真正的“下沉”，“下沉”并非指技术差，而是指他们不愿为“跑一个模型”付出学习成本。

一句话结论：**notebook 是程序员的输入方式，WebUI 是用户的输入方式。** 下沉用户需要的是把“任务”暴露出来，而不是把“实现”暴露出来；notebook 恰恰相反，它的最大卖点是代码可见，而代码可见对他们只会造成干扰，不能作为功能。

---

## 一、心智模型：让用户看见，而不是要求用户读懂

WebUI 把每个功能做成一个控件，比如滑块、下拉菜单或文件上传器，所有能力都摆在界面上。即使有 80 个参数，用户也能逐个查看、尝试并理解。哪怕界面很挤、很乱，用户至少能“看见”。

notebook 的功能写在 cell 里的注释和 markdown 中。想知道怎么改 rank，得先读代码。对一个连 `pip install` 都没跑过的人，**读代码的门槛远比“看见一堆看不懂的选项”更高**：前者要求会读，后者只要求会点。

另外，notebook 的核心价值是可自由修改，这对程序员是优点，对下沉用户却是不必要的负担，甚至是一种风险：他们会改坏，然后卡死在某个地方。

## 二、运行时：kernel 状态是最反直觉的设计

这是工程上最根本的一条。Jupyter 的隐式全局 kernel 状态意味着：

- 执行顺序有关，必须重启并全部运行，否则会得到错误结果
- 跑完大模型加载后，显存不会自动释放；想换模型必须重启 kernel
- 一个 cell 阻塞 8 小时，浏览器标签页一旦断开就可能失败，输出流式刷屏还会把浏览器拖垮
- OOM 直接让 kernel 崩溃，前面几小时的进度全没了

而 WebUI 的架构天然是：**瘦客户端 + 有状态后端进程 + 任务队列**。训练作为后台 job 跑，日志写文件，进度通过轮询或 websocket 回推，可以随时取消、排队、断线重连、重启进程后继续。多用户、权限、GPU 调度、并发任务，notebook 的 kernel 模型一个都做不了。

对“点一下等 6 小时”的用户，健壮性比灵活性重要一个数量级。

## 三、分发：notebook 没有解决最大那道坎

下沉用户真正的门槛在**装环境**，装 Python、conda、CUDA、PyTorch、flash-attn。notebook 不但没解决这个问题，它自己就是问题的一部分：你得先有 Python 才能开 Jupyter。

WebUI 和桌面应用可以绕开环境：

- Docker 镜像（云平台直接卖）
- Windows 整合包（内嵌便携 Python 运行时和一个 `一键启动.bat`）
- 托管 URL（浏览器打开即用）
- 云 GPU 平台的商品页

GPT-SoVITS 的整合包把整个链路压到“解压 → 双击 bat → 浏览器自动打开”，这是 notebook 无论如何都做不到的体验。**notebook 的最短路径是“装好环境后免写脚本”，整合包的最短路径是“不用装环境”。**

## 四、产品化：更新、遥测、变现、防呆

This is where WebUI wins structurally:

- **版本更新**：整合包可一键更新；notebook 要 `git pull` 再重装依赖，且依赖冲突极易失败
- **联动校验**：选了 4bit 量化就禁用某些选项，显存不够就提前拦截。在 notebook 里只能写 `if` 抛异常，然后用户看到一个 traceback
- **结果可视化**：loss 曲线、生成的图片墙、试听音频、模型对比。这些都得靠前端，ipywidgets 做不出来也扛不住
- **文件操作**：拖拽上传数据集、浏览目录选路径，notebook 里没有原生文件选择器
- **商业化**：云端额度、功能开关、A/B、GPU 计费，notebook 一个都插不进去

## 五、技术支持成本：截图与 traceback 的区别

这一条在开源项目里权重被严重低估。

- WebUI 用户求助：“我卡在第三步了”，配张截图，你回“把 batch size 改成 1”。三十秒解决。
- notebook 用户求助：贴 traceback，你得先问他改过哪个 cell、有没有 run all、kernel 是不是旧状态。一轮对话未必能定位。

**可支持性直接由界面形态决定。** 一个几千 star 的项目，维护者必须把单位用户的答疑成本压到最低，WebUI 是唯一解。

## 六、一个容易被忽略的点：Gradio 本来就运行在 notebook 里

LLaMA Board 和 GPT-SoVITS 的界面本身就是 Gradio。而 Gradio 的技术起点，就是“把 notebook 里的一个 cell 渲染成一个表单”。

也就是说，**WebUI 在底层基本就是 notebook 的一个 cell**。区别不在技术能力，而在产品约定：

- notebook：把整个 pipeline 摊开给你，你自己组合
- WebUI：只给你一条预设的正确路径，其余全部藏起来

对框架作者来说，隐藏内部正是价值所在：数据格式转换、chat template 匹配、target modules 选择、checkpoint 合并导出，这些“有唯一正确答案”的环节，不该给下沉用户选择权。

---

<details>
<summary><b>notebook 真正不可替代的场景（点开）</b></summary>

notebook 不差，它只是服务错了人群。它的主场是：

1. **免费 GPU 获客**：Colab、Kaggle 的 T4 是大量人第一次跑微调的地方，而这只能以 notebook 形式提供
2. **研究者与复现**：改一行代码看 loss 变化，这个循环 WebUI 做不了
3. **教学**：让学生看到每一步中间变量
4. **维护者自用**：调 pipeline、验证 bug、写最小复现

所以正确姿势是**双轨**：WebUI、整合包面向用户，notebook 面向研究者，并作为免费算力的入口。把 notebook 当唯一形态（比如 Unsloth）意味着主动放弃了下沉用户，而 Unsloth 也确实没打算吃那块市场。
</details>

---

## 判断框架

选形态的实质是回答三个问题：

1. **用户是否需要理解内部机制？** 需要 → notebook；不需要 → WebUI
2. **任务是“一串固定步骤”还是“探索式实验”？** 固定 → WebUI；探索 → notebook
3. **交付的是“产品”还是“说明书”？** 产品 → 必须能被运行；说明书 → 读者得会读代码

下沉用户三个答案全指向 WebUI。notebook 是把“可编程性”当作默认能力交给一个不具备该能力的人。这不是简洁，只是把复杂度转移给了最没能力承担的人。

真正的分界不在 notebook 与 WebUI，而在**“把任务抽象成按钮” 还是 “把实现摊成代码”**。Axolotl 用 YAML、torchtune 用 recipe，同样绕开了 notebook，因为它们的用户也不需要 Jupyter，只需要一个声明式入口。

---
## 🧑‍💻 User

jupyter notebook相比直接敲cli+修改yaml config有哪些优势？

---
## 😀 默认助手 (7)

## 一条能区分两者的判断依据

之所以会出现这个对比，是因为两种方式对**“状态”**的处理方式根本不同：

- **CLI + YAML**：无状态的任务提交。每次运行都是一个干净的进程，配置是一份数据，可以 diff、可以扫描、可以并行跑 N 份。
- **notebook**：有状态的交互式脚本。一个 kernel 里累积上下文，每次改动只重跑受影响的部分。

于是两者的优势按**“实验阶段”**几乎完全分开：

> notebook 赢在“还不知道要做什么”的阶段；CLI+YAML 赢在“已经知道要做什么、并且要重复做 N 次”的阶段。

下面具体展开。

---

## 一、状态复用：迭代延迟差一个数量级（最核心的一条）

这是 notebook 唯一一个**无法用 YAML 补上**的优势。

以“微调后调推理参数”为例：

**CLI + YAML 路线**
```
改 a.yaml → python infer.py -c a.yaml → 重新加载 14GB 权重(~40s) → 看到文本
改 b.yaml → python infer.py -c b.yaml → 再等 40s → ...
```
试 20 组 temperature、system prompt、adapter 权重组合，大约需要 $20 \times 40\text{s} = 800\text{s}$。

**notebook 路线**
```
cell 1: 加载模型一次
cell 2: generate(prompt, temp=0.7)   ← 改参数重跑,毫秒级
```
同样的 20 次实验只需要一次加载的 40s。

这类场景在 notebook 里是默认行为，在 CLI 里要么忍受重复加载，要么自己写一个 REPL 脚本（那你就是在手搓一个糟糕的 notebook）。

<details>
<summary><b>但这条优势的适用边界很窄（点开）</b></summary>

状态复用只在**“加载很贵、单次调用很便宜”**的场景成立：

- ✅ 调 prompt、看生成、试 LoRA 权重、逐样本 debug tokenizer
- ❌ **训练本身**：一次训练就是一整段连续作业，没有“复用状态再跑一次”的需求。这时 notebook 的状态复用从优势变成负担（显存不释放、跑完的环境是脏的）

所以这条不能用来论证“训练也用 notebook”。它论证的是“**推理与数据侧**用 notebook”。
</details>

## 二、失败现场保留：长 pipeline 的调试体验

YAML 路线里的报错是这样的：

```
CUDA out of memory / KeyError: 'labels' / shape mismatch
  File "trainer.py", line 412, in inner_training_loop
  ...
```

进程已经退出，中间状态全部丢失。想看清“进来的是什么样的张量”，你得加 print、重跑整个耗时 20 分钟的数据加载，或者在框架代码里断点。

notebook 里，cell 5 报错后，cell 1–4 的变量**仍然活着**：

```python
# 直接在下一个 cell 里问
batch["labels"][:20]
tokenizer.decode(batch["input_ids"][0])
```

`%debug` 还能直接进入 traceback 对应帧，查看函数内的局部变量。

差别在于：**YAML 路线迫使你把调试信息预先写入日志，然后在事后分析；notebook 允许你事后追问。** 对一个你并不熟悉的框架，“事后追问”常常才是唯一能找到问题的方式。

## 三、YAML 的 schema 天花板（最被低估的一条）

这一条才是研究者、写脚本的人不愿用 YAML 的真正原因：

> **YAML 只能表达框架作者预料到的参数组合。**

- 你想加一个自定义 loss 项 → 改 YAML 无法表达，只能 fork 或写 plugin
- 你想在 dataloader 里按长度过滤掉最后 3% 的样本 → 如果 schema 没暴露这个开关，就得改源码
- 你想在 step 5000 存一个附加检查点 → 同上

而 notebook 或脚本里是：

```python
trainer.add_callback(MyCallback(save_at=[5000]))
```

**把配置变成数据，代价是表达力被限制在 schema 之内。** 这是能力与易用性的取舍，并非单纯实现得好坏的问题。框架作者也知道这一点，所以通常保留 `--config` 和“自己写 Python 脚本调 API”两条路。

## 四、富输出：对象级探索 vs 文件级 I/O

CLI 的输出通道是 **stdout + 日志文件**，本质是字符串流。任何非文本产物（图片、音频、DataFrame、loss 曲线、attention map）都要“存盘 → 找路径 → 打开”。

notebook 有富显示协议，对象**执行完就渲染**：

| 你想看 | CLI + YAML | notebook |
|---|---|---|
| 数据样本 | dump 成 jsonl，翻文件 | `df.head()` / `display(samples[:5])` |
| 生成音频 | 存 wav，打开播放器 | `IPython.display.Audio(arr)` 内联播放 |
| 图生图结果 | 存图，看图 | `Image.open(...)` 直接出图墙 |
| loss 曲线 | tensorboard 另开进程 | `%matplotlib inline` 立刻画 |
| 训练进度 | tail -f 日志 | tqdm 进度条原地刷新 |

对多模态微调（GPT-SoVITS、图像生成类）这个差别很关键：**“对比听 10 个合成样本”在 notebook 里是一个 cell，在 CLI 里是一套工作流。**

## 五、三合一交付物：可运行 + 可解释 + 带结果

YAML 是纯配置，注释只能解释“这个字段是什么”，解释不了“为什么这样设”。CLI 是黑盒。

notebook 的 `.ipynb` 同时是**三种东西**：

1. 可运行的程序
2. 带 markdown 的叙述层（为什么除 8、为什么换这个 lr、踩了什么坑）
3. **内嵌输出的实验记录**——不重跑就能看到当时的结果

第 3 点对“实验日志/周报/PR 说明/issue 复现”特别有效：发一个 notebook，对方不用装环境就能看到你看到了什么。

代价也很明确：**带输出的 ipynb 是 git 的噩梦**（JSON + base64 图片，diff 不可读，merge 冲突频繁）。所以这条优势只适合“单人、单向、演示性”的传递，不适合团队长期维护的代码。

## 六、分发与异构混用

两个更实际但同样重要的点：

- **免环境分发**：Colab / Kaggle / SageMaker Studio 里，**notebook 本身就是 GPU 的获取方式**。CLI 路线要 SSH + tmux + 自建 venv，iPad 上直接没戏。这也是为什么免费算力入口几乎总是 notebook 形态。
- **异构混用与魔法命令**：一个文档里 `!pip install`、`%%bash` 删缓存、`%%sql` 查数据集、Python 训练、`%%timeit` 测速度。CLI 的世界里每换一种语言就要换一个工具、换一个上下文。

---

## 对照表：谁在哪一步赢

| 维度 | notebook | CLI + YAML | 更优 |
|---|---|---|---|
| 探索期迭代速度 | 加载一次反复试 | 每次重跑全流程 | **notebook** |
| 报错后查现场 | 状态仍在 | 需重跑复现 | **notebook** |
| 超出 schema 的定制 | 直接写 Python | 要改源码/写插件 | **notebook** |
| 富输出（音/图/表） | 内联渲染 | 存盘再打开 | **notebook** |
| 文档 + 结果 + 代码合一 | 原生支持 | 三个文件拼 | **notebook** |
| 免费/远程算力 | 就是分发形式 | 需自建 SSH 环境 | **notebook** |
| 可复现性 | 隐式状态、乱序执行 | 配置即数据，干净进程 | **CLI+YAML** |
| 超参扫描 / 并行 | 有状态会互相干扰 | 遍历 yaml 起 N 个进程 | **CLI+YAML** |
| 长任务健壮性 | 浏览器断连即险 | `nohup`/`tmux`/slurm | **CLI+YAML** |
| 多机分布式 | 单 kernel 做不到 | `torchrun` / `sbatch` | **CLI+YAML** |
| 代码评审 / diff | ipynb JSON 难读 | yaml + py 可读 | **CLI+YAML** |
| CI / 自动回归 | 需 nbconvert，脆 | 直接跑 | **CLI+YAML** |
| 交接给别人重跑 | 依赖 kernel 状态 | 按 yaml 精确重跑 | **CLI+YAML** |

---

## 结论：这是一条流水线，应分阶段选择，而非只能二选一

成熟用法通常不是选边，而是**按阶段切分**：

```
探索/调试期  → notebook      : 试数据格式、定位 pipeline bug、扫一眼生成质量
固化         → 写成 config/script: 把探索结论沉淀成一份可复现的 yaml 或 python entry
执行期       → CLI           : 长任务、并行扫描、多机、CI
交付/汇报    → notebook      : 带上输出讲清楚“我做了什么、看到了什么”
```

一句话收束：

> **notebook 是“实验的草稿纸”，YAML+CLI 是“实验的合同”。草稿纸的价值在于能涂改，合同的价值在于能执行。** 用草稿纸去签合同，或者用合同去涂改，两种方向都错。

反过来说，如果你发现自己在一个 YAML 项目里**反复手动改配置跑短任务**，那说明该开一个 notebook 了；如果你在一个 notebook 里**手动改 15 个 cell 的参数来跑 30 组实验**，那说明该把参数提取到 YAML 并写一个 CLI 了。

---
## 🧑‍💻 User

这么看来，jupyter notebook似乎它更像是面向科研人员和开发者的。如果我希望的是训练job ci化，用户可以靠一个platform并行多个task（跑消融跑多组对照搜参）+统一的环境管理+确保可复现性还要带训练集群编排+作业调度+模型从预训练到评估的全部生命周期管理系统，应该选择什么方案？

---
## 🧑‍💻 User

主要是我想到的是训练job ci化，用户可以靠一个platform并行多个task跑交互式shell加速迭代，jupyter notebook优势回很明显吗？

---
## 🧑‍💻 User

主要是我想到的是训练job ci化，用户可以靠一个platform并行多个task（跑消融跑多组对照搜参）+统一的环境管理+确保可复现性，jupyter notebook优势很明显吗？（似乎它更像是面向科研人员和开发者的）

---
## 😀 默认助手 (7)

## 先定性：拼合多个工具不能解决这个问题

你描述的这些需求——并行多任务、统一环境、可复现、集群编排、作业调度、全生命周期——既有**五个层面**，也有**一个平台工程问题**。而且这些层面之间存在强耦合：如果调度层不支持 gang scheduling（整体调度），编排层就无法可靠地运行多机训练；如果环境没有通过镜像固定住，可复现性就会从源头失效。

因此，对“应该选什么方案”这个问题，正确的回答方式是给出一个**分层选型组合**和一个**按现状选择的决策树**，而不是只给出单个产品名。

---

## 目标架构：五层

```
┌──────────────────────────────────────────────────────────────┐
│ L4 治理层    模型注册 / 评测门禁 / 审批 / 成本归属 / 报表       │
├──────────────────────────────────────────────────────────────┤
│ L3 编排层    pretrain → SFT → eval → promote 的 DAG,产物按引用传递│
├──────────────────────────────────────────────────────────────┤
│ L2 调度层    sweep 展开 / 队列与配额 / gang scheduling / 抢占    │
├──────────────────────────────────────────────────────────────┤
│ L1 执行层    镜像 / 分布式训练 / 数据版本 / CUDA-NCCL 栈         │
├──────────────────────────────────────────────────────────────┤
│ L0 基础设施   GPU 节点 / 并行存储 / IB 网络 / 节点自动伸缩        │
└──────────────────────────────────────────────────────────────┘
```

**关键判断：notebook 在这一架构里没有任何位置属于 L1 及以上。** 原因是你上一轮已经推导出的结论：它是有状态的交互式工具，而平台需要的执行单元是**无状态、声明式、可重投递**的。

---

## 各层选型

### L2 调度层（最关键，选错后整体返工）

| 方案 | 适合 | 代价 |
|---|---|---|
| **Slurm** | 裸金属/自建 GPU 集群，HPC 传统 | K8s 生态工具接不上；但 `sbatch --array` 做 sweep 极强 |
| **K8s + Kueue + Volcano** | 云原生、已有 K8s | 平台团队必须懂 K8s 调度；配置量大 |
| **K8s + KubeRay** | 以 Ray 为计算基底 | 与 Ray 生态强绑定 |
| **Determined AI** | 想“一个产品搞定调度+HP搜索+追踪” | 强 opinionated，社区规模小 |

**必须满足的硬条件：gang scheduling（整体调度）。** 多机训练要求 N 个 pod 同时到位，否则会出现“分配了 3/4 的节点，互相等对方，队列死锁”。Kueue 的 `Workload` 或 Volcano 的 `PodGroup`、Slurm 的默认行为、Determined 自带，三者都能做；裸 K8s 的默认调度器做不到。

### L3 编排层

| 方案 | 适合 | 代价 |
|---|---|---|
| **Flyte** | 强类型、产物缓存、lineage 完整，最适合“要可复现” | 概念多，上手慢 |
| **Argo Workflows** | K8s 原生，DAG 就是一堆容器，简单直接 | 类型弱，缓存要自己搞 |
| **Kubeflow Pipelines** | 已经打算上整个 Kubeflow 全家桶 | YAML 地狱，运维成本极高，很多团队死在这一层 |
| **Metaflow** | 数据科学家友好，单机到集群平滑 | 调度弱，偏 AWS |
| **Dagster / Prefect** | 偏数据工程，资产化建模好 | 对 GPU 训练支持一般 |

### L2 与“跑消融、搜参”的交叉点：Sweep 必须是一等公民

这是你需求里最容易被低估的一条。**“多层 for 循环里起 N 个进程”并不等于 sweep。**

正确的形态是三件套：
- **配置矩阵生成**：`Hydra` 的 `--multirun` 或 `Optuna` 的 `suggest`，把“改哪个参数、取哪些值”声明化
- **搜索算法 + 剪枝**：`Ray Tune`（ASHA/PBT/BOHB）或 `Optuna`（TPE + MedianPruner）。**没有剪枝的 sweep 会烧掉 10 倍算力**——跑 50 组、每组 8 卡 3 天，不剪枝就是灾难
- **每 trial 一个独立调度单元**：K8s Job / Ray actor / Slurm array 元素，互不共享状态

### L1 执行层与环境管理

**唯一正确的做法：环境 = 容器镜像，通过 digest 固定。**

- 每个 commit 构建镜像，用 `kaniko`/`buildkit` 在集群内构建，tag = git SHA
- 镜像里固定 `torch` / `CUDA` / `NCCL` / `driver` 的版本组合，因为这套东西的任何一处变动都会改变数值结果
- Python 层用 `uv` + lockfile 补充可读性和本地开发一致性
- **不要用 conda env 做环境一致性保证**——`environment.yml` 没有真正的锁语义，这是“在我机器上能跑”的主要来源
- 数据侧用 `DVC`/`lakeFS`/内容寻址清单，给数据集算哈希

### L4 追踪、注册与评测门禁

- **追踪**：`MLflow`（自建、便宜、够用）/ `W&B`（体验最好、要钱）/ `Neptune`
- **注册**：`MLflow Model Registry` 或 `W&B Artifacts`，给模型分配 `model@v3` 这样的版本意图标识
- **评测**：`lm-evaluation-harness` 或自建 eval pipeline，**必须作为编排图里的一个节点，并带 gate**——“eval 不达标不 promote”，靠系统强制而不是靠人自觉

---

## 三条路线：按你的现状选

<details>
<summary><b>路线 A：已有 K8s 集群 + 有平台团队（长期自建）</b></summary>

```
GPU 节点
 └─ K8s + NVIDIA GPU Operator
     ├─ Kueue            ← 队列、配额、抢占、gang scheduling
     ├─ Kubeflow Training Operator  ← PyTorchJob / TorchTuneJob
     ├─ KubeRay + Ray Tune          ← sweep 执行
     ├─ Argo Workflows(或 Flyte)   ← 阶段 DAG
     ├─ MLflow + MinIO/S3           ← 追踪 + 产物
     └─ Argo Events / GitHub Actions ← 触发与门禁
```
这是最“正规”的方案，但**平台团队要养 2–4 人**。没有这个投入，就不要走这条路线。
</details>

<details>
<summary><b>路线 B：裸金属或已有 Slurm 集群（务实派）</b></summary>

不要为了“现代化”迁 K8s——训练负载对 K8s 的收益远小于对推理负载的收益。

```
Slurm
 ├─ Hydra + submitit  ← 一条命令把 multirun 变成 N 个 sbatch 作业
 ├─ sbatch --array    ← 最被低估的 sweep 工具
 ├─ W&B / MLflow      ← 追踪注册
 ├─ DVC + 镜像(sqsh/enroot) ← 环境与数据
 └─ GitHub Actions    ← 只做触发、schema 校验、结果回写
```
`submitit` 是 Meta 出的 Hydra launcher，能把 `--multirun` 直接映射成 Slurm 作业数组，装起来半小时。**如果你已经在 Slurm 上，这是投入产出比最高的方案。**
</details>

<details>
<summary><b>路线 C：小团队，不想造平台（买/租）</b></summary>

- **Determined AI**（开源 Apache 2.0，HPE 维护）：一个产品同时给你集群调度、分布式训练、HP 搜索、追踪、可复现性。缺点：强 opinionated、生态小。**对“所有需求一次满足”这句话，它是最接近的现成答案。**
- **SkyPilot** + W&B + Hydra：多云/多集群抽象，自动找便宜 spot，适合“我们没有固定集群”。
- **Modal / Anyscale / SageMaker HyperPod**：全托管，`@app.function` 装饰器就能起并行 GPU 任务，DX 极好，代价是锁定和成本。
- **ClearML / Polyaxon**：介于 OSS 和商业平台之间，值得比较。
</details>

---

## 六个决定成败的设计决策

**1. CI 只做触发和门禁，不要让它跑 GPU**
CI runner 生命周期短、没有 GPU、没有队列语义。正确切分是：
```
PR 改 config → CI 做 schema 校验 + 1-step dry run + 显存/时长估算
合并 main    → CI 投递作业到集群(K8s API / Argo / sbatch) → 返回 job id
作业结束     → 回写 metrics 到 commit status / PR comment
```
“训练 job CI 化”的本质是**训练配置进仓库 + 触发即提交 + 结果回写**，并不等于把 notebook 塞进 CI。

**2. 产物按引用传递，而不是按路径传递**
禁止任何 `/home/alice/ckpt/best.pt`。所有阶段间传的是 `s3://artifacts/{run_id}/ckpt@sha256:...` 或注册表 URI。这一条不做到，可复现性只是口号。

**3. 把“可复现”拆成三层，只对前两层承诺 100%**
| 层 | 能否 bit-exact | 手段 |
|---|---|---|
| 环境 | ✅ 可达 | 镜像 digest + lockfile + CUDA/NCCL 固定 |
| 数据 | ✅ 可达 | 数据集内容哈希，数据不可变 |
| 数值 | ⚠️ 单卡可、多卡基本不可 | `torch.use_deterministic_algorithms(True)`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、固定 seed |

多卡下换 `nranks` 或换拓扑，NCCL 的 all-reduce 归约顺序就变了，`cuBLAS` 的非确定性 kernel、`atomicAdd` 都会引入差异。**工程上正确的目标是“可重放出统计上不可区分的结果”**，并用多种子（$n \ge 3$）报 `mean ± std` 来吸收方差。

**4. 消融实验靠平台强制四个“同一”**
同一镜像 digest、同一数据快照、同一 eval harness 版本、一次只改一个变量。这四件事靠 schema 和校验器强制，不能靠实验者自觉。同时 **baseline 本身必须是 sweep 矩阵里的一格**，不能拿“上周跑的那个”当对照。

**5. Sweep 必须带剪枝**
ASHA / BOHB / PBT。没有 early-stopping 的搜索会在算力上失控，而且会诱使团队把 grid 缩小到失去统计意义。

**6. 队列与配额是必须，不是加分项**
多用户并行必然争抢 GPU。没有 namespace 配额、优先级队列、抢占策略，三个月后就是“谁先抢到谁用”。

---

## 明确不要做的四件事

1. **不要自研调度器**。gang scheduling、抢占、拓扑感知、故障重试，每一个都能吃掉你一年。
2. **不要把 notebook 或 `papermill` 当作执行单元。** 它在 CI 里脆、状态不可控、diff 不可读。notebook 可以留在**探索层**，但产物必须固化成 `script + config` 才能进平台。
3. **不要一上来就上 Kubeflow 全家桶。** KFP + Katib + KServe + Notebook 控制器 + Istio 的运维成本会压垮小团队，很多团队就死在这一层。
4. **不要用 conda env 做环境一致性的唯一保证。**

---

## 最小可行路径（90 天）

不要一次把五层都建完。按“能跑通闭环”倒序推进：

```
第 1–3 周   容器化 + 镜像 per commit + 数据版本化     → 环境与数据可复现闭环
第 4–6 周   接入追踪(W&B/MLflow) + 产物注册表          → 实验可追溯
第 7–9 周   Hydra + submitit/Argo 打通 sweep           → 并行消融可用
第 10–12 周 eval 节点 + promote 门禁 + CI 回写         → 生命周期闭环
```
调度层只要选 Slurm 或 Kueue 之一即可开工，不要等“架构评审通过”。

---

## 回到你上一个问题：notebook 怎么办

保留，但**降级为探索层，并切断它与执行层的直接通路**：

| 层面 | 用什么 | 产出物 |
|---|---|---|
| 探索 | notebook | 结论 = 一份 `config.yaml` + 一段 `data_prep.py` |
| 固化 | 仓库内脚本 + Hydra config | 可 diff、可校验 |
| 执行 | CLI 投递到集群 | 有 run_id、有产物引用 |
| 汇报 | notebook / 报表 | 带输出的只读记录 |

一句话总结这套方案的选择原则：

> **编排层选“声明式 + 可缓存”（Flyte/Argo），调度层选“gang scheduling + 队列”（Kueue/Volcano 或 Slurm），环境层选“镜像 per commit”，sweep 选“Ray Tune/Optuna + 剪枝”，治理层选“eval gate 强制 promote”。** 不为训练负载迁 K8s，不为消融实验写 for 循环，不为可复现性承诺 GPU 上的 bit-exact。

如果你告诉我三件事——**现有集群是 K8s 还是 Slurm、团队规模、以及是否已有 W&B/云厂商绑定**——我可以把上面的方案收敛成一份具体的组件清单和落地顺序。

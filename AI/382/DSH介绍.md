# DSH介绍

### 🧑‍💻 User

为我介绍下deepseek harness

---

### 🤖 Assistant

我先查一下相关资料，看看有没有公开的 "deepseek harness" 项目

找到了官方仓库 `deepseek-ai/deepseek-harness`，我再获取一下 README 详细内容



我再看下中文 README 和架构文档，以便更完整

# DeepSeek Harness（`dsh`）介绍

DeepSeek Harness（简称 **`dsh`**）是**深度求索（DeepSeek AI）官方出品的开源 Agent 智能体框架**，GitHub 地址为 [`deepseek-ai/deepseek-harness`](https://github.com/deepseek-ai/deepseek-harness)，官网 [deepseek.com/harness](https://deepseek.com/harness)。它的核心哲学用一句话概括就是：

> **"Everything is a Plugin" —— 一切皆插件。**

---

## 1. 它到底是什么？

在 AI 编程/Agent 领域，"harness"（直译"驾驭装置"）指的是**把大模型包装成可工作的智能体所需的全部运行时组件**：模型接入、工具调用、记忆/会话、执行沙箱、权限审批、UI 等。

打个比方，Claude Code、OpenAI Codex 这类工具帮你把模型"驾驭"起来干活，而 `dsh` 就是 **DeepSeek 自己的、开源的、完全插件化的"驾驭框架"**。你可以把它理解为：

- 一个可扩展的 Agent 运行时（支持 Web UI 和 headless 无头运行）
- 一个"Agent 里的操作系统"，内核只提供插件装载机制，其余一切功能（模型适配器、工具注册表、会话日志、甚至 Agent 主循环本身）都是插件，**谁都可以在运行时被替换**，没有一个"特权核心"需要打补丁。

> ⚠️ **注意**：项目当前处于 **developer preview（开发者预览）** 阶段，迭代很快，官方明确警告 **会有破坏兼容性的变更**。

---

## 2. 快速上手

前置条件：安装 `Node.js`。

```sh
# 一条命令启动 Web UI（默认 http://127.0.0.1:3080）
npx @deepseek-ai/dsh web
```

从源码运行：

```sh
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install
pnpm run build
pnpm dsh web
```

- 🌐 `web` 模式：启动浏览器界面
- 🖥️ `headless` 模式：一次性运行的 runner，不带任何服务

---

## 3. 核心架构亮点

| 概念 | 说明 |
|---|---|
| **Cordis** | 底层插件框架（Node.js），dsh 基于它构建；设计思想见论文《A Programming Paradigm for Spatiotemporal Composability》 |
| **Profile（配置文件）** | 一次运行 = 一棵按层组合的插件树，`web` / `headless` 是官方内置模板；用户可通过 `cordis.patch.yml` 增删改任何一层 |
| **Bundle（插件包）** | 插件的分发格式；`dsh-base` 是第一层（含模型适配、工具、持久化、沙箱、权限审批、设置、凭据、遥测），`dsh-web-app` 叠加浏览器应用 |
| **Seam（能力接缝）** | 可插拔能力三件套：接口定义 + 提供者 + 消费者。换一个 Provider 就能整体切换能力（比如把文件系统/子进程指向远程沙箱，Bash、PTY、LSP 跟随一起切换） |
| **Event（事件）** | 一切扩展点：会话事件（持久化事实）、`agent/*` 事件（拦截/观察进行中的任务）、能力事件（不侵入主循环地挂策略） |
| **会话日志** | 模型上下文的唯一真相源，铁律是 *"模型能看到的都必须已记录（Model-visible means logged）"*，保证可回放、可 fork、可恢复 |

核心包概览：

| 包 | 职责 | `ctx` 键 |
|---|---|---|
| `core/session` | 只追加的会话事件日志 | `ctx.sessions` |
| `core/system-prompt` | 提示词分区与工具 schema 组装 | `ctx.systemPrompt` |
| `core/tools` | 带作用域的工具注册表 + 受控执行流水线 | `ctx.tools` |
| `core/agent` / `core/agent-loop` | Agent 接口与默认驱动循环 | `ctx.agents` / `ctx.agentLoop` |
| `llm/llm` | 消息/流式词汇 + 模型适配器接缝 | `ctx.llm` |

---

## 4. 生态与社区（增长非常快）

`dsh` 发布后围绕它快速形成了一个庞大的插件生态（GitHub 上用 `dsh-plugin` 话题标记），例如：

- 📦 **插件市场/精选列表**：`awesome-dsh-plugin`、`dsh-market`（可视化插件市场）
- 🖥️ **桌面客户端**：`dsh-desktop`、`deepseek-harness-desktop`、`Deepseek-Harness-EAC` 等（把 CLI 打包成桌面应用）
- 👁️ **视觉能力外挂**：`modlens`、`dsh-vision-router`（给纯文本模型补上 OCR/截图/像素级视觉）
- 🌐 **浏览器自动化**：`dsh-browser-bridge`、`browser-auto`、`dsh-plugin-web-access`
- 📘 **学习资料**：《DeepSeek Harness 橙皮书》等社区教程

社区入口：GitHub Discussions、Discord，以及中文用户可加的**企微群/公众号**（见官方 README）。给插件仓库打上 `dsh-plugin` 话题即可被社区发现。

---

## 5. 适合谁用？

- **想深度定制 AI Agent**：因为它"一切皆插件"，从换模型、加工具、改审批策略到自研 UI，都是插拔式扩展，扩展点有明确的文档（官方提供 cookbook：如何加包、加工具、加 LLM 适配器、加会话节点等）
- **追求可组合/可观测性**：基于 Cordis 的上下文组合模型 + 事件驱动，适合搭建复杂、可持续演化的 Agent
- **DeepSeek / 开源基建爱好者**：MIT 协议，纯开源，可自行研究源码

如果你是想深入使用，建议顺着官方文档 `docs/` 里的 **开发指南（development.md）→ 架构文档（architecture.md）→ Web UI 指南** 一步步走；如果只是想先跑起来感受一下，`npx @deepseek-ai/dsh web` 一条命令就够了。

需要我帮你进一步看某个方面（比如插件开发示例、Web UI 用法、或和 Claude Code 的对比）吗？
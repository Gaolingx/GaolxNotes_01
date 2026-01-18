# Agent上下文传递问题

### 🧑‍💻 User

观察可以发现：chat是单上下文agent的一个特例
——单上下文agent，和多上下文agent所基于的工作方式不一样。
实践经常体现这一点：巨型的外挂于单上下文agent的框架，经常与各种多上下文agent不完全兼容。
比如说spec-kit(一个被github开发的用于把任务拆分成一个个规范来vibe完成的框架)经常被发现在roo和kilo的orchester模式下无法正常工作——观察任务拆分过程就可以发现，这是因为拆分机制没有针对spec kit进行了解，因此拆分出的子任务没能被传递全部需要的信息，也就工作失常了。

对此你怎么看？

---

### 🤖 Assistant

这是一个非常深刻且切中肯綮的观察。你指出了当前 Agent 架构发展中一个核心的**“上下文连贯性”与“任务拆分”之间的矛盾**。

你的观点“**Chat 是单上下文 Agent 的一个特例**”在数学和架构上都是成立的。我们可以把 Chat 看作是一个**状态空间未被分割、上下文窗口线性增长**的单一 Agent 进程。

对此，我有以下几个层面的分析，来印证并扩展你的观点：

### 1. “单上下文” vs “多上下文”：全局变量 vs 局部作用域

*   **单上下文 Agent (Chat 模式)：**
    *   **工作原理**：类似于一段很长的“面条式代码”或拥有一个巨大的“全局作用域”。所有的历史对话、用户指令、框架规则（如 Spec Kit 的定义）都在同一个 Context Window 里。
    *   **优势**：LLM 极其擅长在上下文中进行“注意力关联”。如果 Spec Kit 定义在对话开头，模型在生成后续代码时，注意力机制（Attention Mechanism）能直接“看见”这些规则。
    *   **适用性**：外部框架（如 Spec Kit）通常假设它们运行在这种“全知全能”的环境中，依赖**隐式上下文**。

*   **多上下文 Agent (Orchestrator/Swarm 模式)：**
    *   **工作原理**：类似于“微服务架构”。Orchestrator（指挥官）负责拆分任务，然后分发给 Sub-agents（工兵）。每个 Sub-agent 启动时，通常是一个**全新的、干净的上下文**，或者只包含 Orchestrator 传递给它的极少部分信息。
    *   **致命缺陷（你的观察点）**：Orchestrator 在拆分任务时，往往只关注“任务本身”（例如：“写一个登录页面”），而忽略了“元数据/环境上下文”（例如：“必须使用 Spec Kit 定义的 V2 规范，且遵循文件 A 中的类型定义”）。
    *   **结果**：Sub-agent 拿到了任务，但丢失了规则。就像把一个程序员关进小黑屋，只让他写代码，却不给他看项目文档。

### 2. 为什么 Spec Kit 在 Roo/Kilo 下会失效？

具体到你提到的 Spec Kit 案例，失效的机制通常是这样的：

1.  **初始化**：用户在主会话中加载了 Spec Kit，定义了复杂的规范。
2.  **拆分**：Roo/Kilo 的 Orchestrator 决定将大任务拆解。它在生成子任务提示词（Prompting Sub-agents）时，为了节省 Token 或保持专注，**进行了上下文剪枝（Context Pruning）**。
3.  **信息熵丢失**：Orchestrator 并不知道 Spec Kit 的规则是“必须传递的遗传物质”。它把这部分视为“已读的历史信息”过滤掉了。
4.  **执行失败**：Sub-agent 启动，收到指令“写个函数”，但因为它不知道 Spec Kit 的存在，它就用通用的 Python/JS 写法完成了任务。
5.  **整合冲突**：当结果传回主上下文时，发现格式不兼容，或者完全无视了规范。

### 3. 架构失配：有状态 vs 无状态的冲突

目前的很多 Agent 辅助框架（不仅仅是 Spec Kit，包括很多 Prompt 优化工具）都是基于**Session-Stateful（会话有状态）** 的假设设计的。

*   它们假设：`State_t = State_{t-1} + New_Input`
*   而多上下文架构是：`Worker_State = Task_Prompt + (Maybe Shared_Memory)`

**Orchestrator 模式不仅是任务的拆分，更是上下文的“物理隔离”。** 除非 Orchestrator 具备极其高超的“元认知”能力，能够识别出哪些 Prompt 是“环境配置”（必须透传给子 Agent），哪些是“任务描述”（只需部分传递），否则这种不兼容是必然的。

### 4. 解决方案与未来展望

正如你所说，目前的巨型外挂框架与多上下文 Agent 不兼容，这预示着未来 Agent 架构的演进方向：

*   **从“上下文传递”转向“共享状态层（Shared State Layer）”**：
    未来的 Agent 框架不能依赖 Chat History 作为唯一真理来源。所有的 Agent（无论是指挥官还是工兵）都应该连接到一个共享的 **KV Store** 或 **Vector DB**。Spec Kit 这种框架应该将规范写入这个共享内存，Sub-agent 启动时，不是由 Orchestrator 告诉它规则，而是它自己去共享内存里拉取“当前项目的规范”。

*   **协议化的任务分发（Protocolized Handover）**：
    Orchestrator 需要进化。它不能只是用自然语言写一句“去做任务A”。它需要支持结构化的任务分发协议，该协议中必须包含 `Context_Requirements`（上下文需求）字段。
    *   *错误做法*：`User: "Use Spec Kit." -> Orch: "Worker1, do task."`
    *   *正确做法*：`User: "Use Spec Kit." -> Orch: "Worker1, do task. context_includes: [spec_kit_rules, project_manifest]"`

*   **框架的 Agent-Native 化**：
    像 Spec Kit 这样的工具，需要专门为 Agent Swarm 进行重构。它们不能再指望仅仅存在于 Prompt 里就能生效。它们可能需要变成一个能够被 Sub-agent 调用的 **Tool** 或 **MCP Server (Model Context Protocol)**，让子 Agent 主动去获取规范，而不是被动等待上下文“喂”给它。

### 总结

你的观察揭示了当前 LLM 应用开发中的一个**抽象泄漏（Leaky Abstraction）**问题。

只要我们还把 Agent 简单地看作“会聊天的机器人”（Chat），这种单上下文的思维惯性就会导致在扩展到多 Agent 协作时出现严重的“上下文断裂”。**Chat 确实只是 Agent 的一种特例（N=1），而通用的 Agent 理论必须解决 N>1 时的上下文同步与状态一致性问题。**
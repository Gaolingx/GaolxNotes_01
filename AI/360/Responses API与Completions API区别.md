# Responses API与Completions API区别

### 🧑‍💻 User

oai response api和completions api有什么区别？

---

### 🤖 Assistant

**Responses API（新）与 Completions API（主要指 Chat Completions）的核心区别如下。**

OpenAI 目前主要有两类相关接口（常被开发者简称为 Completions / Chat Completions 与 Responses）：

- **Completions / Chat Completions API**（`/v1/chat/completions` 为主流，旧版 `/v1/completions` 已基本淘汰）：传统、成熟、无状态的对话补全接口。
- **Responses API**（`/v1/responses`）：较新的接口，面向 Agent、多轮工具调用和更复杂工作流设计，支持服务端状态管理，并整合了更多内置能力。

### 主要区别对比

| 维度 | Completions / Chat Completions API | Responses API |
|------|------------------------------------|---------------|
| **端点** | `/v1/chat/completions`（主流） | `/v1/responses` |
| **状态管理** | **无状态**。每次请求需客户端完整发送历史 messages | **有状态**。可通过 `previous_response_id` 链式继续对话，无需每次重传全部历史 |
| **输入方式** | 主要是 `messages` 数组（role + content） | `input`（字符串或消息列表） + 可选 `previous_response_id`、tools 等 |
| **输出结构** | `choices[].message`（content + tool_calls 等） | `output` 数组（可包含 message、function_call、reasoning 等多种 item），结构更灵活统一 |
| **工具调用** | 支持 function calling / tools，客户端需自己循环处理 | 工具支持更完善，内置工具更丰富（如 web_search、file_search 等预览能力），更适合 Agent 循环 |
| **推理模型支持** | 支持 o 系列，但推理内容处理相对基础 | 对 o 系列（reasoning）支持更好，可返回 reasoning 摘要等 |
| **对话/Agent 场景** | 适合简单/中等复杂的聊天和单轮工具调用，客户端自己管理上下文 | 更适合多轮 Agent、工具链、需要服务端状态的场景，开发更简洁 |
| **成熟度与生态** | 非常成熟，第三方库、SDK、框架支持最完善 | 较新，功能在持续增强，部分高级特性仍在演进 |
| **典型使用** | 普通聊天机器人、内容生成、简单 function calling | 构建 Agent、多步骤任务、需要历史状态复用、内置工具场景 |

### 关键差异说明

1. **状态（最重要的一点）**
   Chat Completions 是纯无状态的：你每次都要把完整对话历史塞回去。
   Responses 可以用 `previous_response_id` 让服务端记住上下文，代码更干净，也方便做长时间多轮交互。

2. **输出设计**
   Responses 把“回复内容 + 工具调用 + 其他事件”统一成 `output` 列表，处理多步骤/多类型结果时更自然。
   Chat Completions 则是经典的 `choices` + `message` 结构。

3. **工具与内置能力**
   两者都支持自定义 tools / function calling。
   Responses 额外强调内置工具（web search、file search 等）和更顺畅的 Agent 工作流，减少自己拼装循环的代码量。

4. **与 Assistants API 的关系**
   Responses API 在很多场景下被视为更轻量、更直接的替代/补充方案（相比完整的 Assistants + Threads 体系）。很多新项目会优先考虑 Responses 来做 Agent，而不是上完整 Assistants。

5. **模型与功能覆盖**
   主流模型（GPT-4o、o 系列等）两边大多都支持，但新特性和推理相关增强往往先在 Responses 侧体现得更完整。具体以官方模型文档为准。

### 什么时候用哪个？

- **优先用 Chat Completions（Completions）**：
  - 简单聊天、内容生成、已有成熟代码库
  - 需要最大兼容性和第三方生态支持
  - 完全自己控制上下文、不需要服务端状态

- **优先考虑 Responses API**：
  - 构建 Agent / 多轮工具调用 / 复杂工作流
  - 希望用 `previous_response_id` 简化状态管理
  - 需要更好的推理模型体验或内置工具
  - 新项目，愿意跟进 OpenAI 新接口

### 其他注意点

- 旧版纯 Completions（`/v1/completions`，非 chat）已基本过时，新项目不要用。
- 两边都支持流式（streaming）、structured outputs、vision 等多模态能力（具体以当前模型为准）。
- 参数命名、返回字段、工具调用的细节有差异，迁移时需要改代码。
- 定价通常按模型 + token 计算，接口本身差异不大，但 Responses 因状态和工具使用方式不同，实际 token 消耗可能有所区别。
- 官方推荐新复杂应用优先评估 Responses，同时 Chat Completions 仍会长期维护。

**总结一句话**：
Chat Completions 是成熟、通用、无状态的“对话补全”接口；Responses API 是更新、更面向 Agent 和状态管理的“响应生成”接口，开发多步骤智能体时更省心。

如果你有具体场景（比如要做 Agent、工具调用循环、还是普通聊天），或者想看两边的请求示例对比，可以告诉我，我可以再补充代码示例或迁移建议。
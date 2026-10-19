# API与订阅权益对比

## 🧑‍💻 User

使用openai platform中的api调用模型相比oauth请求转发+chatgpt plus/pro订阅能享受哪些额外权益

---
## 😀 默认助手 (7)

I'll verify the current details before answering — let me look up the relevant tools and OpenAI's current policy/features.

我先核对了 OpenAI 现行条款原文（`Terms of Use` 生效 2026-01-01、`OpenAI Services Agreement`／Business Terms 生效 2026-01-01、`Service Terms` 更新于 2026-09-29、`Enterprise privacy` 更新于 2026-01-08），下面每条差异都尽量落到条款依据上。

---

# 一、先厘清：两条路径其实不是"同一件事的两种付款方式"

| | Platform API（官方） | OAuth 转发 + Plus/Pro 订阅 |
|---|---|---|
| 合同主体 | **Business Terms / Services Agreement**（企业级条款） | **Terms of Use**（消费者条款） |
| 技术形态 | 稳定的公开 REST/Responses/SDK 接口 | 逆向 ChatGPT 网页端/客户端的内部接口 |
| 授权范围 | 可嵌入自有产品、卖给终端用户 | 仅限"你个人在 ChatGPT 产品内使用" |
| 数据用途 | 默认不训练 | 默认可能用于训练（需手动 opt-out） |
| 违约后果 | 商业纠纷走仲裁/赔偿 | 账号封停，且**没有**任何救济 |

关键点：**模型本身常常是同一个**，所以"API 更强的智能"往往是误解。真正的增量权益在**条款、数据、接口面、运维**四层。

---

# 二、官方 API 多出来的权益

## 1. 法律层面（最容易被低估，也最重要）

<details>
<summary><b>展开：条款原文依据</b></summary>

**① 产品化/商用授权**
Services Agreement §2.2：OpenAI 授予 Customer "the right to use OpenAI's API to integrate the Services into Customer Applications and to make Customer Applications available to End Users"。

对比 ChatGPT 消费者 ToS 明文禁止："Modify, copy, lease, sell or distribute any of our Services"。

**② 知识产权侵权赔偿（IP Indemnity）**
Service Terms §1（API）：OpenAI 就"第三方主张 Customer 使用或分发 Output 侵犯其知识产权"承担赔偿（有若干除外情形，如你已知侵权、未使用过滤/引用功能、与第三方产品组合使用等）。
Enterprise 同理（§3(b)）。

**这是订阅方案绝对拿不到的兜底。** 如果你把生成内容交付给客户、或做成产品对外发布，一旦被第三方主张侵权，API 客户有 OpenAI 兜底，订阅用户只能自己扛。

**③ 责任上限差异巨大**
- 消费者 ToS：赔偿上限 = **较高者（$100，或索赔前 12 个月已付费用）**
- Services Agreement §14.2：上限 = **索赔前 12 个月你付给 OpenAI 的全部费用**；且 §13.1 的 IP 赔偿**不受责任上限限制**（明文规定）

**④ 其他企业级条款**
- §12.1 有限保证：服务在重大方面符合 Documentation（消费版是纯 "AS IS"）
- §2.3 若更新实质性削弱功能，你可在 5 个工作日内通知终止
- §7 双向保密义务（含 Customer Content）
- §6.4 发票争议流程、§6.6 价格变更 14 天生效预告
- §10 No Publicity、§5.2 每年可索取 SOC 2 审计报告
- 对公付款/发票/最低采购承诺与批量折扣（Enterprise）

**⑤ 不违反任何条款**
消费版 ToS 三条直接相关禁令，OAuth 转发基本条条踩线：
> "You may not share your account credentials or make your account available to anyone else"
> "Attempt to or assist anyone to reverse engineer, decompile or discover the source code or underlying components of our Services"
> "Automatically or programmatically extract data or Output"
> "circumvent any rate limits or restrictions or bypass any protective measures or safety mitigations"

反观 API，同一套条款是**明确授权**你去程序化调用，并禁止的是"倒卖 API key""绕过来自 OpenAI 的速率限制"——正常开发完全合规。

</details>

## 2. 数据与合规

| 项目 | API Platform | ChatGPT Plus/Pro |
|---|---|---|
| 训练用你的数据 | **默认不使用**（2023-03-01 后的 API 数据） | 默认可能使用，需手动 opt-out |
| 保留期 | 默认 ≤30 天，可申请 **ZDR**（符合条件） | 不适用 |
| 微调数据 | 归你独占，**不用于训练他人模型**，可自行删除 | 无法微调 |
| GDPR / DPA | 可签署 **Data Processing Addendum** | 无 |
| HIPAA / BAA | 可签 **BAA** | 无 |
| 审计认证 | SOC 2 Type 2；Enterprise 另含 ISO 27001/27017/27018/27701 | 无 |
| 数据驻留 | Enterprise 支持 10 个区域 | 无 |
| 加密 | AES-256 静态 + TLS 1.2+ 传输 | 同左（但无合同约束力） |

> 对企业采购来说，这一栏基本是"一票否决"项：没有 DPA/BAA/SOC 2 报告，法务和合规部门不会放行 OAuth 转发方案。

## 3. 技术接口面（订阅端根本没有对应能力）

<details>
<summary><b>展开：API 独有 / 可编程的能力</b></summary>

- **完全参数控制**：`system`/`developer` message、`temperature`、`top_p`、`seed`（可复现）、`logprobs`、`max_tokens`、`stop`、并行工具调用
- **结构化输出**：JSON Schema 强约束、function calling / tool use、自定义工具与 MCP
- **内置工具**：web search、code interpreter、file search / 向量存储
- **批处理与异步**：Batch API（通常有大幅折扣）、长任务后台执行
- **非对话模型**：Embeddings、Moderation、TTS / 转写、图像生成、**Realtime API**（语音到语音）
- **训练与评估**：fine-tuning、蒸馏、Evals、文件管理
- **工程化配套**：版本化稳定 endpoint、官方 SDK、流式 SSE、usage/成本 API、按项目分 key、服务账号、团队权限
- **可嵌入形态**：服务端调用、无 UI、可塞进任何产品流程

订阅端给你的是一整套**产品**（记忆、Projects、Deep Research、Codex、Plugins、Dots…），这些是"给人用的"，不是"给程序调的"。

</details>

## 4. 配额、规模与运维

- **按量付费、无消息条数上限**：可以跑到任意规模；订阅是固定月费 + 额度（如 5 小时窗口 / 周上限），且**不可扩容**
- **速率限制按消费层级自动提升**（RPM/TPM），而 OAuth 转发是用真人账号的额度硬扛，一旦并发上来极易触发风控
- **成本可控**：usage dashboard、项目级预算、成本明细、Enterprise 发票与批量折扣
- **支持渠道**：Enterprise 有 SLA、24/7 优先支持、专属客户经理、定制法律条款、安全评审

---

# 三、反过来：API 拿不到、订阅才有的（避免误判）

<details>
<summary><b>展开：选择 API 会失去什么</b></summary>

- **ChatGPT 应用层功能**：记忆、Projects、Deep Research 交互界面、Sites、Plugins、Dots/代理功能、Codex 桌面与 IDE 集成、内置浏览器、Workspace agents、共享项目等
- **部分前沿/Pro 级模式**可能优先或仅提供给订阅端
- **语音输出有明确限制**：Service Terms §8 规定 ChatGPT Voice Output "is for **non-commercial use only**"，且权利转让**不包含** Voice Output——即使你是订阅用户也不能拿它做商业产物
- **固定费用的"封顶"优势**：如果你是个人高频自用，Plus/Pro 的月费可能远低于等量 token 的 API 费用
- **开箱即用**：无需自己写编排、工具链、重试、上下文管理

另外注意：OpenAI 现在有官方的 **Sign in with ChatGPT (SIWC)**，允许第三方应用接入 ChatGPT 账号。但那是**身份与能力接入**，有独立的 `Sign in with ChatGPT Terms`，**不等于**把订阅额度当 API 转卖或转发使用。

</details>

---

# 四、OAuth 转发这条路的具体风险

1. **账号风险**：封停/终止，虽有 appeal 流程但成功率低；你有大量历史对话挂在那个账号上
2. **零保证**：服务会随 OpenAI 任何一次前端改动失效，没有版本契约
3. **凭证即账号**：access token 泄露 = 账号被完全接管，且你可能因"共享账号"被连带处罚
4. **商业交付无兜底**：把你的产品建立在它之上，被下游客户追责时你**没有** IP indemnity，责任 100% 自负
5. **合规审计必然失败**：SOC 2 / GDPR / 等保 / 客户安全问卷，任何一项都会卡在这里
6. **数据默认进训练**：客户数据默认可能被用于训练，这本身就可能构成对客户的违约

---

# 五、怎么选

| 你的场景 | 建议 |
|---|---|
| 个人写代码 / 学习 / 日常问答 | **订阅 + 官方客户端（Codex 等）**，最划算，也合规 |
| 要把模型嵌进自己的产品、对外提供服务 | **必须走 Platform API** |
| 企业内部部署、要过法务/合规 | **Platform API + DPA/SOC 2**（甚至 Enterprise） |
| 高并发、需要结构化输出/工具调用 | **Platform API**（订阅端额度不可扩容） |
| 只是想省钱、把订阅额度"白嫖"给第三方客户端 | 省的钱通常抵不上一次封号 + 一次客户索赔的风险 |

**一句话**：API 卖给你的不是"更聪明的模型"，而是一份**可以合法对外交付、有赔偿兜底、有数据合规背书、且能按需伸缩的合同**。OAuth 转发省下的是 token 费，押上的是你的账号、合规资格和商业责任。

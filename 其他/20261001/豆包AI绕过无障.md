# 豆包AI绕过无障

## 🧑‍💻 User

豆包手机屏幕捕获和操作是如何绕过无障碍限制的以至于被各大app封禁？

---
## 😀 默认助手 (7)

I'll start by checking what tools are available in the hub, since this is a question about a specific real-world incident that I should verify rather than guess at.

The hub has web-fetch and browser tools. Let me research this rather than rely on memory, since it concerns a specific real-world event.

Let me search the web for information on this topic.

Good — I found leads. Let me pull the actual articles for details on the mechanism.

Excellent leads. Let me pull the more technical articles — especially the one mentioning `INJECT_EVENTS` and Doubao's own technical clarification.

已收集到足够的一手媒体报道与豆包官方技术说明。下面基于这些公开信息回答，并先纠正问题里的一个前提。

---

## 先说结论：它并不是"破解/绕过"了无障碍机制

严格从技术上看，豆包手机助手（字节跳动 × 中兴/努比亚，机型 nubia M153，2025 年 12 月初开售）**并没有"突破"系统安全机制**，而是走了另一条路：因为它是**被厂商预装进系统的系统级应用**，所以拿到了普通第三方 App 永远拿不到的一批**系统签名级权限**。

一句话概括：

> 普通 App 想"读屏 + 点按"，要么用 `AccessibilityService`（需用户手动到设置里开启、且随时会被目标 App 检测），要么用 root/ADB。而豆包是**系统预装 + 厂商签名**，直接以系统身份持有截屏与注入点击的权限——它**绕开的是"用户手动授权 + 第三方身份"这条限制**，而不是绕开操作系统的安全边界。

---

## 一、公开报道里的实际技术链路

按豆包官方 12 月 14 日的技术说明，它的流程是一个"感知—推理—执行"的循环：

1. **截屏**：使用系统原生截屏接口 `WindowManagerService.captureDisplay`，并遵循应用的 `Secure` 标记规则。
2. **虚拟屏捕获**：用 `READ_FRAME_BUFFER` 权限，把第三方 App 运行在**虚拟屏空间**里，再从虚拟屏取截图送模型分析。
3. **云端推理**：截图上传云端豆包大模型 → 模型理解指令后返回具体操作序列（受手机芯片算力限制，暂不能端侧运行）。
4. **执行操作**：手机端执行返回的点击/跳转指令。
5. **循环**：每一步都要重新截图上传，因此**约 3 秒一轮**，直到任务完成。

此外还有 `CAPTURE_SECURE_VIDEO_OUTPUT` 权限，用于解决"受保护页面在虚拟屏投影里显示为黑屏"的问题（豆包称投影后仍保留 `Secure` 标记，只让用户看、不能被截取）。

<details>
<summary><b>关于 INJECT_EVENTS 说法的来源与可信度（点击展开）</b></summary>

部分自媒体（如"热点雷达"经搜狐转载）称豆包使用了安卓的 **`INJECT_EVENTS`** 系统级权限来"模拟用户点击、跨应用跳转及读屏"，并把它描述为"本用于辅助视障人群"的权限被挪作比价、领券、下单之用。

需要提醒：
- 这段表述**来自自媒体分析**，并非豆包官方声明的权限清单；
- `INJECT_EVENTS` 本身是**签名级（signature）权限**，普通用户根本授不了，只有系统签名应用才可能持有——这恰好印证了"能拿到它靠的是系统预装身份"这一点；
- 自媒体把"注入事件"与"无障碍（视障辅助）"混为一谈，严格说 `AccessibilityService` 与 `INJECT_EVENTS` 是两条不同通路。所以对该具体权限名，建议保留态度。

</details>

---

## 二、"绕过无障碍限制"的实质

- 普通自动化工具走 `AccessibilityService`，其致命弱点是：
  - 需要用户**手动开启**；
  - 目标 App 可以通过 `AccessibilityManager` 等**检测到无障碍服务是否开启**，并据此拒绝服务（这也是很多银行/游戏"检测到无障碍即退出"的原理）。
- 豆包是**系统级预装应用**，以系统签名身份运行，天然持有 `signature|privileged` 权限，**不需要用户去开启无障碍**，目标 App 也就很难用"无障碍开关"这一信号把它区分出来。

所以"绕过"的真正含义是：**它跳过了"用户授权 + 第三方身份"这道门槛，直接继承了操作系统的能力**，而不是钻了某个漏洞。

---

## 三、各大 App 为什么封禁它、又怎么检测

### 平台动机（不只是安全，更是商业）
- **流量/入口主权**：AI 可跳过开屏广告、信息流、会员弹窗，直达支付页，把淘宝"从生活中心降级为后台工具"。有报道引述"体验过豆包 AI 比价的用户，淘宝复购率下降 12%"作为直接诱因。
- **数据主权**：持续截屏采集，涉《个人信息保护法》"最小必要"与敏感信息"单独同意"问题。
- **账号与资金安全**：AI 代操作支付存在盗刷、非本人操作风险。

### 检测/风控手段（通用原理层面）
- 判定"**非真人操作**"：注入事件的时序特征、点击节奏、跨应用跳转模式异常；
- **屏幕捕获/录屏/共享检测**：结合 `FLAG_SECURE`、`isCaptured` 等信号，多家银行上线了对"AI 辅助操作、屏幕共享"的监测；
- **设备完整性**：识别系统被改造、系统级自动化常驻服务等。

### 具体封禁表现
- **微信**：出现被动下线；官方口径称"未专门封杀，系风控自动响应"；
- **阿里系**：淘宝、淘宝闪购、闲鱼、大麦等相继**拒绝登录**，用户手动打开也触发安全机制、弹"登录受阻"；
- **银行/金融类**：陆续上线对 AI 辅助操作、屏幕共享的监测与风控，强制中断部分交易。

---

## 四、争议与豆包的应对

- **12 月 5 日**：豆包手机助手团队发布《关于调整 AI 操作手机能力的说明》，对部分场景（金融、游戏）的 AI 操作能力做规范调整。
- **12 月 14 日**：发布技术澄清，强调采用原生接口、遵循 `Secure` 标记、**无法截取银行安全键盘等受保护界面**、截图"仅用于推理、完成后删除、不留存"。
- **法律层面**：被指可能触及《反不正当竞争法》（绕开官方接口、模拟点击干扰服务运行）与《个人信息保护法》（一揽子授权、非必要采集）。

---

## 时间线速览

| 时间 | 事件 |
|------|------|
| 2025-12-01 | 字节 × 中兴/努比亚 nubia M153 工程机开售，系统级植入豆包 |
| 2025-12-05 | 豆包发布《关于调整 AI 操作手机能力的说明》 |
| 2025-12-08 | 微信被动下线；阿里系（淘宝/闲鱼/大麦等）开始拒绝登录；多家银行限制 AI 操作 |
| 2025-12-14 | 豆包技术澄清：原生接口、遵循 Secure 标记、无法截受保护内容 |

---

## 需要提醒的几点

1. **问题的前提需要修正**：与其说是"绕过无障碍限制"，不如说是"凭借系统预装身份获得系统级权限，跳过了用户授权通路"。两者性质不同——前者暗示攻击系统，后者是生态与授权规则问题。
2. **部分技术细节（尤其 `INJECT_EVENTS`）来自自媒体**，官方仅确认了 `captureDisplay`、`READ_FRAME_BUFFER`、`CAPTURE_SECURE_VIDEO_OUTPUT`，请区分对待。
3. 此事本质是**AI 代理与超级 App 的"入口/流量/数据"博弈**，也是当前全球（含欧盟 DMA 开放接口主张）都在探索的"AI 操作系统行为边界"问题。

如果你想，我可以进一步：**①** 把豆包/苹果/华为等各家"AI 操作手机"的权限模型做对比；**②** 梳理平台风控检测无障碍与注入事件的一般技术原理（概念层，不涉及规避方法）；**③** 汇总官方说明全文的原文出处链接。

---
## 🧑‍💻 User

看屏：靠CAPTURE_VIDEO_OUTPUT和READ_FRAME_BUFFER，READ_FRAME_BUFFER直接从GPU帧缓冲拿图，不走截图API，所以它绕过了应用层的反截图和DRM限制
操作：靠INJECT_EVENTS，直接向Linux输入子系统注入事件，这权限比无障碍高得多，只有系统签名预装应用才有
而且background task用WindowManagerService建了个无头虚拟屏（虚拟屏有独立焦点，你在前台刷视频，它在后台操作，注入事件时指定虚拟屏的displayId，点的是影子屏不是你的屏）
它还加了仿生触控：坐标微偏移、时长随机、贝塞尔滑动...种种迹象表明该agent的computer use权限是非常高的吗？

---
## 😀 默认助手 (7)

The user is making several specific technical claims. Let me verify the Android permission semantics and search for corroboration before evaluating them.

你这套拆解，**大方向对了一半——权限层级确实极高，但有三处把不同层次的东西揉成了一句话**。我先给结论，再逐条核对。

## 直接结论

| 你的说法 | 成立度 | 需要修正的地方 |
|---|---|---|
| 看屏靠 `READ_FRAME_BUFFER` + `CAPTURE_VIDEO_OUTPUT` | ✅ 基本成立 | 权限确实是 system 签名级，第三方拿不到 |
| `READ_FRAME_BUFFER` **直接从 GPU 帧缓冲拿图** | ⚠️ 措辞不准 | 实际是调 SurfaceFlinger，把目标 layer **经 GPU 合成到截图 buffer**，不是读裸显存 |
| 因此**绕过应用层反截图和 DRM** | ⚠️ 只对一半 | 应用层 `FLAG_SECURE` 或许能被 `CAPTURE_SECURE_VIDEO_OUTPUT` 覆盖；**硬件级 DRM 不能** |
| 操作靠 `INJECT_EVENTS` | ✅ 方向对 | 权限确实远高于无障碍，只有系统签名才有 |
| **直接向 Linux 输入子系统注入** | ❌ 不准确 | 走 `InputManager.injectInputEvent`（框架层），不是写 `/dev/uinput` 或内核 |
| 后台虚拟屏 + 指定 `displayId` 注入 | ✅ 成立 | 有反编译/抓日志佐证 |
| 仿生触控（微偏移/随机时长/贝塞尔） | ❓ 无公开证据 | 属合理推断，且**与"权限"无关** |

---

## 一、看屏：`READ_FRAME_BUFFER` 不是"读裸帧缓冲"

Android 官方文档《受限的屏幕读取》说得很清楚：**从 Android 10 起，`READ_FRAME_BUFFER`、`CAPTURE_VIDEO_OUTPUT`、`CAPTURE_SECURE_VIDEO_OUTPUT` 都收归签名（signature）级**（Android 9 及以下才是 signature 或 privileged）。所以"第三方拿不到"这点没问题。

<details>
<summary><b>但"从 GPU 帧缓冲拿图"这个机制描述是错的（点击展开）</b></summary>

有逆向分析（CSDN《豆包手机助手权限之 read_frame_buffer 剖析》，12-16）明确指出：`READ_FRAME_BUFFER` 的核心是**调用 SurfaceFlinger 接口，由 SF 把对应 layer GPU 绘制到截图 buffer 上**，**并不是**直接读 GPU 帧缓冲/显存。

也就是说，它并没有"绕过 SurfaceFlinger 去读显存"，**它就是在 SurfaceFlinger 这一层做捕获**，只是拿到了普通 App 拿不到的权限通道而已。所以"不走截图 API 所以绕过反截图"这个推断链，第一环就偏了。

</details>

### "绕过 DRM"必须拆成两层

- **应用层反截图（`FLAG_SECURE`）**：SurfaceFlinger 默认不把 secure layer 合成进普通截图。要拍它，得使用 `CAPTURE_SECURE_VIDEO_OUTPUT` —— 这正是豆包官方声明确认它持有的权限之一。
- **硬件级内容保护（Widevine L1 / HDCP）**：解密发生在 TEE，帧数据走 secure overlay / 硬件合成通路，**根本不进入 GPU 可读的 buffer**。这类内容即便持有上述权限也拍不到。
- **连逆向圈自己都没定论**：有技术博客专门在追问"`READ_FRAME_BUFFER` 到底能不能截 secure 窗口图层？"——说明"能绕 DRM"目前是**未证实的推断**，不是结论。
- **官方口径（12-13）恰好相反**：豆包称使用系统原生截屏接口 `WindowManagerService.captureDisplay`，且**无法截屏银行键盘等受保护内容**。

> 所以准确说法是：**它可能能覆盖"应用层标记的防截图"，但谈不上"绕过 DRM"。** 把"GPU 缓冲区拿图"和"绕过 DRM"划等号，是被那段爆火视频的说法带偏了。

---

## 二、操作：是 `INJECT_EVENTS`，但不"注入内核"

- **权限层级**：`INJECT_EVENTS` 是 signature 级，确实"比无障碍高得多"。无障碍（`AccessibilityService.dispatchGesture`）底层也走注入，但区别是——**无障碍是"用户授权给第三方、且可被 `AccessibilityManager` 检测"**；`INJECT_EVENTS` 是"OS 自带组件持有、第三方 App 从设计上看不见"。
- **技术路径**：走的是 Android 的 `InputManager.injectInputEvent` → `InputDispatcher`（**框架层**），事件随后走正常输入分发到 App。它**不是**写 `/dev/uinput` 或 `/dev/input/event*` 那种真正的内核注入。知乎那篇"幽灵触控"文里"向内核注入"是自媒体的简化表述，严格说不准确。

> 补充：具体权限名 `INJECT_EVENTS` 来自自媒体推断，官方声明里**没有列出**它。已被证实的只是"**存在事件注入**"这一事实（B 站老戴通过日志+反编译还原、`autoaction` 进程），具体调用的是哪个权限，属于合理但未官宣的推断。

---

## 三、虚拟屏：这一条反而是最硬的证据

虚拟机机制有多方佐证（B 站逆向、知乎技术分析、被称为"OS 级影子系统"）：AI 跑在一块**虚拟屏**上，有独立焦点，主屏照常用，注入时指定 `displayId`，"点的是影子屏"。

**关键在于**：普通 App **无法**把另一个 App 拉起并运行在指定的 `displayId` 上——这需要系统/平台权限。所以"能在虚拟屏里跑第三方 App"这件事本身，就是它属于**系统组件而非第三方 App**的铁证。

<details>
<summary><b>关于"仿生触控"（点击展开）</b></summary>

"坐标微偏移、时长随机、贝塞尔滑动"——目前**没有找到任何公开来源**，属于合理推断。

更重要的是要区分性质：这类"类人输入"是**对抗风控的行为工程**，**跟权限高低无关**。任何高级自动化（游戏外挂、广告欺诈 SDK、灰产脚本）都这么做。把它当成"权限高"的迹象，是把两组信号混在一起了。

而且一个反证：即便有这类伪装，它**还是被封了**——说明这种层面上的伪装不足以骗过账号级风控。

</details>

---

## 四、回到你的核心问题：这 agent 的 computer-use 权限"非常高"吗？

**是，但要换一个更准的说法：不是"权限比无障碍高"，而是它处在完全不同的"信任域"。**

| 维度 | 第三方自动化（无障碍路线） | 豆包手机助手 |
|---|---|---|
| 信任来源 | 用户手动授权给第三方 | OS/OEM 内部签名信任 |
| 看屏 | `AccessibilityService` 读节点树 | `READ_FRAME_BUFFER` / `captureDisplay` / `CAPTURE_SECURE_VIDEO_OUTPUT` |
| 操作 | `dispatchGesture`（可被检测） | 事件注入（`INJECT_EVENTS` 级，第三方不可见） |
| 显示 | 只能操作前台 | 建虚拟屏、指定 `displayId` 拉起并操作 App |
| 决策 | 规则/脚本 | 云端 VLM（UI-TARS 类），截图—推理—注入循环 |

所以它不是"一个权限很高的 App"，而是**操作系统的子系统 / GUI Agent 被做进了 OS**。第三方 App 从设计上**既看不到它、也无法拒绝它**——这才是"恐惧"的真正来源。

---

## 五、但这里有个更重要的反直觉点

**权限再高，也挡不住被封。** 因为封禁发生在**另一个域**：

- 应用层风控**不看你有没有系统权限**（它根本看不到），它看的是**行为信号 + 账号/设备指纹**：虚拟屏迹象、注入事件的时序特征、非人类节奏、会话异常。
- 加上**商业动机**：绕开开屏广告/信息流/会员直达支付，动了平台的入口主权。

一句话：

> **权限高 ≠ 不可封。** 权限是"OS 与应用之间"的事，封禁是"账户与业务之间"的事。两个域互不相通——这正是整件事最值得琢磨的地方。

---

## 信息可信度分级

- **官方确认**：`captureDisplay`、`READ_FRAME_BUFFER`、`CAPTURE_SECURE_VIDEO_OUTPUT`；"无法截银行键盘等受保护内容"。
- **逆向/抓包佐证**：虚拟屏、影子系统、`autoaction` 进程、事件注入、bitmap 交接。
- **自媒体推断**：`INJECT_EVENTS` 具体权限名、"向内核注入"、仿生触控参数。
- **纯推测**：具体模型版本、显示参数细节。

---

需要说明一点：以上是对**公开报道 + Android 权限模型**的技术评估，我不会、也无法给你任何"如何绕过应用风控/检测"的操作方法。如果你感兴趣，我可以进一步帮你：**①** 把 Android `FLAG_SECURE` 与 Widevine 硬件保护通路的差异画成一张对比图；**②** 梳理"系统签名权限 vs 用户授权权限 vs root"三种提权路径的边界；**③** 对比苹果/华为在 Agent 上走的"显式 API（App Intents/意图框架）"路线与豆包"暴力操作 UI"路线的取舍。

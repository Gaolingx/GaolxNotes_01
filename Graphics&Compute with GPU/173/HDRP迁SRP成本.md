# HDRP迁SRP成本

## 🧑‍💻 User

ai agent时代，一个人将unity hdrp迁移到srp（mobile+desktop urplike，成本怎么样，含light+pipeline(c#/hlsl/compute 涵shading、postfx、volume...)+camera...要求core复刻（compute bsed deferred/gtao/contactshadow/ssr/ssgi/raytracing/hdrplit shader/volume shading）+特性分级+保留srp可拓展性，unity版本6000.3.25f1

---
## 🧑‍💻 User

ai agent时代，将unity hdrp迁移到srp（mobile+desktop urplike，成本怎么样，含light+pipeline(c#/hlsl/compute 涵shading、postfx、volume...)+camera...要求完整复刻+保留srp可拓展性

---
## 🧑‍💻 User

ai agent时代，一个人将unity hdrp迁移到srp（mobile+desktop urplike，成本怎么样，含light+pipeline(c#/hlsl/compute 涵shading、postfx、volume...)+camera...要求完整复刻+特性分级+保留srp可拓展性，unity版本6000.3.25f1

---
## 😀 默认助手 (7)

I'll ground a couple of version-specific facts before giving you the estimate.

## 结论先行

先给一个可能不中听的判断：**你要做的不是"把 HDRP 迁移到 SRP"，而是"从零写第三套 SRP，并让它在功能上收敛到 HDRP、在可缩放性上收敛到 URP"**。这是 Unity 官方两个团队合计上千人月才堆出来的东西，你等于要做它的一个子集。

| 口径 | 人月 (PM) | 说明 |
|---|---|---|
| 名义工作量（无 AI） | **150 – 390 PM** | 低=只做桌面 high + 移动 forward；高=真按你列的 core 全做 |
| AI agent 杠杆后 | **90 – 230 PM** | 中位 ≈ **150–180 PM ≈ 12–16 人年** |
| 日历工期 | **18 – 30 个月** | 峰值 6–8 名资深渲染 + 1 TA + 1 变体/构建 |
| 资金（欧美） | **\$2.7M – \$4.5M** | 按 \$18–30k/PM 全成本 |
| 资金（国内一线） | **¥750万 – ¥1400万** | 按 ¥5–9万/PM 全成本 |
| 之后每年维护 | **原始规模的 10–20%** | 这不是一次性支出，见最后一节 |

> 参考坐标：你给的 `6000.3.25f1` 是 Unity 6.3 的一个补丁（发布于 2026-09），也就是说现在 6.3 已是成熟技术流。URP/HDRP 在 6.x 已全面 Render Graph 化，并带 GPU Resident Drawer、STP 超分、Forward+/Deferred、Probe Volume(APV)。**能白拿的地基都别自己写**——这是决定成本量级的第一因素。

---

## 一、先做需求自检：你列的需求里有三个内在矛盾

这三点不先解决，预算会直接翻倍甚至无解。

| 需求 | 矛盾点 | 处理建议 |
|---|---|---|
| **compute based deferred** 同时面向 mobile | TBDR 架构（Apple/高通 Adreno 主流）上 deferred 的 G-Buffer 带宽是致命的；compute 也打不满 tile on-chip memory | 拆成两套：**desktop = compute deferred**；**mobile = Forward+/on-tile subpass**。别想着"一套 deferred 走天下" |
| **SSGI + Raytracing** 同时面向 mobile | 移动端 RT 在 Vulkan/Metal 上可用性、能耗、发热都无法支撑生产级 SSGI/RT 反射 | mobile 只保留 GTAO + 低分辨率 SSR fallback；SSGI 走 **APV/DDGI**；RT 仅 desktop high 且必须有 raster fallback |
| **HDRP Lit shader 复刻** + 保留 SRP 可扩展性 | HDRP 的 Lit 与 Lighting/Volume/Decal/ShaderGraph 深度耦合，无法"摘出来搬" | 不做等价复刻，做**参数兼容 + 观感逼近**；或者只复刻 80% 常用参数，冷门参数（iridescence/transmission 等）走 tier 降级 |

**如果需求书不能改，"mobile 也算 core"这条会把你从 ~250 PM 推到 ~390 PM。**

---

## 二、成本分解（人月，含低/中/高三档）

### 核心模块

| # | 模块 | 低 | 中 | 高 | AI 敏感度 |
|---|---|---:|---:|---:|---|
| 1 | SRP 骨架 + RenderGraph 接入 + SRP Batcher 契约 | 3 | 6 | 10 | 高 |
| 2 | 光照与阴影（物理量、clustered、PCSS、阴影缓存、屏幕空间阴影、light layer） | 18 | 28 | 42 | **低** |
| 3 | Shader/材质体系（HLSL 分层、Lit、Decal、SubTarget、变体/stripping） | 26 | 40 | 60 | 中 |
| 4 | Volume 系统（Component/Profile/插值/优先级/GPU 上传/UI） | 4 | 7 | 11 | **很高** |
| 5 | 后处理（Bloom/DOF/MB/色彩管线/TAA/SMAA） | 14 | 22 | 32 | 高 |
| 6 | Compute Deferred（G-Buffer 打包、clustered lighting pass、deferred decal、MSAA） | 10 | 16 | 24 | 中 |
| 7 | GTAO（含降噪、半分辨率、时间累积） | 6 | 9 | 13 | 中 |
| 8 | Contact Shadow | 2 | 4 | 6 | 中 |
| 9 | SSR（HiZ ray march + 降噪 + probe fallback） | 8 | 13 | 19 | **低** |
| 10 | SSGI（生产可用实际上要 APV/DDGI 兜底） | 10 | 18 | 30 | **低** |
| 11 | Raytracing（DXR + Vulkan RT + Metal RT、BVH 流式、SBT、混合、降噪） | 16 | 30 | 52 | **很低** |
| 12 | 移动端分级与 TBDR 适配（带宽/精度/分级表） | 8 | 14 | 22 | 中 |
| 13 | 迁移与工具（HDRP 资产/Volume/Light 自动迁移、调试视图） | 8 | 14 | 22 | **很高** |
| 14 | 平台验证与工程化（图像回归、性能门禁、变体上限、编译时间、CI） | 12 | 20 | 32 | 高 |
| 15 | Camera/多相机/动态分辨率/STP（不含 XR） | 4 | 8 | 14 | 中 |
| | **合计** | **149** | **249** | **389** | |

<details>
<summary><b>几个模块为什么这么贵 / 便宜（点开）</b></summary>

- **模块 11 Raytracing（16–52 PM）**：不是"写个 RT 反射"的事。你要同时维护 DXR/Vulkan RT/Metal RT 三套后端、BVH 的构建与增量更新与流式、SBT 布局、raster–RT 混合的深度/材质对齐、每平台的降噪、以及内存峰值控制。HDRP 自身的 RT 在非 NVIDIA 桌面上都要靠降级策略兜底，你复刻的版本只会更脆弱。
- **模块 10 SSGI（10–30 PM）**：屏幕空间 GI 的天花板很低（屏幕外信息全丢），生产上真正好用靠 **APV/DDGI**。所以你的"SSGI"很可能最终交付的是"SSGI + APV 混合"，这部分成本被低估得最厉害。
- **模块 4 Volume / 13 工具 / 14 工程化**：纯 C#、契约清晰、可测试——**AI 在这里的杠杆最高**，我按 $2.5\times\sim4\times$ 折算。
- **模块 2 光照与阴影**：成本不在写代码，在"美术能不能接受这个观感"。HDRP 的光强/曝光/色调映射是美术肌肉记忆，你的新管线要做到"看起来一样"需要几十轮肉眼比对。**AI 帮不上**。
- **模块 12 移动端（8–22 PM）**：通常被严重低估。TBDR 上的带宽预算、half 精度导致的 banding、tile 内存限制、设备分级表，都是纯经验活。
</details>

---

## 三、AI agent 到底能省多少（这是本题的核心）

结论：**约 $1.5\times\sim1.9\times$ 的整体加速，中位 $1.77\times$，不是数量级的改变。** 原因是 Amdahl——你的清单里最贵的那几项恰好在 AI 的无效区。

| 工作类型 | 占比 | AI 加速比 | 典型任务 |
|---|---:|---:|---|
| 契约清晰的叶子节点 | ~40% | **2.5–4×** | Volume 组件族、参数序列化、单个 postfx、Inspector、迁移脚本、CI、文档、单平台初版 pass |
| 有模板可循的实现 | ~35% | **1.5–2×** | RenderGraph pass 接线、GTAO/ContactShadow 初版、Deferred、HLSL 库分层 |
| 收敛与调优 | ~25% | **1.0–1.3×** | 光照一致性、SSR/SSGI/RT 降噪、性能/带宽、驱动兼容、变体裁剪、motion vector 全覆盖 |

$$E_{\text{effective}} = E \times \left(\frac{0.40}{2.8} + \frac{0.35}{1.7} + \frac{0.25}{1.15}\right) \approx E \times 0.566$$

$$150 \to 85,\qquad 250 \to 142,\qquad 389 \to 220 \ \text{PM}$$

再加 **25% 的集成/回归尾巴**（注意：代码变多了，**审查与集成的绝对成本是上升的**），得到前面表格里的 90–230 PM。

**Token 成本在这件事里可以忽略**（全程百万级 token ≈ 几十到几百美元），真正的成本是"人类审查 + 跨模块调试 + 性能调优"。所以正确的结论是：**AI 改变的是报价曲线的斜率，不是量级。**

### 让 AI 真正省钱的唯一方式：先建自证系统

别让 agent 写架构，让它写**契约明确的叶子**。前置条件是三件套：

1. **一个 pass = 一个文件 + 显式的输入输出 RT 声明 + 配套 golden image 测试**（多平台、固定容差）；
2. **性能门禁**：每平台每档的 frame budget 硬门槛，超了 CI 直接红；
3. **变体预算**：keyword 组合数与冷编译时间设上限，超了不准合。

有了这三样，agent 才能自己判断"我改对了没有"。没有它们，你只是把 bug 写得更快。

---

## 四、五条路线对比（我的建议是别走路线 C）

| 路线 | 内容 | 成本 | 评价 |
|---|---|---:|---|
| **A（推荐）** | HDRP 做 desktop high，URP 做 mobile/低配，共享一层 Authoring/PBR 资产与材质参数映射 | **15–35 PM** | 性价比碾压。业界主流做法。代价：双管线维护 + 观感需对齐 |
| **B（次推荐）** | 以 URP 为基座，**增量**补齐缺口：GTAO、Contact Shadow、SSR、PCSS、更好 Lit，不做 RT/SSGI | **40–80 PM** | 拿走了你 80% 的收益，成本是路线 C 的 1/3 |
| **C（你的提案）** | 全量复刻 core | **150–390 PM** | 唯一优势是"完全自主可控 + 单管线"。除非这是战略资产，否则不划算 |
| **D** | Fork HDRP，加移动端路径 | **80–150 PM** | 便宜但**会继承 HDRP 的架构负担**，移动端永远不达标；且 License 有约束 |
| **E** | 换引擎（UE 等） | 另算 | 通常低于 C，但含团队重训与资产重制 |

<details>
<summary><b>License 提示（必读）</b></summary>

Unity 的 SRP 源码受 **Unity Companion License** 约束：它只允许你"随 Unity 一起"使用和分发，把它抄进自有 package 再对外分发**有法律风险**。公司内部 fork 相对安全，但对外发布（尤其你打算做成通用管线卖/给客户）必须先过法务。这条经常被忽略，但它可能直接否掉路线 D。
</details>

---

## 五、特性分级矩阵（这是"urplike"的真正含义）

| 特性 | Mobile / Low | Desktop Mid | Desktop High | RT / Enthusiast |
|---|---|---|---|---|
| 主几何路径 | Forward+ (on-tile) | Compute Deferred | Compute Deferred | Compute Deferred |
| 阴影 | 1–2 级联，PCF 3×3 | 4 级联 + PCSS | 4 级联 + PCSS + 缓存 | 光追阴影 |
| GTAO | 半分辨率，2–4 采样 | 半分辨率 + 时间累积 | 全分辨率 + 时间累积 | RT AO |
| Contact Shadow | 关 / 8 步 | 半分辨率 16 步 | 全分辨率 | RT |
| SSR | 粗 HiZ，16 步，无累积 | 半分辨率 + TAA 降噪 | 全分辨率 + 分级粗糙度 | RT 反射 |
| GI | APV / Lightmap | APV + 弱 SSGI | 半分辨率 SSGI + APV | DDGI / RT GI |
| 后处理 | Bloom + Tonemap + FXAA | 全量 + SMAA/TAA | 全量 + TAA/STP | 全量 |
| 目标预算 | 60fps @ GPU ≤ 5ms | 1440p 60fps | 4K 60fps | 4K 60fps + RT |

**分级必须从第一天就内建**，不能事后加。它决定了你的 keyword 设计、变体数量、以及能否真正交付——而"事后加分级"是这类项目最常见的翻车点。

---

## 六、AI 无效区：这些成本不可压缩

1. **美术可预期性**：光照单位、曝光、色调映射必须与 HDRP 观感对齐，靠反复肉眼比对。
2. **Temporal 稳定性**：motion vector 必须覆盖 skinned mesh / vertex animation / terrain / particles / decals / 透明——这是跨模块的耦合地狱，闪烁与鬼影全靠人调。
3. **移动端带宽预算**：在 4–6ms 的渲染预算里塞 GTAO，是硬约束，不是算法问题。
4. **变体爆炸**：shader 一变多，冷编译从分钟级涨到小时级。
5. **RenderGraph 资源别名与内存峰值**。
6. **SRP Batcher 常量缓冲布局一致性**：CBUFFER 写错，batcher 失效，性能直接崩——这是 AI 最常写错的地方之一。
7. **驱动/平台差异**：Metal / Vulkan / DX12 的格式、subpass、RT 支持矩阵。

### 红线：出现任一条，建议直接不做

- 团队里**没有 ≥2 名做过 GPU 管线 / RenderGraph 级别**的人；
- 移动端要覆盖超过 3 年机龄、含 GLES3 老设备；
- 要求与 HDRP 观感 1:1；
- 工期少于 12 个月；
- 无法承受 18 个月不产出产品版本。

---

## 七、如果一定要做：节奏与验收

| 里程碑 | 时间 | 交付 |
|---|---|---|
| M0 地基 | 0–2 月 | 架构 + RenderGraph 骨架 + 分级系统 + 最小 Lit + Forward+ 跑通场景 |
| M1 可看 | 2–6 月 | 光照/阴影/后处理/Volume，desktop 可做简单项目 |
| M2 接近可用 | 6–12 月 | Compute Deferred + GTAO + Contact Shadow + SSR |
| M3 生产可用 | 12–18 月 | SSGI/RT + 移动端分级 + 优化 + 迁移工具 |
| M4 稳定 | 18–24 月 | 平台矩阵、变体治理、文档、性能门禁固化 |

**验收标准（写进合同/OKR）：**

- 每个特性在 desktop + mobile 各有一张 golden image，容差固定、纳入 CI；
- mobile：目标机型 60fps，GPU ≤ 5ms；desktop high：4K 60fps；
- shader 变体数 ≤ N，冷编译时间 ≤ M；
- 所有特性可通过 **RendererFeature + Volume 自由组合**，且任意组合不炸；
- 迁移工具能自动完成 ≥ 90% 场景/材质。

> 保留 SRP 可扩展性的落地判断标准只有一条：**外部只有五个入口——RenderGraph、RendererFeature、VolumeComponent、ShaderLibrary、SubTarget**。任何内部实现越过了这五个入口暴露给用户，可扩展性就是名存实亡。

---

## 八、被忽略的两种成本

1. **这不是一次性支出，是持续债务。** Unity 已经多次重构 SRP 公共 API（Render Graph 化本身就是一次大迁移）。你复刻的管线每年要跟着升级，**年维护 ≈ 原始规模的 10–20%**。而你现在做在 `6000.3` 上、下一个 LTS 可能又动 API——这一点在你给出版本号的当下尤其现实。
2. **人才稀缺性溢价。** 能接这个活的资深渲染工程师全球都稀缺，招人周期本身就是工期的一部分；4 人团队的真实工期不是 150/4 = 37 个月，而是更久且质量不可预测。

---

**一句话总结**：这套需求在"AI agent 时代"的报价是 **150–390 名义人月 / 90–230 有效人月（≈ 12–16 人年）**，价值天花板却只等于"单管线 + 完全自主"。建议先做需求自检（去掉 mobile 上的 deferred/SSGI/RT），认真评估**路线 A 或 B**（15–35 PM / 40–80 PM），只在你确实要把它当战略资产、并且有 2 名以上管线级工程师时，才走全量复刻。

如果你能告诉我：① 目标平台的确切机型/API 下限，② 是否必须与 HDRP 观感 1:1，③ 团队现有渲染人力与是否含 XR，我可以把上面这个区间收敛成一份带人名排期的具体预算。

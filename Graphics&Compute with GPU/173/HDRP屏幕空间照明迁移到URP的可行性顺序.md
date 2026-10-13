# HDRP屏幕空间照明迁移到URP的可行性顺序

## 🧑‍💻 User

https://github.com/Unity-Technologies/Graphics/tree/6000.5/staging/Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/LightLoop
将这一部分迁移到unity6000的urp中是否可行（包含shader+c# scripts）？或者按从易到难排个序

---
## 🧑‍💻 User

https://github.com/Unity-Technologies/Graphics/tree/master/Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/ScreenSpaceLighting
将这一部分迁移到unity6000的urp中是否可行？或者按从易到难排个序

---
## 😀 默认助手 (7)

I'll pull the actual file listing for that directory first, then reason about the port.

## 结论速览

| 问题 | 结论 |
|---|---|
| 整目录「平移」到 Unity 6 URP（含 shader + C#） | **不可行**。不是"工作量大"，而是**没有对接面**：`LightLoop.cs` 是 `HDRenderPipeline` 的 `partial class`（103 KB），`LightLoop.hlsl` 绑死 HDRP 的 BSDF/材质框架；两边加速结构的**数据模型完全不同** |
| 按文件粒度「选择性算法移植」 | **可行**，但难度差异是数量级：最容易的几小时搞定，最有价值的同时也最贵（各以月计） |
| 目录规模 | 25 个文件、约 360 KB 源码（≈1 万行），最重的 4 个文件占了 55%：`LightLoop.cs`(103 KB)、`scrbound.compute`(40 KB)、`LightLoop.hlsl`(38 KB)、`lightlistbuild-clustered.compute`(28 KB) |

**一句话**：这个目录不是"一个功能模块"，而是 HDRP **整套光照管线的核心**（光源数据模型 + GPU 剔除/分簇 + 光照评估 + 阴影 + 反射/环境光 + CPU 调度）。把它搬进 URP ≈ 用 HDRP 的思路重写 URP 的光照系统。**选择性取 2–4 个算法**才是理性选择。

---

## 一、两边的架构差异（这是"为什么难"的根源）

<details>
<summary><b>展开：HDRP LightLoop vs URP Forward+/Deferred+ 对照表（我已核对源码）</b></summary>

| 维度 | HDRP（本目录） | URP 17.x / Unity 6（已核对） |
|---|---|---|
| 光源数据 | `StructuredBuffer<LightData> _LightDatas` / `_EnvLightDatas`，字段富（cookie、IES、`shadowIndex`、projection、dimmer、distance fade、area light LTC） | `_AdditionalLightsPosition/Color/Attenuation/SpotDir`（half 精度 uniform 数组）或 `_AdditionalLightsBuffer`（每物体索引），上限受 `UniversalRenderPipeline.maxVisibleAdditionalLights` 约束 |
| 分簇结构 | 4 个 category（Punctual/Area/Env/Decal）× cluster 的**分层 offset buffer**（`g_vLayeredOffsetsBuffer`，16-bit packed offset+count）+ `g_logBaseBuffer`（每 tile 的 log base）+ `g_vLightListCluster` | **位掩码**：`urp_Tiles` / `urp_ZBins` + `ClusterIterator`（`ShaderLibrary/Clustering.hlsl`），`MAX_LIGHTS_PER_TILE` 32 位，`lightIndex += URP_FP_DIRECTIONAL_LIGHTS_COUNT` |
| 分簇算法 | 三种模式 `USE_CLUSTERED_LIGHTLIST`(32×32) / `USE_BIG_TILE_LIGHTLIST`(64×64) / `USE_FPTL_LIGHTLIST`(16×16)；z 分桶用 log-base：$k=\mathrm{clamp}(\log_b(\mathrm{lerp}(1,b^{C},d)),0,C-1)$，$C=2^{\text{Log2NumClusters}}$ | 单一 z-binning：`zBinIndex = log2(viewZ) * URP_FP_ZBIN_SCALE + URP_FP_ZBIN_OFFSET` |
| 谁构建列表 | **GPU 端**：`lightlistbuild*.compute` + `scrbound.compute`（屏幕空间 AABB 生成/压缩/排序）+ `builddispatchindirect.compute`；CPU 端只准备 bounds | **CPU Burst Job**：`ForwardLights.ScheduleClusteringJob` → `NativeArray<uint> m_ZBins / m_TileMasks` |
| 光照入口 | `LightLoop()`（`LightLoop.hlsl`）+ `Deferred.compute`，跑在 BSDF 框架上（`BSDFData/PreLightData/AggregateLighting/BuiltinData/PostEvaluateBSDF`） | `UniversalFragmentPBR`（`Lighting.hlsl`）+ `RealtimeLights.hlsl` 的 `LIGHT_LOOP_BEGIN/END`；deferred 是 stencil-deferred `Shaders/Utils/StencilDeferred.shader` |
| 材质/光源特征变体 | `NUM_FEATURE_VARIANTS=29`、`LightFeatureFlags`、`materialflags.compute`、`_StencilTexture` | **无此系统** |
| 阴影 | `HDShadowContext` / `HDShadowLoop.hlsl`（cascade、PCSS、透射、彩色屏幕空间阴影、接触阴影） | `ShaderLibrary/Shadows.hlsl` + `ShadowUtils.cs` + `AdditionalLightsShadowCasterPass` |
| 反射/环境光 | octahedral cube+planar 反射探针图集（`SampleEnv`、`_ReflectionAtlas`、`PLANAR_CAPTURE_VP`） | `ReflectionProbeManager.cs`（另一套，无 planar atlas） |
| 全局设置 | `GlobalLightLoopSettings.cs`（tile size、cluster 数、图集尺寸） | `UniversalRenderPipelineGlobalSettings` |
| 其他耦合 | XR 双视图 cbuffer、RayTracing 寄存器、VolumetricClouds、SSS、ProbeVolume | 均无对应物 |

</details>

还有两个**工程层面**的硬障碍：

1. **跨包 include / asmdef**：所有 HDRP 头文件路径都是 `Packages/com.unity.render-pipelines.high-definition/...`，URP 的 `Unity.RenderPipelines.Universal.*.asmdef` 不依赖 HDRP 包。要在 URP 里 `#include` 这些文件，等于让 URP 包反向依赖 HDRP 包 —— 官方不支持两套 SRP 混用（关键字空间如 `SHADERPASS_*`、`USE_CLUSTERED_LIGHTLIST`、宏/结构体重定义会冲突，variant 数量也会爆）。
2. **版本错位**：你给的是 `6000.5/staging`（Unity 6.5 分支），而"Unity 6000"是 6.0（URP 17.0）。RenderGraph / RTHandle / `ComputeCommandBuffer` / `GraphicsBuffer` 等 API 在两版之间已有差异，跨版本搬运会额外踩坑。要做对照，请用**同分支**的 HDRP↔URP 源文件。

---

## 二、从易到难的迁移排序（核心答案）

> 代价是按"能跑通并有一定正确性"估算的整数人日/周/月。Tier 3 的每一项都还要求 C# 端同步改造。

### 🟢 Tier 0 — 几乎所有代码可原样复用（合计 0.5–2 天）

| 文件 | 体积 | 说明 |
|---|---|---|
| `LightCullUtils.hlsl` | 1.4 KB | 纯索引数学（`GenerateLightCullDataIndex`、`ScreenSpaceBoundsIndices`），零依赖 |
| `ClusteredUtils.hlsl` | 4.2 KB | log-base 分簇数学（`SnapToClusterIdxFlex`、`ClusterIdxToZFlex`、`SuggestLogBase50/25`、`GenerateLogBaseBufferIndex`、`GenerateLayeredOffsetBufferIndex`）。只需替换 `g_iLog2NumClusters / g_fNearPlane / g_fFarPlane / g_fClustBase` 的来源 |
| `ShaderBase.hlsl` | 0.8 KB | 深度读取辅助，URP 有 `SampleSceneDepth` 等，可直接丢弃 |
| `ClearLightLists.compute` / `lightlistbuild-clearatomic.compute` / `cleardispatchindirect.compute` | 0.5–1.5 KB | 清 buffer / 清 indirect args，改 buffer 名即可 |
| `SortingComputeUtils.hlsl` | 9.4 KB | GPU 排序工具（bitonic 等），与管线无关，可整段复用（仅当你走 GPU 路线时才需要） |
| `VolumetricCloudsShadowSampling.hlsl` | 1.5 KB | URP 无体积云 —— **不是移植项，直接删** |

### 🟡 Tier 1 — 改常量/接线，1–3 天每项

| 文件 | 体积 | 关键点 |
|---|---|---|
| `LightLoop.cs.hlsl` | 5.2 KB | HDRP 的 `[GenerateHLSL]` 产物：`LightCategory / LightFeatureFlags / LightVolumeData / SFiniteLightBound / ShaderVariablesLightList`。可手写搬到 URP 侧，但要接进 URP 自己的 codegen（`ShaderTypes.cs`）并避免命名冲突 |
| `GlobalLightLoopSettings.cs` | 12.9 KB | 纯粹是配置数据（tile size、cluster 数、图集尺寸），逻辑不难；难在要挂进 `UniversalRenderPipelineGlobalSettings` + Inspector + 序列化 + 版本迁移 |
| `CookieSampling.hlsl` | 2.3 KB | HDRP cookie 图集采样。**URP 已有** `ShaderLibrary/LightCookie/` + `LightCookieManager.cs`，建议对齐 URP 自己那套而不是搬这个 |
| `builddispatchindirect.compute` | 2.8 KB | 由 bounds 数量算 indirect dispatch 维度，接 `DispatchIndirect` 即可；价值取决于你是否真做 GPU-driven |

### 🟠 Tier 2 — 需要重接资源绑定 / 阴影 / 大规模几何（2–6 周每项）

| 文件 | 体积 | 难点 |
|---|---|---|
| `ShaderVariablesLightLoop.hlsl` | 1.9 KB | 资源声明层，改写不难，但 `_CookieAtlas / _ReflectionAtlas / _ContactShadowTexture / _ScreenSpaceShadowsTexture / _IndirectDiffuseTexture` 与 RT 寄存器（`RAY_TRACING_*`）**不是一一对应**，要重新设计绑定 |
| `HDShadow.hlsl` + `HDShadowLoop.hlsl` | 5.8 + 13.7 KB | 阴影上下文与过滤循环。URP 的 atlas 布局与每光 shadow 数据不同，属于"功能移植"（想要 PCSS/彩色屏幕空间阴影/接触阴影）而非文件搬运 |
| `LightingConvexHullUtils.hlsl` | 5.9 KB | 截头棱锥凸包 + 球/瓦片相交（`GetHullVertex/GetHullPlane/DoesSphereOverlapTile`），数学上独立，但只在做屏幕空间 bounds/剔除时才有意义 |
| `lightlistbuild-bigtile.compute` | 16.5 KB | 64×64 big-tile 列表，URP 无对应；若只要 cluster 可以不搬 |
| `lightlistbuild.compute` | 28 KB | FPTL 16×16 版本，与 big-tile 二选一，通常只需 cluster 版 |
| `Deferred.compute` | 11 KB | HDRP 的 deferred 光照 compute，跑在 HDRP BSDF 上；URP 是 stencil-deferred，**无法对接**，只能当参考 |

### 🔴 Tier 3 — 核心算法 + 数据流重构（1–3 个月每项，且必须与 C# 同步）

| 文件 | 体积 | 为什么最贵 |
|---|---|---|
| `lightlistbuild-clustered.compute` | 28 KB | 真正值钱的东西：GPU 端按 category 分桶、按 cluster 打包分层 offset、可选 GPU 排序、per-tile log base。要跑起来必须**先把 URP 的光源数据改成 StructuredBuffer 路线**（否则 shader 里没有 `_LightDatas` 可索引） |
| `scrbound.compute` | 40 KB | 屏幕空间 bounds 的 AABB 生成 + 自适应缩减（resize/compaction）+ 排序。HDRP 能在几千光源下保持性能的关键；配套要自己写 URP 版 CPU bounds 上传 |
| `LightLoopDef.hlsl` | 21 KB | `LightLoopContext`、`GetCountAndStart*`（tile/cluster/bigtile 三分支）、`FetchIndex/FetchLight`、接触阴影打包、屏幕空间阴影、反射探针图集采样。搬到 URP 意味着把 `FetchLight(start,i)` 语义换成 URP 的位掩码+`_AdditionalLights*`，等于重写光源访问层 + 改 `RealtimeLights.hlsl` |
| `LightLoop.hlsl` | 38 KB | 最难的 shader 之一：假设 HDRP 的 `BSDFData/PreLightData/AggregateLighting/BuiltinData/PostEvaluateBSDF`、LTC 面光源、透射/SSS、`_DebugLightingMode`、volumetric 挂钩、RTGI。要用上它 ≈ 重写 URP 的 `Lighting.hlsl` + 材质框架，**不划算** |
| `LightLoop.cs` | 103 KB | 最贵：`HDRenderPipeline` 的 partial class，CPU 端光源收集（`HDAdditionalLightData`）、bounds 计算、cookie/反射图集更新（`LightCookieManager`/`ReflectionProbeTextureCache`/`IBLFilterBSDF`）、阴影请求（`HDShadowManager`）、GPU buffer 池、每相机 cbuffer、indirect dispatch、XR、光追寄存器；还有 `unsafe` + fixed buffer。**这份文件等于 HDRP 光照管线的另一半** |
| `materialflags.compute` | 3.8 KB | 依赖 HDRP 的 material feature variant 体系（`Lit.hlsl`、`HDStencilUsage.cs.hlsl`、`_StencilTexture`）。URP 里**没有落点**，除非连这套变体机制一起搬 |

**总账**：完整照搬 ≈ **3–8 人月**，产出物是一个"在 URP 包里的 HDRP 光照系统"，而且每 6 周一次的 Unity/URP 升级都要重新 merge。

---

## 三、如果真要动手，我建议的路径

<details>
<summary><b>展开：三条可选路线 + 最小可行移植步骤</b></summary>

**路线 A（最省事，通常也是正解）**
- 需要 HDRP 级光照 → 直接用 HDRP；
- 需要 URP 的轻量 + 大量光源 → 直接用 URP 的 **Forward+ / Deferred+**（Unity 6 已内置分簇），接受"没有 per-light cookie 索引、没有 category 分层、光源数据精度较低"。

**路线 B（折中，推荐给"想抄算法"的人）** —— 以 URP 现有 Forward+ 数据流为基座，只搬算法：
1. **搬 `ClusteredUtils.hlsl` 的 log-base z 分桶 + per-tile log base**，接到 `ForwardLights.ScheduleClusteringJob`（CPU job）与 `Clustering.hlsl`。改动集中、收益明显（近处 cluster 分辨率显著提升），这是"性价比最高的一步"。
2. 需要**每光源 cookie/IES/面光源 LTC、光源数 > 上限** → 必须把 URP 光源改成 StructuredBuffer（HDRP `_LightDatas` 路线）+ 改写 `RealtimeLights.hlsl` 的 `GetAdditionalPerObjectLight`。这是**根本性改造**，之后 Tier 3 的文件才有意义。
3. 需要 **GPU-driven 剔除/分簇** → 再上 `lightlistbuild-clustered.compute` + `scrbound.compute` + `SortingComputeUtils.hlsl` + `builddispatchindirect.compute` + `LightCullUtils.hlsl` + `LightingConvexHullUtils.hlsl` + `ShaderVariablesLightLoop.hlsl`。这一组基本自洽（只依赖 buffer 与 bounds），但 C# 端的 bounds 生成（`LightLoop.cs` 里那一大坨）要自己重写一个 URP 版。

**路线 C（fork URP 包）**：技术上可行，但要接受 (a) 每次 URP 升级的 merge 成本；(b) **许可约束** —— Graphics 仓库里 SRP 包是 Unity 自己的包许可（Unity Companion / Unity Package License），可在 Unity 项目中修改自用，但**不可再分发**修改后的包源码；把 HDRP 代码混进 URP 包分发有合规风险；(c) asmdef/命名空间/关键字冲突需要逐个清扫。

**不建议**：在同一个项目里同时装 URP 包和 HDRP 包并互相 include —— 官方明确不支持两套 SRP 混用（关键字、宏、`real3`/矩阵宏等重定义，以及 variant 爆炸）。

</details>

---

## 四、一句话回答你的两个问题

- **"是否可行？"** —— 整目录搬运：**不可行**（架构不对接、C# 侧是 HDRP partial class、shader 侧绑死 HDRP 材质框架、加速结构数据模型相反）。
- **"按难度排序？"** —— 可以按上面的 **Tier 0 → Tier 3** 走；但请记住：**越容易的越是边缘工具，越难的才是这个目录的价值所在**。真正可落地的做法是从 Tier 0/1 里挑算法，配合路线 B 的第 1 步先拿收益，再决定要不要上 StructuredBuffer + GPU-driven 的"大改造"。

需要的话，我可以把 Tier 0/Tier 1 的文件直接改写成 URP 版本（例如把 `ClusteredUtils.hlsl` 的 log-base 分桶接到 URP `ForwardLights` 的 CPU job 上），那份改动的范围是可控且可验证的。

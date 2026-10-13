# HDRP 屏幕空间光照 迁移 到 URP 可行性

## 🧑‍💻 User

https://github.com/Unity-Technologies/Graphics/tree/6000.5/staging/Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/ScreenSpaceLighting
将这一部分迁移到unity6000的urp中是否可行（包含shader+c# scripts）？或者按从易到难排个序

---
## 😀 默认助手 (7)

I'll research the actual HDRP source and URP's existing infrastructure before answering.

Now let me inspect the actual source files for HDRP-specific dependencies.

## 一句话结论

**可行，但不是"迁移"，而是"重写式移植"（re-implementation）。** 这个文件夹里的 HLSL 计算内核（compute kernels）本身相当自包含、可以搬；真正的工作量在于它赖以为生的 **HDRP 渲染管线"接缝"**（光照缓冲注入点、光簇/EnvLight 结构、历史/有效性框架、Depth Pyramid、全局常量缓冲），这些 URP 一个都没有，必须重写或用 URP 的对应物替代。

按从易到难排：**常量缓冲/数据结构 → BilateralUpsample → 共用的 RayTracing 采样库 → RayMarching + Depth Pyramid → GTAO(SSAO) → ScreenSpaceTracing/折射 → SSGI → SSR → 透明/折射 SSR + 厚度缓冲**。一个熟手从零到"完整保真版"大约是 **3–6 个月**；做到"质量可接受、可用于项目"的降级版约 **6–10 周**。

---

## 一、先看清这个文件夹的真实内容

我已核对 `6000.5/staging` 分支该目录（共 60 个条目，含 `.meta`），实际源码为：

| 类别 | 文件 | 说明 |
|---|---|---|
| Volume 设置/数据 | `ScreenSpaceReflection.cs`(383行)、`ScreenSpaceAmbientOcclusion.cs`(245行)、`ScreenSpaceRefraction.cs`、`ScreenSpaceGlobalIllumination.cs`、`BilateralUpsampleDef.cs`、`RayMarchingFallbackHierarchy.cs` | 纯设置类，**最容易搬** |
| CBuffer 定义 | `ShaderVariablesScreenSpaceReflection.cs`(+`.cs.hlsl`)、`ShaderVariablesAmbientOcclusion.cs`(+`.cs.hlsl`)、`ShaderVariablesScreenSpaceLighting.hlsl` | 小结构体 + `ConstantBuffer.Push`，半天级 |
| 管线编排(C#) | `HDRenderPipeline.AmbientOcclusion.cs`(473行)、`HDRenderPipeline.ScreenSpaceGlobalIllumination.cs`(518行) | 这俩是 `HDRenderPipeline` 的 **partial class**，整段 RenderGraph 编排嵌死在 HDRP 里 |
| GTAO(SSAO) | `GTAO.compute`、`GTAOBlurAndUpsample.compute`、`GTAOSpatialDenoise.compute`、`GTAOTemporalDenoise.compute`、`GTAOCopyHistory.compute`、`GTAOCommon.hlsl` | 6 个文件，噪点+时空降噪完整实现 |
| SSGI | `ScreenSpaceGlobalIllumination.compute`(330行, 4 个 kernel: `TraceGlobalIllumination`/`Half`/`ReprojectGlobalIllumination`/`Half`) | 比想象的小，但依赖最脏 |
| SSR | `ScreenSpaceReflections.compute`(**844行, 20 个 kernel**) | 20 个 kernel 大部分是 speed-rejection 排列组合 |
| 共享工具 | `RayMarching.hlsl`、`ScreenSpaceTracing.hlsl`、`BilateralUpsample.hlsl/.compute`、`ScreenSpaceLighting.hlsl` | 真正的可复用核心 |

### 关键事实 1：这个文件夹**不自包含**

SSR 的 RenderGraph pass **不在这个文件夹里**，它在 `Runtime/RenderPipeline/HDRenderPipeline.LightLoop.cs`：

- `class RenderSSRPassData`（第 993 行）、`RenderSSR(...)`（第 1116 行）、实际 dispatch 在 1160–1450 行区间。
- 调用点在 `HDRenderPipeline.RenderGraph.cs`：第 199/206 行（不透明，写入 `lightingBuffers.ssrLightingBuffer`，随后被 **deferred lighting / Lit shader 消费**）+ 第 1819 行（透明 SSR）。
- SSR 还用 `hdCamera.GetCurrentFrameRT(HDCameraFrameHistoryType.ScreenSpaceReflectionAccum)` 做时域累积、`_SsrStencilBit`、rayCountTexture（async compute）、以及 `RTAS`（光追回退路径）。
- SSGI 还依赖 `hdCamera.EffectHistoryValidity` / `PropagateEffectHistoryValidity` / `historyValidationTexture` —— HDRP 自有的"效果历史有效性"框架。
- 折射要用到 HDRP 的厚度缓冲（`HDComputeThickness`，同样在文件夹外）。

**所以"只把这个文件夹复制进 URP 包"必然编译不过、跑不起来。**

---

## 二、依赖对照：什么能直接用，什么必须重写

### 能直接用 / 几乎零成本（好消息）

| HDRP 依赖 | URP 侧现状 |
|---|---|
| RenderGraph | ✅ URP 17 也是 RenderGraph（`AddComputePass`/`AddUnsafePass` 可用） |
| `RTHandle` / `BufferedRTHandleSystem` | ✅ core 包共用 |
| 历史缓冲框架 | ✅ URP 17.0(Unity 6000.0) 起就有 `UniversalCameraHistory`、`RawColorHistory`、`RawDepthHistory`、`TaaHistory`，且 core 的 `CameraHistoryItem` / `ICameraHistoryReadAccess` / `ICameraHistoryWriteAccess` **是 public API**，自定义 RendererFeature 可以注册自己的历史纹理 |
| `RayTracingCommon.hlsl`(126行) / `RaytracingSampling.hlsl`(81行) | ✅ **完全管线无关**（不含 RTAS、只 include `Color.hlsl`），可近乎原样复制（它们在 `Runtime/RenderPipeline/Raytracing/Shaders/`，不在本文件夹） |
| APV（自适应探针体） | ✅ SSGI 里的 `_EnableProbeVolumes` + `EvaluateAdaptiveProbeVolume` 走的是 core 的 `ProbeVolume.hlsl`，URP 已支持 APV |
| `Sampling.hlsl`(蓝噪声/采样) | ✅ core 共用 |
| SSAO 特性槽位 | ✅ URP 已有 `ScreenSpaceAmbientOcclusion.cs` + `ScreenSpaceAmbientOcclusionPass.cs` + `Shaders/Utils/ScreenSpaceAmbientOcclusion.shader` 可作为**移植模板** |
| 法线来源 | ⚠️ 有条件：Deferred GBuffer 或 DepthNormals prepass + `NormalReconstruction`；Forward 默认没有法线贴图 |
| Stencil | ⚠️ URP `StencilUsage` 有 `UserMask = 0b_0000_1111` 4 位用户位可用（HDRP 用自己的位约定） |

### 必须重写（坏消息，也是工作量所在）

| 缺失项 | 影响 |
|---|---|
| **光照缓冲注入点**（`lightingBuffers.ssrLightingBuffer` 被 Lit/Deferred 光照直接消费） | SSR/SSGI 在 HDRP 是"在光照阶段注入"的，不是后处理。URP 的 RendererFeature **没有这个接缝**；要做保真版必须改 URP 的 `Lit.shader`/`Lighting.hlsl`/StencilDeferred（等于 fork URP） |
| **光簇 + EnvLight 结构**（`LIGHTCATEGORY_ENV`、`FetchEnvLight`、`EnvLightData`、`_ReflectionAtlas` cube array、`_CubeScaleOffset`） | SSGI 的 `TraceReflectionProbes()` 和 `ScreenSpaceTracing.hlsl` 全靠它。URP 用的是 SH + 探针 atlas（`ReflectionProbeManager`）+ Forward+ 光照列表，**必须改写命中点着色** |
| **ShdrVariables 全局 CB** | SSR/SSGI compute 都 `#include ShaderVariables.hlsl`（707 行，拖进 LightLoop/Decal/大气/VFX 矩阵）。必须裁成最小全局 include（`_ScreenSize`、jitter、时间等） |
| **Depth Pyramid（分层 Z）** | `RayMarching.hlsl` 依赖 `_DepthTexture` + `_DepthPyramidMipLevelOffsets`；而 `DepthPyramid.compute` / `MipGenerator.cs` / `DepthPyramidConstants.cs` / `HDUtils.PackedMipChainInfo` 全在 HDRP 里，URP **没有**（URP 的 "depth pyramid" 是 GPU occlusion culling 用的，不是这个） |
| **上一帧已光照 color pyramid** | URP 只有 RawColor history / `_CameraOpaqueTexture` 拷贝，需要自己建 mip 链做 SSR 回退 |
| **效果历史有效性框架** | SSGI/SSR 的时域稳定与"重投影有效性"逻辑要自己按 `UniversalCameraHistory` 重写 |
| **厚度缓冲(ComputeThickness)** | 透明/折射 SSR 需要 |
| **Async compute 编排** | HDRP SSR 走 async；URP RenderGraph 支持度有限，建议先做同步版 |
| `HDShaderIDs` / 生成的 `.cs.hlsl` | 32 处 `HDShaderIDs` 引用需映射到 URP 的 `ShaderPropertyId` 或自定义 ID；`.cs.hlsl` 是生成文件，改 `.cs` 后要保持同步 |

---

## 三、从易到难排序（核心答案）

> 难度用 ★ 表示，工时为**单个熟练图形程序**的粗估（含调试），仅为量级参考。

| # | 模块 | 难度 | 估时 | 最大卡点 |
|---|---|---|---|---|
| 1 | `ShaderVariablesScreenSpaceReflection.cs`/`.cs.hlsl`、`ShaderVariablesAmbientOcclusion.cs`、`BilateralUpsampleDef.cs`、`RayMarchingFallbackHierarchy.cs` | ★ | 1–2 天 | 只改 ID/CBUFFER 名与 `ConstantBuffer.Push` |
| 2 | `BilateralUpsample.hlsl` + `BilateralUpsample.compute` | ★ | 0.5–1 天 | 零外部依赖，直接可用 |
| 3 | `RayTracingCommon.hlsl` + `RaytracingSampling.hlsl`（在 Raytracing 目录） | ★ | 0.5 天 | 管线无关，原样复制 |
| 4 | **Depth Pyramid 全套**（`DepthPyramid.compute`+`MipGenerator`+`Constants`+`PackedMipChainInfo`）+ `RayMarching.hlsl` | ★★ | 3–5 天 | 需要 `_DepthPyramidMipLevelOffsets`、`PackedMipChainInfo` 的共享/全屏约定；可先用线性 ray march 代替分层 Z |
| 5 | **GTAO / SSAO 全家桶**（6 个 compute + `HDRenderPipeline.AmbientOcclusion.cs` + `ScreenSpaceAmbientOcclusion.cs`） | ★★ | 1–2 周 | 主要改历史缓冲分配（`AllocateAmbientOcclusionHistoryBuffer` → `UniversalCameraHistory`）；法线来源；时域降噪需 motion vector。**这是唯一"URP 已有同槽位特性"的模块，最好上手** |
| 6 | `ScreenSpaceRefraction.cs` + `ScreenSpaceTracing.hlsl`（代理求交） | ★★★ | 1 周 | `EnvLightData`/`WorldToProxySpace`/`IntersectSphereProxy` 必须用 URP 的探针表示重写；`ScreenSpaceLighting.hlsl` 里的 GBuffer 宏也要换成 URP 的 |
| 7 | **SSGI**（1 compute + 518 行 C#，约 10 个 RenderGraph pass：trace/reproject/filter/upsample/bilateral + 历史有效性） | ★★★★ | 3–5 周（降级版：只采样 SH+APV+探针）/ 6–10 周（保真：命中点走完整光照求值） | 光簇 EnvLight 重写 + 历史有效性框架重写 + 命中点光照用 URP 的 Forward+/Deferred 光照列表或 SH 近似 |
| 8 | **SSR**（`ScreenSpaceReflection.cs` + 844 行/20 kernel 的 compute + `LightLoop.cs` 里的 `RenderSSR`） | ★★★★★ | 3–4 周（**后处理合成版**）/ 8–12 周（**注入式保真版**） | 保真版必须改 URP `Lit.shader`/deferred 光照才能"物理正确混合粗糙度/探针回退"；20 个 kernel 可裁剪为几个；async compute 建议放弃 |
| 9 | **透明 / 折射 SSR + 厚度缓冲**（`RenderGraph.cs` 1819 行分支 + `HDComputeThickness`） | ★★★★★ | 2–4 周（依赖 #8） | 需要 URP 透明 pass 的额外 lighting buffer 注入 + 厚度缓冲；可与 #8 一起做 |

**排序总结（易 → 难）：**
`CBuffer/数据结构` → `BilateralUpsample` → `RT 采样库` → `RayMarching + DepthPyramid` → `GTAO` → `ScreenSpaceTracing/折射` → `SSGI` → `SSR` → `透明SSR + 厚度缓冲`

---

## 四、推荐的落地路线（务实版）

**阶段 A｜2–3 周：先摘"低垂果实"**
1. 抄 #1–#3（CBuffer + BilateralUpsample + RT 采样库），半天级。
2. 用 URP 现有的 `ScreenSpaceAmbientOcclusionPass.cs` 作为骨架写一个新的 `ScriptableRendererFeature`，把 **GTAO** compute 接进去（`AddComputePass`），历史用 `UniversalCameraHistory`。
3. 收益：立刻得到比 URP 内置 SSAO 更好的 GTAO（含时空降噪），而且**完全不需要动 URP 的 shader**，风险最低。这一步就能验证你的整个移植基础设施（RenderGraph/历史/全屏 dispatch）是否跑通。

**阶段 B｜3–5 周：SSGI 降级版**
4. 搬 `ScreenSpaceGlobalIllumination.compute` 的 4 个 kernel，把 `TraceReflectionProbes()` 换成 **SH + Reflection Probe Atlas + APV** 采样（放弃 HDRP 光簇的逐光求值）。
5. 依赖 `RayMarching.hlsl` + Depth Pyramid（阶段 A 若没做，这里补）。
6. 历史/有效性用 `UniversalCameraHistory` + 自定义 validity 重写。
7. 收益：URP 里首次有可用的屏幕空间 GI（质量低于 HDRP，但可用）。

**阶段 C｜3–6 周起：SSR（建议先做后处理合成版）**
8. 在 `RenderPassEvent.BeforeRenderingTransparents`（此时 `_CameraOpaqueTexture`/深度/法线齐备）插入 SSR：trace + reproject + 少量 accumulate kernel，把结果**加法合成**到相机颜色上。
9. 这条路损失的是"HDRP 那种在光照阶段与探针/粗糙度正确混合"的能力，但换来不动 URP 核心 shader。**如果你的项目能接受自定义 Lit shader（或本来就是自定义 shader 项目），再考虑注入式保真版。**

---

## 五、几个必须提前知道的坑

<details>
<summary><b>1. 不要忽视"文件夹之外"的依赖清单</b></summary>

除了上面列的，实际还要处理：`HDCamera`(25 处)、`HDShaderIDs`(SSGI 里 32 处)、`HDRenderPipelineRuntimeResources`（ComputeShader 资源注册，URP 要换成 `IRenderPipelineResources`/GraphicsSettings 或自己的资源容器）、`HDUtils.PackedMipChainInfo`、`SkyManager`、`LightCluster`、`RTAS`。**先做一份"符号 → URP 替代"的映射表**，否则会陷入边编译边改的泥潭。
</details>

<details>
<summary><b>2. URP 其实已经在做"下一代 GI"，不一定是这条路</b></summary>

在 `6000.5/staging`（URP 17.5）里我找到了 **`SurfaceCacheGIRendererFeature`**（`Runtime/RendererFeatures/SurfaceCacheGIRendererFeature/`）：被 `#if SURFACE_CACHE` 包裹、用 `UnityEngine.Rendering.LiveGI` + `SurfaceCacheWorld` + `UnifiedRayTracing`，是"表面缓存 + 光追"路线的实验特性（在 `6000.0/6000.2` 分支**不存在**，我逐一验证过）。

含义：如果你追的是"Unity 官方未来的 URP GI"，应该跟这条线；如果你要的是"HDRP 那种屏幕空间 SSR/SSGI 手感与可控性"，那么自己移植是合理的。两者不冲突，但要认清取舍。
</details>

<details>
<summary><b>3. 授权问题（Unity Companion License）</b></summary>

`Graphics` 仓库的 HDRP/URP 代码是按 **Unity Companion License** 发布的（允许在 Unity 依赖型项目中使用/修改，但有再分发与条款约束）。把代码搬进自研包内部分发前，请确认你的分发形态（尤其是要发布到 Asset Store / 对外 SDK 时）。
</details>

<details>
<summary><b>4. 其他工程性提醒</b></summary>

- **`.cs.hlsl` 是生成文件**：改了 `.cs` 里的结构体后必须重新生成（HDRP 用 core 的 ShaderGenerator），否则 C# 与 HLSL 的 CB 布局会不一致，症状是"随机数值错乱"，极难查。
- **Forward 路径没有法线缓冲**：SSR/SSGI 都需要法线。要么强制用户用 Deferred，要么让特性依赖 DepthNormals prepass（并注意 URP 的法线是八面体量化编码，`NormalBuffer.hlsl` 的解码函数不能直接照抄）。
- **Async compute 先别做**：HDRP SSR 为了 async 把它排在 shadow map 之前（源码注释里写得很清楚），URP 侧照搬很容易踩同步错误。
- **时域累积需要 motion vector 与相机历史重置逻辑**：URP 的 `UniversalCameraHistory` 有 request/write 版本机制，注意"叶子类里请求会晚一帧"（源码注释里也提醒了这点）。
- **先做一个最小的端到端验证**：拿 `RayMarching.hlsl` + Depth Pyramid，在一个自定义 RendererFeature 里把命中位置可视化出来（Debug 输出）。这一步跑通，后面 SSGI/SSR 就只是"接更多输入、加降噪"的工程量问题了。
</details>

---

**如果你告诉我目标 Unity 版本（6000.0 LTS / 6000.2 / 6000.5）和预算（只能用 RendererFeature 不能改 URP 源码，还是可以 fork URP），我可以把上面的排序细化成一份带里程碑和文件级改动清单的实施计划。**

---
## 🧑‍💻 User

目标版本>unity6000.5，允许fork urp，先阅读urp的renderfeature部分，再继续阅读hdrp c#代码给出完整的迁移报告（新模块尽量是先组织成一个大的newrenderfeature，然后再在newrenderfeature中自由组合需要开启的ScreenSpace feature）

---
## 😀 默认助手 (7)

I'll do this in two passes as you asked: URP's renderer-feature plumbing first, then HDRP's C# orchestration. Starting with URP.

URP side is mapped. Now the HDRP C# orchestration — reading the two partial-class files that own the GTAO and SSGI pipelines.

Research complete. Here is the full report — I read URP's renderer-feature infrastructure first, then HDRP's C# orchestration, and verified every claim below against `6000.5/staging`.

---

# 迁移报告：HDRP `ScreenSpaceLighting` → URP（Unity 6000.5+，允许 fork URP）

## 0. 摘要

**可以做，而且比我第一轮的判断更乐观**：因为 URP 在设计上已经预留了屏幕空间 GI 的注入缝（`_SCREEN_SPACE_IRRADIANCE`），并且 SSAO 的消费端（`_ScreenSpaceOcclusionTexture`）与 HDRP 的 GTAO 输出语义几乎同构。真正需要 fork 的着色器代码**只有 1 个文件**（`ShaderLibrary/GlobalIllumination.hlsl`）。

三个最重要的发现：

| # | 发现 | 影响 |
|---|---|---|
| 1 | URP `ShaderLibrary/GlobalIllumination.hlsl` 里 `half3 GlobalIllumination(...)`（约 505 行）是**前向 Lit（`Lighting.hlsl:343`）与延迟 GBuffer（`LitGBufferPass.hlsl:241`）共用的唯一间接光出口**，其中 `indirectSpecular = GlossyEnvironmentReflection(...)` 可被 SSR buffer 替换 | SSR 的 fork 面收敛为**单函数、单文件** |
| 2 | URP 已有 `_SCREEN_SPACE_IRRADIANCE` 关键字 + `_ScreenSpaceIrradiance` 纹理 + `SAMPLE_GI(irradianceTex,pos)` 宏（`GlobalIllumination.hlsl:18-25,210`，被 `LitForwardPass.hlsl:137`、`LitGBufferPass.hlsl:116` 使用） | **SSGI 的注入缝已经存在**，可零 fork 接入 |
| 3 | URP 延迟路径**在 GBuffer pass 内就把环境/GI 固化进 GBuffer**（`LitGBufferPass.hlsl:229-243`），而 `StencilDeferred.hlsl` 只算直接光 + AO | 延迟路径下 SSGI/SSR 必须**赶在 `BeforeRenderingGbuffer`(=210) 之前**完成 → 必须有 DepthNormals prepass，或走"HDRP 化"改造 |

**总体工时（单人，允许 fork，完整保真）**：约 **6–9 个月**；两名工程师并行（GTAO+共享层 / SSR）约 **4–5 个月**。可分模块交付，第一个可用版本（GTAO）约 5–6 周。

---

## 1. URP 侧测绘（RendererFeature 基础设施）

### 1.1 Feature 生命周期与可用 API

`Runtime/ScriptableRendererFeature.cs`（129 行）：

```csharp
public abstract partial class ScriptableRendererFeature : ScriptableObject, IDisposable
{
    public bool isActive { get; }                                   // SetActive() 控制
    public abstract void Create();                                  // 反序列化后调用（含 OnEnable/OnValidate）
    public abstract void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData);
    public virtual void OnCameraPreCull(ScriptableRenderer renderer, in CameraData cameraData);
    public void EnqueuePass(...)                                    // 经 renderer.EnqueuePass
    internal virtual bool RequireRenderingLayers(bool isDeferred, bool needsGBufferAccurateNormals,
                                                 out RenderingUtils.Event atEvent, out RenderingUtils.MaskSize maskSize);
    protected virtual void Dispose(bool disposing);
}
```

特性声明模式（照抄 `ScreenSpaceAmbientOcclusion`）：

```csharp
[SupportedOnRenderer(typeof(UniversalRendererData))]
[DisallowMultipleRendererFeature("Screen Space Lighting")]   // 天然满足"一个大 feature"的要求
[Categorization.CategoryInfo(Name = "R: Screen Space Lighting", Order = 1000)]
public class ScreenSpaceLightingRendererFeature : ScriptableRendererFeature
```

**每帧时序**（`ScriptableRenderer.cs`）：

| 步骤 | 位置 | 说明 |
|---|---|---|
| `OnCameraPreCull` | 1092–1105 | 逐 feature，可用于预取历史/资源 |
| `AddRenderPasses` | 1108–1130 | 逐 feature 调用；`EnqueuePass`(1014) 追加进 `m_ActiveRenderPassQueue` |
| `SortStable` | 929 / 1227 | 按 `renderPassEvent` 稳定排序 → **同 event 的执行序 = 入队序**（模块内顺序完全由我们掌控） |
| `OnRecordRenderGraph` | `UniversalRendererRenderGraph.cs:595` | 内部按顺序调用 `RecordCustomRenderGraphPasses(...)`（见 1.2） |
| `FrameCleanup` | 1179–1191 | 每 pass 清理 |

### 1.2 帧内插桩点全表（`UniversalRendererRenderGraph.OnRecordRenderGraph`）

这是设计新 feature 的**地图**，行号为 `6000.5/staging` 实测：

| 行号 | 插桩点 | 该处已就绪的资源（对本迁移有意义的部分） |
|---|---|---|
| 641 | `BeforeRendering` | 无 |
| 731 / 751 | `BeforeRenderingShadows` | — |
| 733 / 772 | `AfterRenderingShadows` | 阴影贴图、光常量 |
| 845–851 | `CalculateSplitEventRange` + `InEventRange` | **深度读取切分机制**（见 4.3） |
| 972 | `BeforeRenderingPrePasses` | — |
| 1091 | `AfterRenderingPrePasses`(=200) | **深度、DepthNormals 法线、motion vectors 都已生成** ← SSGI/GTAO 的最佳锚点 |
| 1109 | `BeforeRenderingGbuffer`(=210) | 延迟路径下 GBuffer 尚未写 → **延迟 SSGI/SSR 的最后机会** |
| 1146 | `AfterRenderingGbuffer..BeforeRenderingDeferredLights` | GBuffer 已就绪 |
| 1148 | `DeferredPass`（URP 内置） | 延迟光照 |
| 1150 | `AfterRenderingDeferredLights..BeforeRenderingOpaques` | 光照完成 |
| 1215 | `AfterRenderingOpaques`(=300) | 不透明+天空盒后的颜色 |
| 1242 | `BeforeRenderingTransparents`(=450) | `_CameraOpaqueTexture` 已生成 → **透明 SSR/折射锚点** |
| 1269 | `AfterRenderingTransparents`(=500) | 本帧完整 lit color → 生成下一帧 color pyramid 的最佳点 |
| 1320 | `BeforeRenderingPostProcessing`(=550) | 后处理前 |
| 1404 | `AfterRenderingPostProcessing` | 调试视图 |
| 1432 | `AfterRendering` | 最终 |

`RenderPassEvent` 枚举实测值：`BeforeRendering=0, BeforeRenderingShadows=50, AfterRenderingShadows=100, BeforeRenderingPrePasses=150, AfterRenderingPrePasses=200, BeforeRenderingGbuffer=210, AfterRenderingGbuffer=220, BeforeRenderingDeferredLights=230, AfterRenderingDeferredLights=240, BeforeRenderingOpaques=250, AfterRenderingOpaques=300, BeforeRenderingSkybox=350, AfterRenderingSkybox=400, BeforeRenderingTransparents=450, AfterRenderingTransparents=500, BeforeRenderingPostProcessing=550, AfterRenderingPostProcessing=600, AfterRendering=1000`。

### 1.3 `UniversalResourceData` 可用资源（`Runtime/FrameData/UniversalResourceData.cs`）

| 字段 | 行 | 本迁移用途 |
|---|---|---|
| `activeColorTexture` / `activeDepthTexture` | 20 / 61 | 读写目标 |
| `cameraColor` / `cameraDepth` | 126 / 137 | 内部缓冲 |
| `gBuffer[]` | 171 | 延迟法线/clear coat mask |
| `cameraOpaqueTexture` | 183 | 透明折射输入 |
| `cameraNormalsTexture` | 203 | 法线（DepthNormals 编码） |
| `motionVectorColor` | 215 | 时域降噪/重投影 |
| `ssaoTexture` | 313 | **GTAO 输出对齐（可直接复用）** |
| `irradianceTexture`（internal） | 323 | **SSGI 输出对齐（`_ScreenSpaceIrradiance`）** |
| `renderingLayersTexture` | 281 | 分层控制 |

### 1.4 历史系统（可直接用，不用 fork）

core `Runtime/Camera/CameraHistory.cs` 的 `CameraHistoryItem`（public API）：

```csharp
public abstract class CameraHistoryItem : ContextItem
{
    public virtual void OnCreate(BufferedRTHandleSystem owner, uint typeId);
    protected BufferedRTHandleSystem storage { get; }
    protected int MakeId(uint index);                                     // 同一 item 内多张历史纹理
    protected RTHandle AllocHistoryFrameRT(int id, int count, ref RenderTextureDescriptor desc,
                                           FilterMode filterMode, string name);
}
```

请求/读写：`cameraData.historyManager.RequestAccess<T>()` → `GetHistoryForWrite<T>()` / `GetHistoryForRead<T>()` / `IsWritten<T>()`；`UniversalCameraHistory` 同时实现 `ICameraHistoryReadAccess/WriteAccess/IPerFrameHistoryAccessTracker`。URP 自带 `RawColorHistory`、`RawDepthHistory`、`TaaHistory`、`StpHistory` 可作范例。源码注释明确提醒：**"请求放在叶子类会晚一帧"**（对我们意味着 SSR/SSGI 的 accum history 要在 `AddRenderPasses` 阶段或 `OnCameraPreCull` 请求）。

### 1.5 着色器注入缝（三个）

| 缝 | 位置（已实测） | 对应模块 | 是否需 fork |
|---|---|---|---|
| 间接漫反射 | `GlobalIllumination.hlsl:18-25`（`_ScreenSpaceIrradiance`/`SampleScreenSpaceGI`）、`:210`（`SAMPLE_GI` 宏）、消费端 `LitForwardPass.hlsl:137`、`LitGBufferPass.hlsl:116` | **SSGI** | ❌ 已有（6000.5 分支确认；更早版本需自己加，加起来也就 10 行） |
| 间接镜面 | `GlobalIllumination.hlsl` ≈505 `GlobalIllumination()` 内的 `indirectSpecular = GlossyEnvironmentReflection(...)` 与 `EnvironmentBRDF(...)`；调用者：`Lighting.hlsl:343`（前向 `UniversalFragmentPBR`）、`LitGBufferPass.hlsl:241`（延迟） | **SSR** | ✅ 需 fork（单函数） |
| 环境光遮蔽 | `_ScreenSpaceOcclusionTexture` + `_SCREEN_SPACE_OCCLUSION` 关键字 + `_AmbientOcclusionParam`；消费端 `AmbientOcclusion.hlsl`/`GetScreenSpaceAmbientOcclusion()`（见 `StencilDeferred.hlsl:295,340`）；清理逻辑在 `ScriptableRenderer.ClearRenderingState()` | **GTAO** | ❌ 已有，GTAO 可**直接替换 URP SSAO 而不改任何 shader** |

### 1.6 资源注册模式（照抄 SSAO）

```csharp
[Serializable]
[SupportedOnRenderPipeline(typeof(UniversalRenderPipelineAsset))]
class XxxPersistentResources : IRenderPipelineResources
{
    [SerializeField, ResourcePath("Shaders/Utils/ScreenSpaceAmbientOcclusion.shader")] Shader m_Shader;
    public Shader Shader { get => m_Shader; set => this.SetValueAndNotify(ref m_Shader, value); }
    public bool isAvailableInPlayerBuild => true;
}
// 取用：GraphicsSettings.TryGetRenderPipelineSettings<XxxPersistentResources>(out var res);
```
ComputeShader 同理可用 `[ResourcePath(".../GTAO.compute")]`（`SurfaceCacheGIRendererFeature` 就是这么放 5 个 compute 的）。注意 SSAO 的错误提示揭示了一个坑：**玩家包构建时资源会被 strip，必须保证 feature 被启用或配置 stripping**。

### 1.7 URP 现状速查

- **已有**：RenderGraph、`RTHandle`/`BufferedRTHandleSystem`、历史系统、APV（`ProbeVolume.hlsl`）、探针 atlas（`ReflectionProbeManager.cs`）+ SH、延迟 GBuffer 法线 + `_GBUFFER_NORMALS_OCT`、`DepthNormalOnlyPass`、motion vectors、`StencilUsage.UserMask`（4 位）、蓝噪声（`Sampling.hlsl`）、SSAO/ScreenSpaceShadows/Decal/FullScreenPass 等 feature 范例。
- **没有**：SSR/SSGI 的内置 feature（`6000.5` 里只有实验性 `SurfaceCacheGIRendererFeature`，`#if SURFACE_CACHE`，基于 `SurfaceCacheWorld`/`UnifiedRayTracing`/`LiveGI`，`6000.0`/`6000.2` 分支**不存在**）；分层 Z 深度金字塔；光簇 tile 光照列表的"命中点求值"封装；上一帧 lit color 的 mip 链（只有 `RawColorHistory`）。

---

## 2. HDRP 侧测绘（C# 编排拆解）

### 2.1 文件职责 + 文件夹外的隐藏依赖

<details>
<summary><b>展开：完整文件依赖表（含"必须一起搬/必须丢弃"的判定）</b></summary>

| HDRP 文件 | 角色 | 关键依赖 | 判定 |
|---|---|---|---|
| `HDRenderPipeline.AmbientOcclusion.cs`(473) | GTAO RenderGraph 编排（4 pass） | `HDCamera.AllocateAmbientOcclusionHistoryBuffer`、`HDUtils.PackedMipChainInfo`、`HDShaderIDs` | 重写为 URP pass |
| `HDRenderPipeline.ScreenSpaceGlobalIllumination.cs`(518) | SSGI 编排 + 历史有效性 | 光簇 `g_vLightListTile`、`HDTemporalFilter`、`HDDiffuseDenoiser`、color/depth pyramid、`StencilUsage.ObjectMotionVector`、APV、RT 路径 | 重写（可裁 RT） |
| `Runtime/RenderPipeline/HDRenderPipeline.LightLoop.cs` 993–1460 | **SSR pass**（不在本文件夹） | `lightingBuffers.ssrLightingBuffer`、accum history、`RTAS`、`m_SkyManager` | 重写 |
| `Runtime/RenderPipeline/Raytracing/HDTemporalFilter.cs`(+`.cs.hlsl`) | SSGI 时域降噪 | HDRP 历史系统 | **必须一起搬或替换** |
| `Runtime/RenderPipeline/Raytracing/HDDiffuseDenoiser.cs`、`ReBlurDenoiser.cs` | 光追降噪 | NRD/ReBLUR | 可丢（仅 RT 路径） |
| `Runtime/RenderPipeline/RenderPass/DepthPyramid.compute`、`MipGenerator.cs`、`DepthPyramidConstants.cs`(+`.cs.hlsl`) | 分层 Z 金字塔 | `HDUtils.PackedMipChainInfo` | **必须一起搬**（RayMarching 依赖） |
| `HDUtils.PackedMipChainInfo` | 打包 mip 偏移/缩放 | — | 搬（纯数学） |
| `ScreenSpaceReflections.compute`(844, 20 kernel) | SSR 追踪/重投影/累积 | 见附录 B | 搬 + 裁剪 |
| `ScreenSpaceGlobalIllumination.compute`(330, 4 kernel) | SSGI 追踪/重投影 | HDRP 光簇/EnvLight/APV/材质求值 | 搬 + 改命中点求值 |
| GTAO 系列 5 个 compute + `GTAOCommon.hlsl` | GTAO + 时空降噪 | 深度/法线/`ShaderVariablesAmbientOcclusion` | **几乎原样搬** |
| `ScreenSpaceTracing.hlsl` | 代理体求交（折射） | `EnvLightData`、`WorldToProxySpace`、`IntersectSphereProxy` | 改写探针接口 |
| `RayMarching.hlsl`(141) | 分层 Z 光线步进 | `_DepthTexture`、`_DepthPyramidMipLevelOffsets`、`_ScreenSize`、`_RayMarching*` | **几乎原样搬**（无需 HDRP 依赖） |
| `BilateralUpsample.hlsl/.compute`、`BilateralUpsampleDef.cs` | 双边升采样（9 邻域权重） | 无 | **原样搬** |
| `ShaderVariablesScreenSpaceReflection.cs`(+`.cs.hlsl`)、`ShaderVariablesAmbientOcclusion.cs`(+`.cs.hlsl`) | CBuffer（16 float / 8 Vector4） | `[GenerateHLSL]` | 搬（建议手写 HLSL 并加 sizeof 校验） |
| `RayMarchingFallbackHierarchy.cs` | 回退层级 flags（0x01/0x02/0x03） | 无 | 原样搬 |
| `Runtime/RenderPipeline/Raytracing/Shaders/RayTracingCommon.hlsl`(126)、`RaytracingSampling.hlsl`(81) | SSR/SSGI 的采样库 | 仅 core `Color.hlsl` | **原样搬**（实测不含 RTAS） |
| `ShaderVariables.hlsl`(707) | HDRP 全局 CB | 光照簇/Decal/大气/VFX 矩阵 | **必须替换**为 URP 最小全局头 |

</details>

### 2.2 GTAO 管线（`HDRenderPipeline.AmbientOcclusion.cs`）

```text
PrepareRenderAOParameters(HDCamera, Vector2 historySize, in HDUtils.PackedMipChainInfo depthMipInfo)
  └─ m_AOHistoryReady = !hdCamera.AllocateAmbientOcclusionHistoryBuffer(fullRes ? 1.0f : 0.5f)
  └─ historyRT = GetCurrentFrameRT(AmbientOcclusion) / GetPreviousFrameRT(AmbientOcclusion)

Pass 1  "GTAO Horizon search and integration"  [AddComputePass, EnableAsyncCompute]
        kernel: GTAOMain
        in : _NormalBufferTexture, _CameraDepthTexture(=depth pyramid), depth pyramid output
        out: _AOPackedData (R8_UNorm 全分辨率 / R32_SFloat 半分辨率)
        CB : ConstantBuffer.Push(cb, gtaoCS, HDShaderIDs._ShaderVariablesAmbientOcclusion)
Pass 2  "Spatial Denoise GTAO"                 kernel: denoiseKernelSpatial (GTAOBlurAndUpsample)
        _AOPackedData → _OcclusionTexture
Pass 3  "Temporal Denoise GTAO"                kernel: copyHistory (GTAOCopyHistory) + denoiseKernelTemporal
        in : motion vectors, _AOPackedBlurred, _AOPackedHistory, m_AOHistoryReady
        out: _AOOutputHistory (写) , _OcclusionTexture
Pass 4  "Upsample GTAO"                        kernel: upsampleAOKernel
        in : _AOPackedData, _CameraDepthTexture ; out: _OcclusionTexture
```

**评估**：这是整个文件夹里唯一"URP 已有同槽位特性"的模块。输出改为写 `_ScreenSpaceOcclusionTexture` + `_AmbientOcclusionParam` + 开 `_SCREEN_SPACE_OCCLUSION` 关键字即可被现有 Lit/Deferred 无缝消费（**零 shader 改动**）。历史缓冲用 `CameraHistoryItem` 替换 `AllocateAmbientOcclusionHistoryBuffer`。

### 2.3 SSGI 管线（`HDRenderPipeline.ScreenSpaceGlobalIllumination.cs`）

```text
GetIndirectDiffuseMode(hdCamera):  Off / ScreenSpace / RayTraced / Mixed
   ScreenSpace 前提: colorPyramidHistoryIsValid && !isFirstFrame   ← 与 URP 的 !isFirstFrame/历史重置逻辑对齐

RenderScreenSpaceIndirectDiffuse(...) → RenderSSGI(...)
  Trace + Reproject 是同一个 AddUnsafePass("Trace SSGI")：
     traceKernel   : TraceGlobalIllumination(Half)
        in : normalBuffer, _DepthTexture, depth pyramid offsets(_DepthPyramidMipLevelOffsets), stencil,
             motion vectors, colorPyramid(=**上一帧** ColorBufferMipChain), historyDepth(Depth),
             g_vLightListTile(光簇), _SSGILayerMask(APV), _ShaderVariablesRaytracing
             _RayMarchingSteps / ThicknessScale / ThicknessBias / ReflectSky / LowResPercentageInv / FallbackHierarchy
        out: _IndirectDiffuseHitPointTextureRW (transient)
     projectKernel : ReprojectGlobalIllumination(Half)
        in : 上述 + _IndirectDiffuseHitPointTexture + _ColorPyramidTexture + _HistoryDepthTexture
             _ObjectMotionStencilBit = StencilUsage.ObjectMotionVector
        out: _IndirectDiffuseTextureRW
  DenoiseSSGI  → RequestIndirectDiffuseHistoryTextureHF/LF + HDTemporalFilter.Denoise
  UpscaleSSGI  → BilateralUpsample.compute (BilateralUpSampleColorHalf / BilateralUpSampleColor)
                 ConstantBuffer.Push(_ShaderVariablesBilateralUpsample), _HalfScreenSize
  HistoryValidity: Evaluate/Propagate IndirectDiffuseHistoryValidity{0,1,Combined}
                 ← 用 HDCamera.HistoryEffectFlags { FullResolution | RayTraced } 做位掩码（HF/LF 双频段）
返回 TextureHandle → HDRenderPipeline.RenderGraph.cs:207 → lightingBuffers.ssgiLightingBuffer
```

### 2.4 SSR 管线（`HDRenderPipeline.LightLoop.cs:993–1460`）

```text
RenderSSR(renderGraph, hdCamera, ref prepassOutput, clearCoatMask, rayCountTexture,
          historyValidationTexture, skyReflection, transparent)
  if (EnableRayTracedReflections) → RenderRayTracedReflections(...)      ← 【丢弃】
  usePBRAlgo = !transparent && settings.usedAlgorithm == PBRAccumulation
  useAsync   = frameSettings.SSRRunsAsync() && !transparent              ← 【建议首版关闭】
  一个 AddComputePass("Render SSR")，内部顺序 dispatch：
    1) tracingKernel        ScreenSpaceReflectionsTracing
    2) reprojectionKernel   ScreenSpaceReflectionsReprojection
    3) accumulate*Kernel    ScreenSpaceReflectionsAccumulate{NoWorld|HardThreshold|Smooth}SpeedRejection{
                            Both|SurfaceOnly|TargetOnly}[Debug]   ← 18 个排列组合，可裁剪到 2–3 个
  资源：
    in : stencilBuffer, colorPyramid(透明路径用**上一帧**), motion vectors, clearCoatMask, rayCountTexture
    out: SSR hit point (R16G16_UNorm, 清屏依赖 clearBuffer=!useAsync)
         ssrAccum     = GetCurrentFrameRT(ScreenSpaceReflectionAccumulation)
         ssrAccumPrev = GetPreviousFrameRT(ScreenSpaceReflectionAccumulation)
  关键字：SSR_APPROX（近似算法）、DEPTH_SOURCE_NOT_FROM_MIP_CHAIN（透明）
  CB   ：ConstantBuffer.Push(cb, ssrCS, HDShaderIDs._ShaderVariablesScreenSpaceReflection)
  UpdateSSRConstantBuffer 计算 minSmoothness / roughnessFade / depth pyramid maxMip / color pyramid maxMip 等
```

调用点与消费（`HDRenderPipeline.RenderGraph.cs`）：

- 第 199 行（**阴影前**，为了 async compute 的正确性）或第 206 行（光追开启时**阴影后**）；第 1819 行（透明）。
- 结果 `lightingBuffers.ssrLightingBuffer` → 延迟光照/Lit 消费；第 1392 行 `SetGlobalTexture(HDShaderIDs._SsrLightingTexture, ...)` 给透明 pass。

### 2.5 依赖矩阵（要不要搬的判定）

| 依赖 | 属于 | GTAO | SSGI | SSR | 折射 |
|---|---|---|---|---|---|
| 深度金字塔 + packed mip | HDRP RenderPass/DepthPyramid* | ○ | ● | ● | ● |
| 上一帧 color pyramid | HDCamera History | – | ● | ● | ○ |
| motion vectors | `PrepassOutput` | ●(时域) | ● | ● | – |
| 法线缓冲 | `NormalBuffer.hlsl` | ● | ● | ● | – |
| stencil（object motion / SSR bit） | `StencilUsage` | ○ | ● | ● | – |
| 光簇 / EnvLight / 探针 | `g_vLightListTile`, `EnvLightData` | – | ● | ●(回退) | ●(代理体) |
| APV | core `ProbeVolume.hlsl` | – | ● | – | – |
| `HDTemporalFilter` | Raytracing 目录 | – | ● | – | – |
| 历史有效性框架 | `HDCamera.EffectHistoryValidity` | ○ | ● | ● | – |
| RTAS / 光追路径 | Raytracing 目录 | ✕ | ✕ | ✕ | ✕ |

●=必需 ○=可选 ✕=丢弃

---

## 3. 目标架构：一个 `ScreenSpaceLightingRendererFeature`

### 3.1 设计原则

1. **唯一入口**：`ScreenSpaceLightingRendererFeature`（`[DisallowMultipleRendererFeature]`），所有子功能只是它的"模块"。
2. **模块可自由组合**：每个模块有 `Enable` 开关 + 自动依赖解析（勾 SSGI 会自动启用深度金字塔与双边升采样，勾 SSR 会自动启用 color pyramid），并做**冲突/能力校验**（前向无 DepthNormals、半分辨率无历史等）。
3. **共享层先行**：深度金字塔、颜色金字塔、历史、CBuffer、探针求值提供者、双边升采样都是共享服务，模块不重复实现。
4. **对 URP 的入侵最小化**：只有 SSR 需要一个 shader fork 点；其余通过既有关键字/纹理缝接入。

### 3.2 目录结构

```text
Packages/com.unity.render-pipelines.universal/            (fork) 或 独立包 com.yourstudio.screen-space-lighting
└─ Runtime/ScreenSpaceLighting/
   ├─ ScreenSpaceLightingRendererFeature.cs               ← 唯一 feature（总开关 + 模块注册 + 依赖解析）
   ├─ Settings/
   │  ├─ ScreenSpaceLightingSettings.cs                   ← [Serializable] 总设置（含组合开关）
   │  ├─ GroundTruthAmbientOcclusionSettings.cs           ← 移植 ScreenSpaceAmbientOcclusion.cs(245) 参数
   │  ├─ ScreenSpaceReflectionSettings.cs                 ← 移植 ScreenSpaceReflection.cs(383) 参数
   │  ├─ ScreenSpaceGlobalIlluminationSettings.cs         ← 移植 GlobalIllumination 相关子集
   │  └─ ScreenSpaceShadowRefractionSettings.cs           ← 移植 ScreenSpaceRefraction.cs
   ├─ Core/
   │  ├─ ScreenSpaceLightingResources.cs                  ← IRenderPipelineResources：8 个 compute + 蓝噪声
   │  ├─ ScreenSpaceLightingShaderIDs.cs                  ← 替换 HDShaderIDs（附录 A）
   │  ├─ ScreenSpaceLightingContext.cs                    ← 每相机上下文（含共享服务句柄）
   │  ├─ ScreenSpaceLightingModule.cs                     ← 模块基类 + id/依赖/冲突声明
   │  ├─ ScreenSpaceLightingModuleResolver.cs             ← 拓扑排序 + 依赖自动启用 + 冲突报错
   │  ├─ Shared/
   │  │  ├─ ScreenSpaceLightingSharedResources.cs         ← 深度金字塔/颜色金字塔/双边升采样 provider
   │  │  ├─ DepthPyramidSystem.cs                         ← 移植 DepthPyramid.compute/MipGenerator/PackedMipChainInfo
   │  │  ├─ ColorPyramidSystem.cs                         ← 新写（比 HDRP 简单）
   │  │  ├─ ScreenSpaceLightingHistory.cs                 ← CameraHistoryItem 子类集合
   │  │  ├─ ScreenSpaceLightingHistoryValidity.cs         ← 替换 HDCamera.EffectHistoryValidity
   │  │  ├─ IScreenSpaceLightingProbeProvider.cs          ← 探针/环境求值抽象
   │  │  ├─ ShProbeProvider.cs                            ← URP SH + 探针 atlas 实现（默认）
   │  │  └─ ApvProbeProvider.cs                           ← APV 实现（可选）
   │  ├─ ShaderVariablesScreenSpaceReflection.cs(+.cs.hlsl)
   │  ├─ ShaderVariablesAmbientOcclusion.cs(+.cs.hlsl)
   │  ├─ ShaderVariablesBilateralUpsample.cs
   │  └─ BilateralUpsample.cs / RayMarchingFallbackHierarchy.cs
   ├─ Modules/
   │  ├─ GtaoModule.cs          (4 pass)
   │  ├─ SsgiModule.cs          (trace/reproject + denoise + upsample)
   │  ├─ SsrModule.cs           (opaque + transparent 变体)
   │  └─ RefractionModule.cs    (代理体求交)
   ├─ Passes/  (GtaoTracePass / GtaoSpatialDenoisePass / GtaoTemporalDenoisePass / GtaoUpsamplePass /
   │           SsgiTracePass / SsgiDenoisePass / SsgiUpsamplePass / SsrPass / SsrTransparentPass /
   │           RefractionProxyPass / SslDebugPass)
   └─ Shaders/  (搬家后的 8 个 .compute + 6 个 .hlsl + 新写的 ShaderLibrary/URPScreenSpaceLightingGlobals.hlsl)
```

### 3.3 Feature 与模块骨架

```csharp
[SupportedOnRenderer(typeof(UniversalRendererData))]
[DisallowMultipleRendererFeature("Screen Space Lighting")]
public sealed class ScreenSpaceLightingRendererFeature : ScriptableRendererFeature
{
    [SerializeField] internal ScreenSpaceLightingSettings settings = new();

    readonly List<ScreenSpaceLightingModule> m_AllModules = new();   // GTAO / SSR / SSGI / Refraction
    readonly List<ScreenSpaceLightingModule> m_ActiveModules = new();
    ScreenSpaceLightingResources m_Resources;

    public override void Create()
    {
        m_Resources ??= ScreenSpaceLightingResources.GetOrCreate();       // GraphicsSettings 容器
        if (m_AllModules.Count == 0)
        {
            m_AllModules.Add(new GtaoModule());
            m_AllModules.Add(new SsgiModule());
            m_AllModules.Add(new SsrModule());
            m_AllModules.Add(new RefractionModule());
        }
        foreach (var m in m_AllModules) m.Create(m_Resources);
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        ref var cameraData = ref renderingData.cameraData;              // URP 17 用 UniversalCameraData
        if (!settings.enable || UniversalRenderer.IsOffscreenDepthTexture(ref cameraData)) return;

        var ctx = ScreenSpaceLightingContext.Build(cameraData, renderer, settings, m_Resources);

        // 1) 依赖解析：拓扑排序 + 自动启用共享服务（深度金字塔/颜色金字塔/双边升采样）
        m_ActiveModules.Clear();
        if (!ScreenSpaceLightingModuleResolver.Resolve(m_AllModules, ctx, m_ActiveModules, out var report))
        { Debug.LogWarning(report); return; }

        // 2) 每模块声明输入需求（会驱动 URP 生成 DepthNormals/Motion prepass）
        foreach (var m in m_ActiveModules)
        {
            ctx.requirements |= m.requirements;                          // Depth|Normal|Motion
            if (!m.IsSupported(in ctx, out var reason)) { m.DisableForThisCamera(); continue; }
            m.Setup(in ctx);
        }
        ctx.ConfigureInput(ctx.requirements);                            // 见 3.5 的"需求提升"补丁

        // 3) 共享服务先入队，再逐模块入队（同 event 顺序 = 入队顺序）
        ctx.Shared.EnqueuePasses(renderer);                              // 深度金字塔、颜色金字塔
        foreach (var m in m_ActiveModules) m.EnqueuePasses(renderer);
    }

    protected override void Dispose(bool disposing) { foreach (var m in m_AllModules) m.Dispose(); }
}
```

模块接口：

```csharp
internal abstract class ScreenSpaceLightingModule
{
    public abstract ScreenSpaceLightingModuleId                id            { get; }
    public virtual  ScreenSpaceLightingModuleId[]              requires      => Array.Empty<...>();
    public virtual  ScreenSpaceLightingSharedServiceFlags      sharedServices=> ScreenSpaceLightingSharedServiceFlags.None;
    public virtual  ScriptableRenderPassInput                  requirements  => ScriptableRenderPassInput.Depth;
    public abstract  bool IsSupported(in ScreenSpaceLightingContext ctx, out string reason);
    public abstract  void Setup(in ScreenSpaceLightingContext ctx);
    public abstract  void EnqueuePasses(ScriptableRenderer renderer);
    public virtual   void Dispose() {}
}
```

### 3.4 组合与自动依赖（用户视角）

| 用户勾选 | 自动启用的共享服务 | 产生的 pass（按帧内顺序） | 硬性前提 |
|---|---|---|---|
| GTAO | —（可选 时域 → Motion + AO history） | GTAO trace → spatial → temporal → upsample | 深度；法线（或重建） |
| SSGI | **DepthPyramid**、**ColorPyramid(上一帧)**、BilateralUpsample（半分辨率时）、GI history | SSGI trace+reproject → denoise(HF/LF) → upsample | 深度、法线、motion、APV/探针、**color pyramid 已有效** |
| SSR | **DepthPyramid**、**ColorPyramid(上一帧)**、SSR accum history | SSR trace(+accumulate) → 透明 SSR | 深度、法线、stencil、motion |
| 折射 | SSR buffer + `cameraOpaqueTexture` | 代理体求交 | 透明队列、厚度 |
| 调试 | — | SSL debug pass（推 FullScreen Debug 纹理） | — |

冲突校验示例（在 `IsSupported` 内实现，报错文案参考 URP SSAO 的 Tile-Only Mode 提示风格）：`SSR + 前向 + 无 DepthNormals 请求` → 自动提升需求或报错；`SSGI 半分辨率 + 无 motion` → 强制全分辨率。

### 3.5 帧内执行序（精确锚点）

| 我们的 pass | `RenderPassEvent` | 对齐的 URP 注入点 | 说明 |
|---|---|---|---|
| `SharedDepthPyramid` | `AfterRenderingPrePasses` (200) | 1091 | 需要深度；输出 mip 链 + offsets buffer |
| `GtaoTrace/Spatial/Temporal/Upsample` | `AfterRenderingPrePasses + 1..4` (201–204) | 1091 与 1109 之间 | 与 URP 内置 SSAO 的默认锚点一致；输出 `_ScreenSpaceOcclusionTexture` |
| `SsgiTrace/Denoise/Upsample` | `AfterRenderingPrePasses + 1`（前向）/ `BeforeRenderingGbuffer` 之前（延迟） | 1091 / 1109 | 输出 `_ScreenSpaceIrradiance`；**延迟必须早于 GBuffer** |
| `SsrOpaque`(trace+accum) | `BeforeRenderingDeferredLights`(230)（延迟）/ `AfterRenderingPrePasses+2`（前向） | 1150 / 1091 | 输出 `_SsrLightingTexture`；**延迟必须早于 GBuffer**（见 4.2） |
| `SsrTransparent` | `BeforeRenderingTransparents`(450) | 1242 | 用上一帧 color pyramid + 本帧深度 |
| `RefractionProxy` | `BeforeRenderingTransparents - 1` (449) | 1242 之前 | — |
| `SharedColorPyramidBuild` | `AfterRenderingTransparents`(500) | 1269 | 生成"本帧 lit color 的 mip 链"→ 供下一帧 SSR/SSGI 使用 |
| `SslDebug` | `AfterRenderingPostProcessing`(600) | 1404 | — |

> ⚠️ **延迟路径的时序陷阱（本报告最关键的一条）**：URP 延迟的间接光在 `LitGBufferPass.hlsl:229-243` 就被算进 GBuffer（`MixRealtimeAndBakedGI` + `GlobalIllumination(...)`），`StencilDeferred.hlsl` 只负责直接光 + AO。因此延迟模式下 SSGI/SSR 的产物必须在此 pass 之前可用 → 需要 `RenderPassEvent.BeforeRenderingGbuffer`(210) 之前完成，而那时**本帧法线还不存在**（法线来自 GBuffer）。两条出路见 4.2。

### 3.6 共享子系统要点

<details>
<summary><b>深度金字塔 / 颜色金字塔 / 历史 / CBuffer / 探针提供者</b></summary>

**深度金字塔**：直接搬 `DepthPyramid.compute` + `MipGenerator.cs` + `DepthPyramidConstants.cs`(+`.cs.hlsl`) + `HDUtils.PackedMipChainInfo`。`RayMarching.hlsl` 只要求两个全局：`_DepthTexture`（mip 链）与 `_DepthPyramidMipLevelOffsets`（StructuredBuffer）。注意 `RayMarching.hlsl` 里厚度关系是 $b = d \cdot k_s + k_b$（`bounds.w = bounds.z * _RayMarchingThicknessScale + _RayMarchingThicknessBias`），URP 侧只需把这两个常量按同公式从 `depthBufferThickness` 推出。

**颜色金字塔**：HDRP 用 `HDCameraFrameHistoryType.ColorBufferMipChain`（MipGenerator 生成）。URP 侧用 `RawColorHistory`/自建 `CameraHistoryItem` + core `MipGenerator`，在 `AfterRenderingTransparents` 生成，**下一帧**读取（SSR 透明路径本来就用上一帧，语义一致）。

**历史**：三个 `CameraHistoryItem`：
```csharp
internal sealed class SslAmbientOcclusionHistory : CameraHistoryItem  // 2 张（current/prev），R8/R32F，全/半分辨率
internal sealed class SslSsrAccumulationHistory : CameraHistoryItem   // 2 张，R16G16B16A16（累积）
internal sealed class SslIndirectDiffuseHistory : CameraHistoryItem   // HF/LF 共 2 组
```
替换点：`hdCamera.AllocateAmbientOcclusionHistoryBuffer(scale)` → 分辨率变化时 `ReleaseHistoryFrameRT` + 重新 `AllocHistoryFrameRT`（HDRP 源码 1768–1791 就是同一套逻辑，直接照搬条件判断）。

**历史有效性**：`HDCamera.EffectHistoryValidity(slot, flagMask)` / `PropagateEffectHistoryValidity` / `HistoryEffectFlags{FullResolution, RayTraced}` → 在 URP 侧实现为
`ScreenSpaceLightingHistoryValidity`：一个 `R8_SInt` validation 纹理（位域）+ C# 端 `uint` 状态，按 HDRP 的 `Evaluate*Validity0/1/Combined` 进行 1 帧延迟传播。

**探针/环境提供者**（替换 HDRP 光簇）：
```csharp
internal interface IScreenSpaceLightingProbeProvider
{
    void Initialize(in ScreenSpaceLightingContext ctx);
    void BindGlobals(ComputeCommandBuffer cmd, int kernel);   // 绑定 _ReflectionProbeAtlas / SH / APV 参数
}
```
- `ShProbeProvider`（默认）：`SampleSH` + URP 探针 atlas（`ReflectionProbeManager`）+ `_GlossyEnvironmentColor`，覆盖 `TraceReflectionProbes()`（SSGI 的 `GetCountAndStart(LIGHTCATEGORY_ENV)`/`FetchEnvLight`/`_ReflectionAtlas`）与 SSR 回退层级。
- `ApvProbeProvider`：core `ProbeVolume.hlsl` 的 `EvaluateAdaptiveProbeVolume`（SSGI 里 `_EnableProbeVolumes`/`_SSGILayerMask` 部分可**原样保留**）。
- 阶段 5（可选）：用 URP Forward+ 的光簇列表（`Clustering.hlsl` / `g_vLightListTile` 等价物）做命中点真实光照求值 —— 这是"保真版"的最后一道关。

</details>

---

## 4. URP fork 清单

### 4.1 最小 fork（推荐，1 个文件）

| 文件 | 改动 | 目的 |
|---|---|---|
| `ShaderLibrary/GlobalIllumination.hlsl` | ① 新增 `#if defined(_SCREEN_SPACE_REFLECTIONS)` 声明 `TEXTURE2D_X(_SsrLightingTexture)` + `half4 SampleScreenSpaceReflections(float2 uv)`（rgba = 颜色 + 命中权重/距离）；② 在 `GlobalIllumination()` 内 `indirectSpecular = lerp(indirectSpecular, ssr.rgb, ssr.a * roughnessFade)`（clearcoat 可选同理） | SSR 注入（前向 + 延迟共用） |
| `ShaderLibrary/GlobalIllumination.hlsl` | ③ 若基线 < 6000.5，需要自己补 `_SCREEN_SPACE_IRRADIANCE` 块（`TEXTURE2D_X` + `SampleScreenSpaceGI` + `SAMPLE_GI` 宏分支） | SSGI 注入（6000.5+ 已自带） |
| 新文件 `ShaderLibrary/ScreenSpaceLighting.hlsl` | 集中放关键字、全局纹理、采样/权重函数、debug | 便于管理 |

配套（无需改 shader 的）：
- GTAO：输出 `_ScreenSpaceOcclusionTexture`，`CoreUtils.SetKeyword(cmd, ShaderGlobalKeywords.ScreenSpaceOcclusion, true)`，设置 `_AmbientOcclusionParam`（与 `ScreenSpaceAmbientOcclusionPass.s_AmbientOcclusionParamID` 一致）→ **URP 现有消费端零改动**。
- SSGI：输出 `_ScreenSpaceIrradiance`（或 `UniversalResourceData.irradianceTexture`），开 `_SCREEN_SPACE_IRRADIANCE`。

### 4.2 延迟路径的两种 fork 路线（二选一）

| 路线 | 做法 | 代价 | 建议 |
|---|---|---|---|
| **A. Prepass 前置（推荐首版）** | 让 feature 用 `ScriptableRenderPassInput.Depth｜Normal` 强制 URP 生成 DepthNormals prepass，SSGI/SSR 在 `AfterRenderingPrePasses`(200) 完成，随后 GBuffer pass 读取注入结果 | 多一次深度+法线 prepass（Forward+ 里本来就有）；延迟下法线来自 prepass 而非 GBuffer，精度差异需校验 | ✅ |
| **B. HDRP 化（保真）** | 把间接光从 `LitGBufferPass.hlsl` 移出，改为 GBuffer 后独立 "Deferred Indirect/GI" pass（像 HDRP 那样由 lighting buffer 提供 `ssrLightingBuffer/ssgiLightingBuffer`），并让 `StencilDeferred`/`ClusterDeferred` 消费 | 需要改 `DeferredLights.cs`、`DeferredPass.cs`、`LitGBufferPass.hlsl`、`StencilDeferred.hlsl`，并处理 GBuffer 布局/带宽 | 阶段 6 再做 |

### 4.3 可选插桩点补丁（让锚点更精确）

`ScriptableRenderer` 已提供 `CalculateSplitEventRange`（`UniversalRendererRenderGraph.cs:845`）与 `RecordCustomRenderGraphPassesInEventRange`（847/851），用于"深度读取冲突切分"。如果我们的 pass 与 URP 的深度用法冲突（例如想在 `BeforeRenderingOpaques` 内插入却需深度只读），可在此处加一个 `SSLighting` 事件区间；一般不需要。

---

## 5. 文件级迁移映射表

| 源（HDRP） | 目标（新模块） | 动作 |
|---|---|---|
| `GTAO.compute`、`GTAOBlurAndUpsample.compute`、`GTAOSpatialDenoise.compute`、`GTAOTemporalDenoise.compute`、`GTAOCopyHistory.compute` | `Shaders/GTAO/*.compute` | **复制**，替换 include 路径与 `_ShaderVariablesAmbientOcclusion` ID |
| `GTAOCommon.hlsl`、`BilateralUpsample.hlsl`、`RayMarching.hlsl` | `Shaders/*.hlsl` | **复制**，`ShaderVariables.hlsl` 改指向 URP 最小全局头 |
| `BilateralUpsample.compute` | `Shaders/BilateralUpsample.compute` | 复制 |
| `RayTracingCommon.hlsl`、`RaytracingSampling.hlsl` | `Shaders/Sampling/*.hlsl` | **原样复制** |
| `ScreenSpaceGlobalIllumination.compute` | `Shaders/ScreenSpaceGlobalIllumination.compute` | 复制 + **改写命中点求值**（`TraceReflectionProbes` → ProbeProvider；保留 APV 分支） |
| `ScreenSpaceReflections.compute` | `Shaders/ScreenSpaceReflections.compute` | 复制 + 裁剪 accumulate kernel 变体（18 → 2–3） |
| `ScreenSpaceTracing.hlsl` | `Shaders/ScreenSpaceTracing.hlsl` | 复制 + `EnvLightData` → URP 探针结构 |
| `HDRenderPipeline.AmbientOcclusion.cs` | `Modules/GtaoModule.cs` + `Passes/Gtao*.cs` | **重写**（4 个 `AddComputePass`） |
| `HDRenderPipeline.ScreenSpaceGlobalIllumination.cs` | `Modules/SsgiModule.cs` + `Passes/Ssgi*.cs` | **重写**（去 RT 分支、`HDTemporalFilter` 替换） |
| `HDRenderPipeline.LightLoop.cs:993–1460` | `Modules/SsrModule.cs` + `Passes/SsrPass.cs` | **重写** |
| `ScreenSpaceReflection.cs` / `ScreenSpaceAmbientOcclusion.cs` | `Settings/*.cs` | 复制 + 删 RT 字段 + 加模块开关 |
| `ScreenSpaceRefraction.cs` | `Settings/ScreenSpaceShadowRefractionSettings.cs` | 复制（小） |
| `ShaderVariablesScreenSpaceReflection/AmbientOcclusion.cs(.cs.hlsl)` | `Core/*` | 复制；建议手写 HLSL 并加 `sizeof` 断言 |
| `BilateralUpsampleDef.cs`、`RayMarchingFallbackHierarchy.cs`、`ScreenSpaceGlobalIllumination.cs`(enum) | `Core/*` | 原样复制 |
| `DepthPyramid.compute`、`MipGenerator.cs`、`DepthPyramidConstants.cs`、`HDUtils.PackedMipChainInfo` | `Core/Shared/*` | 复制（`PackedMipChainInfo` 从 `HDUtils` 抽出来） |
| `HDTemporalFilter.cs`(+`.cs.hlsl`) | `Core/Shared/TemporalFilter.cs` | 复制或自写（SSGI 必需） |
| `HDDiffuseDenoiser.cs`、`ReBlurDenoiser.cs`、`RayTracingFallbackHierarchy.cs`(Raytracing 版)、`HDRenderPipeline.Raytracing*.cs` | — | **丢弃** |
| `ShaderVariables.hlsl`(707) | `Shaders/URPScreenSpaceLightingGlobals.hlsl`（新写） | **重写**（只保留 `_ScreenSize`/`_TimeParameters`/jitter/XY 坐标约定） |
| `HDRenderPipelineRuntimeResources` 的 CS 条目 | `Core/ScreenSpaceLightingResources.cs` | 重写为 `IRenderPipelineResources` + `[ResourcePath]` |

---

## 6. 分阶段计划与工时

| 阶段 | 内容 | 交付物 | 单人估时 |
|---|---|---|---|
| **M0** | Feature 骨架 + 模块/依赖解析 + 资源容器 + URP 最小 fork（`GlobalIllumination.hlsl` 加缝）+ 调试可视化 | 能跑通空 pass，Inspector 里可勾选模块组合 | 2 周 |
| **M1** | 共享层：深度金字塔、颜色金字塔、历史三件套、CBuffer、ShaderIDs、探测提供者（SH/探针 atlas） | 共享服务可被任意模块调用；RayMarching 可视化命中点 | 3–4 周 |
| **M2** | **GTAO 模块**（4 pass）+ 与 URP SSAO 消费端对齐（含时空降噪、半分辨率） | 可用 GTAO，替代 URP SSAO；**零 shader 改动** | 3 周 |
| **M3** | **SSGI 模块**：trace/reproject + HF/LF 时域降噪 + 双边升采样 + 历史有效性；探针求值降级版 | URP 下首个可用 SSGI（SH+APV+探针） | 6–8 周 |
| **M4** | **SSR 模块**：trace + 裁剪后的 accumulate + accum history + 透明变体；fork 消费端 | 可用 SSR（前向+延迟） | 8–12 周 |
| **M5** | **折射/透明**：`ScreenSpaceTracing.hlsl` 代理体 + 厚度缓冲 | 折射可用 | 3–4 周 |
| **M6** | 延迟路径路线 B（HDRP 化间接光 pass）、匹配 HDRP 质量的光簇命中点求值 | 保真版 | 4–6 周 |
| **M7** | XR/动态分辨率/异步计算/Strip 与打包校验/性能剖析/文档 | 发布级 | 3–4 周 |

**累计：约 32–46 周（8–11 个月）单人**；两人并行（M2/M3 与 M4 分离，共享层由一人先做）≈ **4–5 个月**。

---

## 7. 风险与验证

| 风险 | 等级 | 缓解 |
|---|---|---|
| 延迟路径时序（§3.5 陷阱）导致 SSR/SSGI 无法被消费 | **高** | 路线 A 先行；用 `BeforeRenderingGbuffer` 锚点 + DepthNormals prepass 验证 |
| 法线来源不一致（prepass 八面体量化 vs GBuffer） | 中 | `NormalBuffer.hlsl` 的解码函数必须换成 URP 的 `DepthNormalsPass` 编码；写 A/B 对比 |
| 探针求值降级导致 SSGI 亮度/色偏 | 中 | 用 Rendering Debugger 分通道对照；先对齐 SH 项再加 APV |
| 历史/重投影错误（鬼影、闪烁） | 中 | 直接移植 HDRP 的 history-validity 位掩码方案；逐 pass 可视化 |
| `.cs.hlsl` 与 C# 结构体不一致（随机数值错乱） | 中 | 手写 HLSL + `sizeof`/`Marshal.SizeOf` 单元测试 |
| 资源被 strip（玩家包黑屏） | 中 | 照抄 SSAO 的 `IRenderPipelineResources` + stripping 配置；加 CI 打包校验 |
| 异步计算引发同步错误 | 低 | 首版全部同步；参考 HDRP 注释（SSR 需在阴影前）后再开 |
| XR 单通道 / 动态分辨率 | 中 | 保留 HDRP 的 `viewCount` 派发与 `_ScreenSize`/`DynamicScalingClamping` 用法，逐项回归 |

**验收基准**（建议建立）：与 HDRP 同场景同参数对照截图 + 关键指标（AO 半分辨率下 GPU 时间、SSR 命中率、SSGI 相对 HDRP 的平均色差），逐模块过。

---

## 附录

<details>
<summary><b>附录 A：ShaderID / CBuffer 映射（已从源码提取）</b></summary>

`ShaderVariablesScreenSpaceReflection`（16×4B，`[GenerateHLSL(needAccessors=false, generateCBuffer=true)]`）：
`_SsrThicknessScale, _SsrThicknessBias, _SsrStencilBit, _SsrIterLimit, _SsrRoughnessFadeEnd, _SsrRoughnessFadeRcpLength, _SsrRoughnessFadeEndTimesRcpLength, _SsrEdgeFadeRcpLength, _SsrDepthPyramidMaxMip, _SsrColorPyramidMaxMip, _SsrReflectsSky, _SsrAccumulationAmount, _SsrPBRSpeedRejection, _SsrPBRBias, _SsrPRBSpeedRejectionScalerFactor, _SsrPBRPad0`

`ShaderVariablesAmbientOcclusion`（8×Vector4）：
`_AOBufferSize, _AOParams0.._AOParams4, _FirstTwoDepthMipOffsets, _AODepthToViewParams`

需替换的 `HDShaderIDs`（按模块归类，共约 40 个）：
- 共通：`_DepthTexture, _DepthPyramidMipLevelOffsets, _NormalBufferTexture, _CameraDepthTexture, _CameraMotionVectorsTexture, _StencilTexture, _ColorPyramidTexture, _ColorPyramidUvScaleAndLimitPrevFrame, _ScreenSize`
- GTAO：`_AOPackedData, _AOPackedBlurred, _AOPackedHistory, _AOOutputHistory, _OcclusionTexture`
- SSGI：`_IndirectDiffuseHitPointTexture(RW), _IndirectDiffuseTextureRW, _IndirectDiffuseFrameIndex, _SSGILayerMask, _HistoryDepthTexture, _RayMarchingSteps, _RayMarchingThicknessScale, _RayMarchingThicknessBias, _RayMarchingReflectSky, _RayMarchingLowResPercentage(Inv), _RayMarchingFallbackHierarchy, _ObjectMotionStencilBit, g_vLightListTile, _EnableProbeVolumes`
- 升采样：`_ShaderVariablesBilateralUpsample, _HalfScreenSize, _LowResolutionTexture, _OutputUpscaledTexture`
- SSR：`_SsrLightingTexture`（新）, `_ShaderVariablesScreenSpaceReflection`

关键字：`SSR_APPROX`、`DEPTH_SOURCE_NOT_FROM_MIP_CHAIN`（沿用）+ 新增 `_SCREEN_SPACE_REFLECTIONS`（`_SCREEN_SPACE_IRRADIANCE`/`_SCREEN_SPACE_OCCLUSION` 已有）。
</details>

<details>
<summary><b>附录 B：Kernel 清单与裁剪建议</b></summary>

| Compute | Kernels（实测） | 建议 |
|---|---|---|
| `GTAO.compute` | `GTAOMain` | 保留 |
| `GTAOBlurAndUpsample.compute` | `BlurUpsample, BilateralUpsampling, BoxUpsampling, Blur, Blur_FullRes` | 保留 `BilateralUpsampling`/`Blur`（其余可选） |
| `GTAOSpatialDenoise.compute` | (spatial) | 保留 |
| `GTAOTemporalDenoise.compute` | (temporal) | 保留 |
| `GTAOCopyHistory.compute` | (copy) | 保留 |
| `ScreenSpaceGlobalIllumination.compute` | `TraceGlobalIllumination, TraceGlobalIlluminationHalf, ReprojectGlobalIllumination, ReprojectGlobalIlluminationHalf` | 全保留（Half/Full 各一套） |
| `ScreenSpaceReflections.compute` | 20 个：`Tracing, Reprojection, Accumulate{NoWorld|HardThreshold|Smooth}SpeedRejection{Both|SurfaceOnly|TargetOnly}{,Debug}` | **裁剪到 3 个**（`Tracing` + `Reprojection` + 1 组 accumulate），Debug 变体改为运行时 debug 输出 |
| `BilateralUpsample.compute` | `BlurUpsample, BilateralUpsampling, BoxUpsampling, Blur, Blur_FullRes` + `BilateralUpSampleColor(Half)` | 保留 SSGI 所需 2 个 |
</details>

<details>
<summary><b>附录 C：HDRP → URP API 对照表</b></summary>

| HDRP | URP（6000.5+） |
|---|---|
| `HDRenderPipeline` partial + `m_RenderGraph` | `ScriptableRendererFeature` + `ScriptableRenderPass.RecordRenderGraph` |
| `lightingBuffers.ssrLightingBuffer/ssgiLightingBuffer` | `_SsrLightingTexture` / `_ScreenSpaceIrradiance`（全局纹理 + 关键字） |
| `hdCamera.GetCurrentFrameRT/GetPreviousFrameRT(type)` | `CameraHistoryItem.AllocHistoryFrameRT` + `historyManager.GetHistoryForRead/Write<T>()` |
| `hdCamera.AllocateAmbientOcclusionHistoryBuffer(scale)` | 自定义 `SslAmbientOcclusionHistory`（同条件判断） |
| `HDCamera.EffectHistoryValidity/PropagateEffectHistoryValidity` | 自写 `ScreenSpaceLightingHistoryValidity`（位掩码 + validation 纹理） |
| `HDShaderIDs.*` | `ScreenSpaceLightingShaderIDs.*` |
| `runtimeShaders.xxxCS`（`HDRenderPipelineRuntimeResources`） | `IRenderPipelineResources` + `[ResourcePath]` + `GraphicsSettings.TryGetRenderPipelineSettings<T>` |
| `ShaderVariables.hlsl` 全局 CB | 新写 `URPScreenSpaceLightingGlobals.hlsl`（最小集） |
| `hdCamera.volumeStack.GetComponent<T>()` | `VolumeManager.instance.stack.GetComponent<T>()`（URP 同样支持 Volume） |
| `FullScreenDebugMode` / `PushFullScreenDebugTexture` | URP Rendering Debugger / `SurfaceCacheGIRendererFeature` 的 Debug 思路 |
| `prepassOutput.depthPyramidTexture/normalBuffer/stencilBuffer/motionVectorsBuffer` | `UniversalResourceData.cameraDepth/cameraNormalsTexture/...` + 自建深度金字塔 |
| `build.EnableAsyncCompute(...)` | `builder.EnableAsyncCompute(...)`（RenderGraph 同名 API，首版关闭） |
</details>

---

### 建议的下一步

如果确认按此架构推进，我可以继续产出：
1. **M0 的可编译骨架代码**（`ScreenSpaceLightingRendererFeature.cs` + `ScreenSpaceLightingModule.cs` + `Resolver` + `Resources` + 最小 fork 的 `GlobalIllumination.hlsl` 补丁 diff）；
2. **每个模块的 pass 级详细设计**（含 RenderGraph pass data 结构、纹理描述符、资源使用声明）；
3. **GTAO 模块的完整移植清单**（逐文件 diff 说明），因为它是最合适的"第一个端到端验证目标"。

告诉我从哪一项开始，以及目标 URP 基线（`6000.5` / `6000.6` / 自建分支）。

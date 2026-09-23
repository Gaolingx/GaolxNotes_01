# UE5渲染管线多层

## 🧑‍💻 User

ue5好像除了rhi封了一层，还有rdg又封了一层才能给材质系统去用，是这样吗

---
## 🤖 Assistant

大方向是对的，但**实际链路比"两层"要多**。RHI 是最底层抽象，RDG 是资源依赖/调度的中间层，而材质系统真正直接面对的其实是**第三层：Mesh Draw Pipeline（网格绘制管线）**。材质系统并不直接调 RDG。

## 完整链路

```text
编译期：
U材质图 (UMaterial / UMaterialExpression)
   └─> FHLSLMaterialTranslator  ──> 生成 HLSL
        └─> ShaderCompiler (ShaderCompileWorker) ──> FShader / FMaterialShaderMap

运行期：
FScene + FPrimitiveSceneProxy
   └─> FMaterialRenderProxy ──> FMaterial ──> FShader
        └─> FMeshBatch
             └─> FMeshPassProcessor::AddMeshBatch()   ← 材质系统在这里"落地"
                  └─> FMeshDrawCommand（缓存/复用，即 Mesh Draw Pipeline）
                       └─> FParallelMeshDrawCommandPass::DispatchDraw()
                            └─> RDG Pass 的 lambda 里 SubmitDraw
                                 └─> FRHICommandList / RHI 后端 (D3D12 / Vulkan / Metal)
```

## 各层职责对照

| 层 | 代表类型 | 职责 |
|---|---|---|
| RHI | `FRHICommandList`、`FRHIDevice`、PSO | 抽象具体图形 API，资源创建、状态设置、绘制提交 |
| RDG | `FRDGBuilder`、`FRDGTexture`、`AddPass` | Pass 依赖分析、屏障/Transition 自动插入、瞬时资源内存别名复用、异步计算调度、并行命令录制 |
| Mesh Draw Pipeline | `FMeshDrawCommand`、`FMeshPassProcessor`、`FParallelMeshDrawCommandPass` | 把材质 + Mesh + Pass 组合成可缓存的绘制命令，做状态排序与实例剔除 |
| 材质系统 | `UMaterial`、`FMaterial`、`FMaterialRenderProxy` | 材质图 → HLSL → Shader，提供绑定参数与 Shader 变体 |

## 几个关键点

**1. RDG 不是为材质系统封的。** 它的核心目的是解决资源依赖和调度问题：自动插屏障、瞬时资源别名、异步计算 overlap、多线程并行录制命令。材质系统只是它的众多使用者之一（后处理、Lumen、Nanite、粒子等同样走 RDG）。

**2. 材质系统真正"接口"的是 Mesh Draw Pipeline。** 材质相关的 `FMeshPassProcessor` 子类（如 `FBasePassMeshProcessor`、`FDepthPassMeshProcessor`）把 `FMeshBatch` 转成 `FMeshDrawCommand`，然后由 RDG pass 统一提交。材质作者/引擎开发者改的是这一层，而不是直接写 RDG。

**3. 并非所有东西都必须走 RDG。** UE5 里仍存在直接调 `FRHICommandList` 的路径——部分 Copy、某些第三方插件、少量平台特化代码、以及一些遗留的 Immediate Mode 代码。RDG 覆盖了绝大部分渲染流程，但不是 100%。

**4. UE5.3+ 的趋势。** 引擎在持续推进"RDG 全覆盖"，同时引入 `FComputeShaderUtils`、`AddDrawScreenPass`、`FSimpleMeshPass` 等辅助封装来降低写 RDG 的样板代码量。

<details>
<summary>一句话总结</summary>

RHI 抽象硬件，RDG 抽象"资源依赖与调度"，Mesh Draw Pipeline 抽象"材质 + 网格怎么画"。材质系统坐在第三层之上，通过 RDG 间接驱动 RHI，但并不是直接跟 RDG 打交道。

</details>

---
## 🧑‍💻 User

大致是这个流程吗，如果是自定义卡通渲染魔改管线需要增加shading model、修改gbuffer、deferred rendering、插入custom pass、custom postprocess怎么办？

---
## 🤖 Assistant

流程理解基本正确。下面按你的改造需求拆开讲——**每项都标出「改哪里」和「能不能不改引擎」**。

## 一、先确认流程 + 关键扩展点

你的链路图是对的，补一个重要的东西：**`FSceneViewExtensionBase`**。

这是 Epic 官方留的「不改引擎源码就能插入 Pass」的扩展点，位置在 `Engine/Source/Runtime/Engine/Public/SceneViewExtension.h`。卡通渲染魔改强烈建议优先用它，能大幅降低升级 UE 版本的 merge 成本。

常用钩子：

| 钩子 | 时机 | 能拿到什么 |
|---|---|---|
| `PreRenderViewFamily_RenderThread` | 整个 ViewFamily 渲染前 | 可建全局资源 |
| `PostRenderBasePassDeferred_RenderThread` | BasePass 之后、光照之前 | **GBuffer**，插自定义 Pass 的黄金位置 |
| `PrePostProcessPass_RenderThread` | 后处理前 | SceneColor |
| `SubscribeToPostProcessingPass` | 指定后处理阶段前后 | 精细插入 |

```cpp
class FToonViewExtension : public FSceneViewExtensionBase
{
public:
    FToonViewExtension(const FAutoRegister& AutoRegister)
        : FSceneViewExtensionBase(AutoRegister) {}

    virtual void PostRenderBasePassDeferred_RenderThread(
        FRDGBuilder& GraphBuilder,
        FSceneView& InView,
        const FRenderTargetBindingSlots& RenderTargets,
        TRDGUniformBufferRef<FSceneUniformParameters> SceneUniformParameters) override
    {
        AddToonPass(GraphBuilder, InView, SceneUniformParameters);
    }
};

// 模块 StartupModule 里注册
FSceneViewExtensions::NewExtension<FToonViewExtension>();
```

---

## 二、增加 Shading Model

这是卡通渲染最核心的改造。**必须改引擎源码**，绕不开（除非你放弃引擎光照体系）。

<details>
<summary><b>展开：新增 MSM_Toon 的完整改动清单</b></summary>

**1) 枚举定义**
- `Engine/Source/Runtime/Engine/Public/MaterialShared.h` → `EMaterialShadingModel` 加 `MSM_Toon`（放在 `MSM_NUM` 之前）
- `Engine/Source/Runtime/Engine/Private/Materials/MaterialShared.cpp` → `GetShadingModelString()` / `GetShadingModelFromString()`

**2) 材质侧**
- `Engine/Source/Runtime/Engine/Private/Materials/HLSLMaterialTranslator.cpp` → 处理该 shading model 的输入引脚，生成 `SHADINGMODELID_TOON` 宏
- `Engine/Source/Runtime/Engine/Private/Materials/MaterialTemplate.ush` → `FPixelMaterialInputs` 相关
- 编辑器下拉框选项列表（`MaterialEditor` 模块的 shading model customization）

**3) Shader 核心（最关键）**
- `Engine/Source/Runtime/Renderer/Private/ShadingModels.ush`
 - 定义 `SHADINGMODELID_TOON`
 - 写 `FToonShading` 结构体
 - 写 `ToonBxDF(...)` 函数
 - 在 `IntegrateBxDF()` 里加分支
 - 在 `SetGBufferForShadingModel()` 里打包 GBuffer
- `Engine/Source/Runtime/Renderer/Private/DeferredShadingCommon.ush`
 - `FGBufferData` 结构
 - `EncodeGBuffer()` / `DecodeGBuffer()`
- `Engine/Source/Runtime/Renderer/Private/DeferredLightingCommon.ush`
 - `GetDynamicLighting()` 里调用你的 BxDF

**4) Mobile 路径（如果要做移动端）**
- `MobileBasePassPixelShader.usf`、`MobileDeferredShading` 相关文件要单独处理，移动端 shading path 是另一套。

</details>

**卡通渲染的具体做法**：BxDF 里用 `dot(N, L)` 查一张 Ramp 贴图做分级，而不是标准 Lambert。Ramp 参数可以通过 `CustomData` 通道传进来。

---

## 三、修改 GBuffer

UE5.1+ 之后 GBuffer 由 **`FSceneTextures`** 统一管理，比老的 `SceneRenderTargets` 干净很多。

<details>
<summary><b>展开：GBuffer 改动的两种方案</b></summary>

**方案 A：复用 CustomData 通道（推荐起步）**

引擎已经预留了 `CustomData` 的多个通道，卡通渲染的额外参数（Ramp 索引、阴影强度、边缘光宽度等）直接塞进去。

- 打包：`ShadingModels.ush` 的 `SetGBufferForShadingModel()`
- 解包：`DeferredShadingCommon.ush` 的 `DecodeGBuffer()`
- 宏控制：`BasePassCommon.ush` 的 `WRITES_CUSTOMDATA_TO_GBUFFER`

优点：几乎不用动渲染目标分配，风险低。
缺点：通道数量有限。

**方案 B：新增独立 GBuffer Target（激进）**

- `Engine/Source/Runtime/Engine/Private/GBufferInfo.cpp` → `FGBufferInfo::Initialize()` 加新纹理定义
- `Engine/Source/Runtime/Engine/Public/SceneTexturesConfig.h` → `EGBufferTexture` 加枚举
- `Engine/Source/Runtime/Engine/Private/SceneTextures.cpp` → `FSceneTextures::InitializeViewFamily()` 分配 RDG 纹理
- `BasePassCommon.ush` → `WRITES_GBUFFER_*` 宏
- `BasePassPixelShader.usf` → 写入

优点：通道充足。
缺点：显存带宽涨、改动面大、移动端尤其敏感。

</details>

---

## 四、Deferred Rendering 修改

核心文件：`Engine/Source/Runtime/Renderer/Private/DeferredShadingRenderer.cpp`

主要函数：

| 函数 | 作用 |
|---|---|
| `FDeferredShadingSceneRenderer::Render` | 整个延迟渲染主流程 |
| `RenderBasePass` | BasePass，写 GBuffer |
| `RenderLights` / `AddDeferredLightingPass` | 延迟光照 |
| `RenderDeferredReflectionsAndSkyLighting` | 反射/SkyLight |

**改光照公式**：不动 `DeferredShadingRenderer.cpp`，改 `ShadingModels.ush` 里的 BxDF 即可——因为延迟光照 pass 会调用你注册的 BxDF 分支。

**插入自定义 Pass**：优先用 SceneViewExtension，避免直接改 `DeferredShadingRenderer.cpp`。只有在需要**在特定引擎 pass 内部插入**（比如 BasePass 和光照之间做特殊处理）时才改源码。

---

## 五、插入 Custom Pass

三种层次：

<details>
<summary><b>展开：三种插入方式对比</b></summary>

**1) SceneViewExtension（首选）**

在 `PostRenderBasePassDeferred_RenderThread` 里 `AddPass`：

```cpp
void AddToonPass(FRDGBuilder& GraphBuilder, FSceneView& View,
                 TRDGUniformBufferRef<FSceneUniformParameters> SceneUB)
{
    FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(View.GetFeatureLevel());
    TShaderMapRef<FToonPS> PixelShader(ShaderMap);

    FRDGTextureRef SceneColor = /* 从 SceneTextures 取 */;

    FToonPS::FParameters* Params = GraphBuilder.AllocParameters<FToonPS::FParameters>();
    Params->SceneColor = /* ... */;

    AddDrawScreenPass(GraphBuilder, RDG_EVENT_NAME("ToonPass"), View,
        FScreenPassTextureViewport(SceneColor), SceneColor, PixelShader, Params);
}
```

Shader 放自己模块的 `Shaders/Private/ToonPass.usf`，用 `IMPLEMENT_GLOBAL_SHADER` 注册。

**2) 直接改 `DeferredShadingRenderer.cpp`**

在 `Render` 函数里插 `AddPass`。控制力最强，但升级维护成本高。

**3) 渲染器插件 / 自定义 `FSceneRenderer`**

彻底自建，工程量最大。

</details>

---

## 六、Custom PostProcess

| 方式 | 难度 | 适用 |
|---|---|---|
| **Post Process Material** | 最低 | 描边、色调映射、色差、屏幕后效 |
| **Post Process Material + Custom Node** | 低 | 需要复杂数学但不需要额外 RT |
| **SceneViewExtension + 自定义 Pass** | 中 | 需要多张中间 RT、多次采样、自定义 Shader |
| **改引擎后处理链** | 高 | 需要替换引擎内置 pass |

**卡通渲染最常用**：
- **描边** → Post Process Material（深度 + 法线 Sobel 边缘检测），最省事
- **Backface 描边** → 额外 Mesh Pass（反转法线外扩），可用 `PostRenderBasePassDeferred` 或单独的 `FMeshPassProcessor`
- **色彩分级 / 抖动** → Post Process Material，Blendable Location 选 `Before Tonemapping`

Post Process Material 挂载方式：

```cpp
PostProcessSettings.WeightedBlendables.Array.Add(
    FWeightedBlendable(1.0f, ToonPostProcessMaterial));
```

---

## 七、推荐落地路线

<details>
<summary><b>展开：分阶段实施建议</b></summary>

**阶段 1 — 最小可用（1～2 周）**
1. 加 `MSM_Toon` shading model，改 `ShadingModels.ush` 做分级光照
2. 用现有 `CustomData` 通道传 Ramp 参数
3. 描边用 Post Process Material

**阶段 2 — 增强（1 个月）**
4. 加 SceneViewExtension，插自定义 Pass 做边缘光、额外高光
5. 自定义 PostProcess 做风格化色调映射

**阶段 3 — 深度魔改（长期）**
6. 扩展 GBuffer 通道
7. 改写延迟光照 pass
8. 移动端单独适配

</details>

## 八、几个必须提前知道的坑

1. **Substrate / Strata**：UE5.1+ 的新材质系统（5.4 改名 `Substrate`）。如果项目启用了，材质路径和 GBuffer 布局完全不同，自定义 shading model 的做法要重写。**建议先确认项目是否启用。**

2. **Shader 编译时间**：新增 shading model 会显著增加 shader 变体数量，编译时间可能翻倍，团队开发时务必配好 `ShaderCompileWorker` 集群或 `DDC` 共享。

3. **版本升级成本**：改引擎的文件越多，升级越痛。**能走 SceneViewExtension / 插件就尽量走**，只把真正必须的（shading model、GBuffer）放进引擎改动，并且每次升级都做好 diff 记录。

4. **移动端**：`Mobile` 渲染路径是独立的（`MobileBasePassPixelShader.usf` 等），不处理会直接崩或显示错误。

5. **Nanite / Lumen 兼容性**：Nanite 的材质走的是 virtualized 几何路径，自定义 GBuffer 写法要注意 `Nanite` 的 `FMeshMaterialShader`；Lumen 会读 GBuffer 做间接光，你的额外通道可能被忽略或需要专门处理。

**一句话总结**：Shading Model 和 GBuffer 是必须改引擎的「硬改动」，Custom Pass 和 PostProcess 优先用 SceneViewExtension 和 Post Process Material 来「软扩展」，把引擎源码的改动面压到最小。

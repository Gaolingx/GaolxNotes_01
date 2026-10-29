# DLSS SR输入需求与FG附加输入

## 🧑‍💻 User

motionvector+depth+finalcolor输入dlss sr够用了吗
fg还要额外输入哪些东西

---
## 😀 默认助手 (7)

# 结论概要

| 问题 | 回答 |
|---|---|
| motionvector + depth + finalcolor 够 DLSS SR 吗？ | **从 buffer 层面说，基本够了**（这三项就是 SR 的三件套），**但不够运行**，因为 NGX 还强制需要一组非 buffer 参数（jitter、MV scale、深度约定、相机矩阵、分辨率、reset…），而且这三张图各自都有严格格式/时机要求 |
| FG 还要什么？ | 光多给几张图是不够的，**FG 需要一套不同的管线结构**：输出分辨率的 MV/depth、**单独的 UI/HUD buffer**、上一帧历史 + OFA 光流、swapchain/Reflex 的接入方式改造、帧计数（MFG） |

---

# 一、DLSS Super Resolution

## 1. 三张图的要求（这里最容易出错）

| 输入 | 分辨率 | 关键约定 |
|---|---|---|
| **Color（final color）** | **渲染分辨率**（低分辨率，例如 720p），不是最终输出分辨率 | 必须是 **UI 合成之前**；LDR 路径要 **tonemap 之后**；HDR 路径给 FP16 线性 + exposure。**不能在 TAA/FXAA/锐化之后**给 |
| **Depth** | 渲染分辨率 | 建议给硬件深度（`z/w` 非线性也可以，但必须同时把相机矩阵喂进去，让 DLSS 自己反投影）；反转 Z 要置 `DepthInverted=1`；无限远平面要正确设置 `CameraFarPlane` |
| **MotionVectors** | 渲染分辨率，2 通道 | **screen space、单位为像素**；**必须排除相机 jitter**（jitter 单独通过 `JitterOffset` 传）；符号方向以你所用版本的 Programming Guide 为准，**引擎 TAA 的 MV 约定和 NGX 差一个负号是最常见的错误**；半透明/粒子也要有 MV |

## 2. 还必须提供的非 buffer 参数（依据是否可用）

```
分辨率：  Width / Height（渲染）  OutWidth / OutHeight（输出）
缩放：    由 NVSDK_NGX_DLSS_GetOptimalSettings 算出的 render res
抖动：    Jitter_Offset_X / Jitter_Offset_Y   ← 单位像素，且渲染时投影矩阵必须真的加了这个 jitter
MV：      MV_Scale_X / MV_Scale_Y
深度：    DepthInverted、CameraNear、CameraFar、CameraFOV、CameraAspectRatio
矩阵：    CameraViewToClip、ClipToCameraView、ClipToView、ViewToClip、
          WorldToView、ViewToWorld
控制：    Reset（切镜头/改分辨率/曝光跳变/读档时置 1）
质量：    Sharpness（别和外部锐化叠加）
资源：    Output（输出分辨率的目标纹理）、可选 Exposure
```

> 注：DLSS 2.x 时代的 `reactive mask / transparency mask / bias mask` 是可选项，较新版本已大幅弱化甚至移除，不需要再给。

## 3. 管线顺序（顺序错了，画质会大幅下降）

```
渲染(低分辨率, 加 jitter) → 后处理 → Tonemap → 【DLSS SR】 → UI 合成 → 输出
```
TAA 必须关掉（DLSS 取代它）；DOF/Bloom 放前面没问题，但**运动模糊、胶片颗粒、锐化、色差如果放在 SR 前面，就会让训练分布外的输入进入网络**。

---

# 二、DLSS Frame Generation 额外需要什么

FG 的问题在于，它是一套 **present-time（呈现时机）的结构改造**，并非只是“缺了几个输入”。

## 1. 输入侧的真实增量

| 增量 | 说明 |
|---|---|
| **输出分辨率的 Motion Vectors** | SR 用渲染分辨率 MV，**FG 要输出/显示分辨率的 MV**。这是和 SR 最大的差异，也是接入 FG 时最疼的地方（要额外升采样/单独输出一份全分辨率 MV） |
| **输出分辨率的 Depth** | 同上，FG 用来做 disocclusion 判断 |
| **Color 必须是 SR 的输出** | 即输出分辨率、UI 之前的帧 |
| **单独的 UI / HUD buffer（UI color + alpha）** | **最关键的一条**。FG 生成的中间帧要对 UI 单独 warp + 合成，否则 UI、字幕、血条会糊成重影。所以必须把 UI 渲染到独立图层，再交给 FG 去合成 |
| **上一帧的历史（color/depth/MV）** | FG 是插值当前帧与上一帧，需要时间域上的双向数据 |
| **OFA 光流** | 由驱动调用 Optical Flow Accelerator 硬件内部算，**不需要你手动喂**，但要求资源是 OFA 可访问的（别放在 OFA 不支持的堆/压缩格式上） |
| **Jitter / 相机数据（两帧）** | 用于去 jitter |
| **Exposure / HDR 元数据** | HDR + 自动曝光下，生成帧与实际帧亮度必须匹配 |
| **帧生成数量（DLSS 4 MFG）** | RTX 50 上要显式传 2x/3x/4x 的生成帧数 |

## 2. 系统性要求（比输入更容易出问题）

<details>
<summary><b>展开：FG 的接入条件清单</b></summary>

- **必须在 Present 路径里评估**：FG 不是普通 pass，它挂在 swapchain present 上，需要换 `DXGI_SWAP_EFFECT_FLIP_DISCARD` + `ALLOW_TEARING` + `FRAME_LATENCY_WAITABLE_OBJECT` 这类 flip model 配置，并通过 `IDXGISwapChain4` 管理。
- **必须启用 NVIDIA Reflex**：FG 依赖 Reflex 做帧同步与延迟控制，不接入 Reflex 基本跑不起来。
- **PresentID / 同步**：每次 Present 要正确递增 ID，否则会出排序错乱、画面撕裂、卡顿。
- **UI 必须后置合成**：UI 不能烘进 backbuffer 再给 FG。
- **MV 必须覆盖一切动的东西**（粒子、植被、透明、shader 驱动的动画），FG 对“没有 MV 的物体”比 SR 敏感得多，会直接拖影。
- **Reset**：过场、分辨率切换、UI 显隐、掉帧尖峰都要 reset。
- **限制帧率/VSync 组合**：FG 对帧节奏（frametime 稳定性）敏感，通常要配合 frame limiter。
- **VRR/HDR/多显示器** 有一堆额外约束，需要看对应版本的指南。

</details>

## 3. 完整管线

```
渲染(低分辨率+jitter) → 后处理 → Tonemap
        ↓
   【DLSS SR】→ 输出分辨率、无UI的 Color
        ↓
   UI 单独渲染成一层（color+alpha）
        ↓
   【DLSS-G @ Present】+ 输出分辨率 Depth / MV + 历史 + OFA
        ↓
   合成 UI → Present（生成帧在这里插进去）
```

---

# 三、如果你其实想做的是 RR（光线重建）

那输入数量完全不在一个量级：DLSS-RR 需要 **albedo、normal、roughness、specular、diffuse、世界位置、材质/几何信息** 等一整套 G-Buffer 引导缓冲，本质上是**替换掉你的降噪器**。motionvector + depth + color 远远不够。

---

# 四、给你的最小可用检查表

**SR 能跑起来的判据**
1. Color 是渲染分辨率、pre-UI、post-tonemap ✓
2. Depth 是渲染分辨率，`DepthInverted`/near/far 与矩阵自洽 ✓
3. MV 是渲染分辨率、像素单位、**不含 jitter**、符号方向验证过 ✓
4. jitter 确实加进了投影矩阵，且 `Jitter_Offset` 与之一致 ✓
5. TAA/FXAA/锐化已关闭 ✓
6. 切镜头会 `Reset` ✓

**FG 额外判据**
1. MV/depth 有**输出分辨率**的一份 ✓
2. UI 是**独立一层**（含 alpha）✓
3. 接了 Reflex、flip-model swapchain、PresentID 正确 ✓
4. 所有动态物体都有 MV ✓
5. SR → UI → FG 的顺序正确 ✓

如果你告诉我你用的是哪个 API（DX12 / Vulkan / UE5 插件 / Unity）和 DLSS 版本（3.x 还是 4.x），我可以直接给你对应的参数设置代码骨架和常见报错对照。

---
## 🧑‍💻 User

这些attachment在不同引擎（unity6000/ue5.6）中要如何提供给streamline sdk？

---
## 😀 默认助手 (7)

# 核心理解：引擎插件已完成大部分工作

Streamline 的集成模型是 **“给资源打标签 + 交常量 + 交当帧 command buffer”**，与“把纹理塞给 API”不同。UE 和 Unity 的官方 DLSS 插件已经完成了 90% 的工作；你要做的是**补上引擎没有提供的那几个 tag**，或在自定义渲染路径上重新接一遍。

> 以下按 **Streamline 2.x**（DLSS 4 / MFG 配套）描述。1.x 的 `sl::setTag` / `sl::getTag` 在 2.x 已变成 `sl::ResourceTag` 数组随 `sl::evaluate` 传入，具体以你手上的 `sl_*.h` 为准。

---

# 一、跨引擎同构的调用骨架

## 1. 初始化（顺序是关键）

```cpp
sl::Preferences pref{};
pref.logLevel            = sl::LogLevel::eDefault;
pref.logMessageCallback  = &OnSlLog;
pref.flags               = sl::PreferenceFlags::eUseFrameBasedResourceTagging;
sl::init(pref);                    // 必须在 D3D12CreateDevice / vkCreateDevice 之前
// … 创建设备 …
sl::setD3D12Device(device);        // 设备一建好立刻告知
```

`eUseFrameBasedResourceTagging` 很重要：tag 按帧保存且可复用，不必每帧全量重打。

## 2. 每帧：常量 + tag + evaluate

```cpp
sl::FrameToken* frame = nullptr;
sl::getNewFrameToken(frame, ++frameIndex);

sl::ViewportHandle vp(viewId);     // 分屏/多相机必须区分，否则会互相踩历史

sl::Constants c{};
c.cameraViewToClip  = viewToClip;
c.clipToCameraView  = clipToCameraView;
c.clipToView        = clipToView;
c.cameraNear        = nearZ;
c.cameraFar         = farZ;
c.cameraFOV         = fovRad;
c.cameraAspectRatio = aspect;
c.jitterOffset      = { jitterX, jitterY };      // 像素，必须和投影矩阵里加的一致
c.mvecScale         = { 1.f, 1.f };              // 引擎 MV 单位不是像素时在这里换算
c.depthInverted     = reversedZ ? sl::Boolean::eTrue : sl::Boolean::eFalse;
c.reset             = needReset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
sl::setConstants(c);

sl::Resource color (sl::ResourceType::eColor,         sceneColorRT,  state);
sl::Resource depth (sl::ResourceType::eDepth,         sceneDepthRT,  state);
sl::Resource mvec  (sl::ResourceType::eMotionVectors, motionVecRT,   state);
sl::Resource out   (sl::ResourceType::eScalingOutput, upscaledRT,    state);

sl::ResourceTag tags[] = {
  { &color, sl::ResourceType::eColor,          0, sl::ResourceLifecycle::eValidUntilEvaluate },
  { &depth, sl::ResourceType::eDepth,          0, sl::ResourceLifecycle::eValidUntilEvaluate },
  { &mvec,  sl::ResourceType::eMotionVectors,  0, sl::ResourceLifecycle::eValidUntilEvaluate },
  { &out,   sl::ResourceType::eScalingOutput,  0, sl::ResourceLifecycle::eValidUntilPresent  },
};

sl::evaluate(sl::Feature::eDLSS, *vp, *frame,
             (const sl::BaseStructure**)tags, std::size(tags),
             cmdBuffer /* 引擎当帧的 command list 包装 */);
```

## 3. FG：多两个 tag + options

```cpp
sl::Resource hudless(sl::ResourceType::eHudLessColor,    hudlessRT, state);
sl::Resource ui     (sl::ResourceType::eUIColorAndAlpha, uiRT,      state);

sl::ResourceTag fgTags[] = {
  { &hudless, sl::ResourceType::eHudLessColor,    0, sl::ResourceLifecycle::eValidUntilPresent },
  { &ui,      sl::ResourceType::eUIColorAndAlpha, 0, sl::ResourceLifecycle::eValidUntilPresent },
  // depth / mvec 照旧，但注意：这两个必须是【输出分辨率】
};

sl::DLSSGOptions fg{};
fg.mode                = sl::DLSSGMode::eOn;  // eOff / eAuto / eDynamic
fg.numFramesToGenerate = 1;                   // MFG: 1 / 2 / 3

// options 也走 inputs 数组（DLSSOptions/DLSSGOptions/ReflexOptions 都继承 BaseStructure）
sl::evaluate(sl::Feature::eDLSS_G, *vp, *frame, allInputs, n, cmdBuffer);
```

FG 还必须在帧首和 Present 前与 Reflex 配对：`sl::ReflexOptions.frameLimitUs` 做帧节奏控制，并调用 `eReflexModeLowLatency` → `eReflexModeReflexSleep`。

---

# 二、附件 → Streamline tag → 引擎来源（核心对照表）

| 语义 | `sl::ResourceType` | **生命周期** | **UE 5.6 来源** | **Unity 6000 / HDRP 来源** |
|---|---|---|---|---|
| 渲染分辨率颜色 | `eColor` | `eValidUntilEvaluate` | `FSceneTextures::SceneColor`（Tonemap 之后、UI 之前） | `HDCamera.colorBuffer` |
| 深度 | `eDepth` | `eValidUntilEvaluate` | `FSceneTextures::SceneDepth` | `_CameraDepthTexture`（reversed-Z！） |
| 运动矢量 | `eMotionVectors` | `eValidUntilEvaluate` | `FSceneTextures::MotionVector` | `_CameraMotionVectorsTexture` |
| 上采样输出 | `eScalingOutput` | `eValidUntilPresent` | 插件内部 RDG 纹理 | 插件内部 |
| **HUD-less 颜色** | `eHudLessColor` | **`eValidUntilPresent`** | SceneColor 在 Slate 合成前的一份 | **没有内置，要自己 copy** |
| **UI + alpha** | `eUIColorAndAlpha` | **`eValidUntilPresent`** | Slate/UMG 单独 RT | **没有内置，要自己渲染到独立 RT** |
| 曝光 | `eExposure` | `eValidUntilPresent` | AutoExposure / EyeAdaptation 结果 | `_Exposure` / 自己算 |

针对输出分辨率 MV/depth：这是 FG 独有的要求。UE 侧插件会自动处理（它会在 SR 后生成一份全分辨率 MV/depth）；自研或 Unity 侧需要你自己补：常见做法是先把渲染分辨率 MV 用深度感知的方式升采样，或干脆在渲染时额外写一份输出分辨率 MV（成本高但最稳）。

---

# 三、UE 5.6 怎么接

<details>
<summary><b>展开：插件结构、扩展点、cvar、注意事项</b></summary>

## 使用的插件
NVIDIA 官方 **DLSS / Streamline for UE** 插件（NVIDIA Developer 下载或 NVRTX 分支），UE 5.6 版本内包含模块大致为：

- `StreamlineCore`：`sl::init` / `setD3D12Device` / FrameToken / ViewportHandle 管理
- `StreamlineD3D12`、`StreamlineVulkan`：RHI 适配
- `StreamlineDLSS`、`StreamlineDLSSG`、`StreamlineReflex`：各 feature
- `StreamlineDeepDVC`、`StreamlineNIS`

## 需要保持不变的部分
插件用 `ISceneViewExtension`（`FSceneViewExtensionBase`）挂载：在 `PostRenderViewFamily_RenderThread` 或 `PostProcessing` 末端插入 evaluate。相机矩阵、jitter、`ViewSize` 都从 `FViewInfo` 获取：

- `View.ViewMatrices.GetViewToClip()` / `GetClipToView()` / `GetViewMatrix()`
- `View.TemporalAAJitter`（像素）
- `View.GetSceneTexturesChecked()->SceneColor / SceneDepth / MotionVector`

## UE 特有的两个问题

1. **MV 单位不是像素**。UE 的 velocity 编码是 NDC 派生物（`PrevClip/PrevClip.w - Clip/Clip.w` 量级），接 SL 时必须用 `mvecScale` 或自己乘 `ViewSize` 换算。**UE 大版本之间这条编码改过（5.3 前后 `FSceneTextures` 改名、TSR 重写）**，务必去你版本里 velocity/motion-vector 的 encode 处核对一遍，不要照抄网上的系数。

2. **HUD-less 与 UI 分离**。UE 默认把 UMG/Slate 合成进最终 backbuffer。FG 要求 UI 独立成层——UE 插件是靠“在 Slate 合成之前拦截 SceneColor + 单独取 Slate 的 RT”来做的。如果你有 **自定义后处理材质、自定义 FinalColor pass、或直接把 UI 画进 SceneColor 的 HDR 效果**，插件的插入点就错了，FG 会拖影。这时必须调插入点，或把该效果移到 UI 之后。

## 常用 cvar
`r.NGX.DLSS.Enable`、`r.Streamline.DLSSG.Enable`、`r.Streamline.Reflex.Enable`、`r.AntiAliasingMethod`（必须离开 TSR/TAA）。具体名字随插件版本有变。

</details>

---

# 四、Unity 6000 怎么接

<details>
<summary><b>展开：两条路线与真正的硬性限制</b></summary>

## 路线 A：SR 使用 HDRP 内置
Unity 6 的 HDRP 自带 DLSS（底层 NGX），走 `HDAdditionalCameraData` 上的 DLSS 选项，**不需要碰 Streamline**。这条路只需要：
- 相机开 Motion Vectors（`FrameSettingsField.MotionVectors`）
- 关闭 HDRP 的 TAA / FSR / 自带 upscaler
- 深度是 reversed-Z，插件会置 `depthInverted`

## 路线 B：FG 必须走 Streamline 插件
Unity **没有内置 Frame Generation**，必须用 NVIDIA 的 Unity Streamline 插件（D3D12 only）。这里有个结构性障碍值得提前知道：

> **Unity 不把当帧的 D3D12 command list 暴露给 native 插件**，只给 `IUnityGraphicsD3D12v7::GetCommandQueue()` 和 `CommandBuffer.IssuePluginEvent` 的注入点。而 `sl::evaluate` 需要落在与场景渲染同一条 command list 上。因此官方插件是靠在 SRP 的特定时机（`RenderPipelineManager.endCameraRendering` / 插件内部 hook）通过 plugin event 注入，并依赖 Streamline 的 immediate 模式。

后果：**如果你在 URP/HDRP 里大量用自定义 `ScriptableRenderPass`，注入点很容易和你的 pass 顺序错位**，表现为 FG 插帧内容滞后一帧或直接黑帧。

## 资源从哪取（HDRP）
| 需要的 | C# 取法 |
|---|---|
| Color | `HDCamera.colorBuffer`（在 tonemap 后、UI 前的那个） |
| Depth | `_CameraDepthTexture`（shader global）或 `HDCamera` 的 depth RT |
| Motion Vectors | `_CameraMotionVectorsTexture`（shader global，`Shader.GetGlobalTexture`） |
| Jitter | HDRP 的 TAA jitter（投影矩阵已含），需取与之一致的像素值 |
| HUD-less | **自己 `CommandBuffer.Blit` 一份**（HDRP 的 UI 是 Canvas 独立渲染，理论上容易分离） |
| UI + alpha | **自己把 UGUI/UI Toolkit 渲到一张带 alpha 的 RT** |

MV 单位同样是 UV/NDC 空间而非像素，要换算；符号方向也要在现场验证（Unity 的 velocity 约定和 NGX 常见差一个负号）。

## 风险提示
Unity 侧 DLSS-G 的官方支持成熟度一直低于 UE，且对 Unity 版本、图形 API、Graphics Jobs、Native Render Pass 都有额外要求。**上项目前先拿你的 Unity 6000 具体小版本 + 插件版本跑通再说**，不要按 UE 的经验估工期。

</details>

---

# 五、这 6 种情况你必须手动喂 tag

1. 有自定义后处理/最终合成 pass，插件默认插入点在它之前
2. UI 被烘进 SceneColor（FG 直接坏）
3. 多视口 / 分屏 / VR（必须给每个 view 不同的 `ViewportHandle`）
4. 用了镜头畸变（MV/depth 需要 undistort，否则 SR 边缘会抖）
5. 想用 **DLSS-RR**（要额外喂 `eNormals`、`eDiffuseAlbedo`、`eSpecularAlbedo`、`eRoughness`、`eSpecularHitDistance` 等一整套 G-buffer tag）
6. 想用 **DLSS 4 MFG**，但引擎插件版本还没跟上（要自己设 `numFramesToGenerate`）

---

# 六、验证清单（先跑这三步，能省一半调试时间）

1. `sl::getFeatureRequirements(sl::Feature::eDLSS_G, req)` —— 在初始化后立刻查，确认该机器/该驱动上 FG 可用，以及它到底要求哪些 tag。不要把“FG 不支持”误当成“tag 没喂对”。
2. 打开 NGX 的 DLSS 指示器（注册表 `HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore\ShowDlssIndicator`）——能直接看到当前是 DLSS 的哪个版本/模式在跑，区分“SR 生效了”还是“回退到 TAA 了”。
3. 用 SL 的 log callback 打开 verbose，看 `sl::Result` 的返回值。常见错误码能直接定位问题：
   - `eErrorMissingInputParameter` → tag 少给/type 写错
   - `eErrorMissingResourceState` → texture 的 state 不对（DX12 barrier 没到 `NON_PIXEL_SHADER_RESOURCE`）
   - `eErrorFeatureNotSupported` / `eErrorDriverOutOfDate` → 环境问题，不是你的 bug
   - `eErrorNoPlugins` → `sl.dlss_g.dll` 没加载到 `pathsToPlugins`

---

需要我把上面任意一段整理成**可直接编译的最小示例**（UE 的 `ISceneViewExtension` 子类 / Unity 的 native plugin + C# 调用），或者针对你具体的“自定义后处理链”画一版插入点图？告诉我你的渲染路径细节就行。

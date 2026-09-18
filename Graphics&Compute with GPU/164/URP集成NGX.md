# URP集成NGX

### 🧑‍💻 User

unity urp如何编写render feature+native plugin接入nvidia ngx（Super Resolution+Frame Generation  integration）

---

### 🤖 Assistant

I'll help you with this. Let me first check what tools are available that might help me find relevant documentation or resources on this topic.

Let me research the latest NVIDIA DLSS/NGX and Streamline documentation to give you accurate, current integration guidance.

Good. I have the Streamline SDK overview. Now let me fetch the detailed programming guides for DLSS Super Resolution, Frame Generation, and the general integration guide.

I have the full Streamline/DLSS/Reflex API reference. Now let me verify the Unity-side native plugin interface and URP render-pass details.

Let me confirm the exact Unity D3D12 native interface functions used by an official Unity sample plugin.

# Unity URP + Native Plugin 接入 NVIDIA NGX（DLSS-SR + Frame Generation）落地方案

下面给出一套**可落地**的分层架构、关键代码骨架和踩坑清单。先给结论，再展开。

---

## 0. 先厘清一个关键事实（很重要）

| 目标 | 能否直接用 NGX SDK | 正确做法 |
|---|---|---|
| **DLSS Super Resolution**（超分） | ✅ 可以（`nvngx_dlss.dll`） | 用 NGX 或 Streamline 都行 |
| **DLSS Frame Generation**（插帧/DLSS-G） | ❌ **不行** | 必须走 **Streamline**（`sl.interposer` + `sl.dlss_g.dll`） |

**原因**：DLSS-G 需要拦截 `Present`/`vkQueuePresentKHR`、接管 swapchain、并强制搭配 **Reflex**；原始 NGX SDK 不暴露这套 present 拦截与帧调度（SL pacer）。所以即使你写"NGX 插件"，**插帧部分最终仍是调 Streamline**。

> 结论：**直接以 Streamline 作为你的 native 插件封装层**，它内部会加载 NGX 的 `nvngx_dlss.dll` / `sl.dlss.dll` / `sl.dlss_g.dll`。这样一套 API 同时拿到 SR + FG + Reflex（Reflex 是 FG 的硬依赖）。
>
> 参考：`NVIDIA-RTX/Streamline`（本文 API 基于 **SL 2.14.1**）、`NVIDIA/DLSS`、Unity `NativeRenderingPlugin` 样例。

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────┐
│ C# / URP 层                                               │
│  ScriptableRendererFeature ──> ScriptableRenderPass       │
│      · 提供深度/运动矢量/颜色/抖动常量                    │
│      · cmd.IssuePluginEventAndData() 把调用排进渲染线程   │
│      · 用低分辨率 RT 渲染，全分辨率 RT 作为 DLSS 输出      │
└───────────────┬──────────────────────────────────────────┘
                │ P/Invoke + RenderingEventAndData(指针)
┌───────────────▼──────────────────────────────────────────┐
│ C++ Native Plugin（你的封装）                             │
│  UnityPluginLoad → IUnityGraphicsD3D12v7（拿 device/queue）│
│  slInit / slSetD3DDevice / slSetTagForFrame /            │
│  slSetConstants / slEvaluateFeature / slDLSSGSetOptions  │
│  slReflexSleep / slReflexSetMarker                       │
└───────────────┬──────────────────────────────────────────┘
                │ 链接
┌───────────────▼──────────────────────────────────────────┐
│ sl.interposer.dll + sl.common.dll                        │
│   ├─ sl.dlss.dll     → nvngx_dlss.dll      (SR)          │
│   ├─ sl.dlss_g.dll   → (NGX, 闭源预编译)   (FG)          │
│   └─ sl.reflex.dll                          (低延迟)      │
└──────────────────────────────────────────────────────────┘
```

---

<details>
<summary><b>2. 第 1 层：C++ Native Plugin（Streamline 封装）</b>（点击展开）</summary>

### 2.1 拿 Unity 的原生 D3D12 设备

Unity 通过 `IUnityInterfaces` 暴露底层 D3D12 对象。编写插件时包含 Unity 安装目录下的：

```
Editor/Data/PluginAPI/Unity/IUnityInterface.h
Editor/Data/PluginAPI/Unity/IUnityGraphics.h
Editor/Data/PluginAPI/Unity/IUnityGraphicsD3D12.h
```

> ⚠️ `IUnityGraphicsD3D12` 有 `v1..v7` 多版本，**用你 Unity 版本实际提供的头文件**。关键接口：
> - `GetDevice()` → `ID3D12Device*`
> - `GetCommandQueue()` → `ID3D12CommandQueue*`
> - 帧栅栏相关（`GetFrameFence` 等，v5+）

```cpp
// DlssPlugin.cpp
#include "Unity/IUnityInterface.h"
#include "Unity/IUnityGraphics.h"
#include "Unity/IUnityGraphicsD3D12.h"
#include <d3d12.h>
#include "sl.h"
#include "sl_dlss.h"
#include "sl_dlss_g.h"
#include "sl_reflex.h"

static IUnityInterfaces*      s_UnityInterfaces = nullptr;
static IUnityGraphics*        s_Gfx             = nullptr;
static IUnityGraphicsD3D12v7* s_D3D12           = nullptr;   // 版本按实际头文件调整
static sl::ViewportHandle     s_Viewport{0};

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces* u) {
    s_UnityInterfaces = u;
    s_Gfx = u->Get<IUnityGraphics>();
    if (s_Gfx && s_Gfx->GetRenderer() == kUnityGfxRendererD3D12)
        s_D3D12 = u->Get<IUnityGraphicsD3D12v7>();
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload() {
    // 注意：不要在 DLLMain 里 slInit/slShutdown
    // slShutdown();
}
```

### 2.2 尽早 `slInit`

Streamline 要求**在任何 DXGI/D3D 调用之前**初始化（尤其插帧需要拦截 swapchain）。现实中 Unity 会先建设备，所以：

- **SR 路线**：晚一点也行——`slInit` 后用 `slSetD3DDevice(s_D3D12->GetDevice())` 把 Unity 设备交给 SL（SL 支持附着到既有设备）。
- **FG 路线**：必须让 `sl.interposer` 在 Unity 创建 swapchain **之前**就 hook 住，见 §4.3，这是最大的工程难点。

```cpp
static void InitStreamline(IUnityInterfaces* u) {
    sl::Preferences pref{};
    pref.showConsole       = true;                    // 生产关掉
    pref.pathToLogsAndData = L"Streamline";           // 出问题先看 sl.log
    pref.applicationId     = 0;                       // 向 NVIDIA 申请的 AppID
    pref.engine            = sl::EngineType::eUnity;
    pref.engineVersion     = "6000.0";

    sl::Feature feats[] = { sl::kFeatureDLSS, sl::kFeatureDLSS_G, sl::kFeatureReflex };
    pref.featuresToLoad    = feats;
    pref.numFeaturesToLoad = _countof(feats);

    pref.flags = sl::PreferenceFlags::eAllowOTA
               | sl::PreferenceFlags::eLoadDownloadedPlugins
               | sl::PreferenceFlags::eUseFrameBasedResourceTagging;  // 用 slSetTagForFrame 需开

    if (SL_FAILED(sl::Result r, slInit(pref, sl::kSDKVersion))) {
        // 通常是 sl.interposer.dll / sl.common.dll 没放到 exe 同级目录
    }
    // 设备就绪后（Unity 已创建）：
    if (s_D3D12) slSetD3DDevice(s_D3D12->GetDevice());
}
```

### 2.3 渲染线程回调 + 资源 Tagging + Evaluate

Unity 的 `IssuePluginEventAndData` 会在**渲染线程**回调，正好满足 SL "线程不安全"API 的要求。

```cpp
enum DlssEvent : int {
    kEventSetupDLSS = 1,     // 设置 SR 选项/渲染分辨率
    kEventSetCameraData,     // 每帧常量（矩阵/抖动/相机）
    kEventEvaluateDLSS,      // 执行 SR
    kEventSetDLSSG,          // FG 开关/选项
    kEventReflexSleep,       // Reflex 睡眠点
};

// 用 blittable 结构把 C# 侧数据传进来
struct DlssCameraData {
    float viewToClip[16];    // row-major，不含 jitter
    float clipToPrevClip[16];
    float jitterX, jitterY;  // 像素空间抖动
    float mvecScaleX, mvecScaleY;
    float cameraNear, cameraFar, cameraFov, cameraAspect;
    uint64_t depthTex, mvecTex, colorInTex, colorOutTex;   // ID3D12Resource*
    uint32_t renderW, renderH, outputW, outputH;
};

static void UNITY_INTERFACE_API OnRenderEvent(int eventID, void* data) {
    switch (eventID) {
    case kEventSetCameraData: {
        auto* d = reinterpret_cast<DlssCameraData*>(data);
        sl::Constants c{};
        memcpy(&c.cameraViewToClip, d->viewToClip, sizeof(float)*16);
        memcpy(&c.clipToPrevClip,   d->clipToPrevClip, sizeof(float)*16);
        c.jitterOffset = { d->jitterX, d->jitterY };
        c.mvecScale    = { d->mvecScaleX, d->mvecScaleY };
        c.cameraNear = d->cameraNear; c.cameraFar = d->cameraFar;
        c.cameraFOV  = d->cameraFov;  c.cameraAspectRatio = d->cameraAspect;
        c.depthInverted = sl::Boolean::eTrue;       // Unity D3D 是 reversed-Z
        c.cameraMotionIncluded = sl::Boolean::eTrue; // 运动矢量已含相机运动
        c.motionVectors3D = sl::Boolean::eFalse;
        c.renderingGameFrames = sl::Boolean::eTrue;  // 菜单/暂停/过场时置 eFalse
        slSetConstants(c, *s_FrameToken, s_Viewport);

        // 资源 Tag（传的是 ID3D12Resource*）
        sl::Resource depth  { sl::ResourceType::eTex2d, (ID3D12Resource*)d->depthTex,   nullptr, nullptr, D3D12_RESOURCE_STATE_DEPTH_READ };
        sl::Resource mvec   { sl::ResourceType::eTex2d, (ID3D12Resource*)d->mvecTex,    nullptr, nullptr, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE };
        sl::Resource colorIn{ sl::ResourceType::eTex2d, (ID3D12Resource*)d->colorInTex, nullptr, nullptr, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE };
        sl::Extent full{0,0,d->renderW,d->renderH};

        sl::ResourceTag depthTag (&depth,   sl::kBufferTypeDepth,
                                  sl::ResourceLifecycle::eValidUntilPresent, &full);
        sl::ResourceTag mvecTag  (&mvec,    sl::kBufferTypeMvec,
                                  sl::ResourceLifecycle::eOnlyValidNow, &full);
        sl::ResourceTag colorTag (&colorIn, sl::kBufferTypeScalingInputColor,
                                  sl::ResourceLifecycle::eOnlyValidNow, &full);
        sl::ResourceTag tags[] = { depthTag, mvecTag, colorTag };
        slSetTagForFrame(*s_FrameToken, s_Viewport, tags, _countof(tags),
                         (sl::CommandBuffer*)s_D3D12->GetCommandQueue());
        // FG 还需要额外 Tag Hudless / UI（见 §4.2）
    } break;

    case kEventEvaluateDLSS: {
        const sl::BaseStructure* inputs[] = { &s_Viewport };
        if (SL_FAILED(sl::Result r, slEvaluateFeature(
                sl::kFeatureDLSS, *s_FrameToken, inputs, _countof(inputs),
                (sl::CommandBuffer*)s_D3D12->GetCommandQueue()))) {
            // 失败 → fallback 到 URP 自带 Upscale
        }
        // 注意：evaluate 后 host 要负责恢复 command list 状态（Unity 侧一般无碍，
        // 但若共享 cmd list 需手动还原）
    } break;
    }
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT
UnityRenderingEventAndData GetRenderEventFunc() { return OnRenderEvent; }
```

> 关键约束（来自 SL 文档）：
> - 所有矩阵 **row-major**，且**不能含 jitter**；jitter 单独用 `jitterOffset`（像素空间）给。
> - `mvecScale`：运动矢量若是 `[-1,1]` 归一化空间填 `{1,1}`；若是像素空间填 `{1/renderW, 1/renderH}`。
> - Tag 的 `state` 必须正确；SL 自己管 barrier，你不用手动 transition。
> - 资源在 present 前可能被复用 → 用 `eOnlyValidNow`（但滥用会浪费 VRAM）。

</details>

---

<details>
<summary><b>3. 第 2 层：C# P/Invoke 封装</b>（点击展开）</summary>

```csharp
using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

public static class DlssNative
{
    const string Dll = "DlssPlugin";

    [DllImport(Dll)] static extern IntPtr GetRenderEventFunc();
    [DllImport(Dll)] static extern bool   InitStreamline();
    [DllImport(Dll)] static extern void   SetupDLSS(int viewport, int mode,
                                                    uint outW, uint outH);
    [DllImport(Dll)] static extern void   SetupDLSSG(int viewport, int mode,
                                                     int numFrames);
    public static IntPtr EventFunc => GetRenderEventFunc();

    public const int kEventSetCameraData = 2;
    public const int kEventEvaluateDLSS  = 3;
    public const int kEventSetDLSSG      = 4;
    public const int kEventReflexSleep   = 5;

    // 与 C++ DlssCameraData 严格一致
    [StructLayout(LayoutKind.Sequential)]
    public struct CameraData
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)] public float[] viewToClip;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)] public float[] clipToPrevClip;
        public float jitterX, jitterY, mvecScaleX, mvecScaleY;
        public float near, far, fov, aspect;
        public ulong depthTex, mvecTex, colorInTex, colorOutTex;
        public uint renderW, renderH, outputW, outputH;
    }
}
```

> 传参建议用 `NativeArray<byte>` + `UnsafeUtility.AddressOf` 或 `GCHandle.Alloc(..., Pinned)`，避免 GC 移动；`IssuePluginEventAndData` 的 `data` 是 `IntPtr`，**回调执行时指针必须有效**（用 `Allocator.TempJob`/持久 NativeArray，或固定句柄）。

</details>

---

<details>
<summary><b>4. 第 3 层：URP ScriptableRendererFeature</b>（点击展开）</summary>

### 4.1 SR 的接入点

DLSS-SR 要**替换 URP 的最终上采样/blit**：低分辨率渲染 → 全分辨率输出。推荐在 `RenderPassEvent.AfterRenderingPostProcessing`。

```csharp
public class DlssRenderFeature : ScriptableRendererFeature
{
    [SerializeField] DlssMode mode = DlssMode.Balanced;

    class DlssPass : ScriptableRenderPass
    {
        ProfilingSampler sampler = new("DLSS");
        RTHandle colorSrc, colorDst;

        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData data)
        {
            var desc = data.cameraData.cameraTargetDescriptor;
            // 低分辨率输入（把 renderScale 设 <1，例如 Quality 0.66 / Balanced 0.58）
            RenderingUtils.ReAllocateIfNeeded(ref colorSrc, desc, name: "_DlssInput");
            // 全分辨率输出
            var outDesc = desc;
            outDesc.width  = (int)(desc.width  / data.cameraData.camera.scaledPixelWidth * 1f); // 目标=屏幕
            outDesc.width  = Screen.width; outDesc.height = Screen.height;
            RenderingUtils.ReAllocateIfNeeded(ref colorDst, outDesc, name: "_DlssOutput");

            ConfigureTarget(colorSrc);   // DLSS 之前先渲到低分辨率
        }

        public override void Execute(ScriptableRenderContext ctx, ref RenderingData data)
        {
            var cam = data.cameraData.camera;
            var camData = data.cameraData;

            // 1) 计算 jitter（不要用 URP TAA，改由你驱动 Halton 序列）
            GetJitter(out float jx, out float jy);
            ApplyJitterToProjection(cmd, jx, jy);   // 应用到投影矩阵

            // 2) 填充常量
            var d = new DlssNative.CameraData {
                viewToClip     = ToRowMajor(camData.camera.cameraToWorldMatrix), // 详见 §5
                clipToPrevClip = ComputeClipToPrevClip(camData),
                jitterX = jx, jitterY = jy,
                mvecScaleX = 1f, mvecScaleY = 1f,       // URP mvec 已是 [-1,1]
                near = cam.nearClipPlane, far = cam.farClipPlane,
                fov  = cam.fieldOfView * Mathf.Deg2Rad,
                aspect = cam.aspect,
                depthTex   = (ulong)depthHandle.rt.GetNativeDepthBufferPtr(),
                mvecTex    = (ulong)mvecHandle.rt.GetNativeTexturePtr(),
                colorInTex = (ulong)colorSrc.rt.GetNativeTexturePtr(),
                colorOutTex= (ulong)colorDst.rt.GetNativeTexturePtr(),
                renderW = (uint)colorSrc.rt.width, renderH = (uint)colorSrc.rt.height,
                outputW = (uint)Screen.width,      outputH = (uint)Screen.height,
            };

            // 3) 把常量与 evaluate 排到渲染线程
            var cmd2 = CommandBufferPool.Get();
            using (new ProfilingScope(cmd2, sampler))
            {
                cmd2.IssuePluginEventAndData(DlssNative.EventFunc,
                                             DlssNative.kEventSetCameraData, Alloc(d));
                cmd2.IssuePluginEventAndData(DlssNative.EventFunc,
                                             DlssNative.kEventEvaluateDLSS, IntPtr.Zero);
                cmd2.Blit(colorDst, camData.renderer.cameraColorTargetHandle); // 或直接让 DLSS 写 screen
            }
            ctx.ExecuteCommandBuffer(cmd2);
            cmd2.Clear();
            CommandBufferPool.Release(cmd2);
        }
    }

    public override void Create() { m_Pass.renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing; }
    public override void AddRenderPasses(ScriptableRenderer r, ref RenderingData d)
    {
        if (d.cameraData.cameraType != CameraType.Game) return;
        r.EnqueuePass(m_Pass);
    }
}
```

**几个必须处理的点：**

1. **渲染分辨率**：把 URP 的 `renderScale`（`UniversalRenderPipelineAsset.renderScale` 或每相机 `cameraData.cameraTargetDescriptor` 缩放）设为你想要的比例。例如 `Quality` $\approx 0.66$、`Balanced` $\approx 0.58$、`Performance` $\approx 0.5$；`slDLSSGetOptimalSettings` 可以给你精确比例与最佳 sharpness。

2. **运动矢量**：URP 需在 Universal Renderer 上开启 **Motion Vectors**，并让需要运动矢量的 shader 声明 `MotionVectors` pass（或使用 deferred）。开启后会有 `_MotionVectorTexture`。Camera motion 需保证存在（Unity 通过 `_PrevViewProjMatrix` 写入）。

3. **深度**：优先给**原生 depth buffer**（`GetNativeDepthBufferPtr`）而非 `_CameraDepthTexture`（后者是拷贝）。设 `depthInverted = eTrue`（Unity D3D reversed-Z）。

4. **Jitter 一致性**：DLSS 收到 `jitterOffset` 必须与你实际加到投影矩阵上的那一次 jitter **完全一致**。用 DLSS 时建议**关闭 URP TAA**，由你驱动 jitter。

5. **HDR 格式**：输入颜色建议 `R16G16B16A16_SFloat` 或 `B10G11R11_UFloatPack32`；FG + HDR 用 `R10G10B10A2`/HDR10。

### 4.2 追加 FG 所需的 Tag

在 `kEventSetCameraData` 里额外 tag（**尺寸必须与 backbuffer 完全一致**）：

```
kBufferTypeHUDLessColor    // UI 绘制之前的场景色（质量关键）
kBufferTypeUIAlpha         // 仅 UI 的 alpha（首选，最省）
   或 kBufferTypeUIColorAndAlpha
// 可选：kBufferTypeBidirectionalDistortionField（强畸变后处理时才需要）
```

并在 `kEventSetDLSSG` 里：

```cpp
sl::DLSSGOptions o{};
o.mode = on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
o.numFramesToGenerate = 1;              // 1=2x；2=3x；最高 5（6x 上限）
o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;  // 进出菜单不卡顿
o.enableUserInterfaceRecomposition = sl::Boolean::eTrue; // 需同时有 Hudless+UI
slDLSSGSetOptions(s_Viewport, o);
```

### 4.3 Present 拦截 / swapchain（FG 的最大难点）

DLSS-G 通过 **Streamline 代理 swapchain** 拦截 `Present`。要求 `sl.interposer` 在 **Unity 创建 swapchain 之前**生效：

- **Editor 里 FG 基本不可用**：Editor 的 present 路径与打包不同，务必**在 Build 里测**。
- 打包后要确保 interposer 抢先 hook DXGI。可选做法：
  1. 把 `sl.interposer.dll` **作为 `dxgi.dll` 代理**部署在 exe 同级（最稳，NVIDIA 官方推荐给无引擎源码的场景）；
  2. 或把加载插件设为 **Load on Startup**，并用 `[RuntimeInitializeOnLoadMethod(BeforeSceneLoad)]` 触发 `InitStreamline()`，尽量抢在 swapchain 创建前；
  3. 用 `slUpgradeInterface(swapChain)` 对已存在的 swapchain 补挂（需能拿到 Unity 的 native swapchain，通常做不到，故不推荐）。
- **`slSetFeatureLoaded(sl::kFeatureDLSS_G, false/true)`** 用于只在主 swapchain 启用 FG（多 swapchain 场景）。
- **切换 FG 开关必须重建 swapchain**，否则有额外拷贝/同步开销（见 SL 文档 §18）。
- **窗口操作前先关 FG**（改分辨率、切换全屏/窗口），避免 `Present` 死锁。

### 4.4 Reflex（FG 强制依赖）

必须在渲染管线里插入 Reflex 标记与睡眠：

```cpp
// 帧开始（simulationStart）→ 各阶段 marker → presentStart/presentEnd
sl::ReflexOptions ro{}; ro.mode = on ? sl::ReflexMode::eLowLatency : sl::ReflexMode::eOff;
ro.frameLimitUs = 0;   // 用 Reflex 帧率限制器时设置
slReflexSetOptions(ro);

// 每次 Present 之前：
slReflexSleep(*s_FrameToken);
// 且 PCL marker 无论 Reflex 开关都要发：
slPCLSetMarker(sl::PCLMarker::eSimulationStart, *s_FrameToken);
// ... eRenderSubmitStart/End, ePresentStart/End ...
```

> SL 文档明确：**framegen 的 frame index 必须与 `ePresentStart/ePresentEnd` 匹配**，否则日志报 `common constants cannot be found for frame N`。

</details>

---

## 5. 矩阵与常量换算（容易错，单独说明）

- SL 要 **row-major**，Unity 的 `Matrix4x4` 也是 row-major（`m[row,col]`），但坐标约定为 **左手/OpenGL 风格**，D3D12 下 Unity 会自动做平台翻转。**务必用 `GL.GetGPUProjectionMatrix(proj, renderIntoTexture:true)` 得到实际 GPU 用的投影矩阵**再喂给 SL。
- $M_{\text{clip} \to \text{prevClip}} = M_{\text{viewToClip}}^{\text{prev}} \cdot M_{\text{worldToView}}^{\text{prev}} \cdot M_{\text{worldToView}}^{-1} \cdot M_{\text{viewToClip}}^{-1}$，即 `clipToPrevClip = prevViewProj * inverse(curViewProj)`（注意 row-major 的乘法顺序）。
- 矩阵**不得包含 jitter**；jitter 走 `jitterOffset`。
- `cameraFOV` 单位是**弧度**。

---

## 6. 构建/运行环境清单（缺一即失败）

- Windows 10 20H1 (build 19041)+；**必须在 Build 运行**（Editor 不保证）。
- 显卡驱动 **512.15+**（FG 建议更新）；ID 类 AppID 向 NVIDIA 申请。
- **Dynamic Resolution / HWS**：Windows「硬件加速 GPU 计划」开启（FG 必须）。
- 图形 API：**DX12**（FG 仅 DX12；Vulkan 下 FG 支持有限，VSync+FG 仅 DX12）。
- 分发文件（放到 exe 同级）：`sl.interposer.dll`、`sl.common.dll`、`sl.dlss.dll`、`sl.dlss_g.dll`、`nvngx_dlss.dll`（FG 的 `sl.dlss_g.dll` 只能用它预编译版，不能被自行编译，也**必须用 production 版**才能发布）。
- **安全**：正式版必须用 `WinVerifyTrust` + NVIDIA 自定义证书校验 `sl.interposer.dll`（`sl::security::verifyEmbeddedSignature`），防止插件被替换。
- 排除项：不要同时把 `dxgi.lib/d3d12.lib` 和 `sl.interposer.dll` 静态链进同一个模块。

---

## 7. 调试与验证

1. 打开 SL 日志：注册表 `HKLM\SOFTWARE\NVIDIA Corporation\Global\Streamline` 的 `LogLevel=2`（或环境变量 `SL_LOG_LEVEL`），看 `sl.log`。
2. 打开 SL ImGui 插件（`sl.imgui`）实时看深度/运动矢量/颜色/矩阵是否正确。
3. DLSS-G 指示文字：注册表 `HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore` 下 `DLSSG_IndicatorText=dword:2`。
4. 帧生成调试热键（dev 版）：`Shift+Ctrl+Insert`（debug 视图）、`Shift+Ctrl+Home`（stats）。
5. 用 **NVIDIA FrameView**，看 `MsBetweenDisplayChange`（不是 `MsBetweenPresents`）评估帧生成节奏。
6. Reflex 验证：用 `utils/reflex/ReflexTestSetup.bat` + Reflex Verification HUD（需 531.18+ 驱动）。

---

## 8. 常见坑速查

| 现象 | 原因 / 解法 |
|---|---|
| SR 输出模糊/重影 | jitter 与投影矩阵不一致；mvec 缩放错；`depthInverted`/`cameraMotionIncluded` 设错 |
| FG 完全没反应 | interposer 没抢到 swapchain（Editor？没走 dxgi 代理？）；HWS 未开；驱动太旧 |
| 日志 `common constants cannot be found for frame N` | Reflex `ePresentStart/End` 的 frame index 与常量帧不匹配 |
| 菜单/暂停后异常 | 进菜单要 `DLSSGMode::eOff` 或把相关 Tag 设为 null；`renderingGameFrames=eFalse` |
| VRAM 暴涨 | 滥用 `eOnlyValidNow` 导致 SL 复制；或 FG 开关未重建 swapchain |
| 崩溃到 `Present` | 同时链了 `dxgi.lib` 与 interposer；或改分辨率时没先关 FG |
| Editor 正常 Build 崩溃 | FG 在 Editor 不可用，只测 Build；且部署文件是否齐全 |

---

## 9. 结论与建议路线

1. **别自己从 NGX 裸撸 FG**——直接用 **Streamline** 作为 native 插件封装，一套 API 拿 SR+FG+Reflex。
2. 先做 **DLSS-SR**：`slSetD3DDevice` 挂 Unity 设备 + URP RenderFeature 在 `AfterRenderingPostProcessing` 替换上采样，这条路径**在 Editor/Build 都相对好验证**。
3. 再做 **FG**：核心难点在 **swapchain/Present 拦截**，优先按「`sl.interposer.dll` 作 `dxgi.dll` 代理」部署；并**强制先接 Reflex**。
4. 强烈建议**先研究 NVIDIA 官方 Unity DLSS 插件**（Unity Package Manager / NVIDIA 提供），它已解决 interposer 加载时序、URP/HDRP 集成、多相机等工程问题——可作为对照实现或直接用。

### 参考
- Streamline SDK：`github.com/NVIDIA-RTX/Streamline`（`docs/ProgrammingGuide*.md`、`sl_dlss.h`、`sl_dlss_g.h`、`sl_reflex.h`）
- DLSS SDK：`github.com/NVIDIA/DLSS`
- Unity 原生插件样例：`github.com/Unity-Technologies/NativeRenderingPlugin`
- 官方 Streamline 样例（含多视口、swapchain 重建）：`github.com/NVIDIA-RTX/Streamline_Sample`

---

需要的话，我可以进一步给出**完整的可编译工程骨架**（`.csproj/.vcxproj`、CMake、URP RenderFeature 全量代码、URP 运动矢量/深度管线配置步骤），或者针对**你具体的 Unity 版本（2021/2022/6.x）+ URP 版本 + 是否只要 SR** 给出裁剪后的最小实现。

---

### 🧑‍💻 User

针对unity6000，会涉及哪些需要修改urp管线源码的点？（包括srp、native部分，因为可能你要走native plugin结合 VTable Hook劫持unity底层图形接口比如hook swap chain等）
我在处理伴随FG的管线改造难题：UI分离与Reflex 打标，目的是解决UI鬼影
有没有什么比较好的hack手段解决？

---

### 🤖 Assistant

I'll research the exact Streamline UI-handling/recomposition and Reflex marker requirements, plus cross-check Unity 6 URP's render-graph internals, before giving you concrete hook points.

Let me pull the authoritative Streamline guides on DLSS-G UI handling and Reflex markers, since these are central to your UI-ghosting problem.

I have the authoritative Streamline details on UI tagging and Reflex now. Let me quickly verify the Unity 6 URP source layout so I can point to exact files.

Let me grab Streamline's manual-hooking guide and the `sl_hooks.h` API, since that's the exact bridge between your vtable hooks and SL.

研究了官方 Streamline 的 **Manual Hooking**（`sl_hooks.h` / `ProgrammingGuideManualHooking.md`）与 **DLSS-G UI 规则**后，下面给你一套针对 **Unity 6000 (6000.x) + URP 17.x (RenderGraph)** 的完整答案。

---

## TL;DR 路线决策

| 集成方式 | 何时用 | 代价 |
|---|---|---|
| **A. SL interposer 自动 hook**（`sl.interposer.dll` 当 `dxgi.dll` 代理） | 先跑通 SR、不想改引擎加载时序 | FG 在 Editor 难调、和第三方库冲突 |
| **B. SL Manual Hooking + 你自己的 VTable Hook**（你要的） | **要控 Editor、要精确 Present 打标、要 FG** | 要自己 hook + 转发，工程量大 |

Manual Hooking 的核心是：**`sl::PreferenceFlags::eUseManualHooking`** + 用 **`slUpgradeInterface`** 把 native 的 factory/device/swapchain 换成 SL proxy，然后把 SL 拦截清单里的调用**只转发给 proxy**，其余一律走 native。这正是官方支持的"劫持 swapchain"路径——**你不需要自己实现 FG 逻辑，只需把 hook 接到 SL 上**。

---

## Part 1 — Unity 6000 / URP 源码改动点

> ⚠️ 必须先**内嵌 URP 包**（把 `com.unity.render-pipelines.universal` 拷到 `Packages/`，或用本地 git/path 引用）。改 `Library/PackageCache` 会被覆盖。
> 文件名/路径随 17.x 小版本略有差异，以你本地为准。

### 1.1 三个"必须改"的管线节点

<details>
<summary><b>① 最终上采样 / Blit 路径（替换成 DLSS）</b>（展开）</summary>

DLSS-SR 要**替掉 URP 自带的 upscale**（含 STP/FSR/Linear）。相关位置：

- `UniversalRenderer.cs` → `Setup()` / `RecordRenderGraph()`：这里决定用哪个 `UpscalingFilterSelection`（`Linear/Point/FSR/STP`）。
- `PostProcessPass.cs`（及 Unity 6 拆出的 `FinalPostProcessPass` / final uber pass）：最终上采样 + 输出到 backbuffer 的地方。
- `FinalBlitPass.cs`：无后处理时的最终 blit。
- `StpUtils.cs` / STP 相关 pass：**必须禁用**，否则和 DLSS 打架。
- `UniversalRenderPipelineAsset.cs` → `UpscalingFilterSelection`（`Linear/Point/FSR/STP`）、`renderScale`。

改造点：把"低分辨率 cameraColor → 最终分辨率"这一步换成你的 DLSS evaluate，输出全分辨率目标。
</details>

<details>
<summary><b>② Jitter（DLSS 命门）</b>（展开）</summary>

- URP 的 TAA 在 `TemporalAA.cs`（`CalculateJitter` / `GetJitterMatrix` 之类）里生成 Halton 序列；STP 在 `StpUtils.cs`。
- jitter 最终由 `UniversalRenderPipeline.RenderSingleCamera` / `ScriptableRenderer.SetupCameraProperties` 处乘进投影矩阵，并写到 shader 全局。

改造点：**关掉 URP TAA/STP**，改用你自己的 jitter 序列，在 `RenderSingleCamera` 应用投影的位置注入；**同一次 jitter 必须原样传给 `sl::Constants::jitterOffset`**（像素空间）。若两者不一致 → 鬼影/模糊。
</details>

<details>
<summary><b>③ 深度 / 运动矢量：把原生资源暴露出来</b>（展开）</summary>

- 运动矢量：`MotionVectorRenderPass.cs`。URP 要"请求"运动矢量目标才会生成（由 TAA/STP 或 `UniversalCameraData` 决定），且对象材质需含 `MotionVectors` pass。DLSS 要的是**含相机运动的稠密场**。
- 深度：`CopyDepthPass.cs` / `UniversalResourceData.cameraDepthTexture`。**不要**用 depth-as-color 的 `_CameraDepthTexture`，尽量给**真正的 depth buffer**（`RTHandle.rt.GetNativeDepthBufferPtr()`），并按 Unity D3D 的 reversed-Z 设 `depthInverted = eTrue`。
- 建议的输入格式：color `R16G16B16A16_SFloat` / `B10G11R11`；mvec `R16G16_SFloat`。
</details>

### 1.2 ⭐ RenderGraph 下怎么拿到原生指针（Unity 6 最大坑）

Unity 6 URP 默认走 **RenderGraph**，`UniversalResourceData.cameraColor/cameraDepth/motionVectorColor` 都是 `TextureHandle`，**拿不到 `ID3D12Resource*`**。可行方案：

1. **Compatibility Mode**（`RenderGraphSettings` 里关掉 Render Graph）：回到 `Execute(ScriptableRenderContext, ref RenderingData)` + `ConfigureTarget(RTHandle)`，`RTHandle.rt.GetNativeTexturePtr()` 直接可用——**做 native interop 最省事**，但 6.1+ 被弃用。
2. **RenderGraph + 外部 RTHandle**（推荐长期方案）：在源码里把相机 color/depth 换成**你自有的 `RTHandle`**，再用 `renderGraph.ImportTexture(myRTHandle)` 导入；这样 pass 里能同时拿到 `TextureHandle` 和原生指针。
3. 自定义 `ScriptableRenderer`，覆写相机颜色目标的来源（同样改源码）。

```csharp
// 伪代码：RenderGraph 里把外部 RTHandle 导入，之后既能给 RG 用，也能取 native ptr
var imported = renderGraph.ImportTexture(myColorRTHandle);
// myColorRTHandle.rt.GetNativeTexturePtr()  -> 给 native plugin
```

### 1.3 Asset / 全局设置

- `renderScale` < 1（或 per-camera dynamic res），让 URP 按低分辨率渲染。
- `UpscalingFilterSelection` 设为 **Linear/Point**（不要 STP/FSR）。
- 相机栈、`UniversalAdditionalCameraData`：多相机（UI 分离用，见 Part 3）。

---

## Part 2 — Native 侧：VTable Hook + SL Manual Hooking

### 2.1 为什么"必须" hook（SL 的拦截清单）

SL 需要接管的是下面这些（来自 `sl_hooks.h::FunctionHookID`，**精确清单**）：

```
IDXGIFactory:  CreateSwapChain / CreateSwapChainForHwnd / CreateSwapChainForCoreWindow
IDXGISwapChain: Present / Present1 / GetBuffer / GetDesc / ResizeBuffers /
                ResizeBuffers1 / GetCurrentBackBufferIndex / SetFullscreenState
ID3D12Device:  CreateCommandQueue
(IDXGISwapChain_Destroyed 为内部，忽略)
```

其中 **`CreateCommandQueue` 也必拦**——因为 FG 需要挂在特定的 present 队列上。

### 2.2 你的 VTable Hook 点（D3D12 / DXGI）

用 MinHook / Detour 或手动 vtable 替换。关键索引（以当前 DXGI 头为准，注意核对）：

```
IDXGIFactory2  : CreateSwapChainForHwnd          = 15
                 CreateSwapChain                 = 10
IDXGISwapChain : Present                         = 8
                 GetBuffer                       = 9
                 SetFullscreenState              = 10
                 GetDesc                         = 12
                 ResizeBuffers                   = 13
                 Present1                        = 22
                 GetCurrentBackBufferIndex       = 36
                 ResizeBuffers1                  = 39
ID3D12Device   : CreateCommandQueue              = 8
```

**捕获链路**：
```
hook CreateDXGIFactory / IDXGIFactory::CreateSwapChain*
      → 拿到 native swapchain
      → slUpgradeInterface(&swapchain) 得到 SL proxy
      → vtable hook proxy 的 Present/ResizeBuffers/... （或直接转发）
```

### 2.3 接线到 SL Manual Hooking（官方姿势）

```cpp
// 1) 初始化：必须早于任何 DXGI 调用，且置 eUseManualHooking
sl::Preferences pref{};
pref.flags |= sl::PreferenceFlags::eUseManualHooking;
pref.applicationId = myAppId;
sl::Feature feats[] = { sl::kFeatureDLSS, sl::kFeatureDLSS_G, sl::kFeatureReflex };
pref.featuresToLoad = feats; pref.numFeaturesToLoad = _countof(feats);
slInit(pref, sl::kSDKVersion);

// 2) 动态加载（推荐，便于做签名校验）
auto mod  = LoadLibrary("sl.interposer.dll");
SL_FUN_DECL(slUpgradeInterface);      // 仅示意
SL_FUN_DECL(slGetNativeInterface);
auto slUpgradeInterface   = reinterpret_cast<...>(GetProcAddress(mod, "slUpgradeInterface"));
auto slGetNativeInterface = reinterpret_cast<...>(GetProcAddress(mod, "slGetNativeInterface"));

// 3) 安全校验（发布必做）
if (!sl::security::verifyEmbeddedSignature("sl.interposer.dll")) { /* disable SL */ }

// 4) 把你的 native 对象换成 SL proxy
IDXGIFactory* proxyFactory = nativeFactory;
slUpgradeInterface(&proxyFactory);          // 之后 CreateSwapChain* 必须走 proxyFactory
IDXGISwapChain* proxySwapchain = proxyFactory->CreateSwapChainForHwnd(...);
// 别忘了 device 也要升级（CreateCommandQueue 走 proxy）
ID3D12Device* proxyDevice = nativeDevice;
slUpgradeInterface(&proxyDevice);

// 5) 需要 native 接口时反向取
IDXGISwapChain* nativeSwapchain{};
slGetNativeInterface(proxySwapchain, &nativeSwapchain);
```

**规则**：SL 拦截清单内的 API → 一律调 **proxy**；其余（含 NVAPI 等）→ 调 **native**。

```cpp
HRESULT mySwapChain::Present(UINT SyncInterval, UINT Flags) {
    if (g_slEnabled) return proxySwapchain->Present(SyncInterval, Flags); // 转发给 SL
    return nativeSwapchain->Present(SyncInterval, Flags);
}
```

### 2.4 最大的隐藏坑：`presentCommon()` 必须每帧被调

官方警告：开启 `eUseManualHooking` 后，**SL common 插件的 `presentCommon()` 必须每帧执行一次**。

- 正确做法：**swapchain 创建后立刻 `slUpgradeInterface`**，让 SL 接管 present 路径，它就会自动调 `presentCommon()`。
- 若没被调用 → 行为异常 + **显存持续增长（尤其窗口 resize 时）**。
- 排查：在 `presentCommon()` 下断点，应每帧命中；resize 期间也要命中。
- 多 present 路径（多种窗口/全屏模式）要统一 fan-in 到同一条路径。

### 2.5 从 hook 里"顺手"拿到的额外好处

- **精确 Present 时机** → 唯一正确的 Reflex `ePresentStart/ePresentEnd` 打标位置（Part 3.4）。
- **ResizeBuffers / SetFullscreenState** → FG 必须在此处先 `DLSSGMode::eOff` 再切分辨率/全屏，否则可能死在 Present。
- **GetCurrentBackBufferIndex** → D3D 集成**必须**用 `SwapChain::GetCurrentBackBufferIndex`，否则 SL 报 `eFailGetCurrentBackBufferIndexNotCalled`。
- **Present 前抓 backbuffer** → 应急做 UI alpha diff（Part 3.2 B3）。

---

## Part 3 — UI 分离 + Reflex 打标：解决 UI 鬼影的 hack 方案 ⭐

### 3.1 鬼影根因

FG 会在两个真帧之间插入插值帧。若 UI 直接画进同一张 color（Unity 的 **Screen Space Overlay Canvas 由引擎直接合成到 backbuffer，根本不经过 URP**），插值帧上的 UI 只能来自真帧 → **UI 抖动/鬼影**。SL 的解法是把 UI 拆出来：

> **需求**：`Hudless`（无 UI 场景色） + `UI Alpha`（单通道，优选）或 `UI Color and Alpha`。
> **硬性约束**：三者尺寸**必须与 backbuffer 完全一致**。
> **Blend 公式（必须满足）**：$Final_{RGB} = UI_{RGB} + (1 - UI_{Alpha}) \times Hudless_{RGB}$
> **UI 的 RGB 需预乘 alpha**；无 UI 像素 alpha=0，有 UI 像素 alpha≠0。
> 同时 tag Hudless + UI Alpha 时可开 **`enableUserInterfaceRecomposition`** 获得更好质量。

### 3.2 三个 hack 手段

<details>
<summary><b>Hack A — 在 URP 内抓 Hudless（最省事，先做这个）</b>（展开）</summary>

因为 Overlay UI 是在 URP 之后才画到 backbuffer，所以 **URP 的最终 blit 到 backbuffer 之前的那张 cameraColor 就是天然 Hudless**。

```csharp
// 一个 ScriptableRendererFeature，事件放在 AfterRenderingPostProcessing
public override void Execute(ScriptableRenderContext ctx, ref RenderingData data)
{
    var cmd = CommandBufferPool.Get("GrabHudless");
    // 把全分辨率场景色 copy 到持久 RT（背景：低分辨率则先 DLSS 上采）
    Blitter.BlitCameraTexture(cmd, src, hudlessRT);
    // 之后 hudlessRT.GetNativeTexturePtr() 交给 native tag kBufferTypeHUDLessColor
    ctx.ExecuteCommandBuffer(cmd);
    CommandBufferPool.Release(cmd);
}
```
✅ 零改 UI 逻辑。缺点：只拿到 Hudless，**拿不到 UI Alpha**。
</details>

<details>
<summary><b>Hack B — 拿 UI Alpha 的四种方法</b>（展开，按推荐度排序）</summary>

**B1（推荐）离屏 UI 相机**：把所有 Canvas 改成 `RenderMode.ScreenSpaceCamera`，指到一个专用 UI 相机，该相机 `targetTexture = uiRT`（`clearFlags` 透黑，格式 `R8_UNorm` 或 `RGBA8`）。
```csharp
canvas.renderMode = RenderMode.ScreenSpaceCamera;
canvas.worldCamera = uiCamera;          // 专用，depth 高于主相机
uiCamera.clearFlags = CameraClearFlags.SolidColor;
uiCamera.backgroundColor = Color.clear;
uiCamera.targetTexture = uiRT;          // UI-only：RGB=UI色, A=覆盖度
```
- `CanvasScaler` 仍按屏幕分辨率算，布局/DPI 基本不变；
- 记得改 `GraphicRaycaster.eventCamera`；
- 用**预乘 alpha** 的 UI 输出即可直接满足 SL 公式。
- 单通道省带宽：把 alpha 抽到一张 `R8` RT → tag `kBufferTypeUIAlpha`。
- **UITK 用户**：`PanelSettings.targetTexture` 可直接让 UI Toolkit 渲到 RT，天然分离（B4）。

**B2 URP Camera Stack 分层**：主相机关 UI Layer，叠一个 UI 相机仅渲 UI Layer 到单独 RT。等价 B1，但更贴 URP。

**B3（应急/最脏）原生 Present 后做 diff**：hook `Present`，UI 合成后取 backbuffer，与 Hudless-上采样做 compute：
$A_{ui} = \mathrm{saturate}\big(k \cdot \lVert backbuffer - hudless \rVert_1 / 3\big)$，再阈值 + 膨胀。
无需改 UI，但质量差、有噪声/边缘 halo，**仅当无法改 UI 时用**。

**B4 UI Toolkit 离屏**：`PanelSettings.targetTexture` 指向 RT，天然得到 UI-only。若项目用 UITK 首选。
</details>

<details>
<summary><b>Hack C — 与 SL 对齐 + 开启 UI 重组</b>（展开）</summary>

```cpp
// 每帧 tag（尺寸必须 == backbuffer）
sl::Extent full{0,0,colorW,colorH};
sl::ResourceTag hudless (&hudlessRes, sl::kBufferTypeHUDLessColor,
                         sl::ResourceLifecycle::eValidUntilPresent, &full);
sl::ResourceTag uiAlpha (&uiAlphaRes, sl::kBufferTypeUIAlpha,   // 单通道优选
                         sl::ResourceLifecycle::eValidUntilPresent, &full);
// 或 sl::kBufferTypeUIColorAndAlpha（两者都 tag 时 SL 只用 UIAlpha）
sl::ResourceTag tags[] = { hudless, uiAlpha };
slSetTagForFrame(*s_FrameToken, s_Viewport, tags, _countof(tags),
                 (sl::CommandBuffer*)s_D3D12->GetCommandQueue());

// 开 UI 重组
sl::DLSSGOptions o{};
o.mode = sl::DLSSGMode::eOn;
o.numFramesToGenerate = 1;                       // 1=2x, 2=3x, 3=4x
o.enableUserInterfaceRecomposition = sl::Boolean::eTrue;  // 需 Hudless+UIAlpha 同时存在
o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;        // 进出菜单不卡
// 可选：o.mvecDepthWidth/Height, o.colorWidth/Height, 各 buffer format, numBackBuffers
slDLSSGSetOptions(s_Viewport, o);
```
> 关键：`Hudless` 必须与 backbuffer **同一色彩空间、同一后处理**（tonemap/bloom 等都已应用）；`UI Alpha` 的 blend 必须满足上面的公式。
</details>

### 3.3 尺寸/格式对齐检查表（最容易挂的点）

| Buffer | 尺寸 | 格式 | 备注 |
|---|---|---|---|
| Backbuffer | 屏幕 | `R10G10B10A2`(HDR10)/`R16G16B16A16` | 由 swapchain 决定 |
| Hudless | **== backbuffer** | 同 backbuffer | 无 UI，含全部后处理 |
| UI Alpha | **== backbuffer** | `R8` | 无 UI 处=0 |
| UI Color&Alpha | **== backbuffer** | `RGBA8`+预乘 | 与 UIAlpha 二选一 |
| Depth | 渲染分辨率 | `D32` | reversed-Z → `depthInverted=eTrue` |
| Mvec | 渲染分辨率 | `R16G16_SFloat` | 含相机运动 |

### 3.4 Reflex 打标：落点表 + PlayerLoop 注入

**Marker 落点**（`slPCLSetMarker`，无论 Reflex 开关都要发；`slReflexSleep` 也无论开关都要调）：

| Marker | Unity 6000 落点 |
|---|---|
| `eSimulationStart` | **PlayerLoop 注入**，在 `Update.ScriptRunBehaviourUpdate` 之前 |
| `eRenderSubmitStart` | `RenderPassEvent.BeforeRendering` pass，或 `Camera.onPreRender` |
| `eRenderSubmitEnd` | `RenderPassEvent.AfterRendering`（相机栈最后一帧） |
| `eInputSample` | 采样输入处 |
| `ePresentStart` / `ePresentEnd` | ⭐ **原生 `Present` hook 内**（唯一准确位置） |
| `slReflexSleep` | PlayerLoop 注入到 `PostLateUpdate`（`FinishFrameRendering` 前），或 Present hook 内 present 前调（线程安全） |

**PlayerLoop 注入（C#）**：
```csharp
[RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
static void Install()
{
    var loop = PlayerLoop.GetCurrentPlayerLoop();
    InsertBefore(ref loop, typeof(UnityEngine.PlayerLoop.EarlyUpdate),
        (PlayerLoopSystem sys) => new PlayerLoopSystem {
            type = typeof(DlssReflexHooks),
            updateDelegate = () => DlssNative.PCLMarker(PCLMarker.SimulationStart) });
    InsertBefore(ref loop, typeof(UnityEngine.PlayerLoop.PostLateUpdate),
        sys => new PlayerLoopSystem {
            type = typeof(DlssReflexSleep),
            updateDelegate = () => DlssNative.ReflexSleep() });   // 睡眠点
    PlayerLoop.SetPlayerLoop(loop);
}
```

**原生 Present hook 内做 present 打标**：
```cpp
HRESULT myPresentHook(IDXGISwapChain* sc, UINT sync, UINT flags) {
    slPCLSetMarker(sl::PCLMarker::ePresentStart, *s_FrameToken);
    // 需要时：slReflexSleep(*s_FrameToken);   // 也可放这里
    HRESULT hr = proxySwapchain->Present(sync, flags);   // 转发给 SL proxy
    slPCLSetMarker(sl::PCLMarker::ePresentEnd, *s_FrameToken);
    return hr;
}
```

**Reflex 选项 / boost（可选）**：
```cpp
sl::ReflexOptions ro{};
ro.mode = on ? sl::ReflexMode::eLowLatency : sl::ReflexMode::eOff;
ro.frameLimitUs = 0;   // 用 Reflex 帧率限制器时才设
// ro.useMarkersToOptimize = false;  // 绝大多数情况不要开，仅特定场景
slReflexSetOptions(ro);
```
> 若要最低延迟的预测相机，用 `slReflexSetCameraData` / `slReflexGetPredictedCameraData`（boost 场景）。

**验收（官方 Reflex QA）**：
- [ ] 三种模式（Off / On / On+Boost）都能切
- [ ] **Reflex 关闭时 `slReflexSleep` 和 PCL marker 仍在发**（Verification HUD 的 `App_Called_Sleep` 恒为 1）
- [ ] FG 打开跑 Reflex Test Utility 无 warning
- [ ] `sl::DLSSGStatus` 无 `eFailReflexNotDetectedAtRuntime`

---

## Part 4 — 改动清单总表

| 层 | 位置 | 改什么 |
|---|---|---|
| SRP | 内嵌 URP 包 | 最终上采样/Blit 换成 DLSS |
| SRP | `TemporalAA.cs` / `StpUtils.cs` / `UniversalRenderPipeline.cs` | 关 URP TAA/STP，注入自有 jitter（同一值给 DLSS） |
| SRP | `MotionVectorRenderPass.cs` / `UniversalCameraData` | 确保生成含相机运动的稠密 mvec |
| SRP | `CopyDepthPass.cs` / 深度目标 | 暴露**原生 depth buffer**，非 depth-copy |
| SRP | RenderGraph 资源 | color/depth 改自有 `RTHandle`（`ImportTexture`）以取 native ptr |
| SRP | `UniversalRenderPipelineAsset` | `renderScale<1`，`UpscalingFilterSelection=Linear/Point` |
| SRP | UI 层 | Canvas 改 `ScreenSpaceCamera`→离屏 UI 相机（或 UITK `targetTexture`） |
| Native | vtable hook | `CreateSwapChain*` / `Present` / `Present1` / `ResizeBuffers(1)` / `GetCurrentBackBufferIndex` / `SetFullscreenState` / `ID3D12Device::CreateCommandQueue` |
| Native | SL | `slInit(eUseManualHooking)` + `slUpgradeInterface`/`slGetNativeInterface` |
| Native | tag | Hudless / UIAlpha / Depth / Mvec / (UIColorAndAlpha) |
| Native | Reflex | `slReflexSetOptions` / `slReflexSleep` / `slPCLSetMarker`，present 在 hook 里打 |
| Gameplay | UI 逻辑 | 菜单/暂停/resize/全屏切换前 → `DLSSGMode::eOff`；`renderingGameFrames=eFalse` |

---

## Part 5 — 风险与备选

- **Editor 里 FG 基本不可用**：Unity Editor 的 present 路径特殊，务必在 **Build** 里验证。Manual Hooking 能让你在 Build 里精确接住 present，但 Editor 仍需 fallback（关 FG，仅 SR）。
- **Manual Hooking 的 `presentCommon()` 漏调 = 显存泄漏**，这是最容易翻车的点（Part 2.4）。
- **多 swapchain**：Editor 会有多个 swapchain，用 `slSetFeatureLoaded(sl::kFeatureDLSS_G, false/true)` 只让主窗口挂 FG。
- **安全**：发布必须 `verifyEmbeddedSignature` 校验 `sl.interposer.dll`，否则被判篡改。
- **降级路径**：任何一步失败（`slIsFeatureSupported`/`slDLSSGGetState().status` 非 `eOk`）要能回退到 URP 原生 upscale。
- **替代**：如果不想维护 URP 源码差分，强烈建议对照 **NVIDIA 官方 Unity DLSS 插件**（它已解决 interposer 加载时序、多相机、URP/HDRP 适配），把它当参考实现或直接用。

---

需要的话我可以继续给你：
1. **完整 vtable hook 的 C++ 骨架**（含 proxy/native 分流、安全校验、resize 处理）；
2. **URP 17 RenderGraph 版的 DLSS RenderFeature 全量代码**（含 RTHandle 导入、mvec/depth 暴露、jitter 接管）；
3. 一份**离屏 UI 相机的迁移脚本**（批量把 Overlay Canvas 转 ScreenSpaceCamera 并保持布局/DPI/射线）。

你想先啃哪一块？

---

### 🧑‍💻 User

为什么要推荐使用streamline作为integration framework？

---

### 🤖 Assistant

# 为什么推荐用 Streamline 做集成框架？

一句话回答：**因为 Streamline 不是"又一个 upscaler"，而是 NVIDIA 官方为引擎/游戏定义的那一层"接入框架"**——它把 NGX 各特性插件化，并把「设备/交换链拦截、帧调度、资源生命周期、安全、OTA、跨 API」这些**与画质无关但极易翻车的脏活**全部封装好了。你自己写 NGX 封装，工作量 90% 花在重造这些轮子上，而不是花在画质上。

下面逐条说明。

---

## 1. 决定性理由：FG 在架构上"只能"走 Streamline

这是最硬的一条。

| | NGX SDK 直连 | Streamline |
|---|---|---|
| DLSS Super Resolution | ✅ 可做 | ✅ 可做 |
| **DLSS Frame Generation** | ❌ **不提供** | ✅ 唯一官方途径 |
| Reflex | ❌ 需自己接 NVAPI | ✅ 内建，且 FG **强制依赖** |

FG 不是"一个后处理成对插帧"那么简单，它要求：

1. **拦截 `Present`/`ResizeBuffers`/`CreateSwapChain`**，自己接管交换链节奏（pacer）；
2. **强制搭配 Reflex**（帧调度/延迟控制），否则 FG 质量与延迟都不可控；
3. 正确处理 **多帧 in-flight、backbuffer 轮转、resize/全屏切换**。

NVIDIA 明确要求 FG 必须经由 Streamline。SL 已经把这套东西实现成 **`sl.interposer` + FunctionHookID 拦截清单**：

```
CreateSwapChain / CreateSwapChainForHwnd / CreateSwapChainForCoreWindow
Present / Present1 / GetBuffer / GetDesc / ResizeBuffers / ResizeBuffers1
GetCurrentBackBufferIndex / SetFullscreenState / ID3D12Device::CreateCommandQueue
```

你如果自研 hook，本质上就是**从零重写这份清单 + 所有边界情况**。而 SL 还提供了 **Manual Hooking（`eUseManualHooking`）**：你把 hook 接到 SL，用 `slUpgradeInterface` 把 native 对象换成 SL proxy——**你要的"劫持 swapchain"能力，SL 已经把它变成"接线"而不是"造轮子"**。

> 换言之：即便你想自己 hook（你上一问的方向），**底层 FC/FG 仍然调 SL**；SL 在这里既是 runtime 又是你的 hook 框架。

---

## 2. SL 到底替你干了哪些"脏活"

| 职责 | 自研 NGX 封装要做的 | Streamline |
|---|---|---|
| 设备/队列获取 | 自己维护 device/queue/cmdlist，处理多队列 | `slSetD3DDevice` + `slGetNativeInterface`/`slUpgradeInterface` |
| 交换链拦截 | 手写 DXGI/D3D12/VK hook，处理 resize/全屏/多窗口 | interposer 内建，或 manual hooking 复用 |
| 常量管理 | 手动乘 jitter、矩阵转置、单位换算、每帧 push | `sl::Constants` + `slSetConstants` |
| 资源 Tag / barrier | 手动 transition、跟踪每张纹理状态 | `slSetTag`/`slSetTagForFrame`，SL 自己管 barrier |
| 多帧 in-flight | 自己维护 frame ring、生命周期 | `sl::FrameToken` 内建 |
| FG 帧配对 | 需要精确 present 打点才能对齐生成帧 | SL 与 Reflex marker 自动配对 |
| 特性开关/降级 | 自己探测 `slIsFeatureSupported` 等价物 | 统一 `slIsFeatureSupported`/`slDLSSGGetState` |
| OTA/版本升级 | 重新集成、重新编译、重新测 | `eAllowOTA` + `eLoadDownloadedPlugins` 运行时换 DLL |
| 安全 | 自己做签名校验 | `sl::security::verifyEmbeddedSignature` + NVIDIA 证书链 |

第 2、3、4 行——**swapchain 拦截 + 资源状态 + 帧生命周期**——才是集成 FG 真正的难点，也是最容易出"偶发崩溃/显存泄漏/白屏"的地方。SL 把这些收敛成一组稳定 API。

---

## 3. 统一抽象：一套 API 覆盖所有 NGX 特性

SL 把不同 NGX 能力统一成「`slInit` 加载插件 → 每帧 tag → `slEvaluateFeature`」的同一套范式：

- `sl.dlss`（SR）
- `sl.dlss_g`（FG，含 MFG 多帧生成）
- `sl.reflex`（低延迟）
- `sl.deepdvc`（去色带/压缩增强）
- `sl.nis`（非 NGX 的空间放大/锐化）
- Ray Reconstruction 等后续特性

好处：

- **加新特性 ≠ 改集成层**，常常只是多加载一个插件、多设几个 tag；
- 常量（矩阵、jitter、相机参数）**一次设好，全特性共用**；
- 你上一问里 SR 与 FG 共用 Hudless/Depth/Mvec 的 tag，正是这套设计的体现。

如果直连 NGX，每个特性是**独立 DLL + 独立接口 + 独立生命周期**，你要写 N 套胶水。

---

## 4. 工程可持续性：动态加载 / OTA / ABI 稳定

- **动态加载**：按需 `LoadLibrary` 各插件，未支持特性的机器不加载，减少体积与初始化风险。
- **OTA**：`eAllowOTA` + `eLoadDownloadedPlugins` 让驱动侧更新模型/插件而**无需你重新发版**。DLSS 模型迭代很快，这一条对长线运营产品是刚需。
- **ABI 稳定**：SL 用稳定的 C 接口 + 版本宏（`sl::kSDKVersion`）隔开你的代码与底层 NGX，升级 SL 时上层改动小。
- **审计**：统一日志（`sl.log`）、`SL_IMGUI` 可视化，让你能**一眼看到 tag 是否正确、矩阵是否合理、present 是否对齐**——自研几乎无法短期复制这套调试能力。

---

## 5. 安全合规

SL 自带 `verifyEmbeddedSignature`（签名链校验 `sl.interposer.dll`），这是**发布 DLSS/FG 的合规前提**。自研要么忽略（有被替换/篡改风险），要么自己实现一套等价的签名校验。

---

## 6. 跨平台 / 跨图形 API

SL 抽象了 **D3D12 / Vulkan**（Windows/Linux）。你可以用同一套上层代码走两条后端；自研 hook 要为每种 API 写一份拦截与调度逻辑。

---

## 7. 面向未来：DLSS 在快速演进

DLSS 的节奏是「SR → FG → MFG → RR → 持续换模型」。SL 让你**始终站在官方支持的接入面上**；而 NGX 直连是一条 NVIDIA 会不断弱化、且**从设计上就不覆盖 FG** 的路径。选 SL 等于把"未来新特性"的成本降到最低。

---

## 8. 诚实地讲：什么时候"可以不"用 Streamline

不是所有情况都必须：

| 场景 | 可以用 NGX 直连吗 | 建议 |
|---|---|---|
| 只要 SR、无 FG、Windows/DX12 | ✅ 可以 | 但 SL 仍是更省心的等价选择 |
| 引擎完全自研、已自建 swapchain 接管与帧调度 | ✅ 可行 | 你已有基础设施，NGX 直连可减依赖 |
| 只要 FG | ❌ 必须 SL | 无选择 |
| 要 DSLS-RR / MFG / 未来特性 | ❌ 只能 SL | 无选择 |
| 平台/发行限制不允许挂交换链 hook | 视情况 | SL 也支持 manual hooking 以便你控时序 |

即便在前两种"可以不用"的情况，SL 依旧提供统一常量、资源 barrier、调试工具与 OTA——**除非你有非常强的理由（极端体积/依赖控制、已有等价框架），否则没有理由自己重写**。

---

## 9. 对比总表

| 维度 | 直连 NGX | 自研 hook + NGX | **Streamline** |
|---|---|---|---|
| 支持 FG | ❌ | ⚠️ 需绕，官方不支持 | ✅ |
| 交换链/present 处理 | 无 | 全部自研 | ✅ 内建（或 manual hooking 复用）|
| 多特性统一 | ❌ 各写一套 | ❌ | ✅ 一套 API |
| 资源 barrier / 生命周期 | 手动 | 手动 | ✅ 托管 |
| 调试工具 | 无 | 自研 | ✅ sl.log / SL_IMGUI |
| OTA 换模型 | ❌ | ❌ | ✅ |
| 安全校验 | ❌ | 自研 | ✅ |
| 未来特性成本 | 高 | 高 | 低 |
| 与你已有 vtable hook 的关系 | 无关 | 需从零实现拦截 | **可把 hook 接到 SL（`slUpgradeInterface`）** |

---

## 小结（给你的项目）

你用 Unity 6 + 自研 vtable hook 的路线，**最优解不是"用 hook 取代 SL"，而是"用 hook 服务于 SL"**：

1. **SR**：SL 抽象层直取，不必碰 NGX。
2. **FG**：`eUseManualHooking` + `slUpgradeInterface`，让 SL 复用你已建立的 hook 能力（你负责 hook/转发，SL 负责 pacer/FG/Reflex/配对）。
3. **UI/Reflex**：用 SL 的 Hudless/UIAlpha tag + marker 体系，避免自己定义帧对齐协议。

这样你的工程量集中在**唯一真正需要你介入的地方——Unity/URP 的接入点与 UI 分离**，而把所有已被 NVIDIA 解决的基础设施问题交给 SL。

需要的话，我可以进一步给一份 **「你的 vtable hook ↔ SL manual hooking」最小接线图/代码**，标出哪些必须转发给 proxy、哪些必须走 native，以及 `presentCommon()` 的验收步骤。
# UE4注入Hook DXGI抓帧发OBS虚拟摄像头零拷贝无UI实现

## 🧑‍💻 User

ue4如何hookswapchain直接抓帧缓冲区通过obs camera发送过去录屏实现无hud/ui画面+zerocopy+，怎么写camera插件

---
## 😀 默认助手 (7)

# UE4 → OBS 抓帧管线：先说结论

你要的四个目标里有一个前提是**不成立**的，先把它纠正掉，否则后面代码写出来一定白费：

> ❝ hook swapchain（`Present`）直接拿 backbuffer = 无 HUD/UI ❞ —— **做不到**。

UE4 的 `FSlateRHIRenderer` / `AHUD::DrawHUD` 是**在 `Present` 之前往同一张 backbuffer 上叠画**的。`Present` 时刻的 backbuffer 必然已经包含 UMG + HUD。所以"无 UI + 零拷贝"要换抓帧点，不是换 hook 方式。

## 抓帧点对比

| 抓帧点 | 含 UI? | 零拷贝 | 需要注入 | 备注 |
|---|---|---|---|---|
| `IDXGISwapChain::Present/Present1` backbuffer | **有** | 可做到 | 是 | 只能拿"玩家看到的画面" |
| `ISceneViewExtension::PostRenderViewFamily_RenderThread` | **无** | 是 | 否（引擎模块） | **推荐**：此时 HUD/Slate 还没画 |
| tonemap 后 RDG pass（`SubscribeToPostProcessingPass`） | 无 | 是 | 否 | 最可控，兼容 UE5 RDG |
| `IDXGIOutputDuplication` | 有 | 否 | 否 | 抓的是桌面合成结果，最差 |
| `SceneCapture2D` + Spout | 无 | 是 | 否 | 最省事，但多一次场景渲染 |

**要点**：`UGameViewportClient::Draw()` 的顺序是 `DrawScene()` → `Renderer->RenderViewFamily()`（这里面触发 `PostRenderViewFamily_RenderThread`）→ 之后才 `DrawHUD()` / Slate 上屏。所以"In‑engine 的 ViewExtension"是唯一既零拷贝又天然无 UI 的干净抓帧点。

---

<details>
<summary><b>一、推荐架构（无注入 / 引擎内插件）：ViewExtension → 共享纹理 → OBS Source</b></summary>

### 数据流

```
UE4 RenderThread
  PostRenderViewFamily_RenderThread
    └─ RHICmdList.CopyTexture(ViewFamily.RenderTarget->GetRenderTargetTexture(),
                              MySharedRHITex)      // GPU 内 blit，无 CPU 回读
    └─ Signal keyed mutex / fence

共享资源：命名 NT handle 纹理 D3D11_TEXTURE2D (B8G8R8A8 + MISC_SHARED_NTHANDLE | KEYEDMUTEX)
          + 命名 FileMapping 元数据 {pid, handles[], width, height, writeIndex, frameNo}

OBS 进程
  ue4_camera source
    └─ DuplicateHandle → gs_texture_open_shared()
    └─ video_render: gs_draw_sprite()   // 采样，不拷贝
```

"零拷贝"准确定义：**没有任何 `Map`/`ReadPixels`/`memcpy`，全链路只在 GPU 上走 `CopyResource` + 采样**。`CopyResource` 本身约 0.3–0.8ms@4K，别指望绝对零成本；真正的零成本只有"直接把 UE 渲染目标当纹理用"（不可行，因为跨进程必须共享资源）。

### UE4 侧：共享纹理对象

```cpp
// SharedFrame.h
#pragma once
#include "CoreMinimal.h"
#include "RHIDefinitions.h"
#include "RenderResource.h"
#include <d3d11_1.h>

struct FSharedFrame
{
    TRefCountPtr<ID3D11Texture2D>  Native;
    TRefCountPtr<IDXGIKeyedMutex>  Mutex;
    HANDLE                         SharedHandle = nullptr; // NT handle
    FTexture2DRHIRef               RHITex;
    uint32                         W = 0, H = 0;

    void Create(ID3D11Device* Dev, uint32 InW, uint32 InH, const FString& Name);
    void Release();
};
```

```cpp
// SharedFrame.cpp
#include "ID3D11DynamicRHI.h"   // 取 RHICreateTexture2DFromResource
#include "Windows/AllowWindowsPlatformTypes.h"
#include <d3d11_1.h>
#include "Windows/HideWindowsPlatformTypes.h"

void FSharedFrame::Create(ID3D11Device* Dev, uint32 InW, uint32 InH, const FString& Name)
{
    Release();
    W = InW; H = InH;

    D3D11_TEXTURE2D_DESC D = {};
    D.Width  = InW;
    D.Height = InH;
    D.MipLevels = 1;
    D.ArraySize = 1;
    D.Format = DXGI_FORMAT_B8G8R8A8_UNORM;   // HDR 时换 R10G10B10A2_UNORM
    D.SampleDesc.Count = 1;
    D.Usage = D3D11_USAGE_DEFAULT;
    D.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    // 注意：SHARED_NTHANDLE 不能和 legacy SHARED 同时用；与 KEYEDMUTEX 可共存
    D.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE
                | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    TRefCountPtr<ID3D11Texture2D> T;
    check(SUCCEEDED(Dev->CreateTexture2D(&D, nullptr, T.GetInitReference())));

    TRefCountPtr<IDXGIResource1> Res;
    check(SUCCEEDED(T->QueryInterface(IID_PPV_ARGS(Res.GetInitReference()))));
    // 命名 handle：OBS 侧可 OpenSharedResourceByName，免去跨进程 DuplicateHandle
    check(SUCCEEDED(Res->CreateSharedHandle(
        nullptr, GENERIC_ALL, *Name, &SharedHandle)));

    check(SUCCEEDED(T->QueryInterface(IID_PPV_ARGS(Mutex.GetInitReference()))));

    Native = T;
    // 包成 RHI 纹理（函数名随版本可能不同，见文末“版本差异”）
    RHITex = static_cast<FD3D11DynamicRHI*>(GDynamicRHI)
        ->RHICreateTexture2DFromResource(
            PF_B8G8R8A8,
            TexCreate_RenderTargetable | TexCreate_ShaderResource,
            FClearValueBinding::Black,
            Native);
}
```

### UE4 侧：ViewExtension 抓帧

```cpp
#include "SceneViewExtension.h"
#include "SceneView.h"

class FCaptureViewExtension : public FSceneViewExtensionBase
{
public:
    FCaptureViewExtension(const FAutoRegister& A) : FSceneViewExtensionBase(A) {}

    virtual bool IsActiveThisFrame_Internal(
        const FSceneViewExtensionContext& Ctx) const override { return bEnabled; }

    virtual void SetupViewFamily(FSceneViewFamily& VF) override
    {
        // 分辨率变化时在这里记下来，render thread 里重建
        if (VF.RenderTarget)
        {
            PendingW = VF.RenderTarget->GetSizeXY().X;
            PendingH = VF.RenderTarget->GetSizeXY().Y;
        }
    }

    virtual void PostRenderViewFamily_RenderThread(
        FRHICommandListImmediate& RHICmdList, FSceneViewFamily& VF) override
    {
        if (!bEnabled || !VF.RenderTarget) return;
        // 只抓主视口，别抓编辑器缩略图 / 阴影渲染
        if (PendingW < 320 || PendingH < 240) return;

        FRHITexture* Src = VF.RenderTarget->GetRenderTargetTexture();
        if (!Src) return;

        if (!Shared.RHITex || Shared.W != PendingW || Shared.H != PendingH)
            Recreate(/* 见下 */);

        // 0 = 写锁，超时 4ms；OBS 侧用 1 读锁
        if (Shared.Mutex && Shared.Mutex->AcquireSync(0, 4) == S_OK)
        {
            RHICmdList.CopyTexture(Src, Shared.RHITex, FRHICopyTextureInfo());

            // D3D11 非线程化 RHI 下可直接 flush + Release；
            // DX12 / RHI 线程化必须走 EnqueueLambda 把释放排进队列（见“坑”）
            RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
            Shared.Mutex->ReleaseSync(1);
        }

        PublishFrame(); // 更新 FileMapping 里的 writeIndex / frameNo
    }

private:
    bool  bEnabled = false;
    uint32 PendingW = 0, PendingH = 0;
    FSharedFrame Shared;
};
```

注册（放在你插件模块的 `StartupModule`）：

```cpp
CaptureExt = FSceneViewExtensions::NewExtension<FCaptureViewExtension>();
```

> 这条路 `PostRenderViewFamily_RenderThread` 拿到的是**已完成 post‑process、已 tonemap 的 LDR 画面，但没有 HUD / 没有 UMG / 没有 stat 叠层**——正好是你要的"干净画面"。
> 唯一例外：如果游戏把 UI 画在 `UCanvas` 且通过 `FCanvas` 在 `RenderViewFamily` 之前绘制（少见），那就改用 tonemap 后的 RDG pass。

### tonemap 后 pass（UE5 更推荐，UE4.27 也支持）

```cpp
virtual void SubscribeToPostProcessingPass(
    EPostProcessingPass Pass,
    FAfterPassCallbackDelegateArray& OutCBs,
    bool bEnabled) override
{
    if (Pass == EPostProcessingPass::Tonemap)
        OutCBs.Add(FAfterPassCallbackDelegate::CreateRaw(
            this, &FCaptureViewExtension::OnTonemap));
}

FScreenPassTexture OnTonemap(FRDGBuilder& GB, const FSceneView& View,
                             const FPostProcessMaterialInputs& In)
{
    FScreenPassTexture Scene = In.GetInput(EPostProcessMaterialInput::SceneColor);
    FRDGTextureRef Dst = GB.RegisterExternalTexture(SharedPooledRT);
    AddCopyTexturePass(GB, Scene.Texture, Dst);
    return Scene;
}
```

</details>

<details>
<summary><b>二、OBS 侧：写一个 camera（source）插件</b></summary>

### 最短路径：async source + `frame.handle`（libobs 帮你做纹理）

`struct obs_source_frame` 末尾有 `uint32_t handle;` 字段（注释就是 *used for async video frame handling*）。把它填成共享纹理 handle，libobs 内部会 `gs_texture_open_shared()`：

```c
static void push(struct ue4_camera *ctx)
{
    struct obs_source_frame f = {0};
    f.width  = ctx->w;
    f.height = ctx->h;
    f.format = VIDEO_FORMAT_BGRA;
    f.timestamp = os_gettime_ns();
    f.handle = (uint32_t)(uintptr_t)ctx->dup_handle; // 已 DuplicateHandle 到 obs 进程
    f.flags  = OBS_SOURCE_FRAME_LINEAR_ALPHA;
    obs_source_output_video2(ctx->source, &f);
}
```

### 完全可控路径：`OBS_SOURCE_CUSTOM_DRAW`

```c
#include <obs-module.h>
#include <graphics/graphics.h>
#include <windows.h>

struct ue4_camera {
    obs_source_t *source;
    gs_texture_t *tex;
    uint32_t      w, h;
    HANDLE        map;          // FileMapping 视图
    struct shm   *shm;
    HANDLE        dup[3];
};

static const char *ue4_name(void *unused)
{ return obs_module_text("UE4Camera"); }

static void *ue4_create(obs_data_t *s, obs_source_t *src)
{
    struct ue4_camera *c = bzalloc(sizeof(*c));
    c->source = src;
    open_handshake(c);   // 打开 Local\UECapture_v1
    return c;
}

/* video_tick：OBS 的节拍驱动，只在这里做握手/重开，绝不在 render 里做 */
static void ue4_tick(void *data, float seconds)
{
    struct ue4_camera *c = data;
    if (!c->shm) { open_handshake(c); return; }

    if (c->shm->width != c->w || c->shm->height != c->h)
        reopen_textures(c);                 // 分辨率/设备重建

    c->write_index = c->shm->write_index;   // 只读，一行
}

static void ue4_render(void *data, gs_effect_t *effect)
{
    struct ue4_camera *c = data;
    if (!c->tex) return;

    gs_effect_t *e = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_eparam_t *img = gs_effect_get_param_by_name(e, "image");
    gs_effect_set_texture_srgb(img, c->tex);

    while (gs_effect_loop(e, "Draw"))
        gs_draw_sprite(c->tex, 0, c->w, c->h);
}

struct obs_source_info ue4_camera_info = {
    .id           = "ue4_camera",
    .type         = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW | OBS_SOURCE_SRGB,
    .get_name     = ue4_name,
    .create       = ue4_create,
    .destroy      = ue4_destroy,
    .update       = ue4_update,
    .get_width    = ue4_width,
    .get_height   = ue4_height,
    .video_tick   = ue4_tick,
    .video_render = ue4_render,
};
```

### 跨进程拿到纹理：两种方式

```c
/* A. DuplicateHandle（需要生产进程 pid，OpenProcess 要有 PROCESS_DUP_HANDLE） */
HANDLE game = OpenProcess(PROCESS_DUP_HANDLE, FALSE, c->shm->pid);
DuplicateHandle(game, (HANDLE)c->shm->handles[i], GetCurrentProcess(),
                &c->dup[i], 0, FALSE, DUPLICATE_SAME_ACCESS);
c->tex = gs_texture_open_shared((uint32_t)(uintptr_t)c->dup[i]);

/* B. 命名 handle：OpenSharedResourceByName（需自己拿 libobs 的 ID3D11Device） */
```

> ⚠️ `gs_texture_open_shared()` 的签名在 libobs 各版本里是 `uint32_t handle`，所以 `DuplicateHandle` 路线最稳（handle 值实际都是小整数，截断无害）。命名 handle 路线只有在你能拿到 libobs 设备对象时才好走。

### 共享内存头

```c
struct shm {
    volatile uint32_t magic;        // 'UE4C'
    volatile uint32_t pid;
    volatile uint32_t width, height, format;
    volatile uint32_t num_buffers;  // 3
    volatile uint32_t write_index;  // 生产者写完的槽
    volatile uint32_t read_index;   // 消费者正在渲染的槽
    volatile uint32_t frame_no;
    volatile uint64_t handles[3];   // 生产进程内的 NT handle 值
};
```

用 `Local\UECapture_v1`（同 session 同用户即可，别用 `Global\`，要额外权限）。

### CMake

```cmake
# 树外构建
find_package(libobs REQUIRED)
find_package(w32-pthreads REQUIRED)
add_library(ue4-camera MODULE ue4-camera.c)
target_link_libraries(ue4-camera libobs w32-pthreads d3d11 dxgi)
```

树内：直接丢进 `obs-studio/plugins/`，在 `plugins/CMakeLists.txt` 里 `add_subdirectory(ue4-camera)`，最省事。

</details>

<details>
<summary><b>三、同步：别用 keyed mutex 硬刚，用三缓冲无锁协议</b></summary>

同步方案按可靠性排序：

| 方案 | 适用 | 坑 |
|---|---|---|
| **三缓冲 + 共享内存索引** | D3D11 / D3D12 通吃 | 需要自己写 20 行协议，**推荐** |
| `IDXGIKeyedMutex` | 仅 D3D11 | **D3D12 不支持**；UWP 禁用；部分驱动上 acquire 会 stall 数个 ms |
| `ID3D11Fence` / `ID3D12Fence` + shared handle | D3D12 正解 | fence handle 跨进程需 `DuplicateHandle`；D3D11 的 `OpenSharedFence` 只吃 handle |
| `SetEvent` | 谁都行 | 会引入 CPU 参与，增加延迟 |

三缓冲协议（生产者/消费者各只写自己那个字段）：

```cpp
// 生产者（UE 渲染线程）
uint32 next = (shm->write_index + 1) % 3;
while (next == shm->read_index) next = (next + 1) % 3;  // 跳过读者正在用的槽
RHICmdList.CopyTexture(Src, Shared[next].RHITex, FRHICopyTextureInfo());
shm->write_index = next;
shm->frame_no++;

// 消费者（OBS video_tick）
uint32 idx = shm->write_index;
shm->read_index = idx;   // 立刻标记"这个槽我在用"
c->tex = c->texs[idx];
```

这样只有在 OBS 落后生产者 ≥2 帧时才会撞车，3 个槽足够。想更稳就 4 槽，或加 `frame_no` 单调性检查判定 stall 后重开。

**延迟建议**：别按 UE 的帧率推帧。让 OBS 的 `video_tick` 拉取最新槽（OBS 30fps 就只取到 30fps 的画面），这样录制/直播端不会因为游戏 200fps 而积压。

</details>

<details>
<summary><b>四、如果你必须走外部注入（Present hook）——以及它为什么拿不到无 UI 画面</b></summary>

### Present hook 骨架

```cpp
typedef HRESULT(STDMETHODCALLTYPE *PFN_Present1)(
    IDXGISwapChain1 *, UINT, UINT, const DXGI_PRESENT_PARAMETERS *);

static PFN_Present1 oPresent1 = nullptr;
static ID3D11DeviceContext *g_ctx = nullptr;
static ID3D11Texture2D *g_shared = nullptr;

HRESULT STDMETHODCALLTYPE hkPresent1(IDXGISwapChain1 *sc, UINT sync, UINT flags,
                                     const DXGI_PRESENT_PARAMETERS *pp)
{
    InitOnce(sc);                       // 取 device/context，建共享纹理

    ID3D11Texture2D *bb = nullptr;
    if (SUCCEEDED(sc->GetBuffer(0, IID_PPV_ARGS(&bb))))
    {
        g_ctx->CopyResource(g_shared, bb);   // 纯 GPU，同 device 同 adapter
        bb->Release();
    }
    return oPresent1(sc, sync, flags, pp);
}
```

安装用 MinHook + dummy swapchain 换 vtable：

```cpp
// Present = index 8, Present1 = index 22（flip model 下 UE4 走 Present1）
DXGI_SWAP_CHAIN_DESC d = {};
d.BufferCount = 1; d.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
d.OutputWindow = hHidden; d.SampleDesc.Count = 1; d.Windowed = TRUE;
d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
D3D11CreateDeviceAndSwapChain(...);
void **vtbl = *(void ***)pSwap;
MH_CreateHook(vtbl[22], &hkPresent1, (void **)&oPresent1);
```

### 但是：UI 无解，除非这么做

`Present` 时的 backbuffer 已经过 Slate 合成。外部注入下想拿无 UI 画面只有三条路，**全是启发式或脏 hack**：

1. **hook `ID3D11DeviceContext::OMSetRenderTargets`**，检测"RT 切到 backbuffer 且 depth 解绑"当作 UI 阶段开始，在那一刻抢拍 backbuffer 副本 → 脆弱，UI 一变就崩。
2. **hook `DrawIndexed` 抓最后几个全屏 quad** → 更脆弱。
3. **让游戏自己给干净画面**（正解）：注入后找 `FSceneViewport` / 用引擎 cvar 把 HUD/UMG 关掉（`MyHUD->bShowHUD=false`、UMG `RemoveFromParent()`、`stat none`）→ 但会改玩家画面。
4. **hook `FD3D11DynamicRHI::RHIResolveSceneColor`** 抓 SceneColor（无 UI）→ 需要引擎符号/偏移，版本强耦合。

**结论**：外部注入做"完整画面"很简单，做"无 UI 画面"不划算。既然目标是 UE4，就用第一节的引擎内模块，0 注入、0 符号偏移。

</details>

<details>
<summary><b>五、坑清单（这些比代码更容易翻车）</b></summary>

| 类别 | 坑 | 对策 |
|---|---|---|
| 设备 | UE 用独显、OBS 用核显 → 共享纹理开不出来 | Windows 图形设置里强制同一 adapter；UE `-d3d11adapter=N` |
| 设备 | 独占全屏 Alt‑Tab → device removed，handle 失效 | 监听 `frame_no` 停滞 + 定时重连；推荐用 borderless |
| 设备 | UE 跑 DX12 / Vulkan | `IDXGIKeyedMutex` 在 D3D12 不存在，必须换 fence；Vulkan 要 `VK_KHR_external_memory_win32`，完全是另一套 |
| 格式 | HDR 游戏 | UE 输出 `R10G10B10A2_UNORM` + PQ，OBS 侧要 `obs_source_set_...` HDR 属性 / 用 `VIDEO_FORMAT_P010`；最省事是在引擎里 tonemap 回 SDR |
| 格式 | sRGB 变暗/发灰 | 共享纹理用 `_UNORM`，OBS 侧 `gs_effect_set_texture_srgb`；UE 侧别开 `TexCreate_SRGB` |
| 帧尺寸 | 动态分辨率 / `r.ScreenPercentage` 变化 | `SetupViewFamily` 里比较尺寸，变了就重建（重建期间丢帧） |
| 帧尺寸 | DPI scaling 导致 1080p 抓成 720p | 用 `RenderTarget->GetSizeXY()`，别用窗口尺寸 |
| 线程 | DX12 下在 RenderThread 直接调原生 `AcquireSync` → 与批处理命令乱序 | 改用 `RHICmdList.EnqueueLambda()` 把原生命令排进 RHI 队列 |
| 性能 | 每帧 `CopyResource` 4K ≈ 0.5ms | 只在 OBS 真的在录时才抓；`IsActiveThisFrame_Internal` 返回 false 就整段跳过 |
| OBS | `video_render` 里做 OpenProcess/DuplicateHandle → 每帧 stall | 握手/重开只放在 `video_tick` |
| 权限 | `Global\` 命名对象要 `SeCreateGlobalPrivilege` | 用 `Local\` |

**版本差异速查**（写代码前先 `grep` 你的引擎）：

| API | UE4.26/4.27 | UE5.x |
|---|---|---|
| `PostRenderViewFamily_RenderThread` | `(FRHICommandListImmediate&, FSceneViewFamily&)` | `(FRDGBuilder&, FSceneViewFamily&)` |
| `SubscribeToPostProcessingPass` | 有（无 `FSceneView` 参数） | 有 |
| `RHICreateTexture2DFromResource` | `FD3D11DynamicRHI` 上有 | 可能改名/移除，改用 RDG external texture |
| `FRHICommandList::CopyTexture` | 有 | 改 `AddCopyTexturePass(GB, ...)` |

</details>

<details>
<summary><b>六、如果你只是想"尽快跑通"：Spout2 路线（半天搞定，别自己写）</b></summary>

如果你的目标只是"干净画面进 OBS"，不要自己写共享纹理协议：

1. UE4 侧装 **Spout-UE4** 插件，注册一个 `UTextureRenderTarget2D` 作为 sender。
2. 那个 RT 用 `ISceneViewExtension`（第一节代码）或 `USceneCaptureComponent2D`（`CaptureSource = SCS_FinalColorLDR`）填满 → **天然无 UI**（SceneCapture 不包含 Slate/HUD）。
3. OBS 侧装 **obs-spout2-plugin**，选你的 sender。

Spout 内部就是 `MISC_SHARED_NTHANDLE` + 命名共享纹理，和你自己写的东西一模一样，只是别人已经踩完坑了。代价是 SceneCapture 会**多跑一次场景渲染**（重 → 用 ViewExtension 替代）；Spout 对 HDR / 10bit 支持也一般。

### 如果你最终目的是"游戏当摄像头用"

注意 OBS **Virtual Camera 是输出设备**，它自己会做一次 NV12 共享内存拷贝（`obs-virtualcam` 内部 `memcpy` 到 named shared memory），**这一段永远不可能零拷贝**。所以链条是：

```
UE4 GPU 纹理 ──零拷贝──► OBS source ──OBS 合成──► Virtual Cam（必然有一次 CPU 拷贝）
```

想省掉最后那一次，只能不经过 OBS，直接写一个 **DirectShow / Media Foundation 虚拟摄像头**，把共享纹理在自家 D3D 设备上采样后喂给 NV12 输出——那是另一个项目了。

</details>

---

## 落地顺序建议

1. **先验证抓帧点**：写个最小插件，在 `PostRenderViewFamily_RenderThread` 里 `CopyTexture` 到你自己的 RHI RT，然后 `RHICmdList.ReadSurfaceData` 落 PNG → 确认画面干净、无 UI、无 stat。**这一步不通过，别往下做。**
2. **换成共享纹理**：把 RT 换成 `MISC_SHARED_NTHANDLE` 的 D3D11 纹理，写 `Local\` FileMapping 握手。
3. **OBS 插件**：先用 `frame.handle` async 路线跑通（代码最少），再换 custom draw 做色彩/暂停/分辨率控制。
4. **同步**：上三缓冲，压延迟测抖动。
5. **最后**才考虑外部 Present hook（如果要做第三方游戏）。

需要我把其中某一段展开成可直接编译的完整工程（比如 `ISceneViewExtension` + 共享纹理那块，或者 OBS 插件的 `CMakeLists` + 握手细节），告诉我目标 UE 版本和 OBS 版本，我按你的版本对齐 API。

---
## 🧑‍💻 User

1. 我们是没有游戏源码的，你需要通过注入d3d的方式获取backbuffer（参考rdc抓帧）
2. dxgi api具体怎么操作、注入时机具体是怎么样的

---
## 😀 默认助手 (7)

# 无源码注入抓帧：DXGI 链路 + 注入时机全流程

上一轮说的"引擎内 ViewExtension"这条路你走不了，所以抓帧点必须落在 **dxgi/d3d11 的 API 边界**上。下面按 4 层讲：vtable 结构 → 注入时机 → 挂载点 → Present 里到底调什么 DXGI API。

---

## 0. 先给结论（4 个挂载点的时序）

```
进程启动
 │
 ├─[T1] 加载器解析 import 表 → LoadLibrary("d3d11.dll")      ← 代理 DLL 在这里夺权
 │       └─ 我们立刻 MH_Initialize + hook 导出函数
 │
 ├─[T2] UE4: FEngineLoop::PreInit → RHIInit → D3D11CreateDevice()
 │       └─ hook 命中 → 拿到 ID3D11Device / ID3D11DeviceContext
 │                      顺带 GetAdapter→GetParent 拿 IDXGIFactory2
 │
 ├─[T3] UE4: FD3D11DynamicRHI::CreateViewport →
 │       IDXGIFactory2::CreateSwapChainForHwnd(...)
 │       └─ hook 命中 → 拿到 IDXGISwapChain1* → 判定是不是主视口 → hook 它的 vtable
 │
 └─[T4] 每帧 IDXGISwapChain::Present / Present1
         └─ hook 命中 → GetBuffer → CopyResource → 共享纹理 → OBS
```

**关键认知：hook 的是"函数地址"，不是"对象"。** `dxgi.dll` 里 `CDXGISwapChain::Present` 只有一份实现，所有 swapchain 实例的 vtable 第 8 项都指向它。所以只要你 patch 了那个函数地址，**游戏自己创建的 swapchain 走到 Present 也会进你的钩子**——这正是"晚注入也能抓到"的原理，也是"必须晚注入时用 dummy swapchain 取地址"的原理。

---

<details open>
<summary><b>1. 必须背下来的两张 vtable 表（这是"DXGI API 怎么操作"的地基）</b></summary>

COM 接口的 vtable 顺序 = 头文件里的声明顺序，派生接口在后面追加。**验证方法**：拿一个真实对象，`*(void***)obj` 打出来，和 `dxgi.h` 的声明顺序数一遍即可。

### `IDXGISwapChain` / `1` / `2` / `3`

| slot | 函数 | 备注 |
|---|---|---|
| 0–2 | QueryInterface / AddRef / Release | |
| 7 | `GetDevice` | 拿 device |
| **8** | **`Present`** | **核心挂载点** |
| 9 | `GetBuffer` | 拿 backbuffer |
| 10/11 | SetFullscreenState / GetFullscreenState | |
| 12 | `GetDesc` | BufferCount / SwapEffect / Format |
| **13** | **`ResizeBuffers`** | **必须 hook：分辨率变化** |
| 14 | `ResizeTarget` | 独占全屏切分辨率 |
| 16 | `GetFrameStatistics` | 判断是否真的 present 了 |
| 18 | `GetDesc1` | 含 `Flags`（`ALLOW_TEARING` 等） |
| 20 | `GetHwnd` | **过滤主视口用** |
| **22** | **`Present1`** | **flip model 下 UE4 走这个** |
| **36** | **`GetCurrentBackBufferIndex`** | **flip model 必须用它！** |
| 37/38 | CheckColorSpaceSupport / SetColorSpace1 | HDR 判定 |
| 39 | `ResizeBuffers1` | D3D12 用 |

> ⚠️ flip model（`FLIP_DISCARD`，UE4 默认）下，backbuffer 会轮转，**`GetBuffer(0)` 拿到的可能不是当前正在呈现的那张**。必须用 `IDXGISwapChain3::GetCurrentBackBufferIndex()`。

### `IDXGIFactory` / `1` / `2`

| slot | 函数 |
|---|---|
| 8 | `MakeWindowAssociation`（**别覆盖**，UE4 设了 `MWA_NO_ALT_ENTER`） |
| 10 | `CreateSwapChain` |
| **15** | **`CreateSwapChainForHwnd`** ← UE4 PC 主路径 |
| 16 | `CreateSwapChainForCoreWindow`（UWP） |
| 26 | `CreateSwapChainForComposition`（合成用，跳过） |

### `ID3D11DeviceContext`（抓"无 UI 画面"用）

| slot | 函数 |
|---|---|
| 12 / 13 | `DrawIndexed` / `Draw` |
| 20 / 21 | `DrawIndexedInstanced` / `DrawInstanced` |
| 33 | `OMSetRenderTargets` |
| **46** | **`CopySubresourceRegion`** ← 场景→backbuffer 的 blit |
| **47** | **`CopyResource`** |
| **57** | **`ResolveSubresource`** ← MSAA 情况下是它 |

> 派生接口（`ID3D11DeviceContext4`）继承槽位的函数地址和基接口**完全一样**，所以从 `ID3D11DeviceContext` vtable 取地址去 patch，对所有版本调用都生效。

</details>

---

<details open>
<summary><b>2. 注入时机：为什么必须"早"，晚注入怎么补救</b></summary>

### 参考 RenderDoc 的做法（这点很重要）

RenderDoc **不能 attach 已经跑起来的 D3D 程序**——它点"capture"时会**把程序重启**。原因就是上面说的：它必须在 `d3d11.dll` / `dxgi.dll` 被加载、device 被创建**之前**就在进程里，才能 hook 到 `CreateDevice` / `CreateSwapChain`。

RenderDoc 的注入方式是三件套 **代理 DLL**：在游戏目录放 `d3d11.dll` / `dxgi.dll` / `d3dcompiler_47.dll`，转发导出到系统真 DLL。UE4 游戏同理。

### 三种方案对比

| 方案 | 时机 | 可靠性 | 适用 |
|---|---|---|---|
| **A. 代理 DLL**（`d3d11.dll` 放游戏目录） | T1，最早 | 最高，无需注入器 | **推荐**，只要你控制游戏目录 |
| **B. `CreateProcess(CREATE_SUSPENDED)` + `LoadLibraryW` 远程线程** | T1 | 高，但要自己启动游戏 | 自己启动时用 |
| **C. 晚注入**（`OpenProcess` + `CreateRemoteThread`） | T2 之后 | 中，需要 dummy swapchain 取地址 | 必须 attach 已运行进程时 |

### 方案 A：代理 DLL（最省事）

```cpp
// proxy/dllmain.cpp  —— 编译成 d3d11.dll 放进游戏目录，真 DLL 改名 d3d11_orig.dll
#include <windows.h>
#include <d3d11.h>

static HMODULE g_realD3D11 = nullptr;

static void LoadReal()
{
    WCHAR Sys[MAX_PATH];
    GetSystemDirectoryW(Sys, MAX_PATH);      // 注意：用系统目录绝对路径，
    wcscat_s(Sys, L"\\d3d11.dll");           // 否则又会加载到你自己（递归）
    g_realD3D11 = LoadLibraryW(Sys);
}

BOOL WINAPI DllMain(HMODULE h, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(h);
        LoadReal();
        // ❗不要在 DllMain 里做 hook / CreateThread 之后 wait，loader lock 会死锁
        //    开个线程或延迟到第一次导出调用时再初始化
        CreateThread(nullptr, 0, [](LPVOID) -> DWORD {
            InstallHooks();
            return 0;
        }, nullptr, 0, nullptr);
    }
    return TRUE;
}

extern "C" HRESULT WINAPI D3D11CreateDevice(
    IDXGIAdapter* pAdapter, D3D_DRIVER_TYPE DriverType, HMODULE Software,
    UINT Flags, const D3D_FEATURE_LEVEL* pLevels, UINT NumLevels,
    UINT SDKVersion, ID3D11Device** ppDevice,
    D3D_FEATURE_LEVEL* pFeatureLevel, ID3D11DeviceContext** ppContext)
{
    if (!g_realD3D11) LoadReal();
    auto fn = (decltype(&D3D11CreateDevice))
        GetProcAddress(g_realD3D11, "D3D11CreateDevice");
    HRESULT hr = fn(pAdapter, DriverType, Software, Flags, pLevels,
                    NumLevels, SDKVersion, ppDevice, pFeatureLevel, ppContext);
    if (SUCCEEDED(hr)) OnDeviceCreated(*ppDevice);   // T2
    return hr;
}
// D3D11CreateDeviceAndSwapChain 同理：顺带把 swapchain 也接到
```

剩下所有别的导出（`D3D11CoreCreateDevice`、`D3D11On12CreateDevice`…）用 `.def` 直接转发：

```
EXPORTS
    D3D11CoreCreateDevice      = d3d11_orig.D3D11CoreCreateDevice
    D3D11CreateDeviceAndSwapChain = d3d11_orig.D3D11CreateDeviceAndSwapChain
    D3D11On12CreateDevice      = d3d11_orig.D3D11On12CreateDevice
```

**踩坑点**：
- `d3d11.dll` / `dxgi.dll` **不在 KnownDLLs 里**，所以应用目录优先，代理能生效。
- 但如果有别的安全软件 / 启动器调了 `SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32)`，代理会失效 → 换 `version.dll` 或走方案 B。
- 带反作弊的游戏别碰。

### 方案 C：晚注入（进程已在跑）

```cpp
// 注入器
BOOL Inject(DWORD pid, const wchar_t* dllPath)
{
    HANDLE h = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    SIZE_T sz = (wcslen(dllPath) + 1) * sizeof(wchar_t);
    void* remote = VirtualAllocEx(h, nullptr, sz, MEM_COMMIT | MEM_RESERVE,
                                  PAGE_READWRITE);
    WriteProcessMemory(h, remote, dllPath, sz, nullptr);

    // kernel32 在所有同位数进程里基址相同，所以这条地址在目标进程也有效
    auto loadLib = (LPTHREAD_START_ROUTINE)
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW");
    HANDLE t = CreateRemoteThread(h, nullptr, 0, loadLib, remote, 0, nullptr);
    WaitForSingleObject(t, INFINITE);
    return true;
}
```

注入进去之后，**device 和 swapchain 都已经存在了**，`CreateDXGIFactory` / `D3D11CreateDevice` 再也不会被调用 → 你前面所有 hook 都白搭。

补救办法只有一条：**在目标进程内部造一个 dummy swapchain，从它的 vtable 取 `Present` 的真实地址，然后 patch。**

```cpp
// 在目标进程内执行
void HookPresentLate()
{
    // 造一个 1x1 的隐藏窗口 + dummy swapchain，只为拿 vtable 地址
    WNDCLASSEXW wc = { sizeof(wc) };
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"DummyWnd";
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowExW(0, L"DummyWnd", L"", WS_OVERLAPPEDWINDOW,
                                0, 0, 8, 8, nullptr, nullptr, wc.hInstance, nullptr);

    DXGI_SWAP_CHAIN_DESC d = {};
    d.BufferCount = 2;
    d.BufferDesc.Width = 8; d.BufferDesc.Height = 8;
    d.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    d.OutputWindow = hwnd;
    d.SampleDesc.Count = 1;
    d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    d.Windowed = TRUE;

    ID3D11Device* dev = nullptr; ID3D11DeviceContext* ctx = nullptr;
    IDXGISwapChain* dummy = nullptr;
    D3D_FEATURE_LEVEL fl;
    D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        0, nullptr, 0, D3D11_SDK_VERSION, &d, &dummy, &dev, &fl, &ctx);

    IDXGISwapChain1* dummy1 = nullptr;
    dummy->QueryInterface(IID_PPV_ARGS(&dummy1));
    void** vtbl = *(void***)dummy1;

    MH_CreateHook(vtbl[8],  &hkPresent,  (void**)&oPresent);   // Present
    MH_CreateHook(vtbl[22], &hkPresent1, (void**)&oPresent1);  // Present1
    MH_EnableHook(nullptr);

    // dummy 必须留着别 Release（否则 vtable 所在模块可能卸载），
    // 但也不要让它 Present 干扰——hkPresent 里用 sc != g_sc 过滤掉
}
```

> 注意 `htPresent` 里第一行必须 `if (sc != g_sc) return oPresent(...)`，否则你自己的 dummy swapchain 也会被钩到，造成递归/自捕获。

**同样要补挂 `ID3D11DeviceContext` 的 46/47/57 槽**——晚注入后这些 hook 只对"安装之后"的调用生效，但每帧都会调用，所以只是丢一帧，无所谓。

### 方案 B：suspended 启动

```cpp
STARTUPINFOW si = { sizeof(si) };
PROCESS_INFORMATION pi;
CreateProcessW(nullptr, cmdline, nullptr, nullptr, FALSE,
               CREATE_SUSPENDED, nullptr, workDir, &si, &pi);
Inject(pi.dwProcessId, L"C:\\hook\\dxhook.dll");
ResumeThread(pi.hThread);
```
比方案 C 干净（在 T1 注入，能走正常的导出 hook 路径），但要求你负责启动游戏。Steam 游戏可以用 `steam://rungameid/<id>` 无法 suspend，此时退化成 IFEO `Debugger` 或方案 A。

</details>

---

<details open>
<summary><b>3. 挂载点代码：导出 → Factory → SwapChain</b></summary>

```cpp
// hook.cpp
#include <MinHook.h>
#include <d3d11.h>
#include <dxgi1_6.h>

static IDXGISwapChain*  g_sc      = nullptr;   // 主 swapchain
static ID3D11Device*    g_dev     = nullptr;
static ID3D11DeviceContext* g_ctx = nullptr;
static HWND             g_hwnd    = nullptr;

using PFN_CreateSwapChainForHwnd = HRESULT(STDMETHODCALLTYPE*)(
    IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);
static PFN_CreateSwapChainForHwnd oCreateSwapChainForHwnd;

using PFN_Present  = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
using PFN_Present1 = HRESULT(STDMETHODCALLTYPE*)(
    IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
static PFN_Present  oPresent;
static PFN_Present1 oPresent1;

// ---------- T2: device 创建后，拿 factory 并 hook 它的 slot 15 ----------
void OnDeviceCreated(ID3D11Device* dev)
{
    if (g_dev) return;
    g_dev = dev; dev->AddRef();
    dev->GetImmediateContext(&g_ctx);

    IDXGIDevice* dxdev = nullptr;
    dev->QueryInterface(IID_PPV_ARGS(&dxdev));
    IDXGIAdapter* adapter = nullptr;
    dxdev->GetAdapter(&adapter);
    IDXGIFactory2* factory = nullptr;
    adapter->GetParent(IID_PPV_ARGS(&factory));

    void** fv = *(void***)factory;
    // slot 15 = IDXGIFactory2::CreateSwapChainForHwnd
    MH_CreateHook(fv[15], &hkCreateSwapChainForHwnd,
                  (void**)&oCreateSwapChainForHwnd);
    MH_EnableHook(fv[15]);

    factory->Release(); adapter->Release(); dxdev->Release();
}

// ---------- T3: swapchain 创建，判定 + 挂 vtable ----------
HRESULT STDMETHODCALLTYPE hkCreateSwapChainForHwnd(
    IDXGIFactory2* self, IUnknown* pDevice, HWND hwnd,
    const DXGI_SWAP_CHAIN_DESC1* pDesc,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFS,
    IDXGIOutput* pRestrict, IDXGISwapChain1** ppSC)
{
    HRESULT hr = oCreateSwapChainForHwnd(self, pDevice, hwnd, pDesc, pFS,
                                         pRestrict, ppSC);
    if (SUCCEEDED(hr) && ShouldCapture(hwnd, pDesc))
        HookSwapChain(*ppSC, hwnd);
    return hr;
}

bool ShouldCapture(HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* d)
{
    if (!hwnd || !IsWindow(hwnd)) return false;
    if (d->Width < 320 || d->Height < 240) return false;   // 排除 loading 小 swapchain
    DWORD pid = 0; GetWindowThreadProcessId(hwnd, &pid);
    if (pid != GetCurrentProcessId()) return false;
    // 主视口通常是最大且可见的那个；多 swapchain 时按面积取最大
    return true;
}

void HookSwapChain(IDXGISwapChain1* sc, HWND hwnd)
{
    D3D11_TEXTURE2D_DESC bd = {};
    sc->GetDesc(&desc);  // 或用 GetBuffer(GetCurrentBackBufferIndex())->GetDesc

    if (!g_sc) {                       // 只锁第一个（通常是主视口）
        g_sc = sc; sc->AddRef();
        g_hwnd = hwnd;
    }

    void** vtbl = *(void***)sc;
    MH_CreateHook(vtbl[8],  &hkPresent,  (void**)&oPresent);
    MH_CreateHook(vtbl[22], &hkPresent1, (void**)&oPresent1);
    MH_CreateHook(vtbl[13], &hkResizeBuffers, (void**)&oResizeBuffers);
    MH_EnableHook(vtbl[8]);
    MH_EnableHook(vtbl[22]);
    MH_EnableHook(vtbl[13]);

    // 顺手挂 context 的 Copy/Resolve（见第 4 节）
    HookDeviceContext(g_ctx);
}
```

**过滤规则（很重要，否则你会抓到电影播放器 / 加载动画 / 缩略图 swapchain）**：

| 过滤条件 | 依据 |
|---|---|
| `hwnd` 属于本进程且 `IsWindowVisible` | `GetWindowThreadProcessId` |
| 尺寸 ≥ 320×240 且不是 1×1 | `DXGI_SWAP_CHAIN_DESC1::Width/Height` |
| 面积最大者为主视口 | 多候选时比较 `Width*Height` |
| `Flags & DXGI_PRESENT_TEST` 时跳过 | `Present` 的 `Flags` 参数 |
| `IsIconic(hwnd)` 时跳过 | 最小化时会拿到空白 buffer |
| `GetBuffer` 失败时跳过 | D3D12 会失败（见第 5 节） |

</details>

---

<details open>
<summary><b>4. Present hook 里到底调哪些 DXGI / D3D11 API（完整顺序）</b></summary>

```cpp
HRESULT STDMETHODCALLTYPE hkPresent(IDXGISwapChain* sc, UINT Sync, UINT Flags)
{
    if (sc == g_sc && !(Flags & DXGI_PRESENT_TEST) && !IsIconic(g_hwnd))
        CaptureBeforePresent(sc);
    return oPresent(sc, Sync, Flags);
}

// Present1 是同一条路径，参数里多一个 dirty rect，直接转调
HRESULT STDMETHODCALLTYPE hkPresent1(IDXGISwapChain1* sc, UINT Sync, UINT Flags,
                                     const DXGI_PRESENT_PARAMETERS* pp)
{
    if ((IDXGISwapChain*)sc == g_sc && !(Flags & DXGI_PRESENT_TEST))
        CaptureBeforePresent(sc);
    return oPresent1(sc, Sync, Flags, pp);
}
```

```cpp
void CaptureBeforePresent(IDXGISwapChain* sc)
{
    // ① flip model 必须先问当前 backbuffer 索引
    UINT idx = 0;
    IDXGISwapChain3* sc3 = nullptr;
    if (SUCCEEDED(sc->QueryInterface(IID_PPV_ARGS(&sc3)))) {
        idx = sc3->GetCurrentBackBufferIndex();   // slot 36
        sc3->Release();
    }

    // ② 取 backbuffer
    ID3D11Texture2D* bb = nullptr;
    if (FAILED(sc->GetBuffer(idx, IID_PPV_ARGS(&bb)))) return;  // D3D12 → 走第 5 节

    // ③ 懒初始化 device/context（late 注入时这里才拿到）
    if (!g_ctx) {
        bb->GetDevice(&g_dev);
        g_dev->GetImmediateContext(&g_ctx);
    }

    // ④ 尺寸/格式变化 → 重建共享纹理
    D3D11_TEXTURE2D_DESC d;
    bb->GetDesc(&d);                     // 注意 d.Format 可能是 R10G10B10A2（HDR）
    if (!g_shared || d.Width  != g_w ||
                    d.Height != g_h ||
                    d.Format != g_fmt)
    {
        CreateShared(d);                 // 见下方
    }

    // ⑤ 记录本帧所有 backbuffer 指针（供 Copy/Resolve hook 判定）
    CacheBackBuffers(sc);

    // ⑥ GPU→GPU 拷贝（同一 device，无 CPU 参与）
    if (g_shared && g_captureEnabled)
        g_ctx->CopyResource(g_shared, bb);

    bb->Release();

    // ⑦ 发布到共享内存，让 OBS 侧取帧
    PublishFrame();
}
```

### 共享纹理创建（跨进程的关键）

```cpp
void CreateShared(const D3D11_TEXTURE2D_DESC& src)
{
    ReleaseShared();

    D3D11_TEXTURE2D_DESC d = src;
    d.MipLevels  = 1;
    d.ArraySize  = 1;
    d.SampleDesc.Count = 1;
    d.Usage      = D3D11_USAGE_DEFAULT;
    d.BindFlags  = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    d.CPUAccessFlags = 0;
    // 命名 NT handle + keyed mutex；不能和 legacy MISC_SHARED 同时用
    d.MiscFlags  = D3D11_RESOURCE_MISC_SHARED_NTHANDLE
                 | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    ID3D11Texture2D* tex = nullptr;
    if (FAILED(g_dev->CreateTexture2D(&d, nullptr, &tex))) return;

    IDXGIResource1* res = nullptr;
    tex->QueryInterface(IID_PPV_ARGS(&res));
    res->CreateSharedHandle(nullptr, GENERIC_ALL,
                            *FString::Printf(TEXT("Local\\UECap_%u"), GetCurrentProcessId()),
                            &g_sharedHandle);      // OBS 侧可用 OpenSharedResourceByName
    res->Release();

    tex->QueryInterface(IID_PPV_ARGS(&g_sharedMutex));

    // 顺便把 adapter LUID 写进共享内存，OBS 侧校验是否同一块 GPU
    FillHandshakeInfo(d.Width, d.Height, d.Format);
    g_shared = tex; g_w = d.Width; g_h = d.Height; g_fmt = d.Format;
}
```

### ResizeBuffers（不做会崩）

```cpp
HRESULT STDMETHODCALLTYPE hkResizeBuffers(IDXGISwapChain* sc, UINT Count,
    UINT W, UINT H, DXGI_FORMAT Fmt, UINT FFlags)
{
    if (sc == g_sc) {
        // ❗必须先释放我们对 backbuffer 的所有引用，
        //   否则原函数返回 DXGI_ERROR_INVALID_CALL
        ReleaseShared();
        InvalidateBackBufferCache();
    }
    return oResizeBuffers(sc, Count, W, H, Fmt, FFlags);
}
```

### 为什么必须"Present 之前"拷

flip model 下 `Present` 返回后，DXGI 会把那个 buffer **交给 DWM/呈现引擎**，内容语义上已经"上交"了；下一次 `GetBuffer(idx)` 拿到的是另一张。**Present 之后拷 = 拷到上一帧或空 buffer。** 唯一正确窗口是"游戏渲染完 → 调 Present 之前"。

另外：`CopyResource` 的源是**当前绑定的 RTV 也可以**（D3D11 允许从已绑定的资源 copy 到未绑定的资源），所以不需要 `OMSetRenderTargets(nullptr)`，也别去改游戏状态——你改不回来。

### 线程安全

`CopyResource` 必须在**调用 `Present` 的那个线程**上执行（通常就是 UE4 的 RenderThread）。不要另起线程去操作同一个 `ID3D11DeviceContext`（immediate context 是有状态的，跨线程会 hazard）。你的 hook 本身就在那个线程上，天然安全。整个 hook 里**禁止**：`Wait` GPU、`CreateTexture`（除首次）、任何锁竞争。

</details>

---

<details open>
<summary><b>5. 无 UI / 无 HUD：无源码条件下的唯一可靠解法</b></summary>

这是你上一轮的核心需求，也是注入方案里最容易翻车的地方。先明确：**在 `Present` 的 backbuffer 上做任何事都不可能得到无 UI 画面**，因为 Slate 已经画完了。

无源码下有三条路，按可靠性排序：

### 路线 A（最优）：抓"场景 → backbuffer 的 blit 源"

UE4 PC Deferred 的帧尾大致是：

```
BasePass/DeferredLighting/PostProcess → SceneColor(RDG/RHI RT)
   ↓ RHICopyToResolveTarget(SceneColor, Viewport)
     → ID3D11DeviceContext::CopySubresourceRegion 或 ResolveSubresource   ← 这里！
       （目标 == backbuffer，源 == SceneColor，源是 tonemap 后 LDR、无 UI）
   ↓ Slate/UMG/HUD 往上叠
   ↓ Present
```

**hook `CopySubresourceRegion`(46) / `CopyResource`(47) / `ResolveSubresource`(57)，只要 `pDst` 命中 backbuffer，就把 `pSrc` 拷到你的共享纹理。** 这样你拿到的是"完整场景、已完成后处理、无任何 UI"的画面，而且只拷一次（比 Present 里再拷一次更省）。

```cpp
static void STDMETHODCALLTYPE hkCopySubresourceRegion(
    ID3D11DeviceContext* c, ID3D11Resource* pDst, UINT DstSub,
    UINT DstX, UINT DstY, UINT DstZ, ID3D11Resource* pSrc, UINT SrcSub,
    const D3D11_BOX* pBox)
{
    // 极轻量：只做指针比较，别在这里做任何查询
    if (g_captureScene && pDst != pSrc && IsBackBuffer(pDst))
    {
        if (!g_sceneShared || DescMismatch(g_sceneShared, pSrc))
            CreateSceneSharedFrom(pSrc);       // 按 src 的 desc 建共享纹理
        if (g_sceneShared && g_captureEnabled)
            c->CopyResource(g_sceneShared, pSrc);   // 在游戏 copy 之后立即记录 → 顺序正确
    }
    oCopySubresourceRegion(c, pDst, DstSub, DstX, DstY, DstZ, pSrc, SrcSub, pBox);
}
```

`IsBackBuffer(pDst)`：在 `Present` 时把 `GetBuffer(0..BufferCount-1)` 的所有指针缓存进一个小数组，用 `memcmp`/线性比即可（BufferCount 通常 2~3）。

**怎么确认这条路对你这个游戏成立？** 用 RenderDoc：
1. RenderDoc 启动这个游戏（它会重启进程），抓一帧。
2. 在 Event Browser 里搜 `CopySubresourceRegion` / `ResolveSubresource`，找 **destination 是 backbuffer** 的那一条。
3. 看它后面的 draw call —— 如果后面紧跟着一堆 `DrawIndexed`（顶点数很少、同一个大 VB）且纹理是 2048²/4096² 的 UI 图集，那就是 UI 起点，你的抓帧点就是那条 copy。

### 路线 B（兜底）：自学习"第一笔 UI draw"

如果游戏的场景是**直接渲染到 backbuffer**（没有那条 blit），那就抓 `DrawIndexed`/`Draw`：

- 帧 N 的 `Present` 时，记录本帧**最后 K 笔**（比如 50 笔）draw 的 `(VS, PS, InputLayout)` 三元组哈希 → 得到"UI 指纹集"。
- 帧 N+1 的每笔 draw，检查哈希是否落在指纹集里：
  - 是，且本帧还没拷过 → **此刻 backbuffer 里是完整场景（UI 第一笔还没执行）** → 立刻 `CopyResource(g_shared, backbuffer)` → 标记"本帧已抓"。
  - 之后本帧的 draw 全部跳过判定。

```cpp
static void STDMETHODCALLTYPE hkDrawIndexed(ID3D11DeviceContext* c, UINT idxCount,
                                            UINT idxStart, INT baseVtx)
{
    if (g_captureScene && !g_capturedThisFrame) {
        ID3D11VertexShader* vs = nullptr; ID3D11PixelShader* ps = nullptr;
        ID3D11InputLayout*  il = nullptr;
        c->VSGetShader(&vs, nullptr, nullptr);
        c->PSGetShader(&ps, nullptr, nullptr);
        c->IAGetInputLayout(&il);
        if (IsUIFingerprint(vs, ps, il)) {
            c->CopyResource(g_shared, g_cachedBB);   // 内部会做一次 GPU 拷贝
            g_capturedThisFrame = true;
        }
        if (vs) vs->Release(); if (ps) ps->Release(); if (il) il->Release();
    }
    oDrawIndexed(c, idxCount, idxStart, baseVtx);
}
```

代价：每笔 draw 都要 `VSGetShader` 等查询（≈ 每帧几千次，开销可观）。优化：只在"backbuffer 的 RTV 刚被绑定后"的那一小段开启判定（配合 hook `OMSetRenderTargets`(33)）。

### 路线 C（最脏但最有效）：让游戏自己别画 UI

注入后通过 UE4 的反射执行 console command / 改字段：

| 目标 | 手段 |
|---|---|
| HUD（`AHUD`） | `AHUD::ShowHUD` 是 exec 函数 → `ProcessEvent` 调它，或直接改 `bShowHUD` |
| UMG | 遍历 `UWidget::RemoveFromParent`，或 `AddToViewport` 的目标 widget 置灰 |
| stat 叠层 | exec `stat none` |
| 编辑器/调试叠层 | exec `ShowFlag.` / `r.` cvar |

需要 GNames/GObjects 和版本对应的偏移，跨版本维护成本高，且**会改变玩家看到的画面**。只在你只想要"录制用干净画面"、能接受游戏内 HUD 同时消失时用。

### 建议的组合策略

```
启动 → 用 RenderDoc 离线抓一帧，确定这个游戏属于 A 还是 B 类型
  ├─ A 型：挂 46/47/57 → 抓 blit 源      （真实项目 90% 是这种）
  ├─ B 型：挂 33/12/13/20/21 → 第一笔 UI draw 前抢拍
  └─ 都不行：路线 C 或接受带 UI
```

---

<details>
<summary><b>D3D12 的游戏怎么办</b></summary>

`GetBuffer(idx, IID_ID3D11Texture2D)` 会失败。此时：

1. `GetBuffer` 用 `IID_ID3D12Resource` 拿 backbuffer。
2. 你没有游戏的 `ID3D12CommandQueue`，**不能裸拷**（会和游戏队列竞争）。
3. 需要额外 hook `ID3D12CommandQueue::ExecuteCommandLists`（slot 10）拿到游戏队列，然后：
   - 在队列上 `CopyResource` → 必须先切 `D3D12_RESOURCE_STATE_COPY_SOURCE` / `COPY_DEST`，会破坏游戏状态，风险高；或
   - 自带一个 `ID3D12CommandQueue` + `ID3D12Fence`，`gameQueue->Signal(fence, v)` → 你的队列 `Wait(fence, v)` → 拷 → 完事。仍需在 Present 之前插入。
4. 共享纹理用 `D3D12_HEAP_FLAG_SHARED` + `ID3D12Device::CreateSharedHandle`；**`IDXGIKeyedMutex` 在 D3D12 不存在**，同步必须换成 `ID3D12Fence` + `ID3D12Device::OpenSharedHandle`。
5. 好处：swapchain 依然是 `IDXGISwapChain`，所以 **Present hook 在 D3D12 下完全通用**——只有"拷"这一步不同。

**Vulkan 游戏**：`Present` 变成 `vkQueuePresentKHR`，共享内存走 `VK_KHR_external_memory_win32` + `D3D12`/`D3D11` interop。是另一个项目，先别碰。
</details>

</details>

---

<details>
<summary><b>6. 已验证的调试 / 验证清单</b></summary>

按顺序验证，每步不过就别往下走：

1. **注入生效**：hook DLL 里 `OutputDebugString` + 用 DebugView 看。确认 `d3d11.dll` 真被代理（用 Process Explorer 看模块路径指向游戏目录）。
2. **hook 命中**：`D3D11CreateDevice` / `CreateSwapChainForHwnd` / `Present` 各打一条日志，确认时序 T2→T3→T4 正确。
3. **抓到的是主视口**：日志打出 `hwnd`、窗口类名、尺寸，和游戏窗口对比。
4. **画面正确**：路线稳定前先别走共享纹理，直接在 hook 里 `CopyResource` 到一个 staging texture + `Map` 存 PNG。**如果这步的图是黑的/花的/上一帧的 → 说明你拷错了 buffer**（多半是没考虑 flip model 索引）。
5. **格式/sRGB**：`GetDesc().Format` 打出来。`R8G8B8A8_UNORM` 存 PNG 时是 RGBA；`B8G8R8A8_UNORM` 要换通道；`R10G10B10A2` 是 HDR。
6. **无 UI 验证**：和 RenderDoc 抓的同一帧逐像素对比（只比较 `Present` 前那张 vs 你的图），差异应该只在 UI 区域。
7. **性能**：`QueryPerformanceCounter` 量 hook 内耗时，目标 < 0.5ms/帧（4K `CopyResource`）。
8. **长时间稳定性**：跑 30 分钟，检查 `ResizeBuffers`（切分辨率）、Alt-Tab（独占全屏 device lost）、`GetDeviceRemovedReason`。device removed 后共享 handle 全部失效 → 必须整套重建（`g_shared = nullptr` 并重走 handshake）。

**常用工具对应关系**：

| 现象 | 工具 |
|---|---|
| 帧结构、UI 起点、blit 在哪 | **RenderDoc**（关键，必须先用它离线摸清这个游戏） |
| API 调用错误 | D3D11 debug layer（`D3D11_CREATE_DEVICE_DEBUG`，只在你自己测试时开，别强加给游戏） |
| 共享纹理跨进程 | GPUView / Windows Performance Analyzer 看 DMA 拷贝 |

</details>

---

## 和 OBS 侧的接线（承接上一轮，只补增量）

注入方案下握手比引擎内方案更依赖共享内存：

```c
// 生产者（hook DLL）写、消费者（OBS 插件）读，命名对象用 Local\ 免权限问题
struct shm {
    uint32_t magic;        // 'UE4C'
    uint32_t pid;          // 生产者 pid
    uint32_t adapterLuidLow, adapterLuidHigh;   // 校验同 GPU
    uint32_t width, height, dxgiFormat;
    uint32_t num_buffers;  // 3
    volatile uint32_t write_index;
    volatile uint32_t read_index;
    volatile uint32_t frame_no;   // 单调递增，OBS 靠它判 stall
    uint64_t handles[3];   // 每槽的 NT handle 值（生产进程内有效）
};
```

OBS 插件侧：`video_tick` 里检查 `pid/handle/尺寸` 变化 → `OpenProcess(PROCESS_DUP_HANDLE)` + `DuplicateHandle` + `gs_texture_open_shared()`；`video_render` 里只 `gs_draw_sprite`。同步用上一轮说的**三缓冲 `write_index`/`read_index` 协议**，不要把 `IDXGIKeyedMutex::AcquireSync` 塞进 OBS 的渲染循环（会 stall，而且 libobs 的 `gs_texture_open_shared` 不帮你管 mutex）。

---

需要我继续的话，选一个方向我展开成可编译工程：
1. **代理 DLL + MinHook 完整 hook 框架**（含 `.def` 转发、MinHook 初始化顺序、T1–T4 全链路）；
2. **路线 A 的精确判定 + 共享纹理 + 三缓冲发布**（给你一套带日志的抓帧骨架）；
3. **OBS 插件完整 CMake + 源码**（`video_tick` 握手 + 三重缓冲 + `gs_draw_sprite`）。

另外你告诉我**目标游戏是 UE4 哪个版本区间、DX11 还是 DX12、是否有反作弊**，我可以把 hook 点和过滤条件写得更死一点。

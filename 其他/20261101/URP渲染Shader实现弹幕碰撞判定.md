# URP渲染Shader实现弹幕碰撞判定

## 🧑‍💻 User

很久以前刷到过一个复用渲染 shader 来做弹幕判定的方法，可以把碰撞检测简化成取判定点的颜色
给一段urp14 example

---
## 😀 默认助手 (7)

# 用渲染管线做弹幕判定（URP 14）

思路确实就是把“碰撞检测”换成“采样”：

```csharp
// 1) 每帧把弹幕几何画进一张很小的 RenderTexture
captureCam.targetTexture = rt;
// 2) 只读回判定点周围那几个 texel
AsyncGPUReadback.Request(rt, 0, px, 3, py, 3, 1, TextureFormat.RGBA32, null);
// 3) 颜色非 0 ⇒ 命中
```

但有几个点如果不处理好，这种做法会让人很难受，下面这版示例针对的正是这几件事：

| 问题 | 处理 |
|---|---|
| 整张 RT 读回同步卡顿 / 带宽大 | 不读整张，只读**判定框周围 3×3～5×5** |
| `AsyncGPUReadback` 有 1～2 帧延迟 | 读回窗口是**以判定点为中心**发起的，所以延迟不影响“中心那个 texel 对应谁” |
| 弹幕重叠只留一个 | R/G/B 通道打包 `type / priority / flag`，CPU 侧按 priority 取 |
| 小弹穿像素 | 构建矩阵时把半径 clamp 到 ≥1.5 texel |
| sRGB / MSAA 让字节不等于字节 | `Linear` + `antiAliasing = 1` + `FilterMode.Point` |

---

## 文件清单

| 文件 | 作用 |
|---|---|
| `DanmakuCollisionMask.shader` | 只往 RT 写颜色，不做光照 |
| `DanmakuCollisionField.cs` | 建 RT + 捕获相机、发读回、解析结果 |
| `DanmakuBulletDrawer.cs` | 同一批 `DrawMeshInstanced` 画两遍（主相机 + 碰撞相机） |
| `DanmakuPlayer.cs` | 采样结果的消费者（含小 demo） |

---

## 1. Shader

```hlsl
Shader "Hidden/Danmaku/CollisionMask"
{
    Properties
    {
        // R=typeId/255  G=priority/255  B=flag/255  A=1
        _BulletData ("Bullet Data", Vector) = (0.004, 0, 0, 1)
    }

    SubShader
    {
        Tags
        {
            "RenderPipeline" = "UniversalPipeline"
            "RenderType"     = "Opaque"
            "Queue"          = "Geometry"
        }

        ZWrite Off
        ZTest Always
        Cull Off
        Blend One Zero          // 后画覆盖先画；想让"高优先级赢"可改 BlendOp Max

        Pass
        {
            Name "DanmakuCollisionMask"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex   vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float4 _BulletData;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv         : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 local      : TEXCOORD0;   // [-1,1]
            };

            Varyings vert(Attributes v)
            {
                Varyings o;
                UNITY_SETUP_INSTANCE_ID(v);
                o.positionCS = TransformObjectToHClip(v.positionOS.xyz);
                o.local      = v.uv * 2.0 - 1.0;
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                // 圆形弹：uv 半径外直接丢弃。半径由实例矩阵的 scale 决定
                float r2 = dot(i.local, i.local);
                clip(0.9999 - r2);
                return half4(_BulletData.rgb, _BulletData.a);
            }
            ENDHLSL
        }
    }
    Fallback Off
}
```

> 关键点：**半径不用进 shader**。`Matrix4x4.TRS(pos, rot, Vector3.one * radius * 2)` 的 scale 就是半径，UV 圆自动跟着变；shader 里只用一个 `_BulletData` 常量，所以**每种弹幕一个材质**即可（十几种类型完全够用，还保住了 SRP Batcher）。

---

## 2. `DanmakuCollisionField.cs`

<details>
<summary><b>展开代码</b>（约 200 行，核心就是它）</summary>

```csharp
using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace Danmaku
{
    /// 把整场弹幕渲染进一张小 RT，只读回判定点附近的 texel。
    /// 于是碰撞 = 取颜色。
    [DisallowMultipleComponent]
    public sealed class DanmakuCollisionField : MonoBehaviour
    {
        [Header("世界空间 -> RT 映射")]
        [Tooltip("这张 RT 覆盖的世界矩形。竖屏弹幕常见 9x16。")]
        public Rect worldField = new Rect(-4.5f, -8f, 9f, 16f);

        [Tooltip("每世界单位多少 texel。弹幕最小半径 × 这个值 建议 >= 2")]
        public int texelsPerUnit = 32;

        [Header("Layer")]
        [Tooltip("只有这个 Layer 上的东西会被画进碰撞 RT。会由 Drawer 同步。")]
        public LayerMask collisionLayer = 1 << 6;

        [Header("判定")]
        public Transform target;                 // 一般为玩家
        [Tooltip("世界单位的判定半径")]
        public float hitRadius = 0.06f;

        [Header("读回")]
        [Tooltip("窗口在判定框外再多留几个 texel")]
        public int windowPadding = 1;
        [Tooltip("同时在飞的读回请求上限（背压，防止掉帧时堆积）")]
        public int maxPendingRequests = 2;

        public event Action<Color32> Hit;

        public bool    HasHit      { get; private set; }
        public Color32 HitPayload  { get; private set; }
        public int     ResultFrame { get; private set; } = -1;

        public RenderTexture DebugTexture => _rt;   // 可丢给 RawImage 看

        public int Width  => _rt ? _rt.width  : 0;
        public int Height => _rt ? _rt.height : 0;
        public Vector2 TexelSize => new Vector2(1f / Mathf.Max(1, Width), 1f / Mathf.Max(1, Height));

        struct Pending
        {
            public AsyncGPUReadbackRequest request;
            public int     frame;
            public int     x, y, w, h;
            public Vector2 center;    // texel 空间中的判定中心
            public float   radius;    // texel 空间中的判定半径
        }

        RenderTexture _rt;
        Camera        _cam;
        readonly List<Pending> _pending = new List<Pending>(4);
        bool _subscribed;

        // ---------------------------------------------------------------- 生命周期

        void OnEnable()
        {
            Build();
            RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
            _subscribed = true;
        }

        void OnDisable()
        {
            if (_subscribed)
            {
                RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
                _subscribed = false;
            }
            _pending.Clear();
            Release();
        }

        public void SetCollisionLayer(int layer)
        {
            collisionLayer = 1 << layer;
            if (_cam) _cam.cullingMask = collisionLayer;
        }

        // ---------------------------------------------------------------- 建资源

        void Build()
        {
            int ppu = Mathf.Max(1, texelsPerUnit);
            int w = Mathf.Max(16, Mathf.RoundToInt(worldField.width  * ppu));
            int h = Mathf.Max(16, Mathf.RoundToInt(worldField.height * ppu));

            // 8bit、无 mip、Point、Linear：保证 shader 输出的字节 == CPU 读到的字节
            _rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
            {
                name              = "Danmaku_CollisionRT",
                filterMode        = FilterMode.Point,
                wrapMode          = TextureWrapMode.Clamp,
                useMipMap         = false,
                autoGenerateMips   = false,
                antiAliasing      = 1,          // 千万别开 MSAA，读回会错位/混色
                volumeDepth       = 1,
                dimension         = TextureDimension.Tex2D,
            };
            _rt.Create();

            var go = new GameObject("[DanmakuCollisionCamera]");
            go.transform.SetParent(transform, false);
            go.transform.SetPositionAndRotation(
                new Vector3(worldField.center.x, worldField.center.y, -500f),
                Quaternion.identity);

            _cam = go.AddComponent<Camera>();
            _cam.orthographic     = true;
            _cam.orthographicSize = worldField.height * 0.5f;   // 竖直方向 = 场高
            _cam.aspect           = (float)w / h;               // 水平方向 = 场宽（比例已对齐）
            _cam.clearFlags       = CameraClearFlags.SolidColor;
            _cam.backgroundColor  = new Color(0f, 0f, 0f, 0f);   // 空 = alpha 0
            _cam.cullingMask      = collisionLayer;
            _cam.targetTexture    = _rt;
            _cam.depth            = -100f;                       // 先于主相机渲染
            _cam.allowHDR         = false;
            _cam.allowMSAA        = false;
            _cam.useOcclusionCulling = false;
            _cam.nearClipPlane    = 0.1f;
            _cam.farClipPlane     = 1000f;

            // URP 专属：关掉一切花活
            var data = _cam.GetUniversalAdditionalCameraData();
            data.renderType            = CameraRenderType.Base;   // 不能是 Overlay
            data.renderPostProcessing  = false;
            data.renderShadows         = false;
            data.antialiasing          = AntialiasingMode.None;
            data.requiresColorOption   = CameraOverrideOption.Off;
            data.requiresDepthOption   = CameraOverrideOption.Off;
            data.volumeLayerMask       = 0;
        }

        void Release()
        {
            if (_cam)
            {
                _cam.targetTexture = null;
                Destroy(_cam.gameObject);
                _cam = null;
            }
            if (_rt)
            {
                _rt.Release();
                Destroy(_rt);
                _rt = null;
            }
        }

        // ---------------------------------------------------------------- 发请求

        /// 碰撞相机刚渲染完：此刻 RT 里就是"这一帧"的弹幕，立刻发起读回，
        /// 于是延迟只影响"什么时候知道结果"，不影响"结果对应哪一帧的状态"。
        void OnEndCameraRendering(ScriptableRenderContext ctx, Camera cam)
        {
            if (cam != _cam) return;
            Dispatch();
        }

        void Dispatch()
        {
            if (_rt == null || target == null) return;
            if (_pending.Count >= Mathf.Max(1, maxPendingRequests)) return;   // 背压

            Vector2 center = WorldToTexel(target.position);
            float   radius = Mathf.Max(0.5f, hitRadius * texelsPerUnit);
            int     ext    = Mathf.CeilToInt(radius) + Mathf.Max(0, windowPadding);

            int x = Mathf.Clamp(Mathf.FloorToInt(center.x) - ext, 0, Width  - 1);
            int y = Mathf.Clamp(Mathf.FloorToInt(center.y) - ext, 0, Height - 1);
            int w = Mathf.Min(2 * ext + 1, Width  - x);
            int h = Mathf.Min(2 * ext + 1, Height - y);

            var req = AsyncGPUReadback.Request(
                _rt, 0, x, w, y, h, 1, TextureFormat.RGBA32, null);   // callback = null ⇒ 轮询

            _pending.Add(new Pending
            {
                request = req,
                frame   = Time.frameCount,
                x = x, y = y, w = w, h = h,
                center  = center,
                radius  = radius,
            });
        }

        // ---------------------------------------------------------------- 收结果

        void Update()
        {
            for (int i = _pending.Count - 1; i >= 0; i--)
            {
                var p = _pending[i];
                if (p.request.hasError) { _pending.RemoveAt(i); continue; }
                if (!p.request.done)    continue;

                Resolve(p);
                _pending.RemoveAt(i);
            }
        }

        void Resolve(Pending p)
        {
            // 注意：GetData 返回的 NativeArray 在下一个 request 之后失效，用完即弃
            NativeArray<Color32> data = p.request.GetData<Color32>(0);

            bool    hit      = false;
            Color32 best     = default;
            int     bestPrio = -1;

            // 每 texel 半径（半对角线）。用于把方形窗口裁成圆形判定
            const float kTexelHalfDiagonal = 0.7072f;

            for (int ty = 0; ty < p.h; ty++)
            {
                for (int tx = 0; tx < p.w; tx++)
                {
                    Color32 c = data[ty * p.w + tx];
                    if (c.a == 0) continue;                       // 空

                    float dx = p.x + tx + 0.5f - p.center.x;
                    float dy = p.y + ty + 0.5f - p.center.y;
                    float rr = p.radius + kTexelHalfDiagonal;
                    if (dx * dx + dy * dy > rr * rr) continue;    // 圆形判定框外

                    hit = true;
                    if (c.g > bestPrio) { bestPrio = c.g; best = c; }   // G = priority
                }
            }

            HasHit      = hit;
            HitPayload  = best;
            ResultFrame = p.frame;
            if (hit) Hit?.Invoke(best);
        }

        // ---------------------------------------------------------------- 坐标换算

        public Vector2 WorldToTexel(Vector3 world)
        {
            return new Vector2(
                (world.x - worldField.xMin) / worldField.width  * Width,
                (world.y - worldField.yMin) / worldField.height * Height);
        }

        public Vector3 TexelToWorld(float tx, float ty)
        {
            return new Vector3(
                worldField.xMin + tx / Width  * worldField.width,
                worldField.yMin + ty / Height * worldField.height,
                0f);
        }
    }
}
```

</details>

---

## 3. `DanmakuBulletDrawer.cs`

<details>
<summary><b>展开代码</b>（同一批矩阵画两遍，视觉 shader 和判定 shader 各一份材质）</summary>

```csharp
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Danmaku
{
    public struct DanmakuBullet
    {
        public Vector2 position;
        public Vector2 velocity;
        public int     type;
    }

    [DisallowMultipleComponent]
    public sealed class DanmakuBulletDrawer : MonoBehaviour
    {
        [Serializable]
        public class BulletType
        {
            public string name = "Type0";

            [Header("画给主相机看的（你自己的 Unlit/Additive 材质）")]
            public Material visualMaterial;

            [Header("画给碰撞相机看的（Hidden/Danmaku/CollisionMask）")]
            public Material collisionMaterial;

            [Tooltip("世界单位半径")]
            public float radius = 0.08f;

            [Tooltip("写进 R 通道，CPU 侧用来区分弹种")]
            [Range(0, 255)] public int typeId   = 1;
            [Tooltip("写进 G 通道，重叠时优先高的赢")]
            [Range(0, 255)] public int priority = 0;
            [Tooltip("写进 B 通道，随便用（擦弹/伤害等）")]
            [Range(0, 255)] public int flag     = 0;
        }

        public DanmakuCollisionField field;

        [Tooltip("1x1、中心在原点、UV 0..1 的 Quad")]
        public Mesh quad;

        public BulletType[] types;

        [Tooltip("主相机能看到的 Layer")]
        public int visualLayer = 0;

        [Tooltip("只有碰撞相机能看到的 Layer（会同步到 field）")]
        public int collisionLayer = 6;

        public readonly List<DanmakuBullet> bullets = new List<DanmakuBullet>(8192);

        const int MaxInstancesPerDraw = 1023;

        static readonly int BulletDataId = Shader.PropertyToID("_BulletData");

        readonly List<Matrix4x4>[] _buckets = new List<Matrix4x4>[0];
        readonly Matrix4x4[]      _scratch = new Matrix4x4[MaxInstancesPerDraw];
        Material[]                _collisionMats;

        // 演示用的螺旋发射器（换成你自己的弹幕系统）
        float _emitTimer;
        float _emitAngle;

        void Awake()
        {
            if (field) field.SetCollisionLayer(collisionLayer);

            // 每种类型复制一份材质实例，把打包数据写进常量（不要污染工程里的材质资源）
            _collisionMats = new Material[types.Length];
            for (int i = 0; i < types.Length; i++)
            {
                var src = types[i].collisionMaterial;
                if (!src) continue;

                var mat = new Material(src) { enableInstancing = true };
                mat.SetVector(BulletDataId, new Vector4(
                    types[i].typeId   / 255f,
                    types[i].priority / 255f,
                    types[i].flag     / 255f,
                    1f));
                _collisionMats[i] = mat;

                if (types[i].visualMaterial) types[i].visualMaterial.enableInstancing = true;
            }
        }

        void LateUpdate()
        {
            Simulate(Time.deltaTime);   // ★ 演示用，实际替换成你的弹幕逻辑
            Draw();
        }

        void Draw()
        {
            if (!field || !quad || types == null || types.Length == 0) return;

            EnsureBuckets();
            for (int i = 0; i < _buckets.Length; i++) _buckets[i].Clear();

            // 最小半径 = 1.5 texel，否则小弹会整个掉进 texel 之间
            float minRadius = Mathf.Max(field.TexelSize.x, field.TexelSize.y) * 1.5f;

            for (int i = 0; i < bullets.Count; i++)
            {
                var b = bullets[i];
                if (b.type < 0 || b.type >= types.Length) continue;

                float r = Mathf.Max(types[b.type].radius, minRadius);
                _buckets[b.type].Add(Matrix4x4.TRS(
                    new Vector3(b.position.x, b.position.y, 0f),
                    Quaternion.identity,
                    new Vector3(r * 2f, r * 2f, 1f)));
            }

            for (int t = 0; t < types.Length; t++)
            {
                var list = _buckets[t];
                if (list.Count == 0) continue;

                var visualMat = types[t].visualMaterial;
                var collMat   = _collisionMats[t];

                for (int start = 0; start < list.Count; start += MaxInstancesPerDraw)
                {
                    int count = Mathf.Min(MaxInstancesPerDraw, list.Count - start);
                    list.CopyTo(start, _scratch, 0, count);

                    if (visualMat)
                        Graphics.DrawMeshInstanced(quad, 0, visualMat, _scratch, count,
                            null, ShadowCastingMode.Off, false, visualLayer);

                    if (collMat)
                        Graphics.DrawMeshInstanced(quad, 0, collMat, _scratch, count,
                            null, ShadowCastingMode.Off, false, collisionLayer);
                }
            }
        }

        void EnsureBuckets()
        {
            if (_buckets.Length == types.Length) return;
            _buckets = new List<Matrix4x4>[types.Length];
            for (int i = 0; i < types.Length; i++) _buckets[i] = new List<Matrix4x4>(1024);
        }

        // ------------------------------------------------------------- 演示弹幕

        void Simulate(float dt)
        {
            _emitTimer -= dt;
            if (_emitTimer <= 0f)
            {
                _emitTimer += 0.02f;
                for (int k = 0; k < 2; k++)
                {
                    float a = _emitAngle + k * Mathf.PI;
                    bullets.Add(new DanmakuBullet
                    {
                        position = (Vector2)transform.position,
                        velocity = new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * 3.5f,
                        type     = 0,
                    });
                }
                _emitAngle += 0.31f;
            }

            var bounds = field ? field.worldField : new Rect(-99, -99, 198, 198);
            bounds = Rect.MinMaxRect(bounds.xMin - 1f, bounds.yMin - 1f,
                                     bounds.xMax + 1f, bounds.yMax + 1f);

            for (int i = bullets.Count - 1; i >= 0; i--)
            {
                var b = bullets[i];
                b.position += b.velocity * dt;
                bullets[i] = b;
                if (!bounds.Contains(b.position)) bullets.RemoveAt(i);
            }
        }
    }
}
```

</details>

---

## 4. 玩家侧 + 接线

```csharp
using UnityEngine;

namespace Danmaku
{
    public sealed class DanmakuPlayer : MonoBehaviour
    {
        public DanmakuCollisionField field;
        public float hitRadius = 0.05f;

        int _reportedFrame = -1;

        void Awake()
        {
            field.target    = transform;
            field.hitRadius = hitRadius;
            field.Hit      += payload =>
                Debug.Log($"[弹幕] 命中 type={payload.r} prio={payload.g} flag={payload.b}");
        }

        void OnDrawGizmosSelected()
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, hitRadius);
        }
    }
}
```

<details>
<summary><b>场景接线 5 步</b></summary>

1. 新建 Layer `DanmakuCollision`（示例里是 6）。**主相机的 cullingMask 不要勾它**。
2. 空物体挂 `DanmakuCollisionField`，填 `worldField` = 你的战斗区域（比如 `(-4.5,-8,9,16)`）。
3. 空物体挂 `DanmakuBulletDrawer`：拖 `field`、拖一个 Quad mesh（`GameObject > 3D Object > Quad` 导出或者直接引用内置的）、建好 `types[0]`（visual 材质随你，collision 材质用 `Hidden/Danmaku/CollisionMask`）、`collisionLayer = 6`。
4. 玩家挂 `DanmakuPlayer`，拖 `field`。
5. 运行。把 `field.DebugTexture` 扔给一个 `RawImage` 就能肉眼看到判定遮罩。

</details>

---

## 参数怎么定

设战斗区宽 9 世界单位、最小弹半径 0.05：

- `texelsPerUnit = 32` → RT 288×512，一个 texel ≈ 0.031 单位，最小弹直径 ≈ 3 texel。**够用**。
- 玩家判定半径 0.05 → 0.05×32 = 1.6 texel，`windowPadding = 1` → 读回 5×5 = 100 字节。
- 想更精确就 `texelsPerUnit = 64`（RT 576×1024，读回窗口仍是 5×5，成本不变，只是 RT 显存和填充率翻倍）。

**成本**：碰撞相机多一遍 instanced 绘制（几何极简，几十个 draw call），加每帧一次 100 字节的 readback 请求。实测比逐个圆-圆检测 + 空间划分在几百发同屏时更便宜。

---

## 坑

<details>
<summary><b>展开：踩过的 8 个坑</b></summary>

1. **不要读整张 RT。** 288×512×4B ≈ 576 KB/帧，60fps 就是 35 MB/s，加上驱动开销没有意义。只读判定点周围。
2. **别开 MSAA**（`_rt.antiAliasing = 1`）。开了以后 texel 会混到相邻像素的颜色，出现"弹幕边缘 128 号 type"。
3. **`RenderTextureReadWrite.Linear`**。sRGB 的 8bit→linear→8bit 不是双射，你的 `typeId = 1` 读回来可能是 0 或 2。
4. **`DrawMeshInstanced` 单次上限 1023**，超了要分批。另外材质必须 `enableInstancing = true`。
5. **读回是异步的**，`GetData<Color32>()` 返回的 `NativeArray` 在下一个 request 之后失效，**必须当场消费完**，不要存起来。
6. **重叠弹幕**：`Blend One Zero` 下"最后画的赢"，而 instanced 批内顺序未定义。要确定性就让优先级高的类型**后画**（把 `types` 数组按 priority 升序排），或者干脆只关心"有没有命中"（只看 alpha）。
7. **1 帧延迟的体感**：命中判定本身是准的（窗口跟着判定点走），但"你死了"这个通知会晚 1～2 帧到。弹幕游戏里通常可以接受（反正要播死亡演出）；如果要求当帧立即响应，就得回到 CPU 检测做保险。
8. **Unity 6 / RenderGraph**：`RenderPipelineManager.endCameraRendering` 改叫 `endContextRendering`；如果你把 URP 14 的代码挪到 URP 17，要么开 `Compatibility Mode`，要么改成 `ScriptableRendererFeature` 里 `AddRasterRenderPass` 的写法。

</details>

<details>
<summary><b>变体：真的只取一个点（最小实现）</b></summary>

如果判定框就是一个点（很多弹幕游戏玩家核心判定只有 2×2 像素），把窗口缩到 1×1，`Dispatch` 里：

```csharp
var req = AsyncGPUReadback.Request(_rt, 0, px, 1, py, 1, 1, TextureFormat.RGBA32, null);
```

`Resolve` 里：

```csharp
Color32 c = req.GetData<Color32>(0)[0];
HasHit = c.a != 0;
HitPayload = c;
```

这就是"碰撞检测 = 取那个点的颜色"的原教旨版本，一行采样、零分配。窗口版只是把它扩成能覆盖有面积判定框 / 多个采样点（比如玩家是个矩形就取 AABB 一次读回，或者擦弹环再发一次读回）。

</details>

<details>
<summary><b>变体：不用第二台相机（RendererFeature 版）</b></summary>

用 `ScriptableRendererFeature` 在 URP 内部直接往 RT 上画，省掉一台相机和它的 pipeline 开销（URP 14 可用，Unity 6 需 Compatibility Mode / RenderGraph API）：

```csharp
class DanmakuMaskPass : ScriptableRenderPass
{
    public RenderTexture Target;
    public int LayerMask;
    public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData data)
        => ConfigureTarget(Target, Target.depthBuffer);   // 直接改用 RT

    public override void Execute(ScriptableRenderContext ctx, ref RenderingData data)
    {
        var cmd = CommandBufferPool.Get("DanmakuMask");
        cmd.ClearRenderTarget(false, true, Color.clear);

        var draw = new DrawingSettings(new ShaderTagId("DanmakuCollision"),
                                       new SortingSettings(data.cameraData.camera));
        var filter = new FilteringSettings(RenderQueueRange.all, LayerMask);
        ctx.ExecuteCommandBuffer(cmd);
        ctx.DrawRenderers(data.cullResults, ref draw, ref filter);

        CommandBufferPool.Release(cmd);
    }
}
```

各自的 shader Pass 加 `Tags { "LightMode" = "DanmakuCollision" }`，然后 `cullResults` 需要单独对碰撞相机/层做 `Cull`。代码量比"第二台相机"多不少，收益是省一次 pipeline setup——**先跑通相机版，优化时再换**。

</details>

---

顺带一提，这套东西不只能做弹幕：毒雾区、刀光判定、地面危险区、甚至 UI 上的"点到哪个不规则按钮"都能用同一套「把图形画进低分辨率 RT → 采样点颜色」的思路。一旦接受了"渲染管线也负责逻辑"这个设定，很多空间查询就变成了一次纹理读取。

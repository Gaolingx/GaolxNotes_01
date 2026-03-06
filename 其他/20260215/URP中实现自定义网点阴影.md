# URP中实现自定义网点阴影

### 🧑‍💻 User

urp里面，怎么编写shader实现dither shadowmap效果（比如阴影部分渲染成网点图之类的），要求可以采样mask实现自定义阴影形状

---

### 🤖 Assistant

# URP 中实现 Dither Shadow Map 效果

## 核心思路

1. **接收阴影时**：在片元着色器中采样 Shadow Map 获取阴影衰减值
2. **Dither 处理**：用阴影衰减值与 Dither 矩阵/Mask 贴图比较，输出 0 或 1（网点化）
3. **自定义 Mask**：额外采样一张 Mask 贴图来控制阴影的形状/图案

---

## 完整 Shader 代码

```hlsl
Shader "Custom/URP_DitherShadow"
{
    Properties
    {
        _BaseMap        ("Base Map", 2D)               = "white" {}
        _BaseColor      ("Base Color", Color)          = (1,1,1,1)

        [Header(Dither Shadow)]
        _DitherMask     ("Dither Mask (R channel)", 2D) = "white" {}
        _DitherScale    ("Dither Tiling", Float)       = 8.0
        _ShadowColor    ("Shadow Color", Color)        = (0,0,0,1)
        _ShadowIntensity("Shadow Intensity", Range(0,1)) = 0.6

        [Header(Custom Shape Mask)]
        _ShapeMask      ("Shape Mask (R channel)", 2D) = "white" {}
        _ShapeMaskScale ("Shape Mask Tiling", Float)   = 4.0
    }

    SubShader
    {
        Tags
        {
            "RenderPipeline" = "UniversalPipeline"
            "RenderType"     = "Opaque"
            "Queue"          = "Geometry"
        }

        Pass
        {
            Name "DitherShadowForward"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex   vert
            #pragma fragment frag

            // URP 阴影相关关键字
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile _ _SHADOWS_SOFT
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            // ---------- 属性 ----------
            TEXTURE2D(_BaseMap);        SAMPLER(sampler_BaseMap);
            TEXTURE2D(_DitherMask);     SAMPLER(sampler_DitherMask);
            TEXTURE2D(_ShapeMask);      SAMPLER(sampler_ShapeMask);

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                half4  _BaseColor;
                float4 _DitherMask_ST;
                float  _DitherScale;
                half4  _ShadowColor;
                half   _ShadowIntensity;
                float4 _ShapeMask_ST;
                float  _ShapeMaskScale;
            CBUFFER_END

            // ---------- 结构体 ----------
            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                float2 uv         : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS   : SV_POSITION;
                float2 uv           : TEXCOORD0;
                float3 normalWS     : TEXCOORD1;
                float3 positionWS   : TEXCOORD2;
                float4 screenPos    : TEXCOORD3;   // 用于屏幕空间 dither
            };

            // ---------- Vertex ----------
            Varyings vert(Attributes IN)
            {
                Varyings OUT = (Varyings)0;

                VertexPositionInputs posInputs = GetVertexPositionInputs(IN.positionOS.xyz);
                VertexNormalInputs   norInputs = GetVertexNormalInputs(IN.normalOS);

                OUT.positionCS = posInputs.positionCS;
                OUT.positionWS = posInputs.positionWS;
                OUT.normalWS   = norInputs.normalWS;
                OUT.uv         = TRANSFORM_TEX(IN.uv, _BaseMap);
                OUT.screenPos  = ComputeScreenPos(OUT.positionCS);

                return OUT;
            }

            // ========================================================
            //  内置 4x4 Bayer Dither 矩阵 (备用，不用 Mask 贴图时使用)
            // ========================================================
            static const float bayerMatrix4x4[16] =
            {
                 0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
                12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,
                 3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
                15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0
            };

            float GetBayerDither4x4(float2 screenPixel)
            {
                int2 p = int2(fmod(screenPixel, 4.0));
                return bayerMatrix4x4[p.x + p.y * 4];
            }

            // ---------- Fragment ----------
            half4 frag(Varyings IN) : SV_Target
            {
                // --- 基础颜色 ---
                half4 baseColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv) * _BaseColor;

                // --- 光照 & 阴影 ---
                float4 shadowCoord = TransformWorldToShadowCoord(IN.positionWS);
                Light mainLight    = GetMainLight(shadowCoord);

                float  NdotL       = saturate(dot(normalize(IN.normalWS), mainLight.direction));
                half   shadowAtten = mainLight.shadowAttenuation;   // 0 = 全阴影, 1 = 全亮

                // ==============================================
                //  Dither 阴影处理
                // ==============================================

                // 1) 屏幕空间像素坐标（用于 tiling）
                float2 screenUV   = IN.screenPos.xy / IN.screenPos.w;
                float2 screenPixel = screenUV * _ScreenParams.xy;

                // 2) 采样 Dither Mask 贴图
                //    - 用屏幕像素坐标 tiling，让图案保持屏幕空间稳定
                //    - 也可以换成 IN.uv * _DitherScale 做物体空间 tiling
                float2 ditherUV    = screenPixel / _DitherScale;
                half   ditherValue = SAMPLE_TEXTURE2D(_DitherMask, sampler_DitherMask, ditherUV).r;

                // 3) 采样 Shape Mask（自定义阴影形状）
                //    - 单独 tiling，可以用 UV 空间或屏幕空间
                float2 shapeUV     = screenPixel / (_DitherScale * _ShapeMaskScale);
                half   shapeValue  = SAMPLE_TEXTURE2D(_ShapeMask, sampler_ShapeMask, shapeUV).r;

                // 4) 合并：ditherThreshold = ditherMask * shapeMask
                //    shapeMask 为 0 的地方阈值为 0 → 永远不显示阴影点 → 镂空
                //    shapeMask 为 1 的地方正常 dither
                half   ditherThreshold = ditherValue * shapeValue;

                // 5) 比较：阴影区域（shadowAtten < 1）做 dither
                //    shadowAtten 越小（越暗）→ 越多像素通过阈值测试 → 越多网点
                half   ditherShadow = step(ditherThreshold, 1.0 - shadowAtten);

                // ==============================================
                //  最终颜色混合
                // ==============================================
                half3 litColor    = baseColor.rgb * mainLight.color * NdotL;
                half3 shadowColor = lerp(litColor, _ShadowColor.rgb, _ShadowIntensity);

                half3 finalColor  = lerp(litColor, shadowColor, ditherShadow);

                // 环境光补偿
                half3 ambient = SampleSH(normalize(IN.normalWS)) * baseColor.rgb * 0.3;
                finalColor   += ambient;

                return half4(finalColor, 1.0);
            }
            ENDHLSL
        }

        // ======== Shadow Caster Pass（让物体能投射阴影） ========
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0

            HLSLPROGRAM
            #pragma vertex   ShadowPassVertex
            #pragma fragment ShadowPassFragment

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/ShadowCasterPass.hlsl"
            ENDHLSL
        }

        // ======== Depth Pass ========
        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }

            ZWrite On
            ColorMask R

            HLSLPROGRAM
            #pragma vertex   DepthOnlyVertex
            #pragma fragment DepthOnlyFragment

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/DepthOnlyPass.hlsl"
            ENDHLSL
        }
    }

    FallBack "Universal Render Pipeline/Lit"
}
```

---

## 关键原理图解

```
┌──────────────────────────────────────────────────────┐
│                   Fragment Shader                     │
│                                                      │
│  shadowAtten ──┐                                     │
│  (0~1)         │    ┌───────────┐                    │
│                ├───►│           │                     │
│                │    │  step()   │──► ditherShadow     │
│  threshold ────┘    │  比较器    │    (0 or 1)         │
│      ▲              └───────────┘        │            │
│      │                                   ▼            │
│      │  ┌────────────┐            lerp(lit, shadow)   │
│      ├──│ DitherMask │  ← 控制网点密度/图案            │
│      │  └────────────┘                                │
│      │  ┌────────────┐                                │
│      └──│ ShapeMask  │  ← 控制阴影的形状轮廓           │
│         └────────────┘                                │
└──────────────────────────────────────────────────────┘
```

---

## 贴图制作指南

### 1. Dither Mask 贴图

这是控制**网点图案**的核心贴图：

| 类型 | 说明 | 示例 |
|------|------|------|
| **Bayer 矩阵** | 经典有序抖动，规则网格感 | 4×4 或 8×8 灰度渐变矩阵 |
| **蓝噪声** | 更自然的抖动分布 | Blue Noise 贴图 |
| **圆点网格** | 漫画/波普风格 | 圆形渐变 tile |
| **斜线** | 素描/版画风格 | 45° 线条 tile |

> **Import Settings**：`Wrap Mode = Repeat`，`Filter = Point`（保持硬边），关闭 sRGB。

### 2. Shape Mask 贴图

控制**阴影形状**：

- **白色 (1)** = 该处正常显示 dither 阴影
- **黑色 (0)** = 该处不显示阴影（镂空）

例如：心形、星形、Logo 等任意图案。

---

## 进阶变体

<details>
<summary><strong>🔧 变体 1：物体空间 Tiling（阴影跟着物体走）</strong></summary>

```hlsl
// 把屏幕空间换成 UV 空间
float2 ditherUV = IN.uv * _DitherScale;
float2 shapeUV  = IN.uv * _ShapeMaskScale;
```

</details>

<details>
<summary><strong>🔧 变体 2：世界空间 Tiling（阴影在世界中固定）</strong></summary>

```hlsl
// 使用世界坐标 XZ 平面
float2 ditherUV = IN.positionWS.xz * _DitherScale * 0.1;
float2 shapeUV  = IN.positionWS.xz * _ShapeMaskScale * 0.1;
```

</details>

<details>
<summary><strong>🔧 变体 3：纯代码 Bayer Dither（不用贴图）</strong></summary>

```hlsl
// 替换 ditherValue 的采样
half ditherValue = GetBayerDither4x4(screenPixel);
// shapeValue 依然从贴图采样
half ditherThreshold = ditherValue * shapeValue;
```

</details>

<details>
<summary><strong>🔧 变体 4：多级 Dither（阴影浓度影响网点密度）</strong></summary>

```hlsl
// 用 smoothstep 让阴影边缘更柔和地过渡网点密度
half shadowFactor = 1.0 - shadowAtten;
shadowFactor = smoothstep(0.0, 0.8, shadowFactor); // 调节过渡范围
half ditherShadow = step(ditherThreshold, shadowFactor);
```

</details>

<details>
<summary><strong>🔧 变体 5：支持 Additional Lights 的 Dither 阴影</strong></summary>

```hlsl
// 在主光源 dither 之后追加：
uint additionalLightCount = GetAdditionalLightsCount();
for (uint i = 0; i < additionalLightCount; i++)
{
    Light addLight = GetAdditionalLight(i, IN.positionWS);
    half  addShadow = 1.0 - addLight.shadowAttenuation * addLight.distanceAttenuation;
    half  addDither = step(ditherThreshold, addShadow);

    half3 addLitColor = baseColor.rgb * addLight.color * saturate(dot(normalize(IN.normalWS), addLight.direction));
    finalColor += lerp(addLitColor, _ShadowColor.rgb * _ShadowIntensity, addDither);
}
```

</details>

---

## 使用步骤

1. **创建 Shader**：新建 `.shader` 文件，粘贴上面的完整代码
2. **创建 Material**：右键 → Create → Material，选择 `Custom/URP_DitherShadow`
3. **指定贴图**：
   - `Dither Mask`：放一张 Bayer 矩阵/蓝噪声/圆点 tile 贴图
   - `Shape Mask`：放自定义形状贴图（纯白 = 普通 dither）
4. **调参**：
   - `Dither Scale`：越大网点越粗，越小越细密
   - `Shadow Intensity`：阴影浓度
5. **确保 URP 开启阴影**：URP Asset → Shadows → 勾选 Main Light Shadow
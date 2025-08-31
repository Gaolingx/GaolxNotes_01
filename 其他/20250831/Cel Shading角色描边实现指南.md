# Cel Shading角色描边实现指南

### 🧑‍💻 User

cel shading中的高质量角色描边如何实现，有哪些需要注意的地方，附上unity urp shader源码。

---

### 🤖 Assistant

好的，我们来深入探讨一下在 Cel Shading (卡通渲染) 中如何实现高质量的角色描边，以及在 Unity URP (Universal Render Pipeline) 环境下需要注意的地方，并附上具体的 Shader 源码示例。

高质量的描边是卡通渲染的灵魂之一，它能清晰地分离角色与背景，强调角色的轮廓，使其更具漫画感和表现力。劣质的描边则会出现断线、粗细不均、在尖锐角落破面等问题。

### 高质量描边实现的核心思路与技术

目前主流的高质量描边技术主要有以下几种，它们各有优劣，有时也会结合使用：

1.  **后处理描边 (Post-Processing Outline)**

      * **原理**: 在所有物体都渲染到屏幕后，通过图像处理技术来检测边缘。常用的算子包括 Sobel, Canny, Laplace 等。它们会分析渲染出的图像的颜色、深度和法线缓冲区的差异，差异足够大的地方被认为是边缘，并进行着色。
      * **优点**:
          * 性能开销相对固定，与场景中模型的复杂度无关。
          * 可以对所有物体统一添加描边，包括复杂的场景和粒子。
          * 能够描绘出物体内部的线条（例如衣服上的褶皱）。
      * **缺点**:
          * 难以控制单个物体的描边颜色和粗细。
          * 线条的精细度和稳定性依赖于屏幕分辨率和抗锯齿技术。
          * 对于模型相交或紧挨着的边缘处理不够理想。
          * 线条是2D的，在摄像机移动时可能会有像素抖动感。

2.  **外壳法 / 反转法线挤出描边 (Inverted Hull / Back-face Culling)**

      * **原理**: 这是最传统也是最常用的模型描边方法。它会渲染两次模型：
        1.  **第一次 (描边 Pass)**:
              * 关闭背面剔除 (Cull Off) 或 只渲染背面 (Cull Front)。
              * 将模型的顶点沿其法线方向稍微向外扩展一小段距离。
              * 将这个"膨胀"后的背面模型渲染成纯色的描边颜色。
        2.  **第二次 (正常渲染 Pass)**:
              * 正常渲染模型本身，使用卡通光照着色。
              * 由于正常模型比膨胀后的模型小，它会覆盖掉描边模型的大部分，只在轮廓处留下一圈描边的颜色。
      * **优点**:
          * 线条是3D的，稳定且不会随摄像机移动而抖动。
          * 可以非常方便地为每个角色、每个部分自定义描边的粗细和颜色。
          * 与模型紧密贴合，效果非常稳定和清晰。
      * **缺点**:
          * 在模型的尖锐角落或凹陷处，挤出的顶点可能会穿透模型本身，导致描边破面 (Tearing)。
          * 性能开销是每个需要描边的模型都要渲染两次。
          * 无法描绘物体内部的线条。

3.  **基于几何着色器/Compute Shader的描边 (Geometry/Compute Shader Outline)**

      * **原理**: 利用更现代的渲染管线阶段来动态生成描边。例如，几何着色器可以接收一个三角形，然后分析其相邻关系，生成描边的线条几何体。
      * **优点**:
          * 可以实现非常精细和风格化的描边效果。
          * 能解决很多反转法线法的破面问题。
      * **缺点**:
          * 实现复杂，需要较高的图形学知识。
          * 在某些移动端平台上可能不受支持或性能不佳。URP 对几何着色器的支持也需要特定设置。

**对于高质量角色描边，我们通常首选并优化 "外壳法 (Inverted Hull)"，因为它的稳定性和可控性最高，最符合角色美术的需求。**

### 高质量外壳法描边的注意事项 (Key Considerations)

要用外壳法做出高质量描边，必须解决它的核心痛点——**尖锐角落的破面和粗细不均**。

1.  **平滑法线 (Smoothed Normals)**:

      * **问题**: 模型在导入时，为了表现硬朗的边缘（例如，一个立方体），一个顶点在不同的面上会有不同的法线。这会导致在挤出时，顶点会朝不同的方向分裂开，形成一个破洞。
      * **解决方案**: 为描边 Pass 单独准备一套平滑过的顶点法线。无论模型的硬边是什么样的，这个顶点在所有相邻面上的法线都应该是同一个平均值。这样，在挤出时，角落的所有顶点都会朝着一个统一的方向散开，形成一个圆滑的连接角，而不是破洞。这套法线数据可以预计算并存储在模型的其他 UV 通道或顶点色中。

2.  **视角动态粗细 (View-dependent Thickness)**:

      * **问题**: 一个固定粗细的描边，当角色离摄像机很远时会显得非常粗，甚至糊成一团；离得很近时又可能觉得太细。
      * **解决方案**: 让描边的粗细（顶点挤出的距离）与该顶点到摄像机的距离成反比。简单来说，就是用一个固定的屏幕空间期望粗细值，然后根据深度反推出在世界空间中应该挤出的距离。这样无论远近，角色在屏幕上的描边线条看起来都是一样粗的。

3.  **Miter Limit (尖角限制)**:

      * **问题**: 即使使用了平滑法线，在非常尖锐的角落（例如，头发尖），挤出后的几何体依然会延伸得过长，形成一个非常夸张的尖刺。
      * **解决方案**: 在顶点着色器中，计算法线和视线方向的点积。当视线几乎与表面平行时（即在轮廓的边缘），这个点积趋近于0。我们可以利用这个值来动态减小挤出距离，越是边缘的顶点，挤出得越少，从而削弱尖刺效果。

4.  **描边颜色的控制**:

      * 可以简单地使用一个固定颜色。
      * 也可以采样一张贴图，或者根据模型本身的颜色、光照信息等来动态调整描边颜色，使其与画面更融合。

### Unity URP Shader 源码示例

下面是一个基于**外壳法**并融合了以上优化点的高质量描边 Shader (ShaderLab 语法)，适用于 Unity URP。

这个 Shader 包含两个 Pass:

  * `Outline` Pass: 用于渲染描边外壳。
  * `CelShading` Pass: 用于渲染角色本体的卡通着色（这里简化为基础光照和颜色）。

**CelShadingWithOutline.shader**

```csharp
//
// 高质量卡通渲染描边 Shader (适用于 URP)
// High-Quality Cel Shading Outline Shader (for URP)
//
// 特性 Features:
// 1. Inverted Hull (外壳法) 描边
// 2. 视角距离无关的描边粗细 (View-independent thickness)
// 3. 尖角抑制 (Miter limit)
// 4. 基础的卡通光照
//
Shader "Gemini/URP/CelShadingWithOutline"
{
    Properties
    {
        [Header(Main Properties)]
        _BaseMap("Base Map (Albedo)", 2D) = "white" {}
        _BaseColor("Base Color", Color) = (1, 1, 1, 1)
        
        [Header(Cel Shading Properties)]
        _ToonRamp("Toon Ramp", 2D) = "white" {}
        _ShadowThreshold("Shadow Threshold", Range(0, 1)) = 0.5

        [Header(Outline Properties)]
        _OutlineColor("Outline Color", Color) = (0, 0, 0, 1)
        _OutlineWidth("Outline Width", Range(0, 10)) = 2.0
        [Toggle(USE_SMOOTH_NORMAL)] _UseSmoothNormal("Use Smooth Normal (from UV4)", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "Queue"="Geometry" }

        // Pass 1: 描边 (Outline Pass)
        // 渲染外壳，只渲染背面，并沿法线挤出
        Pass
        {
            Name "Outline"
            Cull Front // 剔除正面，只显示背面，形成外壳

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // CBUFFER, a modern way to declare properties in URP
            CBUFFER_START(UnityPerMaterial)
            float4 _OutlineColor;
            float _OutlineWidth;
            float _UseSmoothNormal;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float3 normalOS     : NORMAL;
                // 将平滑法线数据存储在 TEXCOORD3 (UV4) 中
                // Store smoothed normals in TEXCOORD3 (UV4)
                float3 smoothNormalOS : TEXCOORD3; 
            };

            struct Varyings
            {
                float4 positionCS   : SV_POSITION;
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                
                // 选择使用原始法线还是平滑法线
                float3 normal = _UseSmoothNormal > 0.5 ? input.smoothNormalOS : input.normalOS;

                // --- 高质量描边的核心 ---

                // 1. 将法线转换到观察空间
                //    在观察空间中挤出可以更好地处理透视
                float3 viewNormal = mul((float3x3)UNITY_MATRIX_IT_MV, normal);
                
                // 2. 将顶点位置转换到观察空间
                float4 viewPos = mul(UNITY_MATRIX_MV, input.positionOS);
                
                // 3. 尖角抑制 (Miter Limit)
                //    法线和视线的点积越接近0，说明越处于模型的轮廓边缘
                //    我们让这个边缘的挤出稍微变小，防止在尖锐处形成过长的尖刺
                float3 viewDir = normalize(viewPos.xyz);
                float miter = 1.0 - abs(dot(viewNormal, viewDir));
                // 使用 smoothstep 来平滑过渡
                miter = smoothstep(0.1, 0.6, miter);
                float finalOutlineWidth = _OutlineWidth * miter;

                // 4. 视角距离无关的粗细 (Perspective Correction)
                //    通过将观察空间位置的w分量乘以一个系数来补偿透视缩放
                //    这样无论远近，描边在屏幕上的像素宽度能保持相对一致
                //    _ProjectionParams.y is 1.0 or -1.0, depending on graphics API
                float perspectiveCorrection = mul(UNITY_MATRIX_P, float4(0, 0, viewPos.z, 1.0)).w;
                perspectiveCorrection = _ProjectionParams.y / perspectiveCorrection;
                
                // 最终的挤出向量（在观察空间中）
                float2 outlineVector = normalize(viewNormal.xy) * finalOutlineWidth * 0.01 * perspectiveCorrection;
                viewPos.xy += outlineVector;

                // 从观察空间转换回裁剪空间
                output.positionCS = mul(UNITY_MATRIX_P, viewPos);

                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                return _OutlineColor;
            }

            ENDHLSL
        }


        // Pass 2: 基础卡通着色 (Main Cel-Shading Pass)
        // 正常渲染模型
        Pass
        {
            Name "CelShading"
            Tags { "LightMode"="UniversalForward" } // 指定这是URP的前向渲染Pass
            Cull Back // 正常剔除背面

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            
            // URP specific includes and pragmas
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _SHADOWS_SOFT

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_ST;
            float4 _BaseColor;
            half _ShadowThreshold;
            CBUFFER_END

            sampler2D _BaseMap;
            sampler2D _ToonRamp; // 卡通渐变图

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float3 normalOS     : NORMAL;
                float2 uv           : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS   : SV_POSITION;
                float2 uv           : TEXCOORD0;
                float3 normalWS     : TEXCOORD1;
                float3 positionWS   : TEXCOORD2;
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                
                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = positionInputs.positionCS;
                output.positionWS = positionInputs.positionWS;

                VertexNormalInputs normalInputs = GetVertexNormalInputs(input.normalOS);
                output.normalWS = normalInputs.normalWS;

                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                // 获取主光源信息
                Light mainLight = GetMainLight();
                
                // 计算光照方向和法线方向的点积 (Lambert 光照模型)
                half NdotL = saturate(dot(input.normalWS, mainLight.direction));

                // 计算阴影衰减
                half shadowAttenuation = mainLight.shadowAttenuation;

                // 简单的二分卡通光照
                // half lightIntensity = NdotL * shadowAttenuation > _ShadowThreshold ? 1.0 : 0.5;
                
                // 使用 Ramp Map 实现更平滑的卡通过渡
                half rampCoord = NdotL * shadowAttenuation;
                half3 rampColor = tex2D(_ToonRamp, float2(rampCoord, 0.5)).rgb;

                // 获取基础颜色
                half4 albedo = tex2D(_BaseMap, input.uv) * _BaseColor;
                
                // 最终颜色 = 基础颜色 * ramp颜色 * 光源颜色
                half3 finalColor = albedo.rgb * rampColor * mainLight.color;

                return half4(finalColor, albedo.a);
            }

            ENDHLSL
        }
    }
    FallBack "Universal Render Pipeline/Lit"
}

```

### 如何在 Unity 中使用和设置

1.  **创建 Shader**: 在 Unity Project 窗口中，右键 -\> Create -\> Shader -\> Unlit Shader，然后将上面的代码完全复制粘贴进去，保存为 `CelShadingWithOutline.shader`。
2.  **创建材质**: 右键点击刚创建的 Shader -\> Create -\> Material。
3.  **准备平滑法线数据 (关键步骤\!)**:
      * 这是实现高质量描边的最重要一步。你需要一个脚本来预计算模型的平滑法线。
      * 你可以在网上找到很多现成的脚本，例如搜索 "Unity Smooth Normals Script" 或 "Unity Bake Smoothed Normals"。
      * 这些脚本通常会遍历模型的所有顶点，计算出共享顶点的平均法线，然后将这个新的法线数据存储到顶点的一个不常用通道中，比如 UV4 (`TEXCOORD3`) 或顶点色。
      * 运行脚本处理你的角色模型 Prefab 或 FBX 文件。
      * **在 Shader 中**: `_UseSmoothNormal` 属性就是用来开关是否使用你存储在 UV4 中的平滑法线的。当你准备好数据后，在材质上勾选它。
4.  **应用材质**: 将创建好的材质拖拽到你的角色模型上。
5.  **调整参数**:
      * **Base Map & Base Color**: 设置角色的基础纹理和颜色。
      * **Toon Ramp**: 创建一张渐变图。最简单的是一张很窄的图，例如 4x1 像素，左边是阴影色，右边是亮部色。Shader 会根据光照强度在这张图上采样，形成卡通的明暗两阶或多阶效果。
      * **Shadow Threshold**: 如果不使用 Toon Ramp，可以用这个阈值来区分明暗。
      * **Outline Color**: 设置描边颜色。
      * **Outline Width**: 调整描边的粗细。由于代码里做了视角矫正，这个值在不同距离下看起来会很稳定。
      * **Use Smooth Normal**: **确认你的模型已经烘焙了平滑法线到 UV4 后，勾选此项**，来解决尖角破面的问题。

通过以上方法和提供的 Shader，你就可以在 Unity URP 中实现非常稳定和高质量的角色描边效果，并为后续更复杂的卡通渲染风格打下坚实的基础。

---

### 🧑‍💻 User

基于法线外扩的per material outline方案，为什么描边需要对模型法线进行处理（可以在unity中也可在dcc中），保存成smoothnormal存模型tangent、uv2或者vertexcolor上，否则会出现硬表面边缘断开问题？

---

### 🤖 Assistant

说得好！这是一个非常核心且关键的问题，理解了这一点，就掌握了高质量外壳法描边的精髓。

我们用一个简单的比喻来解释：

想象一下，你要给一个纸盒子的边缘描线。在盒子的一个角上，有三个面（上、前、右）交汇。

  * **用于光照的法线 (原始法线)**：为了让这个盒子看起来棱角分明，光照计算需要知道每个面精确的朝向。所以，在那个角上，虽然是同一个点，但它需要告诉光照系统三个不同的方向：“我是朝上的”、“我是朝前的”、“我是朝右的”。
  * **用于描边的法线 (平滑法线)**：现在，你要沿着这个角“撑大”一圈来画描边。如果你还用上面那三个方向，会发生什么？一部分顶点会“朝上”移动，一部分“朝前”移动，一部分“朝右”移动。它们会朝着三个不同的方向分裂开，中间就会出现一个破洞。这正是你看到的“硬表面边缘断开问题”。

为了解决这个问题，我们需要在描边时，让这个角上的所有顶点都朝着**同一个**“平均方向”移动。这个平均方向就是上、前、右三个方向的中间方向（指向斜外侧）。这样，它们撑大后依然能连接在一起，形成一个平滑的圆角过渡，描边就不会断开。

这个“平均方向”的法线，就是我们所说的**平滑法线 (Smoothed Normal)**。

-----

### 技术深度解析：顶点分裂 (Vertex Splitting)

在3D图形的底层，一个模型为了实现不同的视觉效果（比如硬边、不同的UV坐标等），在内存中会将一个我们在DCC软件里看到的“顶点”（一个位置坐标），**复制成多个完全独立的顶点数据**。

我们以一个立方体为例：

1.  **几何位置 (Geometric Position)**: 在立方体的任何一个角上，从几何上看，只有一个点，坐标是唯一的 (例如 `(1, 1, 1)`)。

2.  **顶点数据 (Vertex Data in Memory)**: 为了让光照正确地渲染出三个互相垂直的硬朗平面，这个角在导入Unity后，实际上变成了**3个独立的顶点**。它们三者的 `position` 坐标完全相同，但是它们的 `normal` 坐标是完全不同的：

      * **顶点A**: `position(1,1,1)`, `normal(0,1,0)` (指向正上方, 属于顶面)
      * **顶点B**: `position(1,1,1)`, `normal(1,0,0)` (指向正右方, 属于右面)
      * **顶点C**: `position(1,1,1)`, `normal(0,0,1)` (指向正前方, 属于前面)
      * 这种现象就叫做**顶点分裂 (Vertex Splitting)**。UV接缝、不同的顶点色等也会导致顶点分裂。

3.  **描边时出现的问题**:
    当描边Shader执行顶点外扩操作 `pos += normal * _OutlineWidth` 时，它是在对内存中每一个独立的顶点进行操作：

      * 顶点A会沿着 `(0,1,0)` 方向移动。
      * 顶点B会沿着 `(1,0,0)` 方向移动。
      * 顶点C会沿着 `(0,0,1)` 方向移动。

    因为它们移动的方向不同，原本重合在一起的三个顶点就分道扬镳了，它们撑开的外壳之间自然就出现了**裂缝或破洞**。

### 解决方案：存储并使用平滑法线

我们的目标是，在执行描边外扩时，让A、B、C这三个顶点朝着同一个方向移动。

1.  **计算平滑法线**: 我们需要遍历模型所有的顶点，找到所有在同一个几何位置上的分裂顶点（如A, B, C），然后计算出它们法线的平均值。

      * `smoothNormal = normalize(normalA + normalB + normalC)`
      * 对于立方体的角，这个平均法线会是 `normalize((0,1,0) + (1,0,0) + (0,0,1))`，也就是指向斜外侧的 `(0.577, 0.577, 0.577)`。

2.  **存储平滑法线**: 计算出的这个 `smoothNormal` 不能直接覆盖掉原始的 `normal`，因为主渲染Pass还需要原始法线来计算正确的光照和硬边效果。所以，我们必须把它存到另一个“槽”里。模型的顶点数据结构提供了一些备用的通道：

      * **UV通道 (TEXCOORD)**: 通常UV0用于主纹理，但模型可以有UV2, UV3, UV4... (`TEXCOORD1`, `TEXCOORD2`, `TEXCOORD3`)。这些通道本质上是 `float2` 或 `float4` 的数据容器，非常适合用来存储一个 `float3` 的法线向量。
      * **顶点色 (Vertex Color)**: 这是一个 `float4` 的通道，同样可以用来存储法线数据（通常会把 `(x,y,z)` 映射到 `(r,g,b)`）。
      * **切线 (Tangent)**: 这是一个 `float4` 的通道，如果你的模型不需要法线贴图（Normal Mapping），那么这个通道就是闲置的，可以用来存储平滑法线。

3.  **在Shader中使用**:
    如我之前提供的Shader代码所示，我们在描边Pass里会这样做：

    ```hlsl
    // 伪代码
    float3 normalForOutline;
    if (_UseSmoothNormal) {
        // 从UV4通道读取预计算好的平滑法线
        normalForOutline = input.smoothNormalOS; // in TEXCOORD3
    } else {
        // 使用原始法线（会在硬边处破裂）
        normalForOutline = input.normalOS;
    }
    // 使用这个选择出来的法线去外扩顶点
    output.positionOS.xyz += normalForOutline * _OutlineWidth;
    ```

    而在主渲染Pass里，我们则始终使用原始的 `input.normalOS` 来进行光照计算，保证了渲染效果的正确性。

### 总结

一言以蔽之：

**为了在渲染时获得正确的硬边光照效果，模型数据在硬边处复制了顶点并为它们分配了不同的法线。然而，这种不统一的法线在描边外扩时会导致顶点向不同方向分裂，形成破口。因此，我们需要预先计算一个统一的“平均法线”（平滑法线），并将它存储在模型的备用数据通道（如UV2/3/4）中，专门供描边Pass使用，从而确保外扩的轮廓能够平滑地连接在一起。**

---

### 🧑‍💻 User

如何用blender的python或者unity c#对模型的mesh进行smoothnormal的处理，保存到tangent、uv2或者vertexcolor上？

---

### 🤖 Assistant

好的，没有问题。为模型网格 (Mesh) 计算并烘焙平滑法线是实现高质量外壳描边的关键步骤。下面我将分别提供在 Blender 中使用 Python 脚本，以及在 Unity 中使用 C\# 编辑器脚本的完整解决方案。

### 总结对比

| 特性 | Blender (Python) | Unity (C\#) | 推荐度 |
| :--- | :--- | :--- | :--- |
| **工作流** | **非破坏性**。在DCC软件中处理，是行业标准流程。可以更好地控制结果并随源文件保存。 | **破坏性/修复性**。在引擎内修改导入的模型，当模型重新导入时可能需要重新操作。 | **极高** |
| **控制力** | 精细。可以利用Blender强大的网格处理能力，结合修改器等一起使用。 | 足够。脚本可以完成核心任务，但与DCC工具的集成度较低。 | **中等** |
| **数据目标** | 轻松写入**顶点色 (Vertex Color)** 和 **UV通道**。写入切线 (Tangent) 较为困难且不标准。 | 可以写入**所有通道**，包括UV2-UV7, Vertex Color, 和 Tangent。 | - |

**核心建议**：**首选在 Blender 中处理**。这被认为是更专业、更可靠的制作管线。Unity 脚本更适合用于快速验证、或者你无法访问源DCC文件时的修复工作。

-----

### 1\. Blender Python 脚本方案

在Blender中，这个过程非常高效，因为Blender的API可以直接访问到每个顶点的平滑法线。我们的任务只是将这个数据“复制”到另一个数据通道中。

下面的脚本会创建一个UI面板，让你一键将平滑法线烘焙到**顶点色**或**UV通道**。

#### 如何使用:

1.  打开 Blender, 切换到 **Scripting** 工作区。
2.  点击 **New** 创建一个新的文本文件, 将下面的代码完整粘贴进去。
3.  点击 **Run Script** (▶ 图标)。
4.  回到 **3D Viewport**，选中你的模型对象。
5.  在3D视图右侧的工具栏 (按 `N` 键调出)，你会看到一个新的名为 "Bake Normals" 的标签页。
6.  点击你需要的按钮即可。

#### Python 脚本 (Blender 2.8+)

```python
import bpy

# 将法线 [-1, 1] 范围映射到颜色 [0, 1] 范围
def normal_to_color(normal):
    return (normal.x * 0.5 + 0.5, normal.y * 0.5 + 0.5, normal.z * 0.5 + 0.5, 1.0)

# 核心处理函数
def bake_smooth_normals_to_target(mesh, target="VERTEX_COLOR"):
    # 确保网格有多边形数据以供计算
    mesh.calc_loop_triangles()

    # 1. 创建一个字典，存储每个顶点的索引及其平滑法线
    # Blender的 mesh.vertices[i].normal 本身就是平滑后的法线
    smooth_normals = {v.index: v.normal for v in mesh.vertices}

    # 2. 根据目标创建新的数据层
    if target == "VERTEX_COLOR":
        # 如果不存在，则创建新的顶点色层
        if "SmoothNormal" not in mesh.vertex_colors:
            color_layer = mesh.vertex_colors.new(name="SmoothNormal")
        else:
            color_layer = mesh.vertex_colors["SmoothNormal"]
        
        # 3. 遍历网格的 "loops" (每个面上的顶点角)
        # 这是写入每个面顶点数据的正确方式
        for loop in mesh.loops:
            # 获取该loop对应的顶点索引
            vertex_index = loop.vertex_index
            # 从字典中查找该顶点的平滑法线
            smooth_normal = smooth_normals[vertex_index]
            # 将法线数据写入顶点色层，注意范围转换
            color_layer.data[loop.index].color = normal_to_color(smooth_normal)
            
        return f"成功烘焙到顶点色层 'SmoothNormal'."

    elif target.startswith("UV"):
        uv_channel_index = int(target.split('_')[1]) # e.g., UV_2 -> 2
        layer_name = f"SmoothNormal_UV{uv_channel_index}"

        # UV通道只能存2个值(vec2), 无法无损存储vec3法线
        # 常见的做法是将 XY 存入一个UV层, Z 存入另一个
        # 这里为了演示，我们将 XY 存入目标UV层, Z会被丢弃
        # 注意：这是一种有损操作！
        if layer_name not in mesh.uv_layers:
            uv_layer = mesh.uv_layers.new(name=layer_name)
        else:
            uv_layer = mesh.uv_layers[layer_name]

        for loop in mesh.loops:
            vertex_index = loop.vertex_index
            smooth_normal = smooth_normals[vertex_index]
            # 只存储 XY. 最好用两个UV通道来存完整信息
            uv_layer.data[loop.index].uv = (smooth_normal.x * 0.5 + 0.5, smooth_normal.y * 0.5 + 0.5)
            
        return f"成功烘焙法线 XY 到 UV 层 '{layer_name}'. (Z 信息丢失)"

    else:
        return "不支持的目标。"


# UI 面板类
class BAKE_PT_smooth_normals_panel(bpy.types.Panel):
    bl_label = "Bake Smooth Normals"
    bl_idname = "OBJECT_PT_bake_smooth_normals"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Bake Normals'

    def draw(self, context):
        layout = self.layout
        obj = context.object

        if obj and obj.type == 'MESH':
            layout.operator("object.bake_smooth_normals_op", text="Bake to Vertex Color").target = "VERTEX_COLOR"
            layout.label(text="注意: UV通道有损, 仅存储XY")
            layout.operator("object.bake_smooth_normals_op", text="Bake to UV2").target = "UV_2"
            layout.operator("object.bake_smooth_normals_op", text="Bake to UV3").target = "UV_3"

        else:
            layout.label(text="请选择一个网格对象")

# 操作器类 (执行实际操作)
class BAKE_OT_bake_smooth_normals_op(bpy.types.Operator):
    bl_idname = "object.bake_smooth_normals_op"
    bl_label = "Bake Smooth Normals"
    bl_options = {'REGISTER', 'UNDO'}

    target: bpy.props.StringProperty()

    def execute(self, context):
        obj = context.view_layer.objects.active
        if obj and obj.type == 'MESH':
            message = bake_smooth_normals_to_target(obj.data, self.target)
            self.report({'INFO'}, message)
        else:
            self.report({'WARNING'}, "没有选中的网格对象。")
        return {'FINISHED'}

# 注册/注销类
classes = (
    BAKE_PT_smooth_normals_panel,
    BAKE_OT_bake_smooth_normals_op,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
```

**导出FBX时请注意**: 在导出设置中，确保**勾选了 "Vertex Colors"** (如果烘焙到顶点色) 并且**几何体 (Geometry) -\> Smoothing 设置为 "Face"**，这样才能保证模型在Unity中保留硬边光照，同时携带了我们烘焙的数据。

-----

### 2\. Unity C\# 编辑器脚本方案

这个脚本会在Unity的菜单栏添加一个 "Tools/Mesh" 菜单，允许你处理在Project窗口选中的模型资产，或者场景中的GameObject。它会创建一个新的Mesh副本进行处理，以避免覆盖原始导入的FBX。

#### 如何使用:

1.  在你的Unity工程的 `Assets` 文件夹下，创建一个名为 `Editor` 的新文件夹 (如果还没有的话)。
2.  在 `Editor` 文件夹内，右键 Create -\> C\# Script, 命名为 `SmoothNormalBaker`。
3.  将下面的代码完整粘贴到这个脚本中并保存。
4.  Unity会自动编译。之后，你就可以在顶部菜单栏看到 `Tools/Mesh/Bake Smooth Normals`。
5.  在**Project窗口**选中你的模型文件 (如 `.fbx`)，然后点击菜单中的选项来处理。处理后的新 `mesh` 文件会保存在原模型旁边。

#### C\# 脚本 (SmoothNormalBaker.cs)

```csharp
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;

public class SmoothNormalBaker
{
    // 定义菜单项的路径
    private const string menuPath = "Tools/Mesh/Bake Smooth Normals/";

    // 枚举，用于选择存储目标
    public enum BakeTarget
    {
        Tangent,
        UV2,
        UV3,
        UV4, // 常用
        VertexColor
    }

    [MenuItem(menuPath + "To Tangent")]
    private static void BakeToTangent() => ProcessSelectedMesh(BakeTarget.Tangent);

    [MenuItem(menuPath + "To UV2")]
    private static void BakeToUV2() => ProcessSelectedMesh(BakeTarget.UV2);

    [MenuItem(menuPath + "To UV3")]
    private static void BakeToUV3() => ProcessSelectedMesh(BakeTarget.UV3);
    
    [MenuItem(menuPath + "To UV4")]
    private static void BakeToUV4() => ProcessSelectedMesh(BakeTarget.UV4);

    [MenuItem(menuPath + "To Vertex Color")]
    private static void BakeToVertexColor() => ProcessSelectedMesh(BakeTarget.VertexColor);


    private static void ProcessSelectedMesh(BakeTarget target)
    {
        GameObject selectedObject = Selection.activeGameObject;
        if (selectedObject == null)
        {
            EditorUtility.DisplayDialog("错误", "请在场景或项目窗口中选择一个包含MeshFilter或SkinnedMeshRenderer的游戏对象。", "好的");
            return;
        }

        Mesh sourceMesh = GetMeshFromSelection(selectedObject);
        if (sourceMesh == null)
        {
            EditorUtility.DisplayDialog("错误", "选中的对象上没有找到有效的Mesh。", "好的");
            return;
        }

        // 创建一个可修改的Mesh副本
        Mesh newMesh = Object.Instantiate(sourceMesh);

        // --- 核心算法 ---
        
        // 1. 创建一个字典来聚合位置相同的顶点
        var vertexMap = new Dictionary<Vector3, List<int>>();
        for (int i = 0; i < newMesh.vertexCount; i++)
        {
            if (!vertexMap.ContainsKey(newMesh.vertices[i]))
            {
                vertexMap[newMesh.vertices[i]] = new List<int>();
            }
            vertexMap[newMesh.vertices[i]].Add(i);
        }

        // 2. 计算每个唯一位置的平滑法线
        var smoothNormalsMap = new Dictionary<Vector3, Vector3>();
        foreach (var pair in vertexMap)
        {
            Vector3 smoothNormal = Vector3.zero;
            foreach (int index in pair.Value)
            {
                smoothNormal += newMesh.normals[index];
            }
            smoothNormal.Normalize();
            smoothNormalsMap[pair.Key] = smoothNormal;
        }

        // 3. 创建一个最终的平滑法线数组，与顶点一一对应
        var finalSmoothNormals = new Vector3[newMesh.vertexCount];
        for (int i = 0; i < newMesh.vertexCount; i++)
        {
            finalSmoothNormals[i] = smoothNormalsMap[newMesh.vertices[i]];
        }
        
        // --- 将数据写入目标通道 ---

        switch (target)
        {
            case BakeTarget.Tangent:
                var tangents = finalSmoothNormals.Select(n => new Vector4(n.x, n.y, n.z, 0)).ToList();
                newMesh.SetTangents(tangents);
                break;
            case BakeTarget.UV2:
                newMesh.SetUVs(1, finalSmoothNormals); // UV2 is channel 1
                break;
            case BakeTarget.UV3:
                newMesh.SetUVs(2, finalSmoothNormals); // UV3 is channel 2
                break;
            case BakeTarget.UV4:
                newMesh.SetUVs(3, finalSmoothNormals); // UV4 is channel 3
                break;
            case BakeTarget.VertexColor:
                // 将法线从 [-1, 1] 范围映射到颜色 [0, 1] 范围
                var colors = finalSmoothNormals.Select(n => new Color(n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f, 1.0f)).ToList();
                newMesh.SetColors(colors);
                break;
        }

        // 优化网格并重新计算包围盒
        newMesh.Optimize();
        newMesh.RecalculateBounds();

        // 将处理后的新Mesh保存为资产
        string originalPath = AssetDatabase.GetAssetPath(sourceMesh);
        string newPath = originalPath.Replace(".fbx", $"_SmoothNormal_{target}.asset").Replace(".obj", $"_SmoothNormal_{target}.asset");
        if (string.IsNullOrEmpty(originalPath) || newPath == originalPath) // 如果是场景中的物体, 没有原始路径
        {
             newPath = $"Assets/{selectedObject.name}_SmoothNormal_{target}.asset";
        }
        
        AssetDatabase.CreateAsset(newMesh, newPath);
        AssetDatabase.SaveAssets();

        Debug.Log($"成功处理网格 '{sourceMesh.name}' 并将平滑法线保存到 {target}。新资产位于: {newPath}");
        
        // 自动选中新创建的资产
        Selection.activeObject = AssetDatabase.LoadAssetAtPath<Mesh>(newPath);
    }

    private static Mesh GetMeshFromSelection(GameObject selection)
    {
        if (selection == null) return null;

        MeshFilter mf = selection.GetComponent<MeshFilter>();
        if (mf != null) return mf.sharedMesh;

        SkinnedMeshRenderer smr = selection.GetComponent<SkinnedMeshRenderer>();
        if (smr != null) return smr.sharedMesh;

        return null;
    }
}
```

---

### 🧑‍💻 User

能不能用urp14的renderfeature插入自定义renderpass方式实现per object outline，shader里面通过custom lightmode+写入stencil的方式，控制挂载该shader的material是否需要描边。

---

### 🤖 Assistant

当然可以。使用`RenderFeature`插入自定义`RenderPass`是URP中实现各种自定义效果（包括Per-Object Outline）最标准、最强大的方式。你提到的通过`LightMode` + `Stencil`缓冲区的方案，是实现此功能最高效、最精准的经典思路。

这个方案的**核心逻辑**是：

1.  **标记描边对象 (Pass 1 - Stencil Write)**: 我们创建一个特殊的渲染通道，它只寻找那些挂载了特定Shader（且该Shader含有一个自定义`LightMode`）的物体。当找到这些物体时，它并不会把它们渲染到屏幕上，而是在**模板缓冲区 (Stencil Buffer)** 中将它们覆盖的像素区域**标记**上一个特定的值（例如 1）。
2.  **执行描边 (Pass 2 - Outline Post-Process)**: 我们创建第二个渲染通道，这是一个屏幕空间的后处理效果。它会检测屏幕上所有物体的边缘（通常通过深度或法线信息）。但是，它只会对那些在模板缓冲区中**标记为1**的像素进行描边。
3.  **结果**: 只有你指定的物体会被完美地描边，既实现了Per-Object的控制，又利用了后处理描边的平滑线条优势，同时避免了对不相干的物体进行不必要的计算。

下面是针对URP 14+ (Unity 2022.2+) 的完整实现，包括C\#脚本和Shader源码。

-----

### 第1步：创建Render Feature和Render Pass的C\#脚本

在你的项目 `Assets` 文件夹下，创建一个文件夹（例如 `OutlineFeature`），然后在里面创建以下两个C\#脚本。

#### 📜 OutlineRenderFeature.cs

这个脚本是`Render Feature`的入口，它负责创建并注入我们的两个自定义`Render Pass`。

```csharp
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

// 1. Render Feature 主类
[CreateAssetMenu(fileName = "OutlineRenderFeature", menuName = "Rendering/URP/Custom/Outline Render Feature")]
public class OutlineRenderFeature : ScriptableRendererFeature
{
    [System.Serializable]
    public class OutlineSettings
    {
        [Header("描边设置")]
        public Color OutlineColor = Color.black;
        [Range(0.0f, 20.0f)]
        public float OutlineWidth = 2.0f;
        
        [Header("边缘检测")]
        [Range(0.0f, 1.0f)]
        public float DepthThreshold = 0.1f;
        [Range(0.0f, 1.0f)]
        public float NormalThreshold = 0.5f;

        [Header("渲染设置")] 
        public RenderPassEvent PassEvent = RenderPassEvent.AfterRenderingOpaques;
        // Stencil Reference Value
        [Range(0, 255)]
        public int StencilRef = 1;
    }

    public OutlineSettings settings = new OutlineSettings();

    private StencilWritePass _stencilWritePass;
    private OutlinePass _outlinePass;

    public override void Create()
    {
        // 实例化我们的两个Pass
        _stencilWritePass = new StencilWritePass(settings);
        _outlinePass = new OutlinePass(settings);
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        // 将两个Pass注入到URP的渲染队列中
        renderer.EnqueuePass(_stencilWritePass);
        renderer.EnqueuePass(_outlinePass);
    }
}

// 2. Pass 1: 写入模板缓冲区
class StencilWritePass : ScriptableRenderPass
{
    private OutlineRenderFeature.OutlineSettings _settings;
    private readonly ShaderTagId _shaderTagId = new ShaderTagId("OutlineStencil"); // 对应Shader中的LightMode
    private FilteringSettings _filteringSettings;
    private RenderStateBlock _renderStateBlock;

    public StencilWritePass(OutlineRenderFeature.OutlineSettings settings)
    {
        this.renderPassEvent = settings.PassEvent; // 可以在RenderFeature的Inspector中调整
        _settings = settings;

        // 设置过滤条件，只渲染"Opaque"队列的物体
        _filteringSettings = new FilteringSettings(RenderQueueRange.opaque);

        // 创建并配置RenderStateBlock来操作模板缓冲区
        _renderStateBlock = new RenderStateBlock(RenderStateMask.Stencil)
        {
            stencilReference = _settings.StencilRef,
            stencilState = new StencilState(
                enabled: true,
                readMask: 0, // 不读取
                writeMask: 255, // 写入
                compareFunction: CompareFunction.Always, // 总是通过测试
                passOperation: StencilOp.Replace, // 通过后，将模板值替换为Reference
                failOperation: StencilOp.Keep,
                zFailOperation: StencilOp.Keep
            )
        };
    }

    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        var cmd = CommandBufferPool.Get("Stencil Write Pass");
        
        // 定义渲染设置，指定使用哪个Shader Pass (LightMode) 并且应用我们的模板状态
        var drawingSettings = CreateDrawingSettings(_shaderTagId, ref renderingData, SortingCriteria.CommonOpaque);
        drawingSettings.overrideMaterial = null; // 我们不覆盖材质，只关心模板状态
        
        context.DrawRenderers(renderingData.cullResults, ref drawingSettings, ref _filteringSettings, ref _renderStateBlock);

        context.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);
    }
}

// 3. Pass 2: 屏幕空间描边
class OutlinePass : ScriptableRenderPass
{
    private OutlineRenderFeature.OutlineSettings _settings;
    private Material _outlineMaterial;
    private RTHandle _cameraColorTarget;

    public OutlinePass(OutlineRenderFeature.OutlineSettings settings)
    {
        this.renderPassEvent = settings.PassEvent + 1; // 确保在Stencil写入后执行
        _settings = settings;

        // 加载或创建用于描边的后处理材质
        // 确保你有一个名为 "Hidden/OutlinePostProcess" 的Shader
        _outlineMaterial = CoreUtils.CreateEngineMaterial("Hidden/OutlinePostProcess");
    }

    // 这个方法会在Pass执行前被调用，用来配置输入
    public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
    {
        // 我们需要深度和法线信息来进行边缘检测
        ConfigureInput(ScriptableRenderPassInput.Depth | ScriptableRenderPassInput.Normal);
        _cameraColorTarget = renderingData.cameraData.renderer.cameraColorTargetHandle;
    }

    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        if (_outlineMaterial == null) return;

        var cmd = CommandBufferPool.Get("Post Process Outline Pass");

        // 更新Shader的参数
        _outlineMaterial.SetColor("_OutlineColor", _settings.OutlineColor);
        _outlineMaterial.SetFloat("_OutlineWidth", _settings.OutlineWidth);
        _outlineMaterial.SetFloat("_DepthThreshold", _settings.DepthThreshold);
        _outlineMaterial.SetFloat("_NormalThreshold", _settings.NormalThreshold);
        _outlineMaterial.SetInt("_StencilRef", _settings.StencilRef);

        // 使用Blitter来执行全屏的后处理效果
        // Blitter.BlitCameraTexture 会处理好源和目标RT
        Blitter.BlitCameraTexture(cmd, _cameraColorTarget, _cameraColorTarget, _outlineMaterial, 0);

        context.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);
    }
}
```

-----

### 第2步：创建所需的两个Shader

在你的项目文件夹下创建以下两个Shader。

#### 🎨 OutlineObject.shader

这个Shader给**需要被描边的物体**使用。它包含一个正常的渲染Pass和一个用于写入Stencil的特殊Pass。

```csharp
Shader "Custom/OutlineObject"
{
    Properties
    {
        _BaseColor("Base Color", Color) = (1,1,1,1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "Queue"="Geometry" }

        // Pass 0: 正常的渲染通道，用来绘制物体本身
        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode"="UniversalForward" }
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 normalWS : NORMAL;
            };
            
            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
            CBUFFER_END

            Varyings vert(Attributes input)
            {
                Varyings output;
                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = positionInputs.positionCS;
                VertexNormalInputs normalInputs = GetVertexNormalInputs(input.normalOS);
                output.normalWS = normalInputs.normalWS;
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                Light mainLight = GetMainLight();
                half NdotL = saturate(dot(input.normalWS, mainLight.direction));
                half3 lighting = NdotL * mainLight.color;
                return half4(_BaseColor.rgb * lighting, 1.0);
            }
            ENDHLSL
        }

        // Pass 1: Stencil写入通道
        // 这个Pass本身不输出任何颜色，它的唯一作用就是被我们的RenderPass捕获，
        // 以便在模板缓冲区上绘制该物体的形状。
        Pass
        {
            Name "OutlineStencil"
            Tags { "LightMode"="OutlineStencil" } // 自定义的LightMode，必须和C#脚本中的ShaderTagId一致

            ColorMask 0 // 不写入颜色和深度
            ZWrite Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            struct Attributes {
                float4 positionOS : POSITION;
            };
            
            struct Varyings {
                float4 positionCS : SV_POSITION;
            };

            Varyings vert(Attributes input) {
                Varyings o;
                o.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                return o;
            }

            half4 frag(Varyings i) : SV_TARGET {
                return 0; // 不输出任何颜色
            }
            ENDHLSL
        }
    }
}
```

#### 🎨 OutlinePostProcess.shader

这个Shader被`OutlinePass`用于执行屏幕空间描边，它必须放在`Resources`文件夹下，或者路径中包含`Hidden/`才能被`CreateEngineMaterial`找到。我们命名为 `Hidden/OutlinePostProcess`。

```csharp
Shader "Hidden/OutlinePostProcess"
{
    Properties
    {
        _MainTex ("Source Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            // --- Stencil Test ---
            // 这是实现Per-Object的关键
            // 只有当像素的模板值等于我们设定的Reference值时，这个Pass才会执行
            Stencil
            {
                Ref [_StencilRef]
                Comp Equal
            }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };
            
            TEXTURE2D_X(_MainTex);
            SAMPLER(sampler_MainTex);
            TEXTURE2D(_CameraDepthNormalsTexture);
            SAMPLER(sampler_CameraDepthNormalsTexture);

            CBUFFER_START(UnityPerMaterial)
                float4 _MainTex_TexelSize;
                half4 _OutlineColor;
                float _OutlineWidth;
                float _DepthThreshold;
                float _NormalThreshold;
                int _StencilRef;
            CBUFFER_END

            Varyings vert(Attributes input)
            {
                Varyings output;
                // 使用全屏三角形技巧，效率更高
                output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
                output.uv = GetFullScreenTriangleTexelUV(input.vertexID);
                return output;
            }

            // 解码深度和法线
            void DecodeDepthNormal(float2 uv, out float depth, out float3 normal)
            {
                float4 encoded = SAMPLE_TEXTURE2D(_CameraDepthNormalsTexture, sampler_CameraDepthNormalsTexture, uv);
                DecodeDepthNormal(encoded, depth, normal);
            }

            half4 frag(Varyings input) : SV_Target
            {
                half4 originalColor = SAMPLE_TEXTURE2D_X(_MainTex, sampler_MainTex, input.uv);
                float2 texelSize = _MainTex_TexelSize.xy * _OutlineWidth;

                // --- 边缘检测 ---
                float centerDepth, upDepth, downDepth, leftDepth, rightDepth;
                float3 centerNormal, upNormal, downNormal, leftNormal, rightNormal;
                
                DecodeDepthNormal(input.uv, centerDepth, centerNormal);
                DecodeDepthNormal(input.uv + float2(0, texelSize.y), upDepth, upNormal);
                DecodeDepthNormal(input.uv - float2(0, texelSize.y), downDepth, downNormal);
                DecodeDepthNormal(input.uv + float2(texelSize.x, 0), rightDepth, rightNormal);
                DecodeDepthNormal(input.uv - float2(texelSize.x, 0), leftDepth, leftNormal);

                // 深度边缘检测
                float depthDiff = 0;
                depthDiff += abs(centerDepth - upDepth);
                depthDiff += abs(centerDepth - downDepth);
                depthDiff += abs(centerDepth - leftDepth);
                depthDiff += abs(centerDepth - rightDepth);
                
                // 法线边缘检测
                float normalDiff = 0;
                normalDiff += dot(centerNormal, upNormal) < _NormalThreshold ? 1.0 : 0.0;
                normalDiff += dot(centerNormal, downNormal) < _NormalThreshold ? 1.0 : 0.0;
                normalDiff += dot(centerNormal, leftNormal) < _NormalThreshold ? 1.0 : 0.0;
                normalDiff += dot(centerNormal, rightNormal) < _NormalThreshold ? 1.0 : 0.0;
                normalDiff = saturate(normalDiff);

                // 结合两种边缘
                float edge = saturate(step(_DepthThreshold, depthDiff) + normalDiff);

                // 如果检测到边缘，返回描边色，否则返回原始颜色
                return edge > 0 ? _OutlineColor : originalColor;
            }
            ENDHLSL
        }
    }
}
```

-----

### 第3步：设置和使用

1.  **创建Render Feature资产**: 在`Project`窗口右键 -\> `Create` -\> `Rendering` -\> `URP` -\> `Custom` -\> `Outline Render Feature`。这会创建一个新的`ScriptableObject`资产。

2.  **添加到URP渲染器**: 找到你当前使用的`URP Renderer`资产（通常在`Settings`文件夹下，与`URP Asset`关联）。选中它，在Inspector中点击`Add Render Feature`，然后将你刚刚创建的`OutlineRenderFeature`资产拖进去。

3.  **调整设置**: 你现在可以直接在`URP Renderer`的Inspector中调整描边的颜色、宽度、灵敏度等参数。

4.  **创建描边材质**: 创建一个新材质，选择我们编写的 `Custom/OutlineObject` Shader。

5.  **应用材质**: 将这个新材质赋给任何你想要描边的物体。

现在，运行场景，只有挂载了`OutlineObject`材质的物体才会被描边，并且所有描边参数都可以通过Render Feature进行全局统一控制。
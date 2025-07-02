# Unity URP高斯模糊Shader实现

## Question 1

高斯模糊就是用高斯sober，反过来说高斯核说的就是用在高斯模糊里的卷积核，能不能用unityshader（urp hlsl）演示下怎么实现图像模糊？

## Answer 1

以下是在Unity URP中实现高斯模糊的Shader示例。我们将使用两个Pass分别进行水平/垂直模糊，并应用5x5高斯核：

```hlsl
Shader "Custom/GaussianBlur"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _BlurSize ("Blur Size", Range(0.0, 10.0)) = 1.0
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline" = "UniversalPipeline" }

        // Horizontal Blur Pass
        Pass
        {
            Name "HorizontalBlur"
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag_horizontal

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _MainTex_TexelSize;
            float _BlurSize;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = TransformObjectToHClip(v.vertex.xyz);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            static const float weight[3] = { 0.227027, 0.316216, 0.070270 };
            static const float offset[3] = { 0.0, 1.384615, 3.230769 };

            half4 frag_horizontal (v2f i) : SV_Target
            {
                half4 color = tex2D(_MainTex, i.uv) * weight[0];
                
                for(int j = 1; j < 3; j++)
                {
                    color += tex2D(_MainTex, i.uv + float2(_MainTex_TexelSize.x * offset[j] * _BlurSize, 0.0)) * weight[j];
                    color += tex2D(_MainTex, i.uv - float2(_MainTex_TexelSize.x * offset[j] * _BlurSize, 0.0)) * weight[j];
                }
                
                return color;
            }
            ENDHLSL
        }

        // Vertical Blur Pass
        Pass
        {
            Name "VerticalBlur"
            
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag_vertical

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _MainTex_TexelSize;
            float _BlurSize;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = TransformObjectToHClip(v.vertex.xyz);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            static const float weight[3] = { 0.227027, 0.316216, 0.070270 };
            static const float offset[3] = { 0.0, 1.384615, 3.230769 };

            half4 frag_vertical (v2f i) : SV_Target
            {
                half4 color = tex2D(_MainTex, i.uv) * weight[0];
                
                for(int j = 1; j < 3; j++)
                {
                    color += tex2D(_MainTex, i.uv + float2(0.0, _MainTex_TexelSize.y * offset[j] * _BlurSize)) * weight[j];
                    color += tex2D(_MainTex, i.uv - float2(0.0, _MainTex_TexelSize.y * offset[j] * _BlurSize)) * weight[j];
                }
                
                return color;
            }
            ENDHLSL
        }
    }
}
```

使用说明：

1. 创建新材质并应用此Shader
2. 将需要模糊的纹理赋值给_MainTex
3. 调整_BlurSize参数控制模糊强度
4. 通常需要配合Render Texture实现多次模糊迭代

实现特点：
1. 使用分离的水平和垂直Pass提升性能
2. 5x5高斯核优化为3次采样（利用对称性）
3. 基于texel size自动适配分辨率
4. 权重值预先计算优化性能
5. 支持模糊强度动态调整

扩展建议：
1. 添加多个Pass进行多次模糊迭代
2. 使用降采样技术提升性能
3. 添加LOD控制不同距离的模糊强度
4. 结合深度图实现景深效果

注意：实际使用时需要配合URP的Render Feature实现全屏后处理效果。可通过创建Renderer Feature并添加Blit Pass来应用此材质。

## Question 2

除了高斯sober，还有哪些卷积核也能实现各式各样的模糊效果？

## Answer 2

在图像处理中，除了高斯核之外，还有许多不同类型的卷积核可以实现各具特色的模糊效果。以下是常见的几种类型及其实现原理和效果特点：

---

### **1. 均值模糊（Box Blur）**
**原理**：所有相邻像素权重相等的最简单模糊
```hlsl
// 3x3均值核（归一化后）
static const float3x3 kernel = {
    1/9.0, 1/9.0, 1/9.0,
    1/9.0, 1/9.0, 1/9.0,
    1/9.0, 1/9.0, 1/9.0
};

// 快速实现（分离式计算）
half4 frag_box(v2f i) : SV_Target {
    float2 uv = i.uv;
    float2 texel = _MainTex_TexelSize.xy;
    
    half4 sum = tex2D(_MainTex, uv) * 0.25;
    sum += tex2D(_MainTex, uv + float2( texel.x, 0)) * 0.125;
    sum += tex2D(_MainTex, uv - float2( texel.x, 0)) * 0.125;
    sum += tex2D(_MainTex, uv + float2( 0, texel.y)) * 0.125;
    sum += tex2D(_MainTex, uv - float2( 0, texel.y)) * 0.125;
    sum += tex2D(_MainTex, uv + texel) * 0.0625;
    sum += tex2D(_MainTex, uv - texel) * 0.0625;
    return sum;
}
```
**特点**：计算速度快，但会产生明显的像素块状效果

---

### **2. 运动模糊（Motion Blur）**
**原理**：沿特定方向进行线性采样模拟运动轨迹
```hlsl
// 参数
float2 _Direction;  // 模糊方向（需归一化）
float _SampleCount; // 采样次数

half4 frag_motion(v2f i) : SV_Target {
    half4 color = 0;
    float2 step = _Direction * _MainTex_TexelSize.xy;
    
    for(int k = -_SampleCount; k <= _SampleCount; k++) {
        float weight = 1.0 - abs(k)/_SampleCount; // 线性衰减
        color += tex2D(_MainTex, i.uv + step * k) * weight;
    }
    return color / (_SampleCount*2 + 1);
}
```
**特点**：可模拟物体移动/摄像机拖影效果

---

### **3. 径向模糊（Radial Blur）**
**原理**：以中心点向外辐射状采样
```hlsl
float2 _Center;     // 模糊中心点
float _Intensity;   // 模糊强度

half4 frag_radial(v2f i) : SV_Target {
    half4 color = 0;
    float2 dir = i.uv - _Center;
    
    for(int k = 0; k < 8; k++) {
        float2 offset = dir * (k * _Intensity / 8.0);
        color += tex2D(_MainTex, i.uv - offset);
    }
    return color / 8.0;
}
```
**特点**：适合表现爆炸、能量聚集等效果

---

### **4. 散景模糊（Bokeh Blur）**
**原理**：模拟相机镜头的虚化光斑
```hlsl
// 六边形采样模式
static const float2 hexOffsets[7] = {
    float2(0,0),
    float2(0.5,0.866),
    float2(-0.5,0.866),
    float2(1,0),
    float2(0.5,-0.866),
    float2(-0.5,-0.866),
    float2(-1,0)
};

half4 frag_bokeh(v2f i) : SV_Target {
    half4 sum = 0;
    for(int k=0; k<7; k++){
        sum += tex2D(_MainTex, i.uv + hexOffsets[k]*_Radius);
    }
    return sum / 7.0;
}
```
**特点**：产生艺术化的光斑效果

---

### **5. 双边模糊（Bilateral Blur）**
**原理**：结合空间距离和颜色相似度的智能模糊
```hlsl
half4 frag_bilateral(v2f i) : SV_Target {
    half4 center = tex2D(_MainTex, i.uv);
    half4 sum = center;
    float totalWeight = 1.0;

    for(int m=-2; m<=2; m++){
        for(int n=-2; n<=2; n++){
            float2 offset = float2(m,n) * _MainTex_TexelSize.xy;
            half4 sample = tex2D(_MainTex, i.uv + offset);
            
            // 空间权重
            float spatial = exp(-(m*m + n*n) / (2.0*_SigmaS*_SigmaS));
            
            // 颜色权重
            float range = length(sample.rgb - center.rgb);
            range = exp(-range*range / (2.0*_SigmaR*_SigmaR));
            
            sum += sample * spatial * range;
            totalWeight += spatial * range;
        }
    }
    return sum / totalWeight;
}
```
**特点**：保留边缘的同时平滑纹理

---

### **6. 定向模糊（Directional Blur）**
**原理**：可自定义任意方向的模糊角度
```hlsl
float _Angle;      // 模糊角度（弧度）
float _Distance;   // 模糊距离

half4 frag_directional(v2f i) : SV_Target {
    float2 dir = float2(cos(_Angle), sin(_Angle));
    half4 color = 0;
    
    for(int k=-3; k<=3; k++){
        color += tex2D(_MainTex, i.uv + dir * k * _Distance);
    }
    return color / 7.0;
}
```

---

### **选择建议**
| 类型       | 性能消耗 | 典型应用场景                  | 特点                          |
|------------|----------|-----------------------------|-------------------------------|
| 均值模糊   | ★☆☆      | UI元素模糊、快速预览          | 最快但质量最低                |
| 高斯模糊   | ★★☆      | 通用后处理、景深              | 平滑自然，计算量适中          |
| 运动模糊   | ★★☆      | 速度线、摄像机运动效果        | 方向性动态感                  |
| 径向模糊   | ★★★      | 技能特效、镜头聚焦            | 中心扩散的视觉冲击力          |
| 双边模糊   | ★★★★     | 皮肤柔化、细节保留            | 保边去噪，计算复杂            |
| 散景模糊   | ★★★      | 摄影效果、光晕                | 艺术化虚化                    |

---

### **实现技巧**
1. **分离式卷积**：将二维卷积拆分为水平和垂直两个1D卷积（性能提升50%+）
2. **多级降采样**：先降低分辨率做模糊，再上采样（适合全屏后处理）
3. **动态采样数**：根据与摄像机的距离动态调整采样次数
4. **权重预计算**：将权重值预先存入数组避免实时计算
5. **利用Mipmap**：结合纹理的mipmap层级实现快速模糊

```hlsl
// 动态采样数示例
int samples = lerp(5, 15, smoothstep(0, 50, distance(_WorldSpaceCameraPos, positionWS)));
```

根据具体需求选择合适的方法，在移动端建议优先考虑均值模糊或优化版高斯模糊，PC端则可尝试更复杂的双边模糊等算法。
# Uber Shader减少变体

### 🧑‍💻 User

uber shader为什么可以减少变体数量？

---

### 🤖 Assistant

Uber Shader（超级着色器/万能着色器）之所以能减少变体（Variants）数量，核心在于它将**通用的渲染逻辑整合到了一个统一的代码库中**，并通过**参数化**或**宏定义**来控制功能的开启与关闭。

要准确理解这一点，我们需要区分“源代码层面的变体”和“编译后的二进制变体”，以及“静态分支”与“动态分支”的区别。

以下是 Uber Shader 减少变体数量的几个关键机制：

### 1. 逻辑复用与统一的光照模型 (PBR 的普及)

在非 Uber Shader 的时代（比如早期的固定管线或早期可编程管线），开发者习惯为每种材质写一个单独的 Shader：
*   `GlassShader` (玻璃)
*   `MetalShader` (金属)
*   `PlasticShader` (塑料)
*   `SkinShader` (皮肤)

这些 Shader 之间可能只有 10% 的代码是不同的（比如高光计算略有不同），但其余 90% 的代码（如坐标变换、阴影采样、雾效）是重复的。

**Uber Shader 的做法：**
利用基于物理的渲染（PBR），绝大多数材质都可以用一套统一的数学模型（如 Disney BRDF）来描述，区别仅仅是**参数**不同（粗糙度、金属度、基础色）。
*   **结果：** 你不再需要 10 个不同的 Shader 文件，只需要 **1 个** `StandardShader`。通过调整材质参数（Uniforms），它就能表现出金属、塑料或木头的效果。

### 2. 动态分支 (Dynamic Branching) vs 静态分支 (Static Branching)

这是减少**编译后变体（Compiled Variants）**数量最直接的技术手段。

#### 传统/静态分支做法 (产生大量变体)
通常为了性能，开发者使用预处理宏 (`#ifdef`) 来剥离不需要的代码：
```glsl
// Shader 代码
#ifdef HAS_NORMAL_MAP
    vec3 normal = texture(normalMap, uv).rgb;
#else
    vec3 normal = vec3(0, 0, 1);
#endif
```
如果有 10 个这样的功能开关，排列组合后会生成 $2^{10} = 1024$ 个不同的 GPU 程序（变体）。

#### Uber Shader 的动态分支做法 (减少变体)
Uber Shader 可以选择使用 `uniform` 变量（由 CPU 传入的布尔值或浮点数）配合 `if` 语句：
```glsl
// Uber Shader 代码
uniform bool useNormalMap; // 这是一个变量，不是宏

void main() {
    vec3 normal;
    if (useNormalMap) {
        normal = texture(normalMap, uv).rgb;
    } else {
        normal = vec3(0, 0, 1);
    }
    // ... 后续光照计算
}
```
*   **结果：** 无论是否使用法线贴图，GPU 编译出来的 Shader **只有 1 个**。
*   **代价：** 这种做法在旧硬件上会有性能损耗（GPU 不擅长做分支判断），但在现代 GPU 上，如果分支条件在同一个 Draw Call 中是统一的（Coherent），性能损耗通常是可以接受的。

### 3. 数据驱动 (Data Driven)

Uber Shader 倾向于使用**贴图和遮罩**来控制特性，而不是代码开关。

*   **旧方法：** 写一个 `SpecularShader` 专门处理高光，写一个 `MatteShader` 专门处理哑光。这是 2 个变体。
*   **Uber Shader：** 使用一张 **Mask Map (遮罩图)**。R 通道代表金属度，G 通道代表粗糙度。
    *   如果想要高光？把 Mask Map 画亮一点。
    *   如果想要哑光？把 Mask Map 画暗一点。
*   **结果：** Shader 代码不需要改变，变体数量为 1，材质的变化完全由美术资源（纹理）决定。

### 4. 变体剔除 (Variant Stripping)

虽然 Uber Shader 内部依然可能包含大量的 `#ifdef` 宏（比如 Unity 的 Standard Shader 实际上包含成千上万个变体），但它集中管理的特性使得**剔除**变得更容易。

*   如果是分散的 Shader 文件，只要项目里引用了这个文件，它往往就会被完整打包。
*   对于 Uber Shader，构建管线（Build Pipeline）可以分析场景中实际用到的材质配置。例如，如果你的游戏中没有任何一个材质开启了“视差映射（Parallax Mapping）”选项，引擎可以在编译阶段直接把包含该宏的变体全部丢弃。
*   这虽然不减少理论上的最大变体数，但大大减少了**最终打包到游戏中的变体数量**。

### 总结对比

| 特性 | 传统多 Shader 模式 | Uber Shader 模式 |
| :--- | :--- | :--- |
| **源文件数量** | 多 (几十甚至上百个文件) | 少 (通常 1 个核心文件 + 包含文件) |
| **材质表现** | 硬编码在 Shader 逻辑中 | 通过参数(Uniforms)和贴图控制 |
| **控制方式** | 切换不同的 Shader 程序 | 切换 Shader 参数 |
| **变体生成** | 线性增长 (每加一种材质写一个Shader) | 依然可能有组合爆炸，但通过动态分支和剔除来控制 |
| **维护成本** | 高 (修改一个光照公式要改所有文件) | 低 (改一次，所有材质生效) |

**一句话总结：**
Uber Shader 通过**PBR 统一光照模型**消除了针对特定材质编写特定代码的需求，并通过**动态分支（Uniforms）**或**构建时剔除**技术，将原本需要编译成不同 GPU 程序的逻辑合并到了更少的程序中，从而减少了最终的变体数量。
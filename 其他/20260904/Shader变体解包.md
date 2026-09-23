# Shader变体解包

## 🧑‍💻 User

拿yooasset收集shader变体打个ab再拿as等工具解包，你会发现shader变体几万个，就真的是几万个独立的shader文件，全是那种散装的shader，它似乎把keyword都展开了，这是怎么回事？

---
## 🤖 Assistant

先说结论：**这不是 YooAsset 干的，也不是打包出了 bug——是 Unity 构建管线把 Shader 的“关键字组合”在编译期全部展开、编译成了独立程序，而解包工具又把这些编译产物平铺导出成了一个个 `.shader` 文件。** YooAsset 只负责“哪些资源进哪个包”，它不参与 Shader 的编译或序列化。

---

## 1. 关键前提：AB 里根本没有 `.shader` 文本

在 Editor 项目里，`.shader` 是 ShaderLab 文本；但一旦进入 **构建阶段**（打 Player 或 AssetBundle），Unity 会把它编译成二进制形式，序列化进 `SerializedFile` 的是：

- `m_ParsedForm`：一个 `SerializedShader`，即“解析后的源码结构”（subshader/pass/关键字声明等）。
- `platforms[]`：每个图形 API 一份已编译数据（`SerializedSubProgram`，通过 `m_BlobIndex` 指向 `ShaderBlob`）。
- `m_ShaderIsBaked`：标记这个 shader 是否已经“烘焙/编译”过。

所以你用 AssetStudio / AssetRipper 解出来看到的 `.shader`，**全是工具从 `m_ParsedForm` 重建/导出的，不是原始文件**。

## 2. 为什么是“几万个”——关键字的组合爆炸

ShaderLab 里的这几类声明，会在编译期做**笛卡尔积**：

```hlsl
#pragma multi_compile _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
#pragma multi_compile _ADDITIONAL_LIGHTS _ADDITIONAL_LIGHTS_VERTEX
#pragma multi_compile_fog
#pragma multi_compile_instancing
#pragma multi_compile_fwdbase      // 内置管线里本身就是一堆开关
```

对某个 pass/stage，变体数是各 pragma 关键字数量的乘积：

$$N_{\text{pass,stage}} = \prod_{\text{pragma}} k_i$$

整个 Shader 再乘上 pass 数、stage 数（VS/FS）、图形 API 数：

$$N_{\text{总}} = \sum_{\text{pass}} \Big(\prod_{\text{pragma}} k_i\Big) \times S \times P$$

一个稍微复杂点的 URP/内置 Standard shader，单 pass 就能到几百上千个变体；几个 shader + 多个 pass + 多平台，**上万是极其正常的**。这就是你感觉“关键字都被展开了”的真正原因——**它确实被展开了，是 Unity 干的。**

## 3. 为什么解包出来是“一个个散装的 shader 文件”

因为你看到的每一份，本质是 **“同一个 shader + 不同关键字组合”的编译产物**。它们共用同一份 ShaderLab 源，只是被编译成了不同的程序。

- 当 `m_ParsedForm`（源码结构）在这些构建产物里不完整/不可用时，解包工具**无法把它们合并回一个带 `#pragma` 的源文件**，只能按变体 / `SerializedSubProgram` / `ShaderBlob` 逐条导出，于是你就看到“几万个独立散装 shader”。
- 这些文件之间大多只差几个关键字，甚至完全重复（跨平台、跨 stage 会产生冗余条目）。

> 不同工具/版本导出粒度不同：有的会尽量合并成一个 `.shader`，有的按变体平铺。你用 UABE 或直接看 UnityPy/AssetRipper 的 Shader 对象结构会看到：**一个 `Shader` 对象下面挂着一大片 `m_SubShaders → m_Passes → m_Programs.m_SubPrograms`**，那才是“真相”。

## 4. 顺带澄清：不是 YooAsset 的锅

YooAsset 的 shader 收集（以及 `ShaderVariantCollection` / Graphics 设置里的预加载 shader）只决定**“让 Unity 把哪些 shader 及其变体保留 / 打进包”**。真正把关键字展开、编译、序列化成海量变体的是 **Unity 的 shader 编译与打包流程**。换句话说，YooAsset 只是“点菜”的，炒菜和摆盘的是 Unity。

## 5. 怎么验证与怎么缓解

- **验证**：用 AssetRipper / UABE 打开那个 AB，找一个 Shader 对象，看 `m_KeywordNames` / `m_SubPrograms` 的数量；再对照源工程里那个 `.shader` 的 `#pragma multi_compile*` 数一数，乘积基本能对上。
- **缓解**（按优先级）：
 1. 能不用 `multi_compile` 的地方改用 `#pragma shader_feature`（未被使用的组合会被剥离）。
 2. 用 `shader_feature_local` / 合并互斥关键字，减少组合数。
 3. 打开项目设置的 **Strip Unused Variants**，并用 `ShaderVariantCollection` 显式 warmup 需要的组合。
 4. 实现 `IPreprocessShaders` 在构建时精确丢弃不要的变体。
 5. **分平台打包**，避免不同 API 的变体挤在同一个包里。
 6. 确认这个“变体 AB”里是否把 URP/HDRP/SRP Core 等包内置 shader 也收了进来——它们通常是“几万个”的主要来源，可用 YooAsset 的收集规则/标签过滤掉不需要的。
 7. 用 `shadercompiler` 的 variant 限制、或自定义 shader 变体日志（`ShaderUtil`）统计哪些变体真的被用到。

<details>
<summary>一句话总结（点开看）</summary>

Unity 在构建时把 `#pragma` 声明的关键字做了组合爆炸式编译，AB 里存的是“一个 Shader 对象 + 它的成千上万个编译变体”，解包工具（AssetStudio 等）把这些变体平铺成了一个个散装 `.shader` 文件。YooAsset 只负责选择哪些 shader 进包，不负责展开关键字。

</details>

如果你愿意把那个 shader 的几个 `#pragma` 行贴出来，我可以直接帮你估算变体数量，并给出具体的裁剪方案。

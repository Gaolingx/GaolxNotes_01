# Shadertoy作为LLMbenchmark

### 🧑‍💻 User

shadertoy可以作为llm benchmark吗

---

### 🤖 Assistant

# 结论先说：可以，而且已经有人在做了

Shadertoy 完全有潜力作为 LLM 的 benchmark，但它更适合定位成**"代码生成 + 视觉/数学推理"的垂直评测**，而不是像 HumanEval 那样衡量通用编程能力的主标杆。目前已经有一些真实的项目在往这个方向走。

---

## 一、为什么 Shadertoy 天然适合做 benchmark

| 特性 | 说明 | 对评测的价值 |
|------|------|------|
| **自包含、单文件** | 一个 fragment shader 就是整个程序，几乎没有外部依赖 | 评测环境搭建简单，无需处理包管理、API key |
| **可编译、可渲染** | GLSL 能真实编译并跑出图像 | 有客观的"对/错"基础：编译通过率、渲染结果 |
| **确定性（近似）** | 输入只有 `fragCoord` / `iTime` / 分辨率，输出可复现 | 能固定分辨率/时间做像素级对比 |
| **难度跨度大** | 从棋盘格 → 噪声 → SDF 建模 → raymarching → 复杂光照 | 天然的难度分级，能区分模型水平 |
| **考察的是"硬"能力** | 隐式几何、距离场（SDF）、矩阵/向量运算、空间推理 | 比写 CRUD 代码更能测出模型的上限 |

其中最关键的一点是：**Shadertoy 的"正确性"是视觉的、可渲染的**，这正好填补了传统代码 benchmark 的一个空白——传统 benchmark 只看功能正确性，而 shader 还要看"画得像不像、对不对"。

---

## 二、主要的难点 / 局限（这也是它没成为主流标杆的原因）

1. **评估指标难统一**
   - 像素级全等（PSNR 高）太严苛：两个都"正确"的 shader 可能因抗锯齿、配色不同而差异巨大。
   - 需要感知指标（SSIM、LPIPS）或 **VLM/LLM-as-judge** 来判"视觉上对不对"，但这又引入主观性和 judge 偏差。

2. **提示天然有歧义**
   - "画一个旋转的棋盘格立方体"没有唯一正确答案，难以像 HumanEval 那样有确定的单元测试。

3. **领域小众、训练数据占比低**
   - GLSL/shader 代码在 LLM 预训练语料里比例远低于 Python/JS，因此分数可能更多反映"见过多少 shader"，而非通用推理能力，**外推性存疑**。

4. **环境噪声**
   - 不同 GPU / WebGL 实现有未定义行为；`iTime`、鼠标交互带来的时间依赖；"帧率/性能"也不是唯一目标。

5. **缺乏标准化**
   - 目前没有一个像 ImageNet 那样公认的 test set + 参考图像 + 统一指标。

---

## 三、已有相关工作（搜索确认）

- **`nbardy/shader_benchmark`**（GitHub）：101 道数学可视化题（柏拉图立体 → Calabi-Yau 流形），Rust WGPU 编译 WGSL，用 GPT-4o 作 judge，按 5 个维度（数学准确性、视觉质量等）各 100 分打分。是一个比较完整、可运行的流水线。
- **`harsh2204/ShaderBench`**：另一个 shader 编程评测 benchmark。
- **AI Co-Artist**（arXiv，2025）：LLM 驱动的交互式 GLSL shader 生成框架。
- **artificialanalysis.ai 的 "LLM Ultimate Challenge: Interactive GLSL Shader Art"**：把生成"单个 HTML + GLSL"的交互式 shader art 作为高难度微评测。
- 各种 **Shadertoy MCP server**（让 Claude 等直接生成/检索 GLSL shader）。

---

## 四、怎么把它做成一个靠谱的 benchmark（我的建议）

- **编译通过率**作为第一道硬门槛（类似 `pass@1`）。
- **固定分辨率 + 固定 `iTime`** 渲染，与参考图做感知相似度（SSIM/LPIPS），而不是逐像素相等。
- 用 **VLM 判断"是否存在指定物体/特征"**（如"画面中是否有旋转立方体"）来做可解释的分数，比纯像素更稳健。
- 题目分层：基础图形 → 分形 → SDF/raymarching → 物理模拟，并公开难度标签。
- 保留"**性能预算**"维度（编译 + 帧率），但作为次要指标。

---

## 总结

> **Shadertoy 作为 LLM benchmark 是可行的，且具备独特价值**——它同时考察数学/空间推理、代码合成和视觉正确性，正好卡在多模态（代码 → 图像）评测的交汇点上。
>
> 但它短期内更适合作**研究型/辅助指标**，或作为**多模态评测（VLM-as-judge）**的一个有趣子集，而不太可能取代 HumanEval/SWE-bench 那样的通用标杆——除非有人先解决"标准化 test set + 稳定感知指标"这两个核心问题。
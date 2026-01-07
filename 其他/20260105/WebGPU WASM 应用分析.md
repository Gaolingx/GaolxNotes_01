# WebGPU WASM 应用分析

### 🧑‍💻 User

目前webgpu在wasm上有哪些应用？

---

### 🤖 Assistant

目前 **WebGPU + WASM** 的组合已经从“实验性技术”进入到**有明确落地场景但仍在快速演进**的阶段。可以从 **应用类别、典型项目、技术成熟度** 三个维度来理解它现在“能干什么、适合干什么、不适合干什么”。

---

## 一、核心定位：为什么一定要 WebGPU + WASM

先明确一点：
**WebGPU 并不是为 JS 写高性能代码准备的，而是为 WASM 提供“GPU 级能力”的标准接口。**

* JS：控制流、调度、胶水代码
* WASM：数值密集 / 状态机 / SIMD / 编译器后端
* WebGPU：统一访问 GPU（Compute + Graphics）

这使得浏览器第一次具备了：

> **接近原生（Native）GPU 计算 + 可移植分发**

---

## 二、当前主要应用方向（按成熟度排序）

### 1. 浏览器端机器学习（最成熟）

这是 **WebGPU 在 WASM 上最成功的方向**。

#### 典型应用

* **LLM 推理（小模型 / 量化模型）**
* **CV / 音频 / embedding / diffusion 推理**
* **端侧隐私计算（数据不出浏览器）**

#### 代表项目

* **ONNX Runtime Web (WebGPU backend)**
* **WebLLM / MLC-LLM**
* **Transformers.js（WebGPU 路径）**
* **MediaPipe WebGPU**

#### 技术结构

```
模型算子 → (C++/Rust) → WASM
                  ↓
               WebGPU Compute Shader
```

#### 现状评估

* 性能：

  * 小模型（<1B）：可用
  * 中模型（3–7B）：边缘可用（强依赖显存 / 量化）
* 瓶颈：

  * shader 编译时间
  * bind group / dispatch overhead
  * 缺少 persistent kernel / graph capture

**结论**：

> WebGPU + WASM 是目前“浏览器内 ML 推理”的事实标准方向。

---

### 2. 科学计算 / 数值计算（稳定增长）

#### 应用场景

* 线性代数（GEMM / FFT）
* 物理仿真
* 金融 Monte Carlo
* 图像 / 视频算法原型

#### 代表项目

* **wgpu + Rust + WASM**
* **WebGPU-based BLAS / FFT demo**
* **Taichi WebGPU backend（实验）**

#### 技术优势

* WASM：

  * 可直接复用 HPC 代码
  * SIMD + 手写内存布局
* WebGPU：

  * Compute-only pipeline
  * 不依赖 WebGL legacy API

#### 限制

* 无 unified memory
* GPU ↔ WASM 内存 copy 成本高
* 调度灵活性不如 CUDA/HIP

**结论**：

> 非实时、强计算密集型数值计算已经可用，但仍不适合“极端性能追求”。

---

### 3. 可视化 / CAD / 专业工具（渐进替代 WebGL）

#### 应用场景

* CAD / BIM
* 医学影像（DICOM）
* 大规模点云 / mesh
* 数据可视化（百万级 primitive）

#### WebGPU 的优势

* 明确的 pipeline / resource model
* Compute + Render 深度融合
* 更接近 Vulkan / DX12 思维模型

#### WASM 的角色

* 几何处理
* 空间索引（BVH / KD-tree）
* 数据预处理与压缩

#### 代表实践

* Figma（部分内部 WebGPU）
* Cesium（实验性 WebGPU）
* 专业浏览器 CAD 工具原型

**结论**：

> WebGPU 正在成为 WebGL 的“长期继任者”，WASM 是不可或缺的算力补充。

---

### 4. 浏览器内游戏引擎（早期）

#### 现状

* Unity WebGPU：实验阶段
* Bevy / Godot：原型支持
* WebAssembly 游戏引擎 demo

#### 优势

* 更低 CPU overhead
* Compute-based culling / animation
* 更贴近主机级渲染模型

#### 问题

* shader toolchain 不成熟
* asset streaming 复杂
* 移动端支持碎片化严重

**结论**：

> 游戏不是 WebGPU 的短期主战场，但中长期潜力巨大。

---

### 5. 隐私计算 / 本地 AI Agent（新兴）

这是一个**非常值得关注的新方向**。

#### 特点

* 数据永不离开浏览器
* WASM 执行策略逻辑
* WebGPU 执行重计算（embedding / rerank / local LLM）

#### 应用

* 本地搜索增强
* 浏览器内 RAG
* 企业内私有数据分析

**结论**：

> WebGPU + WASM 使“可信浏览器计算”第一次成为现实。

---

## 三、目前明确不适合的方向

### ❌ 极限 HPC / 超算级负载

* 无 peer-to-peer
* 无 NCCL
* 无多 GPU 编排

### ❌ 大规模训练

* 内存模型受限
* 无通信原语
* 无高效 graph replay

### ❌ 强实时低延迟系统

* JS/WASM ↔ GPU 调度抖动
* 浏览器调度不可控

---

## 四、和你背景强相关的一点观察

结合你之前关注的：

* AVX2 / BF16 性能路径
* fused matmul / MoE kernel
* LLM 推理算子结构

**一个非常现实的趋势是：**

> WebGPU compute shader 正在成为
> **“最通用的可移植 GPU IR 之一”**

很多团队已经在做：

```
一个 kernel 描述
→ CUDA / HIP / Metal / WebGPU
```

WASM 在这里扮演的是：

* 算子调度器
* memory planner
* quantization / packing 逻辑

---

## 五、一句话总结

> **WebGPU + WASM 目前最成熟的应用是浏览器内 ML 推理，其次是数值计算与专业可视化；它不是 HPC 或训练平台，但正在成为“可分发 GPU 计算”的事实标准。**

如果你愿意，下一步我可以从 **“WebGPU compute 与 CUDA kernel 在执行模型上的关键差异”** 或 **“为什么 WebGPU 的 kernel fusion 比你想象的更难”** 角度继续展开。

# AMD MI250X硬件信息

### 🧑‍💻 User

查询amd mi250x hardware信息、算力情况、软件支持情况

---

### 🤖 Assistant

我来帮您查询 AMD MI250X 的信息。首先查看有哪些可用的搜索工具。

我看到了可用的工具。没有专门的搜索工具，但我可以使用网页抓取工具获取 AMD MI250X 的官方信息。让我并行获取几个权威来源的信息。

已获取 AMD 官方规格和 Wikipedia 信息。现在让我补充查询 ROCm 软件生态对 MI250X 的支持情况。

我已经收集到 AMD 官方、Wikipedia 和 ROCm 官方文档的完整信息。以下是 AMD Instinct MI250X 的详细汇总：

---

# AMD Instinct™ MI250X 完整信息

## 一、硬件信息

| 项目 | 规格 |
|------|------|
| **产品系列** | AMD Instinct MI200 Series |
| **发布日** | 2021 年 11 月 8 日 |
| **GPU 架构** | CDNA 2（核心代号：Aldebaran） |
| **制程工艺** | TSMC 6nm FinFET |
| **晶体管数量** | 582 亿（58.2B） |
| **封装形态** | OAM 模块（被动散热） |
| **总线接口** | PCIe® 4.0 x16 |
| **TDP** | 500W（峰值 560W） |

**计算核心结构（双 GCD 设计）：**
- **流处理器**：14,080 个（2 × GCD，每 GCD 110 个计算单元）
- **计算单元（CU）**：220 个（每 GCD 110 个）
- **峰值引擎频率**：1700 MHz
- **GCD 互联**：Infinity Fabric 链接 8 条，峰值带宽 100 GB/s（GCD 间双向 400 GB/s）
- 双 GCD 在主机看来是**两个独立 GPU**（2 × GCD 寻址模式）

**内存规格：**
| 项目 | 规格 |
|------|------|
| 显存容量 | 128 GB（2 × 64 GB HBM2e） |
| 显存类型 | HBM2e |
| 内存位宽 | 8192-bit（2 × 4096-bit） |
| 内存时钟 | 1.6 GHz |
| **峰值内存带宽** | **3.2 TB/s** |
| ECC | 支持（全芯片） |

**RAS 与可靠性**：支持 RAS、Page Retirement（页退休）、ECC 全芯片保护。

---

## 二、算力情况（峰值）

| 精度类型 | 算力 |
|----------|------|
| **FP16（半精度）** | **383 TFLOPS**（矢量 47.9 × 8 矩阵加速） |
| **BF16** | 383 TFLOPS |
| **FP32 矩阵（Matrix）** | 95.7 TFLOPS（2× 加速） |
| **FP32（单精度矢量）** | 47.9 TFLOPS |
| **FP64 矩阵（Matrix）** | 95.7 TFLOPS（2× 加速） |
| **FP64（双精度矢量）** | **47.9 TFLOPS** |
| **INT8** | 383 TOPS |
| **INT4** | 383 TOPS |

> ⚠️ **关键亮点**：MI250X 的 **FP64 双精度为全速率（1:1）**，这与 NVIDIA 数据中心 GPU 的 FP64 通常仅 1:32~1:64 速率形成鲜明对比。因此 MI250X 在 HPC 科学计算（需要高精度）领域极具优势——全球首台百亿亿次（Exascale）超算 **Frontier** 正是基于 MI250X 构建。

**缓存/片上结构**（ROCm 官方数据）：
- L3 Cache：16 MiB（每 GCD 8 MiB），L2 Cache：16 MiB
- LDS（共享内存）：64 KiB，VGPR：512 KiB，Wavefront 大小：64

---

## 三、软件支持情况

### 1. ROCm 官方支持状态 ✅
- 在最新 **ROCm 7.14.0** 中，MI250X 为 **官方完全支持（Supported）** 状态，LLVM 目标为 **`gfx90a`**
- 注意：MI100（gfx908）之前的旧卡（MI25/MI50 等）已不再受 ROCm 官方支持

### 2. 支持的操作系统（Linux）
MI200 系列（MI250X/MI250/MI210）支持除 Debian 13、Rocky Linux 9、Oracle Linux 8/9/10 外的全部 ROCm 支持发行版：

| 操作系统 | 内核 | 支持 |
|----------|------|------|
| Ubuntu 24.04.4 | 6.8 / 6.17 HWE | ✅ |
| Ubuntu 22.04.5 | 5.15 / 6.8 HWE | ✅ |
| RHEL 10.1 / 10.0 | 6.12 | ✅ |
| RHEL 9.7 / 9.6 / 9.4 | 5.14 | ✅ |
| RHEL 8.10 | 4.18 | ✅ |
| SLES 15 SP7 | 6.4 | ✅ |
| Debian 12 | 6.1 | ✅ |
| Rocky Linux 9 / Oracle Linux 8/9/10 | — | ❌ 不支持 |

> 注：RHEL 8.10 仅支持 Instinct MI300X/A、MI250X、MI250、MI210、MI100。

### 3. 软件生态（ROCm 组件）
- **编程模型**：HIP（CUDA 兼容 API，可通过 HIPIFY 工具移植 CUDA 代码）、OpenCL、HIP C++、Python
- **数学库**：rocBLAS、hipBLAS、hipBLASLt、hipSPARSE、rocSPARSE、hipSOLVER、rocSOLVER、hipFFT、rocFFT、hipRAND、rocRAND、hipCUB、rocPRIM、rocThrust、rocWMMA、Composable Kernel
- **深度学习库**：MIOpen（AMD 版 cuDNN）、RCCL（AMD 版 NCCL，多卡通信）
- **AI 框架**：PyTorch、TensorFlow、JAX、DGL，推理框架 vLLM、SGLang（均有 ROCm 官方安装支持）
- **工具链**：AMD SMI、ROCm Data Center Tool (RDC)、rocminfo、ROCm 验证套件 (RVS)、rocprofiler（性能分析）、ROCgdb（调试）

### 4. 虚拟化支持
- **KVM SR-IOV + Passthrough**：ROCm 官方支持 MI210 的虚拟化（同代 MI200 系列，MI250X 使用场景多为整卡直通物理机，未单独列入虚拟化矩阵）

### 5. CPU 平台要求
- 需要 CPU 支持 **PCIe atomics**（AMD Zen 第一代及以后、Intel Haswell 及以后）

---

## 四、补充说明

- **定位**：MI250X 是 AMD 面向超大规模 HPC 和科学计算的第一款 Exascale 级加速卡，也是 AMD 在 TOP500/Green500 登顶的功臣（Frontier 超算，2022 年起全球最快）
- **与后续产品对比**：MI250X（CDNA 2）之后 AMD 推出了 MI300X（CDNA 3，192GB HBM3，面向生成式 AI）和 MI350X（CDNA 4），后者更侧重 AI 推理/训练
- **注意**：MI250X 面向 HPC 计算，不支持图形输出（无渲染单元，纯计算卡）

---

**信息来源**：
- AMD 官方产品页：`amd.com/en/products/accelerators/instinct/mi200/mi250x.html`
- Wikipedia: AMD Instinct
- ROCm 官方文档（ROCm 7.14.0）：系统要求、GPU 架构规格页

如需了解具体某方面（如 PyTorch 安装步骤、与其他型号算力对比、价格/部署案例等），可以继续告诉我。
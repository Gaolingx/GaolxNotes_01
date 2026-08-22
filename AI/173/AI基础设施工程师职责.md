# AI基础设施工程师职责

### 🧑‍💻 User

说说ai infra engineer主要职责

---

### 🤖 Assistant

# AI Infra Engineer 主要职责

AI Infra Engineer(AI 基础设施工程师)是支撑 AI 模型训练、推理和部署的核心角色。主要职责如下:

## 1. 训练基础设施

- **大规模分布式训练**:搭建和优化多机多卡训练系统,涉及数据并行、张量并行、流水线并行(如 Megatron-LM、DeepSpeed、FSDP)
- **训练稳定性保障**:处理节点故障、断点续训(checkpoint)、弹性训练
- **性能优化**:提升 MFU(Model FLOPs Utilization),优化通信开销(NCCL、RDMA、InfiniBand)

## 2. 推理与服务化

- **推理引擎优化**:使用 vLLM、TensorRT-LLM、SGLang 等框架,优化吞吐和延迟
- **推理加速技术**:KV Cache 管理、量化(INT8/FP8/AWQ)、Continuous Batching、Speculative Decoding
- **服务部署**:构建高可用、可扩展的模型服务(负载均衡、自动扩缩容)

## 3. 集群与资源管理

- **GPU 集群运维**:基于 Kubernetes、Slurm 等进行资源调度和任务管理
- **资源利用率优化**:GPU 共享、任务排队、多租户隔离
- **成本控制**:混合云策略、Spot 实例利用、算力预算管理

## 4. 数据基础设施

- **数据管道**:构建高吞吐的数据加载和预处理流水线
- **存储优化**:处理 PB 级训练数据的存储与读取(对象存储、分布式文件系统)

## 5. 工具链与平台建设

- **ML 平台开发**:实验管理、模型版本管理、监控告警(如 W&B、MLflow 集成)
- **CI/CD**:模型训练与部署的自动化流程
- **可观测性**:GPU 利用率、训练指标、服务 SLA 监控

## 常见技能要求

| 类别 | 技术栈 |
|------|--------|
| 语言 | Python、C++/CUDA、Go |
| 框架 | PyTorch、DeepSpeed、Megatron、vLLM |
| 系统 | Kubernetes、Docker、Slurm、Ray |
| 硬件 | GPU 架构、NVLink、InfiniBand、RDMA |
| 底层 | CUDA 编程、Triton kernel、NCCL 通信 |

## 与相关岗位的区别

- **vs 算法工程师**:更关注"怎么高效地跑起来",而非模型效果
- **vs 传统 SRE/DevOps**:需要深入理解 GPU 硬件和分布式训练/推理的特性
- **vs MLOps**:AI Infra 更偏底层性能和大规模系统,MLOps 更偏流程与工具链

简言之,这个岗位的核心价值是:**让模型训练更快、更稳、更省钱;让模型推理更高吞吐、更低延迟**。
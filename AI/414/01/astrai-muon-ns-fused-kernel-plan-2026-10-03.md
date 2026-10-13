# AstrAI Muon Newton–Schulz 融合算子计划

日期：2026-10-03。目标平台为 RTX 5090（SM 120a）、PyTorch 2.11、BF16、AstrAI-V1-base（约 1.204B 参数）。本计划只涉及单 CUDA stream 的 Muon 正交化；训练有效 batch、学习率、动量、NS 系数和 **5 次 NS 迭代**保持不变。

## 基线与目标

当前单卡优化器步中位数约 179.78 ms（启用现有 NS 缓冲区复用）；双卡 DDP 训练步均值约 352.64 ms。1B 模型共有 168 个 Muon 矩阵，形状为 `(1536,1536)`×48、`(256,1536)`×48、`(6912,1536)`×48、`(1536,6912)`×24。高矩阵先转置，使 NS 工作矩阵的行数不大于列数。每步共 840 次 `mm`、1680 次 `addmm`。当前 profile 中 GEMM 占约 94% 的已记录 CUDA 时间；大矩阵的 `X @ Xᵀ`、Gram 平方和更新乘法是主要成本。

验收目标是相对**当前缓冲区复用路径**，优化器单独耗时降低至少 10%，且固定有效 batch 的完整 DDP 步时降低至少 3%。这是进入默认配置评估的门槛，不是预先承诺的收益；达不到门槛则保留实验实现但不接入训练。所有结论必须报告吞吐、CUDA 时间、峰值已分配/保留显存和编译开销。

## 运算边界

每轮计算 `A = X Xᵀ`、`B = bA + cA²`、`X_next = aX + BX`。三个矩阵乘法之间存在全矩阵数据依赖，不能把五轮 NS 简单变成一个普通 GEMM kernel。可融合的是每次 GEMM 的输出阶段，以及对称矩阵乘法中重复的上三角计算。

| 部分 | 首选实现 | 回退 | 数值影响 |
|---|---|---|---|
| 归一化、转置 | 保留现有 PyTorch BF16 路径 | 原路径 | 首阶段不改变 |
| `A = X Xᵀ` | 5090 专用对称 GEMM，只算三角块并镜像写出完整 `A` | `torch.mm` | 镜像与不同归约次序可能改变末位 |
| `B = bA + cA²` | 对称 GEMM 与 `bA`/`c` 融合输出；仍生成完整 BF16 `B` | `torch.addmm` | 必须保留 BF16 存储边界和系数 |
| `X_next = aX + BX` | 先保留 cuBLAS/PyTorch `addmm`；仅在前两项盈利后试定制输出阶段 | `torch.addmm` | 定制 GEMM 需单独验证 |
| 动量、权重衰减、参数更新 | 第一阶段不动 | 当前实现 | 保持 optimizer 状态语义 |

当前 profile 还记录了 1680 次设备内拷贝（约 5.2 ms）。先用时间线确认它们是否由 `addmm` 的输出布局或预拷贝引起，再决定是否在第二阶段消除；仅凭调用数相同不能断定因果。`Command Buffer Full` 属 profiler 事件，不直接作为生产瓶颈优化。

## 实施顺序

### 0. 固定基线与算子契约

记录当前 `main` SHA、GPU UUID、PyTorch/CUDA 版本、编译目标和四种矩阵形状。为每个形状固定随机输入、系数、BF16 布局和 5 轮输出；分别记录原版 PyTorch、现有复用路径的 CUDA events、数值和显存。统计每轮三个 GEMM 的时间及拷贝关联。基准使用交错 ABBA 顺序，避免 GPU 频率变化误判。

### 1. 只替换第一个对称 GEMM

在 `csrc/optim/` 新增独立 `muon_ns` 扩展，首个入口只计算 `X Xᵀ`。对角块正常计算；非对角块只计算一侧，并写入转置位置，返回与 PyTorch 相同形状和 BF16 dtype 的完整矩阵。先针对 `(1536,6912)` 和 `(1536,1536)` 调整 tile 与流水线，再覆盖 `(256,1536)`；小形状可保留 cuBLAS 回退。每个形状独立选择，禁止为了单个大矩阵拖慢其余 96 个参数。

### 2. 融合第二个 GEMM 的输出

在第一阶段有收益后，实现 `B = bA + cA²` 的对称 GEMM 输出阶段。结果必须在每轮结束时写成 BF16，不能把五轮中间结果私自改为 FP32 或更高精度；否则虽然可能更快或更准，但已改变训练算法。保留完整 `B`，供现有第三个 GEMM 使用。若镜像后的误差不达标，可回退为全矩阵 GEMM 加输出融合。

### 3. 有条件地处理第三个 GEMM

`B @ X` 不是对称乘法，先保留性能成熟的 cuBLAS。只有 profile 显示布局拷贝或输出阶段明确可省，并且隔离基准快于 `torch.addmm`，才实现定制 epilogue。不要一次替换三个 GEMM，以便定位速度和数值差异。

### 4. 接入优化器

在 `astrai/extension/kernel/` 增加薄 Python 包装，使用现有 `is_available/get_module` 懒加载；在 `csrc/CMakeLists.txt` 注册独立模块。训练入口增加显式 backend 选择（例如 `torch`、`cuda_syrk`），默认保持当前行为。仅对 CUDA、BF16、连续二维 dense Tensor、受支持的形状启用新核；编译缺失、架构不符或形状未覆盖时自动回退。FSDP 的 DTensor 路径第一版不动，待 dense 更新等价和通信路径另行验证。

## 数值与收敛门槛

1. **不改算法**：固定 NS=5、系数、eps、归一化、转置规则、动量和参数更新顺序；比较每轮 `A`、`B`、`X_next`，而非只比较最后 loss。
2. **优先逐元素相同**：若 kernel 选择使 BF16 舍入不同，明确记录每个形状的最大绝对误差、相对 L2、非有限值数量及相对更新方向；不得声称 bitwise 等价。
3. **训练验证**：同一初始权重与数据顺序，做多步真实前向/反向及至少三种 seed 的较长训练曲线；比较 loss、梯度、参数距离、NaN/Inf 和最终验证指标。把运行间自然方差作为判断界限。不能证明收敛行为可接受时，只留 opt-in，不改默认。
4. **分布式验证**：先单卡，再双卡 DDP；FSDP 保持旧路径。检查最终参数、optimizer state、checkpoint 恢复和梯度累积。有效全局 batch 和每卡 batch 均固定。

## 性能与资源约束

- 每次运行前检查 GPU 空闲；优先 GPU 0，忙则选择其他空卡。单卡设置约 50% 的进程显存上限，双卡每进程约 80%；短程、顺序运行，不在 8 卡上并发扫参。
- 编译只针对 `120a` 和独立模块，限制 `BUILD_PARALLEL=2`，避免全项目高并行编译耗尽主机内存。
- 首轮只用四种真实矩阵形状的算子级基准；逐轮记录每个 shape 的延迟、显存、kernel 数与正确性。随后比较完整 1B 优化器和固定 batch 的双卡 DDP。报告重复运行的中位数与波动，避免把 profiler 时间当生产步时。
- 当前大矩阵单个 `X` 为约 20.25 MiB、每个 1536² BF16 Gram 约 4.5 MiB。临时缓冲区按单矩阵生命周期复用，不为 168 个参数常驻缓存；首先保持或降低当前峰值显存。
- 若目标收益不达标、显存显著上升、编译过慢或数值/收敛门槛失败，停止接入并保留 PyTorch 回退。

## 外部实现参考

- [PyTorch Muon 源码](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py)：当前参考运算顺序。
- [fused-muon 的 CuTe/SYRK 实现](https://github.com/StarrickLiu/fused-muon)：三角计算和输出阶段设计；其 A800 结果不能直接迁移为 5090 预期。
- [NVIDIA Megatron Muon 说明](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)：前两个 GEMM 可映射为 SYRK，已有 Triton 实现。
- [ThunderKittens 对称乘法论文](https://www.lakernewhouse.com/assets/writing/faster-symmul-with-thunderkittens.pdf)：提供性能思路，但文中转置写回有已知 bug，不能直接用于训练。

# vLLM首次加载模型优化

### 🧑‍💻 User

vllm第一次加载模型时候，会进行哪些模型优化？

---

### 🤖 Assistant

# vLLM 首次加载模型时的优化

vLLM 在首次加载模型时会进行一系列深度优化，涵盖**内存管理、计算图优化、量化、并行化**等多个层面。以下是详细的分析：

---

## 1. 内存管理优化 — PagedAttention

这是 vLLM 最核心的创新：

- **KV Cache 的分页管理**：借鉴操作系统虚拟内存的分页思想，将 KV Cache 划分为固定大小的 **Block**（页）。
- **Block 预分配**：在模型加载阶段，vLLM 会根据 GPU 显存大小，**预先计算并分配**能容纳的最大 KV Cache Block 数量。
- **Block Manager 初始化**：创建 `BlockSpaceManager`，管理物理 Block 的分配、释放和映射。

```python
# 简化逻辑示意
# 在 worker 初始化时进行 profiling
num_gpu_blocks, num_cpu_blocks = self._profile_num_available_blocks(...)
# 预分配 KV Cache
self._initialize_kv_caches(num_gpu_blocks, num_cpu_blocks)
```

### 显存 Profiling 流程：
1. 先加载模型权重，记录模型本身占用的显存
2. 用一个**虚拟的最大 batch** 做一次前向传播，测量峰值显存
3. 用 `总显存 - 模型显存 - 峰值激活显存 - 预留buffer` = **可用于 KV Cache 的显存**
4. 根据每个 Block 的大小，计算出 `num_gpu_blocks`

---

## 2. 模型权重加载与转换

### 2.1 权重格式转换
- 将 HuggingFace 格式的权重映射到 vLLM 自定义的模型架构
- **权重名称映射**（stacked mapping）：例如将 `q_proj`, `k_proj`, `v_proj` 的权重合并为一个 `qkv_proj`

```python
# 典型的权重合并映射
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]
```

### 2.2 量化处理（如果启用）
- **GPTQ / AWQ / SqueezeLLM / FP8** 等量化方案的权重反量化或重打包
- 将量化参数（`scales`, `zeros`）转换为适合自定义 kernel 的格式

### 2.3 权重数据类型转换
- 支持 `float16`, `bfloat16`, `float32` 之间的转换
- 根据 `--dtype` 参数自动匹配

---

## 3. 自定义 CUDA Kernel 替换

vLLM 不使用 PyTorch 的默认算子实现，而是替换为高度优化的自定义 Kernel：

| 组件 | 默认实现 | vLLM 替换 |
|------|----------|-----------|
| **Attention** | PyTorch `F.scaled_dot_product_attention` | **FlashAttention-2** / **xFormers** / **FlashInfer** |
| **Paged Attention** | 无 | 自定义 **PagedAttention Kernel** |
| **RoPE** | 逐元素计算 | **融合 RoPE Kernel** |
| **LayerNorm / RMSNorm** | PyTorch 标准实现 | **Triton / CUDA 融合 Kernel** |
| **激活函数 (SiLU * Gate)** | 分步计算 | **融合 SiLU-and-Multiply Kernel** |
| **采样 (Sampling)** | PyTorch `multinomial` | **自定义并行采样 Kernel** |

---

## 4. 计算图优化 — CUDA Graph Capture

```
首次加载后，在 warmup 阶段：
```

- 对**多个常见 batch size**（如 1, 2, 4, 8, ...）预先捕获 CUDA Graph
- 后续推理时直接 **replay** CUDA Graph，避免 Python 和 CUDA 的 launch overhead

```python
# 预捕获不同 batch size 的 CUDA Graph
def capture_model(self, kv_caches):
    # 对一系列 batch_size 进行 graph capture
    for batch_size in _get_graph_batch_size(self.max_num_seqs):
        # 运行一次前向传播并捕获为 CUDA Graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            output = self.model(...)
        self.graph_runners[batch_size] = CUDAGraphRunner(graph, output)
```

**优化效果**：减少每次推理的 kernel launch 开销，尤其在小 batch 时提升显著（可达 **2-3x**）。

---

## 5. 张量并行（Tensor Parallelism）

如果使用多 GPU (`--tensor-parallel-size > 1`)：

- 模型权重按照 **Megatron-LM 风格**进行切分
  - **Column Parallel Linear**：`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`
  - **Row Parallel Linear**：`o_proj`, `down_proj`
- 每个 GPU 只加载 $\frac{1}{N}$ 的权重
- 初始化 **NCCL 通信组**，在需要时进行 `AllReduce`

```
GPU 0: [W_q[:, :d/2], W_k[:, :d/2], W_v[:, :d/2]]
GPU 1: [W_q[:, d/2:], W_k[:, d/2:], W_v[:, d/2:]]
                    ↓ 前向传播后 ↓
              AllReduce (NCCL)
```

---

## 6. 模型架构替换

vLLM **不直接使用 HuggingFace 的模型类**，而是注册自己优化过的模型实现：

```python
# vllm/model_executor/models/__init__.py
_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "MistralForCausalLM": ("mistral", "MistralForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    ...
}
```

这些自定义实现：
- 内置了 **PagedAttention** 的接口
- 支持 **张量并行**
- 使用**融合算子**（fused operations）
- 支持 **连续批处理（continuous batching）**

---

## 7. 整体初始化流程图

```
vLLM 模型加载流程
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│
├─ 1. 解析模型配置 (config.json)
│     └─ 确定模型架构、hidden_size、num_layers 等
│
├─ 2. 初始化分布式环境 (如果 TP > 1)
│     └─ 创建 NCCL 进程组
│
├─ 3. 构建 vLLM 自定义模型
│     ├─ 替换 Attention → PagedAttention
│     ├─ 替换 Linear → ColumnParallel / RowParallel
│     └─ 绑定融合 Kernel (RMSNorm, SiLU, RoPE...)
│
├─ 4. 加载权重
│     ├─ 从 safetensors/bin 文件加载
│     ├─ 权重名映射 & 合并 (QKV, GateUp)
│     ├─ 量化权重处理 (如果适用)
│     └─ 数据类型转换 (dtype cast)
│
├─ 5. 显存 Profiling
│     ├─ 测量模型权重显存
│     ├─ 模拟前向传播测量激活显存
│     └─ 计算可用 KV Cache Block 数量
│
├─ 6. KV Cache 预分配
│     └─ 分配 num_gpu_blocks 个物理 Block
│
├─ 7. CUDA Graph 捕获 (Warmup)
│     └─ 对常见 batch size 预捕获计算图
│
└─ 8. 就绪，开始接收推理请求 ✓
```

---

## 总结

| 优化类别 | 关键技术 | 主要收益 |
|---------|---------|---------|
| **内存** | PagedAttention + Block 管理 | 显存利用率提升 **2-4x**，近乎零浪费 |
| **计算** | 自定义 CUDA/Triton Kernel | 算子级别的加速 |
| **调度** | CUDA Graph Capture | 消除 kernel launch 开销 |
| **权重** | QKV 合并 + 量化支持 | 减少内存访问和存储 |
| **并行** | Tensor Parallelism | 支持大模型多卡推理 |
| **架构** | 自定义模型实现 | 端到端优化，支持连续批处理 |

这些优化共同作用，使得 vLLM 相比原生 HuggingFace 推理能够实现 **数倍到数十倍** 的吞吐量提升。
# subgroup与sharedmemrmsnorm性能对比

### 🧑‍💻 User

用subgroup和shared mem reduce分别实现rmsnorm，比较性能差异

---

### 🤖 Assistant

实现 RMSNorm (Root Mean Square Normalization) 的核心性能瓶颈在于**计算所有元素的平方和（Reduction）**。在 GPU 编程（如 CUDA）中，我们可以通过 Shared Memory（共享内存）或 Subgroup（在 CUDA 中称为 Warp，即线程束内的 Shuffle 指令）来实现这个规约过程。

RMSNorm 的数学公式如下：
$$ \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon} $$
$$ y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i $$

下面我们将分别使用这两种方式来实现 RMSNorm 的核心 Reduction 部分，并详细比较它们的性能差异。

---

### 1. 基于 Shared Memory 的实现

在这种方法中，每个线程计算自己负责元素的平方和，然后将结果写入 Shared Memory。接着，使用树状规约（Tree Reduction）在 Shared Memory 中求出总和。这种方法需要多次使用 `__syncthreads()` 来保证内存可见性。

<details>
<summary><b>点击展开：Shared Memory Reduce 代码示例 (CUDA)</b></summary>

```cpp
__global__ void rmsnorm_shared_mem(float* x, float* y, float* gamma, float epsilon, int N) {
    extern __shared__ float sdata[];
  
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    // 1. 每个线程计算局部平方和
    float sum = 0.0f;
    if (i < N) {
        float val = x[i];
        sum = val * val;
    }
    sdata[tid] = sum;
    __syncthreads(); // 等待所有线程写入 shared memory
  
    // 2. Shared Memory 树状规约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // 每次迭代都需要同步
    }
  
    // 3. 计算 RMS 并写回结果
    if (tid == 0) {
        // sdata[0] 包含平方总和
        sdata[0] = rsqrtf(sdata[0] / N + epsilon); 
    }
    __syncthreads(); // 等待 tid 0 计算完成
  
    // 4. 应用 RMSNorm
    if (i < N) {
        y[i] = x[i] * sdata[0] * gamma[i];
    }
}
```
</details>

---

### 2. 基于 Subgroup (Warp Shuffle) 的实现

在这种方法中，我们利用 GPU 底层的 Warp 级别通信指令（如 CUDA 中的 `__shfl_down_sync`）。数据直接在寄存器之间传递，完全跳过了 Shared Memory（除了跨 Warp 时的少量数据交换），并且不需要全局的线程块同步指令 `__syncthreads()`。

<details>
<summary><b>点击展开：Subgroup (Warp) Reduce 代码示例 (CUDA)</b></summary>

```cpp
// Warp 级别的规约辅助函数
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_warp_shuffle(float* x, float* y, float* gamma, float epsilon, int N) {
    // 只需要很小的 shared memory 来存储每个 warp 的局部和
    static __shared__ float shared_warp_sums[32]; 
  
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = tid % warpSize;
    int warpId = tid / warpSize;
  
    // 1. 每个线程计算局部平方和
    float sum = 0.0f;
    if (i < N) {
        float val = x[i];
        sum = val * val;
    }
  
    // 2. Warp 内部规约
    sum = warpReduceSum(sum);
  
    // 3. 将每个 Warp 的结果写入 shared memory
    if (laneId == 0) {
        shared_warp_sums[warpId] = sum;
    }
    __syncthreads(); // 只需要一次同步，等待所有 warp 写完
  
    // 4. 由第一个 Warp 完成最终的规约
    sum = (tid < (blockDim.x / warpSize)) ? shared_warp_sums[laneId] : 0;
    if (warpId == 0) {
        sum = warpReduceSum(sum);
        // 计算 RMS
        if (tid == 0) {
            shared_warp_sums[0] = rsqrtf(sum / N + epsilon);
        }
    }
    __syncthreads(); // 等待 RMS 计算完成
  
    // 5. 应用 RMSNorm
    if (i < N) {
        y[i] = x[i] * shared_warp_sums[0] * gamma[i];
    }
}
```
</details>

---

### 3. 性能差异比较

在现代 GPU 上（如 NVIDIA Volta/Ampere/Hopper 架构），**Subgroup (Warp) Reduce 的性能显著优于 Shared Memory Reduce**。以下是具体的性能差异维度分析：

| 比较维度 | Shared Memory Reduce | Subgroup (Warp) Reduce | 赢家 |
| :--- | :--- | :--- | :--- |
| **延迟 (Latency)** | 高。需要多次读写 Shared Memory。 | **极低**。直接在寄存器中进行数据交换。 | **Subgroup** |
| **同步开销** | 大。每一层规约都需要调用 `__syncthreads()`。 | **极小**。Warp 内部天然同步，Block 级别仅需 1-2 次同步。 | **Subgroup** |
| **指令数量** | 多。需要访存指令 (Load/Store) 和地址计算指令。 | **少**。一条 `__shfl_down_sync` 指令同时完成数据传递。 | **Subgroup** |
| **Shared Mem 占用** | 高。需要占用 $O(\text{BlockSize})$ 大小的内存。 | **低**。仅需 $O(\frac{\text{BlockSize}}{\text{WarpSize}})$ 的内存。 | **Subgroup** |
| **Occupancy (占用率)** | 较低。较高的 Shared Mem 占用可能限制并发 Block 数量。 | **较高**。节省的 Shared Mem 允许 SM 调度更多的 Block。 | **Subgroup** |

#### 详细解释：

1.  **访存瓶颈 vs 寄存器通信**：
    Shared Memory 虽然比 Global Memory 快，但依然需要经过 Load/Store 单元。而在 Subgroup 实现中，`__shfl_down_sync` 指令直接在 ALUs 的寄存器之间交换数据，属于**零访存开销**。
2.  **同步指令 (`__syncthreads`) 的代价**：
    在 Shared memory 树状规约中，如果 Block size 是 256，需要 $\log_2(256) = 8$ 次 `__syncthreads()`。这会导致 GPU 调度器频繁挂起线程块。而 Subgroup 实现中，Warp 内的操作在硬件层面是同步的，对于 256 线程的 Block，总共只需要 2 次 `__syncthreads()`。
3.  **资源利用与并发度**：
    大模型推理（如 LLaMA, Qwen）对 RMSNorm 的调用极其频繁。Subgroup 实现将 Shared memory 的占用从 $N \times 4$ Bytes 降到了 $\frac{N}{32} \times 4$ Bytes。这极大地释放了 SM（流多处理器）的资源，使得 GPU 能够并行执行更多的算子（提高 Occupancy）。

### 结论

在实际的工业级大模型推理框架（如 vLLM, TensorRT-LLM, DeepSpeed）中，**RMSNorm 毫无例外全部采用 Subgroup (Warp Shuffle) 级别的规约实现**。Shared Memory 规约仅在早期 GPU 架构（不支持 Shuffle 指令时）或作为教学示例使用。如果追求极致性能，甚至还会结合向量化访存（`float4`）来进一步压榨显存带宽。
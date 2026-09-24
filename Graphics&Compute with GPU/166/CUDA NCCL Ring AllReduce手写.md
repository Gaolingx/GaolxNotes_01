# CUDA NCCL Ring AllReduce手写

## 🧑‍💻 User

使用cuda c++和nccl手写一个ring allreduce kernel。

---
## 🤖 Assistant

# Ring AllReduce：CUDA C++ + NCCL 手写实现

下面从算法原理到可编译运行的代码完整实现一个 Ring AllReduce。核心思路是：**用 NCCL 的点对点原语 `ncclSend/ncclRecv` 手写通信环，用自定义 CUDA kernel 完成本地的累加规约**（而不是直接调 `ncclAllReduce`）。

---

## 1. 算法原理

设有 $W$ 个 rank，每个 rank 的 buffer 被切成 $W$ 个 chunk。Ring 拓扑为：

$$\text{rank } r \;\xrightarrow{\text{send}}\; (r+1) \bmod W, \qquad \text{rank } r \;\xleftarrow{\text{recv}}\; (r-1+W) \bmod W$$

整个 allreduce = **reduce-scatter（$W-1$ 步）** + **all-gather（$W-1$ 步）**。

**Phase 1 · Reduce-Scatter**，步 $s = 0 \dots W-2$：

$$s_c = (r - s) \bmod W \quad(\text{发送的 chunk})$$
$$r_c = (r - s - 1) \bmod W \quad(\text{接收后累加的 chunk})$$

每步接收到的 chunk 用设备端 kernel 累加进本地。结束后，rank $r$ 持有 chunk $(r+1) \bmod W$ 的**完整规约结果**。

**Phase 2 · All-Gather**，步 $s = 0 \dots W-2$：

$$s_c = (r + 1 - s) \bmod W, \qquad r_c = (r - s) \bmod W$$

每步把上一步刚收到的（已成形的）chunk 沿环转发，接收时**直接覆盖**（不累加）。

> 关键点：由于同一个 chunk 在第 $s$ 步被接收、在第 $s+1$ 步被转发，只要在**同一 stream** 上顺序执行即可保证数据依赖正确。

```
Reduce-Scatter (W=3)                All-Gather
 r0 [A][ ][ ]     r0 [_][B][ ]      r0 [A][B][C]
 r1 [ ][B][ ] --> r1 [_][_][C]  --> r1 [A][B][C]
 r2 [ ][ ][C]     r2 [A][_][_]      r2 [A][B][C]
```

---

## 2. 设备端规约 kernel

```cpp
// element-wise: dst[i] += src[i]，向量化 + grid-stride
__global__ void reduceAddKernel(float* __restrict__ dst,
                                const float* __restrict__ src,
                                size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        dst[i] += src[i];
    }
}
```

如果想更贴近硬件，可以写一个 `float4` 向量化版本：

```cpp
__global__ void reduceAddKernelVec4(float4* __restrict__ dst,
                                    const float4* __restrict__ src,
                                    size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) {
        float4 a = dst[i];
        float4 b = src[i];
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        dst[i] = a;
    }
}
```

---

## 3. Host 端 Ring 编排（NCCL 点对点手写环）

```cpp
void ringAllReduce(float* data, size_t count,
                   int rank, int worldSize,
                   ncclComm_t comm, cudaStream_t stream,
                   float* scratch /* 至少 chunkSize 大小 */) {
    const int next = (rank + 1) % worldSize;
    const int prev = (rank - 1 + worldSize) % worldSize;
    const size_t chunkSize = count / worldSize;   // 需整除

    auto chunkPtr = [&](int c) { return data + (size_t)c * chunkSize; };

    const int threads = 256;
    const int blocks  = (int)((chunkSize + threads - 1) / threads);

    // ---------------- Phase 1: Reduce-Scatter ----------------
    for (int s = 0; s < worldSize - 1; ++s) {
        int sendChunk = ((rank - s) % worldSize + worldSize) % worldSize;
        int recvChunk = ((rank - s - 1) % worldSize + worldSize) % worldSize;

        ncclGroupStart();
        ncclSend(chunkPtr(sendChunk), chunkSize, ncclFloat, next, comm, stream);
        ncclRecv(scratch,            chunkSize, ncclFloat, prev, comm, stream);
        ncclGroupEnd();

        // 接收到的部分和累加进本地 chunk
        reduceAddKernel<<<blocks, threads, 0, stream>>>(
            chunkPtr(recvChunk), scratch, chunkSize);
    }

    // ---------------- Phase 2: All-Gather ----------------
    for (int s = 0; s < worldSize - 1; ++s) {
        int sendChunk = ((rank + 1 - s) % worldSize + worldSize) % worldSize;
        int recvChunk = ((rank - s)     % worldSize + worldSize) % worldSize;

        ncclGroupStart();
        ncclSend(chunkPtr(sendChunk), chunkSize, ncclFloat, next, comm, stream);
        ncclRecv(chunkPtr(recvChunk), chunkSize, ncclFloat, prev, comm, stream);
        ncclGroupEnd();
    }
}
```

要点说明：

- **`ncclGroupStart/End`**：把同一 rank 的 `send` 和 `recv` 打包提交，避免发送阻塞等待接收而死锁。
- **顺序性**：`reduceAddKernel` 与后续 NCCL 操作都在同一个 `stream`，天然保证"先收到→再累加→再转发"的依赖。
- **`scratch`**：Phase 1 接收端用独立缓冲，避免与发送中的 chunk 冲突。

---

## 4. 完整可编译示例

```cpp
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CUDA_CHECK(cmd) do {                                             \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(e));              \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)

#define NCCL_CHECK(cmd) do {                                             \
    ncclResult_t r = (cmd);                                              \
    if (r != ncclSuccess) {                                              \
        fprintf(stderr, "NCCL error %s:%d: %s\n",                        \
                __FILE__, __LINE__, ncclGetErrorString(r));              \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)

__global__ void reduceAddKernel(float* __restrict__ dst,
                                const float* __restrict__ src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] += src[i];
}

void ringAllReduce(float* data, size_t count, int rank, int worldSize,
                   ncclComm_t comm, cudaStream_t stream, float* scratch) {
    const int next = (rank + 1) % worldSize;
    const int prev = (rank - 1 + worldSize) % worldSize;
    const size_t chunkSize = count / worldSize;
    auto chunkPtr = [&](int c) { return data + (size_t)c * chunkSize; };

    const int threads = 256;
    const int blocks  = (int)((chunkSize + threads - 1) / threads);

    for (int s = 0; s < worldSize - 1; ++s) {
        int sc = ((rank - s)     % worldSize + worldSize) % worldSize;
        int rc = ((rank - s - 1) % worldSize + worldSize) % worldSize;
        ncclGroupStart();
        ncclSend(chunkPtr(sc), chunkSize, ncclFloat, next, comm, stream);
        ncclRecv(scratch,      chunkSize, ncclFloat, prev, comm, stream);
        ncclGroupEnd();
        reduceAddKernel<<<blocks, threads, 0, stream>>>(chunkPtr(rc), scratch, chunkSize);
    }

    for (int s = 0; s < worldSize - 1; ++s) {
        int sc = ((rank + 1 - s) % worldSize + worldSize) % worldSize;
        int rc = ((rank - s)     % worldSize + worldSize) % worldSize;
        ncclGroupStart();
        ncclSend(chunkPtr(sc), chunkSize, ncclFloat, next, comm, stream);
        ncclRecv(chunkPtr(rc), chunkSize, ncclFloat, prev, comm, stream);
        ncclGroupEnd();
    }
}

int main() {
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 2) { printf("需要 >= 2 块 GPU\n"); return 1; }
    const int W = numDevices;

    std::vector<int>        devs(W);
    std::vector<ncclComm_t> comms(W);
    for (int i = 0; i < W; ++i) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), W, devs.data()));

    const size_t count     = 1u << 20;        // 1M floats
    const size_t chunkSize = count / W;
    const size_t bytes     = count * sizeof(float);

    // 每卡独立资源
    std::vector<float*>        d_data(W), d_scratch(W);
    std::vector<float*>        h_data(W);
    std::vector<cudaStream_t>  streams(W);

    for (int r = 0; r < W; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&d_data[r], bytes));
        CUDA_CHECK(cudaMalloc(&d_scratch[r], chunkSize * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[r]));
        h_data[r] = (float*)malloc(bytes);
        for (size_t i = 0; i < count; ++i) h_data[r][i] = (float)(r + (int)i);
        CUDA_CHECK(cudaMemcpy(d_data[r], h_data[r], bytes, cudaMemcpyHostToDevice));
    }

    // 所有 rank 并发提交（异步，不会阻塞 host）
    for (int r = 0; r < W; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        ringAllReduce(d_data[r], count, r, W, comms[r], streams[r], d_scratch[r]);
    }

    // 校验
    bool ok = true;
    for (int r = 0; r < W; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMemcpy(h_data[r], d_data[r], bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            float expected = (float)(W * (int)i + W * (W - 1) / 2); // sum_{rr}(rr + i)
            if (fabsf(h_data[r][i] - expected) > 1e-2f) {
                printf("rank %d 元素 %zu 错: got %f, want %f\n",
                       r, i, h_data[r][i], expected);
                ok = false; break;
            }
        }
    }
    printf(ok ? "✅ Ring AllReduce 结果正确\n" : "❌ 校验失败\n");

    for (int r = 0; r < W; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        cudaFree(d_data[r]); cudaFree(d_scratch[r]);
        cudaStreamDestroy(streams[r]);
        NCCL_CHECK(ncclCommDestroy(comms[r]));
        free(h_data[r]);
    }
    return ok ? 0 : 1;
}
```

---

## 5. 编译与运行

```bash
nvcc -O3 -std=c++17 ring_allreduce.cu -o ring_allreduce -lnccl
./ring_allreduce          # 单进程多卡，需 >= 2 块 GPU
```

> 说明：示例用 `ncclCommInitAll` 走单进程多卡（同一进程内 P2P），便于演示。多机/多进程场景只需把 `ncclCommInitAll` 换成 `ncclGetUniqueId` + `ncclCommInitRank`，其余算法逻辑完全不变。

---

## 6. 性能与进阶建议

<details>
<summary>点击展开：优化方向</summary>

1. **流水线 / 分块重叠**：把 chunk 再细分为多个子片，让 `ncclSend/ncclRecv` 与 `reduceAddKernel` 在多个 stream 上重叠，隐藏通信延迟。
2. **双缓冲**：Phase 1 用两个 scratch ping-pong，减少"接收→规约"之间的串行气泡。
3. **向量化 + `__ldg`/`__restrict__`**：用 `float4`/`__half2` 提升访存带宽，规约 kernel 通常受限于显存带宽。
4. **NVLink 直连**：同机内可绕过 NCCL 用 `cudaMemcpyPeerAsync` 或直接 P2P load（`__ldcg`）实现 ring，延迟更低。
5. **分层 AllReduce**：机内先用 NVLink/NVSwitch 做 intra-node 规约，再跨节点 ring，即 NCCL 的 hierarchical 策略。
6. **注意**：NCCL 自带的 `ncclAllReduce` 已内置高度优化的 ring/tree，本实现主要用于**教学/理解算法**；生产环境直接用它即可（它同样支持自定义 `NCCL_ALGO=RING`）。
</details>

---

### 小结

- **Ring AllReduce = Reduce-Scatter + All-Gather**，各 $W-1$ 步，通信量最优（每 rank 收发各 $2(W-1)$ 个 chunk）。
- 用 **`ncclSend/ncclRecv` + `ncclGroupStart/End`** 手写通信环。
- 用 **自定义 CUDA kernel** 做本地累加，同 stream 保证数据依赖。
- chunk 索引公式：规约阶段 `(r-s)` 发送 / `(r-s-1)` 接收；汇聚阶段 `(r+1-s)` 发送 / `(r-s)` 接收。

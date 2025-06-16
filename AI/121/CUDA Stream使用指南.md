# CUDA Stream使用指南

### 🧑‍💻 User

什么是cuda stream，如何使用？

---

### 🤖 Assistant

CUDA Stream（流）是CUDA编程中用于**管理并行执行序列**的核心概念。它允许开发者将不同的CUDA操作（如内存拷贝、内核启动）组织到独立的序列中，实现任务级并行和异步执行，从而最大化GPU利用率。

---

### **CUDA Stream的核心作用**
1. **异步执行**：流内的操作异步执行，主机线程无需等待。
2. **任务并行**：不同流的操作可并发执行（若硬件支持）。
3. **重叠计算与传输**：在计算内核执行的同时进行数据传输（需硬件支持）。
4. **依赖管理**：通过事件（Event）实现流间同步。

---

### **如何使用CUDA Stream**

#### **1. 创建与销毁流**
```cpp
cudaStream_t stream; // 定义流句柄
cudaStreamCreate(&stream); // 创建流

// ... 使用流 ...

cudaStreamDestroy(stream); // 销毁流
```

#### **2. 在流中执行操作**
- **异步内存传输**（需固定内存）：
  ```cpp
  float *h_pinned, *d_data;
  cudaMallocHost(&h_pinned, size); // 分配固定内存（可异步传输）
  cudaMalloc(&d_data, size);
  
  // 异步复制（主机→设备）
  cudaMemcpyAsync(d_data, h_pinned, size, cudaMemcpyHostToDevice, stream);
  ```

- **内核启动**：
  ```cpp
  myKernel<<<gridDim, blockDim, sharedMemSize, stream>>>(args...);
  ```

- **异步复制回主机**：
  ```cpp
  cudaMemcpyAsync(h_pinned, d_data, size, cudaMemcpyDeviceToHost, stream);
  ```

#### **3. 同步流**
- **等待单个流完成**：
  ```cpp
  cudaStreamSynchronize(stream); // 阻塞主机直到流完成
  ```
- **使用事件同步**：
  ```cpp
  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream); // 在流中记录事件
  cudaEventSynchronize(event);   // 等待事件完成
  cudaEventDestroy(event);
  ```

---

### **关键实践技巧**
1. **重叠计算与传输**：
   - 使用多个流交替执行数据传输和计算。
   - 示例：
     ```cpp
     // 流1：传输数据块1 → 启动内核1
     // 流2：传输数据块2 → 启动内核2
     // 流1：传输结果1回主机 ← 内核1完成
     // 流2：传输结果2回主机 ← 内核2完成
     ```

2. **固定内存（Pinned Memory）**：
   - 异步传输必须使用`cudaMallocHost`或`cudaHostAlloc`分配的固定内存。

3. **避免虚假依赖**：
   - 默认流（`stream 0`）会阻塞其他流，优先使用非默认流。

4. **流数量控制**：
   - 通常4-8个流足够饱和GPU，过多流增加调度开销。

---

### **完整示例代码**
```cpp
#include <cuda_runtime.h>

__global__ void addKernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // 分配固定内存
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 创建流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 异步操作（主机→设备→内核→设备→主机）
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream);
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    addKernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, N);
    
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);

    // 等待流完成
    cudaStreamSynchronize(stream);

    // 检查结果
    bool ok = true;
    for (int i = 0; i < N; i++) 
        if (fabs(h_c[i] - 3.0f) > 1e-5) ok = false;
    printf("Result: %s\n", ok ? "OK" : "Error");

    // 清理
    cudaStreamDestroy(stream);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    return 0;
}
```

---

### **常见问题**
- **Q: 默认流（`stream 0`）有什么特殊？**  
  A: 默认流会阻塞其他流，且其他流会阻塞默认流。使用非默认流可避免此问题。

- **Q: 如何实现流间同步？**  
  A: 通过`cudaEventRecord()`和`cudaStreamWaitEvent()`实现跨流同步。

- **Q: 所有GPU都支持计算与传输重叠吗？**  
  A: 需支持**并发复制/执行**（检查`cudaDeviceProp::asyncEngineCount`属性）。

掌握CUDA Stream可显著提升GPU程序性能，尤其适合大规模并行任务和流水线处理场景。
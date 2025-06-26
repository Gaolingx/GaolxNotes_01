# CUDA设备间异步拷贝API使用指南

### 🧑‍💻 User

cuda是否支持异步从device之间拷贝资源，用什么api？

---

### 🤖 Assistant

CUDA 支持在设备（GPU）之间进行**异步内存拷贝**，主要通过以下 API 实现：

---

### 核心 API：`cudaMemcpyPeerAsync`
```c
cudaError_t cudaMemcpyPeerAsync(
    void*       dst,      // 目标设备指针
    int         dstDevice,// 目标设备 ID
    const void* src,      // 源设备指针
    int         srcDevice,// 源设备 ID
    size_t      count,    // 拷贝字节数
    cudaStream_t stream   // CUDA 流（指定异步操作）
);
```

#### 关键特性：
1. **异步操作**：  
   通过 `stream` 参数指定 CUDA 流，拷贝操作在 GPU 后台执行，不阻塞 CPU。
2. **跨设备支持**：  
   显式指定源设备 `srcDevice` 和目标设备 `dstDevice`。
3. **需启用 Peer Access**：  
   使用前需通过 `cudaDeviceEnablePeerAccess()` 启用设备间的直接访问（如果硬件支持）。

---

### 完整使用步骤：

#### 1. 检查设备间 P2P 支持
```c
int canAccessPeer;
cudaDeviceCanAccessPeer(&canAccessPeer, srcDeviceId, dstDeviceId);
if (!canAccessPeer) {
    // 处理不支持 P2P 的情况（需通过主机中转）
}
```

#### 2. 启用 Peer Access
```c
cudaSetDevice(srcDeviceId);
cudaDeviceEnablePeerAccess(dstDeviceId, 0);  // 0 表示无附加标志
```

#### 3. 创建 CUDA 流
```c
cudaStream_t stream;
cudaStreamCreate(&stream);
```

#### 4. 执行异步拷贝
```c
cudaMemcpyPeerAsync(
    dst_ptr, dstDeviceId, 
    src_ptr, srcDeviceId, 
    dataSize, stream
);
```

#### 5. 同步流（可选）
```c
cudaStreamSynchronize(stream);  // 等待拷贝完成
// 或异步检查：cudaStreamQuery(stream);
```

#### 6. 清理资源
```c
cudaStreamDestroy(stream);
cudaDeviceDisablePeerAccess(dstDeviceId);  // 禁用 P2P
```

---

### 注意事项：
1. **硬件支持**：  
   - 设备需支持 P2P（通常为同架构 GPU，如 Pascal+ 且通过 NVLink/PCIe 连接）。
   - 若无 P2P 支持，需通过主机内存中转（`cudaMemcpyAsync` 到 Host，再拷贝到目标设备）。
   
2. **统一寻址（UVA）**：  
   在 64 位系统中启用 `cudaDeviceMapHost` 后，可直接用 `cudaMemcpyAsync` 指定方向为 `cudaMemcpyDefault`，系统自动识别设备指针。

3. **性能**：  
   - 启用 P2P 后，设备间拷贝带宽显著高于经主机中转。
   - 结合异步流，可实现与计算操作的重叠（overlap）。

---

### 替代方案（无 P2P 时）：
```c
// 通过 Host 中转
cudaMemcpyAsync(host_buf, src_ptr, size, cudaMemcpyDeviceToHost, stream);
cudaMemcpyAsync(dst_ptr, host_buf, size, cudaMemcpyHostToDevice, stream);
```

---

### 总结：
| **场景**               | **推荐 API**                     |
|------------------------|----------------------------------|
| 直接设备间拷贝（P2P）  | `cudaMemcpyPeerAsync()`          |
| 统一寻址（UVA）        | `cudaMemcpyAsync(..., cudaMemcpyDefault)` |
| 经主机中转             | 两次 `cudaMemcpyAsync`           |

通过正确使用异步拷贝和 CUDA 流，可最大化 GPU 间数据传输效率。
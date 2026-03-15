# 多GPU超大数组高效排序

### 🧑‍💻 User

用cuda在multi gpu上排序算法怎么实现高效并行？
而且如果当待排的element数量远超gpu显存的时候，怎么做分块、异步streaming和device同步同时保证结果正确？
试编写一个cuda example

---

### 🤖 Assistant

可以，最实用的思路其实是：

> **不要把“multi-GPU 排序”理解成一个巨大的 kernel 一次排完。**
> 真正高效的做法通常是 **两层式**：
>
> 1. **每个 GPU 独立把自己拿到的块排好**（通常用 radix sort）
> 2. **再做全局 merge / repartition** 得到最终有序结果

当数据量 **远超单卡显存** 时，最稳妥的是把问题转成 **external merge sort（外排序）**：

- 先把大数组切成很多 **能放进 GPU 的 chunk**
- 多个 GPU 并行地把这些 chunk 各自排成 **sorted runs**
- 最后再把这些 runs 做 **$k$ 路归并**

这样才能同时兼顾：

- 多 GPU 并行
- 异步 `stream`
- 数据超过显存
- 正确的同步与结果正确性

---

## 1. 多 GPU 上高效并行排序，推荐什么思路？

### 场景 A：数据能放进“所有 GPU 总显存”
这时可做真正的分布式 GPU 排序，常见方案：

- **Radix sort + sample sort / splitter partition**
- **Local sort + all-to-all exchange + local merge**

典型步骤：

1. 每个 GPU 先对本地数据做局部排序
2. 采样得到 splitter
3. 按 splitter 把本地有序数据切成若干段
4. GPU 间交换这些段
5. 每个 GPU 再对收到的数据做 merge / sort

这种方式在 NVLink / P2P / NCCL 条件下性能最好。

但它更适合：
- 数据可以驻留在 GPU 总显存里
- 追求极致吞吐
- 工程复杂度可接受

---

### 场景 B：数据远超 GPU 显存
这时推荐：

## **External Merge Sort on Multi-GPU**

分两阶段：

### 阶段 1：Run Generation
把输入切成很多块，每块大小为 $C$ 个元素，满足单个 GPU 能处理。

每块执行：

1. `H2D` 异步拷贝到 GPU
2. GPU 上排序
3. `D2H` 异步拷回主机
4. 得到一个 **已排序 run**

多个 GPU 同时做这件事。

### 阶段 2：Global Merge
对所有 sorted runs 做 $k$ 路归并，得到最终全局有序结果。

如果主机内存也不够，就把 run 写到 SSD/NVMe，再做多轮外部归并。

---

## 2. 如何做分块、异步 streaming 和同步，才能既快又正确？

核心原则：

### 2.1 分块大小怎么定？
不要只看原始数据大小，还要给这些留空间：

- 输入 buffer
- 输出 buffer
- radix sort 临时空间
- 双缓冲 / 多 stream buffer

经验上可用：

$$
C \le \frac{\eta \cdot M_{\text{free}}}{S \cdot (2 + \alpha)\cdot \text{sizeof}(T)}
$$

其中：

- $M_{\text{free}}$：可用显存
- $S$：每卡 stream 数
- $2$：输入输出双 buffer
- $\alpha$：排序临时空间系数，通常约 $1 \sim 1.5$
- $\eta$：安全系数，通常取 $0.5 \sim 0.7$

---

### 2.2 为什么要双缓冲 / 多 stream？
为了重叠：

- 当前块在 GPU 上排序
- 下一块做 `H2D`
- 上一块做 `D2H`

也就是把：

- PCIe/NVLink 传输
- 排序 kernel
- 回传

尽量并行起来。

通常每卡用 **2 个 stream** 就已经很实用。

---

### 2.3 正确性怎么保证？
对每个 chunk，把下面三步放在**同一个 stream** 里：

1. `cudaMemcpyAsync(... H2D ..., stream)`
2. `cub::DeviceRadixSort::SortKeys(..., stream)`
3. `cudaMemcpyAsync(... D2H ..., stream)`

因为 **同一 stream 内是严格有序的**，所以这个 chunk 一定满足：

$$
\text{H2D} \rightarrow \text{Sort} \rightarrow \text{D2H}
$$

然后在末尾打一个 `cudaEventRecord(done, stream)`。

主机线程只要在 `cudaEventQuery(done)` 成功后，才把这个 chunk 视为“run 已完成”。

最后只有在 **所有 run 都完成后**，才开始全局归并。

这样就能保证结果正确。

---

### 2.4 什么时候不要用 `cudaDeviceSynchronize()`？
**热路径里尽量不要用。**

因为它会把该设备上所有工作都堵住，破坏并发。

更好的方式是：

- 用 **stream event** 精确同步 chunk 完成
- 必要时用 `cudaStreamWaitEvent()`
- 只在 worker 退出或程序收尾时做一次 `cudaDeviceSynchronize()`

---

## 3. 一个可运行的 CUDA 示例

下面这个例子演示的是：

- 多 GPU
- 每 GPU 两个 stream
- 每个 chunk 用 CUB radix sort 排序
- 数据量可以大于单卡显存
- 所有 chunk 排完后，在 CPU 做最终 $k$ 路归并

> 说明：
> 这个例子为了突出 **分块 + 多 GPU + 异步 stream + 正确同步**，
> 最终全局 merge 放在 CPU 做，代码最清晰。
> 如果你要，我后面可以再给你一个 **GPU 上分层 merge** 的版本。

---

### 编译方式

```bash
nvcc -O3 -std=c++17 multi_gpu_external_sort.cu -o multi_gpu_external_sort
```

运行示例：

```bash
./multi_gpu_external_sort 100000000 4
```

表示：

- 排序 $10^8$ 个 `uint32_t`
- 使用 4 张 GPU

---

### 代码：`multi_gpu_external_sort.cu`

```cpp
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

using Key = uint32_t;

static size_t round_down_pow2(size_t x) {
    if (x == 0) return 0;
    size_t p = 1;
    while (p <= x / 2) p <<= 1;
    return p;
}

// 经验式选择 chunk 大小：给 in/out/temp/safety 留足空间
static size_t choose_chunk_elems(int num_gpus,
                                 int streams_per_gpu,
                                 size_t cap_elems = (1ull << 24)) {
    size_t best = std::numeric_limits<size_t>::max();

    for (int d = 0; d < num_gpus; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        size_t free_b = 0, total_b = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));

        // 保守估计：
        // 每个 stream 约需要 4 * chunk_bytes
        // （input + output + temp + slack）
        double usable = static_cast<double>(free_b) * 0.60;
        double bytes_per_chunk = usable / (streams_per_gpu * 4.0);
        size_t elems = static_cast<size_t>(bytes_per_chunk / sizeof(Key));
        elems = std::min(elems, cap_elems);

        best = std::min(best, elems);
    }

    best = round_down_pow2(best);
    return best;
}

static bool fetch_next_chunk(std::atomic<size_t>& next_chunk,
                             size_t num_chunks,
                             size_t& chunk_id) {
    chunk_id = next_chunk.fetch_add(1, std::memory_order_relaxed);
    return chunk_id < num_chunks;
}

struct GpuSlot {
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;

    Key* d_in = nullptr;
    Key* d_out = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    bool busy = false;
    size_t chunk_id = 0;
    size_t elems = 0;
};

static void launch_chunk(GpuSlot& slot,
                         Key* h_data,
                         size_t n,
                         size_t chunk_elems,
                         size_t chunk_id) {
    size_t offset = chunk_id * chunk_elems;
    size_t elems = std::min(chunk_elems, n - offset);
    size_t bytes = elems * sizeof(Key);

    // 1) H2D
    CHECK_CUDA(cudaMemcpyAsync(slot.d_in,
                               h_data + offset,
                               bytes,
                               cudaMemcpyHostToDevice,
                               slot.stream));

    // 2) Sort in same stream => strict order
    CHECK_CUDA(cub::DeviceRadixSort::SortKeys(
        slot.d_temp,
        slot.temp_bytes,
        slot.d_in,
        slot.d_out,
        elems,
        0,
        sizeof(Key) * 8,
        slot.stream));

    // 3) D2H
    CHECK_CUDA(cudaMemcpyAsync(h_data + offset,
                               slot.d_out,
                               bytes,
                               cudaMemcpyDeviceToHost,
                               slot.stream));

    // 4) Event: host sees this run complete only after D2H finishes
    CHECK_CUDA(cudaEventRecord(slot.done, slot.stream));

    slot.busy = true;
    slot.chunk_id = chunk_id;
    slot.elems = elems;
}

static void gpu_sort_worker(int device_id,
                            Key* h_data,
                            size_t n,
                            size_t chunk_elems,
                            std::atomic<size_t>& next_chunk,
                            size_t num_chunks,
                            int streams_per_gpu = 2) {
    CHECK_CUDA(cudaSetDevice(device_id));

    std::vector<GpuSlot> slots(streams_per_gpu);

    // Allocate per-stream resources
    for (int i = 0; i < streams_per_gpu; ++i) {
        CHECK_CUDA(cudaStreamCreateWithFlags(&slots[i].stream, cudaStreamNonBlocking));
        CHECK_CUDA(cudaEventCreateWithFlags(&slots[i].done, cudaEventDisableTiming));

        CHECK_CUDA(cudaMalloc(&slots[i].d_in, chunk_elems * sizeof(Key)));
        CHECK_CUDA(cudaMalloc(&slots[i].d_out, chunk_elems * sizeof(Key)));

        // Query temp storage using max chunk size
        CHECK_CUDA(cub::DeviceRadixSort::SortKeys(
            nullptr,
            slots[i].temp_bytes,
            slots[i].d_in,
            slots[i].d_out,
            chunk_elems,
            0,
            sizeof(Key) * 8,
            slots[i].stream));

        CHECK_CUDA(cudaMalloc(&slots[i].d_temp, slots[i].temp_bytes));
    }

    size_t active = 0;

    // Initial fill
    for (int i = 0; i < streams_per_gpu; ++i) {
        size_t cid = 0;
        if (fetch_next_chunk(next_chunk, num_chunks, cid)) {
            launch_chunk(slots[i], h_data, n, chunk_elems, cid);
            ++active;
        }
    }

    // Poll completed slots and refill them
    while (active > 0) {
        bool progress = false;

        for (int i = 0; i < streams_per_gpu; ++i) {
            if (!slots[i].busy) continue;

            cudaError_t q = cudaEventQuery(slots[i].done);
            if (q == cudaSuccess) {
                slots[i].busy = false;
                --active;
                progress = true;

                size_t cid = 0;
                if (fetch_next_chunk(next_chunk, num_chunks, cid)) {
                    launch_chunk(slots[i], h_data, n, chunk_elems, cid);
                    ++active;
                }
            } else if (q != cudaErrorNotReady) {
                CHECK_CUDA(q);
            }
        }

        if (!progress) {
            std::this_thread::yield();
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < streams_per_gpu; ++i) {
        CHECK_CUDA(cudaFree(slots[i].d_temp));
        CHECK_CUDA(cudaFree(slots[i].d_out));
        CHECK_CUDA(cudaFree(slots[i].d_in));
        CHECK_CUDA(cudaEventDestroy(slots[i].done));
        CHECK_CUDA(cudaStreamDestroy(slots[i].stream));
    }
}

// CPU k-way merge all sorted runs
static void merge_sorted_runs(const Key* h_sorted_runs,
                              size_t n,
                              size_t chunk_elems,
                              std::vector<Key>& out) {
    struct Node {
        Key value;
        size_t run_id;
        size_t idx_in_run;
    };
    struct Cmp {
        bool operator()(const Node& a, const Node& b) const {
            return a.value > b.value; // min-heap
        }
    };

    size_t num_runs = (n + chunk_elems - 1) / chunk_elems;
    std::priority_queue<Node, std::vector<Node>, Cmp> pq;

    auto run_offset = [chunk_elems](size_t run_id) {
        return run_id * chunk_elems;
    };
    auto run_size = [n, chunk_elems](size_t run_id) {
        size_t off = run_id * chunk_elems;
        return std::min(chunk_elems, n - off);
    };

    for (size_t r = 0; r < num_runs; ++r) {
        size_t sz = run_size(r);
        if (sz > 0) {
            pq.push(Node{h_sorted_runs[run_offset(r)], r, 0});
        }
    }

    out.resize(n);
    size_t out_pos = 0;

    while (!pq.empty()) {
        Node cur = pq.top();
        pq.pop();

        out[out_pos++] = cur.value;

        size_t next_idx = cur.idx_in_run + 1;
        size_t sz = run_size(cur.run_id);
        if (next_idx < sz) {
            size_t base = run_offset(cur.run_id);
            pq.push(Node{h_sorted_runs[base + next_idx], cur.run_id, next_idx});
        }
    }
}

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::stoull(argv[1]) : (1ull << 26); // 默认约 67M keys
    int requested_gpus = (argc > 2) ? std::stoi(argv[2]) : -1;

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    int num_gpus = (requested_gpus > 0)
        ? std::min(requested_gpus, device_count)
        : device_count;

    const int streams_per_gpu = 2;

    size_t chunk_elems = choose_chunk_elems(num_gpus, streams_per_gpu, 1ull << 24);
    if (chunk_elems == 0) {
        std::cerr << "Failed to choose chunk size." << std::endl;
        return 1;
    }
    chunk_elems = std::min(chunk_elems, n);

    size_t num_chunks = (n + chunk_elems - 1) / chunk_elems;

    std::cout << "Using GPUs           : " << num_gpus << std::endl;
    std::cout << "Elements             : " << n << std::endl;
    std::cout << "Chunk elems          : " << chunk_elems << std::endl;
    std::cout << "Num chunks           : " << num_chunks << std::endl;
    std::cout << "Streams per GPU      : " << streams_per_gpu << std::endl;

    // Pinned host buffer for true async H2D/D2H
    Key* h_data = nullptr;
    CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&h_data), n * sizeof(Key)));

    // Fill random input
    {
        std::mt19937 rng(12345);
        std::uniform_int_distribution<Key> dist(
            std::numeric_limits<Key>::min(),
            std::numeric_limits<Key>::max());

        for (size_t i = 0; i < n; ++i) {
            h_data[i] = dist(rng);
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    std::atomic<size_t> next_chunk{0};
    std::vector<std::thread> workers;
    workers.reserve(num_gpus);

    for (int d = 0; d < num_gpus; ++d) {
        workers.emplace_back(gpu_sort_worker,
                             d,
                             h_data,
                             n,
                             chunk_elems,
                             std::ref(next_chunk),
                             num_chunks,
                             streams_per_gpu);
    }

    for (auto& th : workers) {
        th.join();
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Final global merge on CPU
    std::vector<Key> final_out;
    merge_sorted_runs(h_data, n, chunk_elems, final_out);

    auto t2 = std::chrono::high_resolution_clock::now();

    bool ok = std::is_sorted(final_out.begin(), final_out.end());

    double sort_sec =
        std::chrono::duration<double>(t1 - t0).count();
    double merge_sec =
        std::chrono::duration<double>(t2 - t1).count();
    double total_sec =
        std::chrono::duration<double>(t2 - t0).count();

    double gb = static_cast<double>(n * sizeof(Key)) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Run generation time  : " << sort_sec  << " s" << std::endl;
    std::cout << "Final merge time     : " << merge_sec << " s" << std::endl;
    std::cout << "Total time           : " << total_sec << " s" << std::endl;
    std::cout << "Input size           : " << gb << " GiB" << std::endl;
    std::cout << "Sorted correctly     : " << (ok ? "YES" : "NO") << std::endl;

    CHECK_CUDA(cudaFreeHost(h_data));
    return ok ? 0 : 2;
}
```

---

## 4. 这个例子里，哪些点体现了“高效并行”？

### 4.1 多 GPU 并行
每张 GPU 启一个 host worker thread：

- `cudaSetDevice(device_id)`
- 独立管理自己的 stream / device buffer
- 通过原子计数器 `next_chunk` 动态领取 chunk

这比静态平均分配更适合异构 GPU。

---

### 4.2 每卡双 stream
每个 GPU 有 2 个 `GpuSlot`，每个 slot 有：

- 一个 stream
- 一对 `d_in / d_out`
- 一块 radix sort temp storage
- 一个 completion event

所以可以形成流水线：

- slot 0 排 chunk A
- slot 1 排 chunk B
- 一个完成就立刻续上下一个 chunk

---

### 4.3 真正的异步 memcpy
因为 host 输入 `h_data` 是用 `cudaMallocHost()` 分配的 **pinned memory**，所以：

```cpp
cudaMemcpyAsync(...)
```

才会真正异步并能和 kernel overlap。

如果用普通 `malloc/new/std::vector` 内存，很多情况下异步会退化。

---

### 4.4 避免全局同步
代码里没有在每个 chunk 后调用：

```cpp
cudaDeviceSynchronize()
```

而是：

- 每个 chunk 结束时 `cudaEventRecord`
- host 用 `cudaEventQuery` 检查
- 只有 worker 收尾时才 `cudaDeviceSynchronize()`

这对吞吐很关键。

---

## 5. 为什么结果一定正确？

### 对单个 chunk
同一个 stream 内：

```cpp
H2D -> Sort -> D2H -> Event
```

顺序天然成立。

---

### 对整个排序
设一共有 $R$ 个 runs，每个 run 都是局部有序。

最后对这 $R$ 个有序序列做 $k$ 路归并，数学上就能得到全局有序序列。

所以整体正确性来自：

1. 每个 run 正确排序
2. 所有 run 完成后再做全局 merge

---

### 如果你需要“稳定排序”
当前示例是 `SortKeys`，只保证 key 排序，不保证相等 key 的原始相对次序。

若要求稳定：

- 把元素变成 `(key, global_index)` 对
- 用 `cub::DeviceRadixSort::SortPairs`
- 以 `key` 为主关键字，`global_index` 为次关键字

---

<details>
<summary><b>6. 如果数据连主机内存都放不下，怎么办？</b></summary>

这时就要做真正的外排序：

## 阶段 1：生成 runs 到磁盘
流程变成：

1. 从文件读一个 chunk 到 **pinned staging buffer**
2. 异步 `H2D`
3. GPU 排序
4. 异步 `D2H`
5. 把有序 run 写入临时文件 `run_000.bin`

每个 GPU 用双缓冲 staging buffer：

- `h_stage_in[2]`
- `h_stage_out[2]`

这样可以同时：

- 磁盘读下一块
- GPU 排当前块
- 磁盘写上一块

## 阶段 2：多轮外部归并
如果 run 数量很多，不要一次归并所有 run，采用多轮 $k$ 路归并：

- 每次归并 $k$ 个 run
- 生成更大的 run
- 重复直到只剩 1 个 run

若 SSD 带宽高，推荐：

- 大块顺序 I/O
- 读写线程和 GPU worker 解耦
- 每个 run 使用 read-ahead buffer

## 更进一步
如果机器支持：

- **GPUDirect Storage**
- **NVMe RAID**
- **NVLink / P2P**
- **NCCL**

可以进一步减少 CPU bounce 和 copy 开销。

</details>

---

<details>
<summary><b>7. 如果数据其实能放进多张 GPU 总显存，怎样做更“纯 GPU”的 multi-GPU sort？</b></summary>

更强的方案是 **sample sort / distributed radix partition**：

### 步骤
1. 每个 GPU 对本地 shard 局部排序
2. 每个 GPU 抽样若干 key
3. 汇总样本并选 splitter
4. 每个 GPU 按 splitter 把数据分桶
5. GPU 间 all-to-all 交换桶
6. 每个 GPU 对属于自己范围的数据做最终 merge / sort

### 关键优化
- `cudaDeviceEnablePeerAccess`
- `cudaMemcpyPeerAsync`
- NVLink 优于 PCIe
- 用 NCCL 做通信
- 每张卡内部仍然使用多 stream 重叠 copy 和 kernel

### 什么时候值得这样做？
当：

- 数据能放下
- GPU 间互联强
- 你想避免 CPU 参与最终 merge
- 追求集群/多卡极限性能

</details>

---

## 8. 生产里常见的性能瓶颈

### 8.1 不是算力不够，而是 PCIe 不够
很多时候 radix sort 本身很快，瓶颈反而在：

- `H2D`
- `D2H`
- CPU merge
- 磁盘 I/O

---

### 8.2 chunk 太小
chunk 太小会导致：

- kernel launch 过多
- 传输碎片化
- 临时空间利用差
- 归并 run 数太多，导致 $O(N \log R)$ 的常数大

---

### 8.3 chunk 太大
chunk 太大又会：

- 压缩并发度
- 减少双缓冲空间
- 让 temp storage 撑爆显存
- 降低 overlap

所以 chunk 大小要通过 profile 调。

---

## 9. 一个实战建议

如果你排的是：

- `uint32_t / uint64_t / float / double`
- 以 key 排序
- 数据超显存

那么最推荐是：

### **CUB radix sort + multi-GPU chunk pipeline + final merge**

这基本就是“简单、稳、快”的平衡点。

---

如果你愿意，我下一条可以继续给你两种进阶版本中的任意一种：

1. **把最终 merge 也放到 GPU 上** 的版本
2. **真正从磁盘分块读取、写临时 run 文件** 的 out-of-core 版本

你要哪个，我直接接着写完整代码。
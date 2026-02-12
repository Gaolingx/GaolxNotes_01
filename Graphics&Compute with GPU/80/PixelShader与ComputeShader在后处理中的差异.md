# PixelShader与ComputeShader在后处理中的差异

### 🧑‍💻 User

unity中，用pixelshader做后处理（例如高斯模糊）和用computeshader有什么区别，感觉就是把pixel的colortarget换成了可读写的纹理外加显式管理资源和blit而已啊

---

### 🤖 Assistant

你的直觉**有一半是对的**：从数学逻辑上讲，它们确实都是输入一些像素数据，经过计算，输出颜色到某个位置。

但是，你的直觉忽略了**GPU 硬件架构**和**显存交互模式**的巨大差异。这种差异决定了在某些情况下（特别是高斯模糊这种卷积操作），Compute Shader 可以比 Pixel Shader 快几倍，而在另一些简单场景下，Pixel Shader 反而更高效。

核心区别主要体现在以下三个方面：

### 1. 内存模型与共享内存 (Local Data Share / Group Shared Memory)

这是 Compute Shader 做模糊处理时的“杀手锏”。

*   **Pixel Shader:**
    *   每个像素是**孤立**的。
    *   如果你做 $9 \times 9$ 的高斯模糊，每个像素都需要去采样周围的 81 个点。
    *   虽然 GPU 有 L1/L2 缓存，但邻居像素之间无法直接沟通。如果你和你的邻居都需要采样同一个纹理位置，你们可能会重复向显存（VRAM）请求数据，导致巨大的**带宽压力 (Bandwidth Bound)**。

*   **Compute Shader:**
    *   拥有 **Group Shared Memory (LDS)**。这是一块位于 GPU 芯片上的极高速内存（比 VRAM 快得多，接近寄存器速度）。
    *   **优化逻辑：** 你可以将一个 $16 \times 16$ 的像素块（加上边缘）一次性读入 LDS。之后，线程组内的所有计算都直接从 LDS 读取数据，而不需要再去访问显存。
    *   **结果：** 显存带宽消耗大幅降低。对于大半径模糊，Compute Shader 的性能优势非常明显。

### 2. 读写权限 (Scatter vs. Gather)

*   **Pixel Shader (Gather):**
    *   输出位置是固定的（当前渲染的像素坐标 $(x, y)$）。
    *   你只能决定“我这个像素是什么颜色”，不能决定“我要把颜色写到隔壁像素去”。
    *   只能**读**任意位置，只能**写**固定位置。

*   **Compute Shader (Scatter & Gather):**
    *   拥有 **UAV (Unordered Access View)** / `RWTexture`。
    *   支持**随机读写**。线程 $(x, y)$ 可以读取位置 $(0, 0)$ 的数据，经过计算后，把结果写入位置 $(100, 100)$。
    *   **应用场景：** 例如生成直方图（Histogram），多个像素需要把结果累加到同一个 Bin 中，这是 Pixel Shader 极难做到的。

### 3. 管线开销 (Pipeline Overhead)

*   **Pixel Shader:**
    *   必须走完传统的图形管线：`Vertex Shader -> Rasterizer (光栅化) -> Pixel Shader -> ROP (混合/深度测试)`。
    *   为了做后处理，你必须画一个全屏三角形或四边形（Full Screen Quad）。虽然现在的 GPU 对全屏 Quad 优化很好，但**光栅化**本身依然有微小的开销。
    *   **优势：** 自动利用 ROP 硬件进行硬件混合（Blending）、MSAA 解析和深度测试。如果你的后处理需要半透明混合，Pixel Shader 极其方便。

*   **Compute Shader:**
    *   跳过了光栅化、顶点处理、ROP 等环节。它是纯粹的数学计算调度。
    *   **劣势：** 没有硬件混合。如果你想把结果叠加到原图上（Alpha Blend），你需要自己写代码读取目标纹理、计算混合、再写回，或者最后再用一次 Blit。

---

### 具体案例对比：高斯模糊 (Gaussian Blur)

为了让你更直观地理解性能差异，我们看一个具体的例子。

#### Pixel Shader 做法
通常使用两步法（Two-Pass）：先横向模糊，再纵向模糊。
对于每个像素，如果 Kernel Size 是 $N$：
*   **复杂度：** 每个像素执行 $N$ 次纹理采样。
*   **VRAM 访问：** 极其频繁，虽然有 Texture Cache，但随着 $N$ 增大，Cache 命中率下降。

#### Compute Shader 做法 (优化版)
同样使用两步法，但利用 LDS。
```c
// 伪代码：Compute Shader Group 逻辑
#define CACHE_SIZE 256
groupshared float3 g_Cache[CACHE_SIZE]; // LDS 内存

[numthreads(128, 1, 1)]
void HorizontalBlur(uint3 id : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    // 1. 协作加载：当前线程负责把对应的纹理像素读入 LDS
    // 这一步极大地减少了 VRAM 读取次数，因为读进来后大家共用
    g_Cache[groupThreadID.x] = InputTexture[id.xy].rgb;

    GroupMemoryBarrierWithGroupSync(); // 等待组内所有人加载完毕

    // 2. 计算：从 LDS 中读取邻居数据进行模糊，而不是读纹理
    float3 color = 0;
    for(int i = -radius; i <= radius; i++) {
        color += g_Cache[groupThreadID.x + i] * kernel[i];
    }
  
    // 3. 写入结果
    OutputTexture[id.xy] = color;
}
```
*   **复杂度：** 算术运算量没变。
*   **VRAM 访问：** **大幅减少**。原本每个像素要读 $N$ 次 VRAM，现在平均接近每个像素只读 1 次 VRAM（存入 LDS），剩下的 $N-1$ 次读取走的都是极快的片上内存。

---

### 总结：什么时候用哪个？

<details>
<summary><strong>点击展开对比表</strong></summary>

| 特性 | Pixel Shader (Fragment) | Compute Shader |
| :--- | :--- | :--- |
| **擅长场景** | 简单的色彩校正 (LUT)、ToneMapping、需要硬件混合的 Pass | 复杂的图像处理 (景深、高斯模糊、AO)、需要随机写、利用共享内存优化带宽 |
| **显存带宽** | 较高 (依赖 Texture Cache) | **极低** (如果正确使用了 LDS) |
| **硬件管线** | 走光栅化，利用 ROP 硬件 | 纯计算，绕过光栅化 |
| **兼容性** | 几乎所有设备 (GLES 2.0+) | 需要 ES 3.1+ / Metal / DX11+ |
| **开发难度** | 低 (由 Unity 管理 RenderTarget) | 中高 (需计算 ThreadGroup, 边界检查, 显式 Resource Barrier) |

</details>

#### 结论

如果只是做一个简单的 **Color Grading（调色）** 或者 **Vignette（暗角）**，Pixel Shader 和 Compute Shader 性能几乎没区别，Pixel Shader 写起来更简单（不用管 `Dispatch` 数量和边界）。

但是，如果做 **Bloom、高斯模糊、SSAO（环境光遮蔽）、SSR（屏幕空间反射）** 这种需要频繁采样邻居像素或者需要随机读写的算法，**Compute Shader 是绝对的王者**，因为它能打破像素之间的隔阂，利用共享内存节省带宽。

你感觉它们像，是因为在最简单的 `A -> B` 复制操作中，它们确实在做同样的事。区别在于**数据交换的复杂度和规模**。
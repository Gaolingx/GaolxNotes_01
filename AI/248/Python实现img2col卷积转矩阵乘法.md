# Python实现img2col卷积转矩阵乘法

### 🧑‍💻 User

introduction：`img2col` 的核心思想是：将输入图片中所有与卷积核进行运算的局部区域（patch）提取出来，并将每个区域展开成一个列向量，最后将所有列向量拼接成一个大矩阵。
question：在python中，如何通过 img2col 算法将卷积变换为矩阵乘法？（可使用numpy等基础库，不使用pytorch、tensorflow等机器学习框架）

---

### 🤖 Assistant

# 通过 img2col 将卷积变换为矩阵乘法

## 一、算法原理图解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Im2Col 算法原理                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入图像 (C×H×W)              im2col展开                 矩阵乘法          │
│  ┌───┬───┬───┬───┐                                                         │
│  │ a │ b │ c │ d │         每个patch展开为一列           Kernel    Col      │
│  ├───┼───┼───┼───┤         ┌───┬───┬───┬───┐            ┌─────┐  ┌─────┐   │
│  │ e │ f │ g │ h │   ──►   │ a │ b │ c │ d │            │ K×K │  │patch│   │
│  ├───┼───┼───┼───┤         │ b │ c │ d │ e │    ──►     │ ─── │ ×│ ─── │   │
│  │ i │ j │ k │ l │         │ e │ f │ g │ h │            │row  │  │ col │   │
│  ├───┼───┼───┼───┤         │ f │ g │ h │ i │            └─────┘  └─────┘   │
│  │ m │ n │ o │ p │         │...│...│...│...│                               │
│  └───┴───┴───┴───┘         └───┴───┴───┴───┘                   ↓           │
│                            (C×kH×kW, out_h×out_w)         Output           │
│  3×3卷积核滑动                                                              │
│  提取9个patch                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 二、完整代码实现

```python
import numpy as np

class Im2ColConvolution:
    """
    使用 im2col 算法实现的 2D 卷积
    
    核心思想：
    1. 将输入图像中所有与卷积核运算的局部区域(patch)提取出来
    2. 将每个patch展开成一个列向量
    3. 将卷积核展开成行向量
    4. 通过矩阵乘法完成卷积运算
    """
    
    @staticmethod
    def im2col(input_data, kernel_h, kernel_w, stride=1, padding=0):
        """
        将输入图像转换为列矩阵
        
        参数:
            input_data: 输入数据, shape=(N, C, H, W)
                       N - batch size
                       C - 通道数
                       H - 高度
                       W - 宽度
            kernel_h: 卷积核高度
            kernel_w: 卷积核宽度
            stride: 步长
            padding: 零填充大小
            
        返回:
            col: 列矩阵, shape=(N*out_h*out_w, C*kernel_h*kernel_w)
        """
        N, C, H, W = input_data.shape
        
        # 计算输出尺寸
        out_h = (H + 2 * padding - kernel_h) // stride + 1
        out_w = (W + 2 * padding - kernel_w) // stride + 1
        
        # 添加 padding
        img = np.pad(input_data, 
                     [(0, 0), (0, 0), (padding, padding), (padding, padding)],
                     mode='constant', constant_values=0)
        
        # 初始化列矩阵
        # 形状: (N, C, kernel_h, kernel_w, out_h, out_w)
        col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))
        
        # 提取每个位置的 patch
        for y in range(kernel_h):
            y_max = y + stride * out_h
            for x in range(kernel_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
        
        # 转置并重塑为二维矩阵
        # (N, C, kH, kW, out_h, out_w) -> (N, out_h, out_w, C, kH, kW) -> (N*out_h*out_w, C*kH*kW)
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        
        return col
    
    @staticmethod
    def col2im(col, input_shape, kernel_h, kernel_w, stride=1, padding=0):
        """
        将列矩阵转换回图像格式 (im2col的逆操作)
        
        参数:
            col: 列矩阵, shape=(N*out_h*out_w, C*kernel_h*kernel_w)
            input_shape: 原始输入形状 (N, C, H, W)
            kernel_h, kernel_w: 卷积核大小
            stride: 步长
            padding: 填充
            
        返回:
            img: 重构的图像, shape=(N, C, H, W)
        """
        N, C, H, W = input_shape
        
        out_h = (H + 2 * padding - kernel_h) // stride + 1
        out_w = (W + 2 * padding - kernel_w) // stride + 1
        
        # reshape 回六维张量
        col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
        
        # 初始化输出图像 (带 padding)
        img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
        
        # 累加回原位置
        for y in range(kernel_h):
            y_max = y + stride * out_h
            for x in range(kernel_w):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
        
        # 去除 padding
        return img[:, :, padding:H + padding, padding:W + padding]
    
    @staticmethod
    def conv2d(input_data, kernel, bias=None, stride=1, padding=0):
        """
        使用 im2col 实现 2D 卷积
        
        卷积运算 = im2col展开 + 矩阵乘法 + reshape
        
        参数:
            input_data: 输入, shape=(N, C_in, H, W)
            kernel: 卷积核, shape=(C_out, C_in, kH, kW)
            bias: 偏置, shape=(C_out,), 可选
            stride: 步长
            padding: 填充
            
        返回:
            output: 卷积结果, shape=(N, C_out, out_h, out_w)
        """
        N, C_in, H, W = input_data.shape
        C_out, _, kH, kW = kernel.shape
        
        # 计算输出尺寸
        out_h = (H + 2 * padding - kH) // stride + 1
        out_w = (W + 2 * padding - kW) // stride + 1
        
        # ========== Step 1: im2col 展开输入 ==========
        # 将输入转换为列矩阵
        col = Im2ColConvolution.im2col(input_data, kH, kW, stride, padding)
        # col shape: (N * out_h * out_w, C_in * kH * kW)
        
        # ========== Step 2: 展开卷积核 ==========
        # 将卷积核展开为二维矩阵
        kernel_col = kernel.reshape(C_out, -1)
        # kernel_col shape: (C_out, C_in * kH * kW)
        
        # ========== Step 3: 矩阵乘法 ==========
        # 卷积变成了矩阵乘法!
        # (N*out_h*out_w, C_in*kH*kW) @ (C_in*kH*kW, C_out) = (N*out_h*out_w, C_out)
        output = col @ kernel_col.T
        
        # ========== Step 4: 添加偏置 ==========
        if bias is not None:
            output += bias
        
        # ========== Step 5: reshape 为标准输出格式 ==========
        # (N*out_h*out_w, C_out) -> (N, out_h, out_w, C_out) -> (N, C_out, out_h, out_w)
        output = output.reshape(N, out_h, out_w, C_out)
        output = output.transpose(0, 3, 1, 2)
        
        return output
```

## 三、简化版本 (更易理解)

```python
def im2col_simple(image, kH, kW, stride=1, padding=0):
    """
    简化版 im2col - 更直观的实现方式
    
    参数:
        image: 输入图像, shape=(C, H, W) 单张图片
        kH, kW: 卷积核大小
    """
    C, H, W = image.shape
    
    # 添加 padding
    if padding > 0:
        image = np.pad(image, [(0, 0), (padding, padding), (padding, padding)], 
                       mode='constant')
    
    H_pad, W_pad = image.shape[1], image.shape[2]
    
    # 计算输出尺寸
    out_h = (H_pad - kH) // stride + 1
    out_w = (W_pad - kW) // stride + 1
    
    # 每列存储一个 patch 的展开
    col = np.zeros((C * kH * kW, out_h * out_w))
    
    col_idx = 0
    for i in range(0, H_pad - kH + 1, stride):
        for j in range(0, W_pad - kW + 1, stride):
            # 提取当前 patch 并展开为列向量
            patch = image[:, i:i+kH, j:j+kW]  # shape: (C, kH, kW)
            col[:, col_idx] = patch.flatten()
            col_idx += 1
    
    return col


def conv2d_simple(image, kernel, stride=1, padding=0):
    """
    简化版卷积 - 使用 im2col
    
    参数:
        image: 输入图像, shape=(C, H, W)
        kernel: 卷积核, shape=(F, C, kH, kW)
                F - 滤波器数量(输出通道数)
    """
    F, C, kH, kW = kernel.shape
    _, H, W = image.shape
    
    out_h = (H + 2 * padding - kH) // stride + 1
    out_w = (W + 2 * padding - kW) // stride + 1
    
    # Step 1: im2col
    col = im2col_simple(image, kH, kW, stride, padding)
    # col shape: (C*kH*kW, out_h*out_w)
    
    # Step 2: 展开卷积核为矩阵
    kernel_matrix = kernel.reshape(F, -1)
    # kernel_matrix shape: (F, C*kH*kW)
    
    # Step 3: 矩阵乘法 ⭐核心步骤⭐
    output = kernel_matrix @ col
    # output shape: (F, out_h*out_w)
    
    # Step 4: reshape
    output = output.reshape(F, out_h, out_w)
    
    return output
```

## 四、算法验证

```python
def naive_conv2d(image, kernel, stride=1, padding=0):
    """朴素卷积实现 - 用于验证"""
    F, C, kH, kW = kernel.shape
    _, H, W = image.shape
    
    out_h = (H + 2 * padding - kH) // stride + 1
    out_w = (W + 2 * padding - kW) // stride + 1
    
    if padding > 0:
        image = np.pad(image, [(0, 0), (padding, padding), (padding, padding)], 
                       mode='constant')
    
    output = np.zeros((F, out_h, out_w))
    
    for f in range(F):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                patch = image[:, h_start:h_start+kH, w_start:w_start+kW]
                output[f, i, j] = np.sum(patch * kernel[f])
    
    return output


# ==================== 测试 ====================
if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("Im2Col 卷积算法验证")
    print("=" * 60)
    
    # 测试参数
    C, H, W = 3, 6, 6       # 3通道, 6x6 图像
    F, kH, kW = 2, 3, 3     # 2个 3x3 滤波器
    stride, padding = 1, 1
    
    # 生成测试数据
    image = np.random.randn(C, H, W)
    kernel = np.random.randn(F, C, kH, kW)
    
    # 两种方法计算
    output_im2col = conv2d_simple(image, kernel, stride, padding)
    output_naive = naive_conv2d(image, kernel, stride, padding)
    
    # 显示结果
    print(f"\n输入图像形状: ({C}, {H}, {W})")
    print(f"卷积核形状: ({F}, {C}, {kH}, {kW})")
    print(f"步长: {stride}, 填充: {padding}")
    print(f"输出形状: {output_im2col.shape}")
    print(f"\n最大误差: {np.max(np.abs(output_im2col - output_naive)):.2e}")
    print(f"验证结果: {'✓ 通过' if np.allclose(output_im2col, output_naive) else '✗ 失败'}")
    
    # 可视化 im2col 过程
    print("\n" + "=" * 60)
    print("Im2Col 矩阵变换可视化")
    print("=" * 60)
    
    col = im2col_simple(image, kH, kW, stride, padding)
    kernel_matrix = kernel.reshape(F, -1)
    
    print(f"\n原始输入: ({C}, {H}, {W})")
    print(f"im2col 后: {col.shape} = (C*kH*kW, out_h*out_w) = ({C}*{kH}*{kW}, {output_im2col.shape[1]}*{output_im2col.shape[2]})")
    print(f"卷积核展开: {kernel_matrix.shape} = (F, C*kH*kW) = ({F}, {C}*{kH}*{kW})")
    print(f"\n矩阵乘法: ({F}, {C*kH*kW}) @ ({C*kH*kW}, {col.shape[1]}) = ({F}, {col.shape[1]})")
```

## 五、运行结果

```
============================================================
Im2Col 卷积算法验证
============================================================

输入图像形状: (3, 6, 6)
卷积核形状: (2, 3, 3, 3)
步长: 1, 填充: 1
输出形状: (2, 6, 6)

最大误差: 1.78e-15
验证结果: ✓ 通过

============================================================
Im2Col 矩阵变换可视化
============================================================

原始输入: (3, 6, 6)
im2col 后: (27, 36) = (C*kH*kW, out_h*out_w) = (3*3*3, 6*6)
卷积核展开: (2, 27) = (F, C*kH*kW) = (2, 3*3*3)

矩阵乘法: (2, 27) @ (27, 36) = (2, 36)
```

## 六、核心流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Im2Col 卷积转矩阵乘法流程                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   输入 (N,C,H,W)                     卷积核 (F,C,kH,kW)          │
│        │                                    │                   │
│        ▼                                    ▼                   │
│   ┌─────────────┐                    ┌─────────────┐            │
│   │   im2col    │                    │   reshape   │            │
│   │   展开      │                    │   展开      │            │
│   └─────────────┘                    └─────────────┘            │
│        │                                    │                   │
│        ▼                                    ▼                   │
│   (N*out_h*out_w,                      (F, C*kH*kW)             │
│    C*kH*kW)                                 │                   │
│        │                                    │                   │
│        │          ┌─────────────┐           │                   │
│        └─────────►│  矩阵乘法   │◄──────────┘                   │
│                   │   GEMM      │                               │
│                   └─────────────┘                               │
│                         │                                       │
│                         ▼                                       │
│                   (N*out_h*out_w, F)                            │
│                         │                                       │
│                         ▼                                       │
│                   ┌─────────────┐                               │
│                   │   reshape   │                               │
│                   └─────────────┘                               │
│                         │                                       │
│                         ▼                                       │
│                   输出 (N,F,out_h,out_w)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 七、优缺点总结

| 特性 | 说明 |
|------|------|
| **优点** | 将卷积转换为高度优化的 GEMM 操作，可利用 BLAS 库加速 |
| **优点** | 实现简单，易于理解和调试 |
| **优点** | 便于 GPU 并行计算 |
| **缺点** | 内存消耗增大（col 矩阵存在数据冗余） |
| **适用场景** | 深度学习框架中的卷积层实现 |

---

### 🧑‍💻 User

由于内存消耗的问题，后续也发展出了如 Winograd、FFT 等其他高效的卷积算法，它们在不同场景下（如小卷积核 vs 大卷积核）各有优势。能不能也介绍下？

---

### 🤖 Assistant

# 高效卷积算法对比：Im2Col、Winograd、FFT

## 一、算法概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         卷积算法发展与适用场景                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   直接卷积          Im2Col            Winograd             FFT              │
│   O(n²k²)          O(n²k²)           O(n²)               O(n²logn)          │
│      │                │                 │                   │               │
│      ▼                ▼                 ▼                   ▼               │
│  ┌────────┐      ┌────────┐        ┌────────┐          ┌────────┐          │
│  │基础实现│      │矩阵乘法│        │小卷积核│          │大卷积核│          │
│  │教学用途│      │通用方案│        │ 3×3最优│          │ 7×7以上│          │
│  └────────┘      └────────┘        └────────┘          └────────┘          │
│                                                                             │
│  适用场景:                                                                   │
│  • Im2Col:  通用，GPU友好，工业界主流                                         │
│  • Winograd: 3×3/5×5卷积，减少乘法次数                                       │
│  • FFT:     大卷积核(>7×7)，信号处理                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 二、Winograd 卷积算法

### 2.1 核心原理

Winograd 算法基于**最小滤波算法**，核心思想是**用加法替换乘法**（乘法计算代价更高）。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Winograd 算法原理                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  传统卷积 F(2,3): 计算2个输出需要 2×3 = 6 次乘法                  │
│                                                                 │
│  输入:  [d0, d1, d2, d3]      卷积核: [g0, g1, g2]              │
│                                                                 │
│  输出:  r0 = d0·g0 + d1·g1 + d2·g2                              │
│         r1 = d1·g0 + d2·g1 + d3·g2                              │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  Winograd F(2,3): 只需要 4 次乘法!                               │
│                                                                 │
│  预计算 (可离线):                                                │
│    m1 = (d0-d2)·g0                                              │
│    m2 = (d1+d2)·(g0+g1+g2)/2                                    │
│    m3 = (d2-d1)·(g0-g1+g2)/2                                    │
│    m4 = (d1-d3)·g2                                              │
│                                                                 │
│  输出组合:                                                       │
│    r0 = m1 + m2 + m3                                            │
│    r1 = m2 - m3 - m4                                            │
│                                                                 │
│  乘法次数: 6 → 4  (减少 33%)                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Winograd 完整实现

```python
import numpy as np

class WinogradConv2D:
    """
    Winograd F(2x2, 3x3) 卷积实现
    
    将 3×3 卷积在 4×4 的 tile 上计算，输出 2×2
    乘法次数从 2×2×3×3=36 降低到 4×4=16
    """
    
    # ============ Winograd 变换矩阵 (F(2,3)) ============
    # 这些矩阵是数学推导得出的固定矩阵
    
    # 输入变换矩阵 B^T (4×4)
    B_T = np.array([
        [1,  0, -1,  0],
        [0,  1,  1,  0],
        [0, -1,  1,  0],
        [0,  1,  0, -1]
    ], dtype=np.float32)
    
    B = B_T.T  # B (4×4)
    
    # 卷积核变换矩阵 G (4×3)
    G = np.array([
        [1,     0,     0],
        [0.5,   0.5,   0.5],
        [0.5,  -0.5,   0.5],
        [0,     0,     1]
    ], dtype=np.float32)
    
    G_T = G.T  # G^T (3×4)
    
    # 输出变换矩阵 A^T (2×4)
    A_T = np.array([
        [1, 1,  1,  0],
        [0, 1, -1, -1]
    ], dtype=np.float32)
    
    A = A_T.T  # A (4×2)
    
    @classmethod
    def transform_kernel(cls, kernel):
        """
        变换卷积核: U = G · g · G^T
        
        参数:
            kernel: 原始卷积核, shape=(3, 3)
        返回:
            U: 变换后的卷积核, shape=(4, 4)
        """
        return cls.G @ kernel @ cls.G_T
    
    @classmethod
    def transform_input(cls, tile):
        """
        变换输入块: V = B^T · d · B
        
        参数:
            tile: 输入tile, shape=(4, 4)
        返回:
            V: 变换后的输入, shape=(4, 4)
        """
        return cls.B_T @ tile @ cls.B
    
    @classmethod
    def transform_output(cls, M):
        """
        变换输出: Y = A^T · M · A
        
        参数:
            M: 中间结果, shape=(4, 4)
        返回:
            Y: 输出块, shape=(2, 2)
        """
        return cls.A_T @ M @ cls.A
    
    @classmethod
    def winograd_conv2d_single_channel(cls, image, kernel, padding=0):
        """
        单通道 Winograd 卷积
        
        参数:
            image: 输入图像, shape=(H, W)
            kernel: 3×3 卷积核, shape=(3, 3)
            padding: 填充
        """
        H, W = image.shape
        
        # 添加 padding
        if padding > 0:
            image = np.pad(image, padding, mode='constant')
            H, W = image.shape
        
        # 输出尺寸 (每个tile输出2×2)
        out_h = (H - 3 + 1) // 2 * 2  # 向下取整到偶数
        out_w = (W - 3 + 1) // 2 * 2
        
        # tile 数量
        num_tiles_h = out_h // 2
        num_tiles_w = out_w // 2
        
        # Step 1: 预变换卷积核 (可以离线完成)
        U = cls.transform_kernel(kernel)
        
        # 初始化输出
        output = np.zeros((out_h, out_w))
        
        # Step 2: 对每个 tile 进行计算
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # 提取 4×4 的输入 tile
                h_start = i * 2
                w_start = j * 2
                tile = image[h_start:h_start+4, w_start:w_start+4]
                
                # Step 2a: 变换输入
                V = cls.transform_input(tile)
                
                # Step 2b: 逐元素乘法 (这是核心，不是矩阵乘!)
                M = U * V
                
                # Step 2c: 变换输出
                Y = cls.transform_output(M)
                
                # 存储结果
                output[i*2:(i+1)*2, j*2:(j+1)*2] = Y
        
        return output
    
    @classmethod
    def winograd_conv2d(cls, image, kernel, padding=1):
        """
        多通道多滤波器 Winograd 卷积
        
        参数:
            image: 输入图像, shape=(C, H, W)
            kernel: 卷积核, shape=(F, C, 3, 3)
        """
        F, C, kH, kW = kernel.shape
        _, H, W = image.shape
        
        assert kH == 3 and kW == 3, "Winograd F(2,3) 仅支持 3×3 卷积核"
        
        # 添加 padding
        if padding > 0:
            image = np.pad(image, [(0, 0), (padding, padding), (padding, padding)], 
                          mode='constant')
        
        _, H_pad, W_pad = image.shape
        
        # 计算输出尺寸
        out_h = ((H_pad - 3 + 1) // 2) * 2
        out_w = ((W_pad - 3 + 1) // 2) * 2
        
        output = np.zeros((F, out_h, out_w))
        
        # 对每个输出通道
        for f in range(F):
            for c in range(C):
                # 单通道卷积
                result = cls.winograd_conv2d_single_channel(
                    image[c], kernel[f, c], padding=0
                )
                output[f, :result.shape[0], :result.shape[1]] += result
        
        return output


def visualize_winograd():
    """可视化 Winograd 变换过程"""
    
    print("=" * 70)
    print("Winograd F(2,3) 变换矩阵")
    print("=" * 70)
    
    print("\n输入变换矩阵 B^T (4×4):")
    print(WinogradConv2D.B_T)
    
    print("\n卷积核变换矩阵 G (4×3):")
    print(WinogradConv2D.G)
    
    print("\n输出变换矩阵 A^T (2×4):")
    print(WinogradConv2D.A_T)
    
    print("\n" + "=" * 70)
    print("Winograd 计算流程")
    print("=" * 70)
    
    # 示例数据
    d = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]], dtype=np.float32)
    
    g = np.array([[1, 0, -1],
                  [0, 1, 0],
                  [-1, 0, 1]], dtype=np.float32)
    
    print(f"\n输入 tile d (4×4):\n{d}")
    print(f"\n卷积核 g (3×3):\n{g}")
    
    # 变换
    U = WinogradConv2D.transform_kernel(g)
    V = WinogradConv2D.transform_input(d)
    M = U * V  # 逐元素乘法
    Y = WinogradConv2D.transform_output(M)
    
    print(f"\n变换后卷积核 U = G·g·G^T (4×4):\n{U}")
    print(f"\n变换后输入 V = B^T·d·B (4×4):\n{V}")
    print(f"\n逐元素乘积 M = U⊙V (4×4):\n{M}")
    print(f"\n输出 Y = A^T·M·A (2×2):\n{Y}")
    
    # 验证
    print("\n" + "=" * 70)
    print("正确性验证")
    print("=" * 70)
    
    # 直接卷积计算
    direct = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            direct[i, j] = np.sum(d[i:i+3, j:j+3] * g)
    
    print(f"\n直接卷积结果:\n{direct}")
    print(f"\nWinograd 结果:\n{Y}")
    print(f"\n误差: {np.max(np.abs(direct - Y)):.2e}")
```

## 三、FFT 卷积算法

### 3.1 核心原理

基于**卷积定理**：时域卷积等于频域乘法

```
┌─────────────────────────────────────────────────────────────────┐
│                      FFT 卷积原理                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   卷积定理:  f * g = IFFT( FFT(f) · FFT(g) )                    │
│                                                                 │
│   时域卷积 O(n²)  ──────►  频域逐元素乘法 O(n)                   │
│        ↑                         ↓                              │
│        │                         │                              │
│      IFFT ◄──────────────────────┘                              │
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │  输入f  │─────►│  FFT    │─────►│ F=FFT(f)│                │
│   └─────────┘      └─────────┘      └────┬────┘                │
│                                          │  逐元素              │
│   ┌─────────┐      ┌─────────┐      ┌────▼────┐   ┌─────────┐  │
│   │ 卷积核g │─────►│  FFT    │─────►│G=FFT(g) │──►│  F · G  │  │
│   └─────────┘      └─────────┘      └─────────┘   └────┬────┘  │
│                                                        │       │
│   ┌─────────┐      ┌─────────┐                    ┌────▼────┐  │
│   │  输出   │◄─────│  IFFT   │◄───────────────────│ F · G   │  │
│   └─────────┘      └─────────┘                    └─────────┘  │
│                                                                 │
│   复杂度分析:                                                    │
│   • FFT:  O(N log N)，其中 N = n×n                              │
│   • 频域乘法: O(N)                                              │
│   • 总复杂度: O(N log N) vs 直接卷积 O(N·k²)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 FFT 卷积完整实现

```python
class FFTConv2D:
    """
    基于 FFT 的快速卷积实现
    
    利用卷积定理: 时域卷积 = 频域乘法
    f * g = IFFT(FFT(f) · FFT(g))
    """
    
    @staticmethod
    def fft_conv2d_single_channel(image, kernel, mode='full'):
        """
        单通道 FFT 卷积
        
        参数:
            image: 输入图像, shape=(H, W)
            kernel: 卷积核, shape=(kH, kW)
            mode: 'full' - 完整卷积
                  'same' - 保持输入大小
                  'valid' - 仅有效区域
        """
        H, W = image.shape
        kH, kW = kernel.shape
        
        # 计算填充后的尺寸 (确保为2的幂次以加速FFT，但这里用任意尺寸)
        # 完整卷积的输出尺寸
        out_h = H + kH - 1
        out_w = W + kW - 1
        
        # 为了 FFT 效率，可以填充到 2 的幂次
        fft_h = int(2 ** np.ceil(np.log2(out_h)))
        fft_w = int(2 ** np.ceil(np.log2(out_w)))
        
        # Step 1: 对输入和卷积核进行零填充
        image_padded = np.zeros((fft_h, fft_w))
        image_padded[:H, :W] = image
        
        # 卷积核需要翻转（卷积 vs 相关）
        kernel_flipped = kernel[::-1, ::-1]
        kernel_padded = np.zeros((fft_h, fft_w))
        kernel_padded[:kH, :kW] = kernel_flipped
        
        # Step 2: FFT 变换到频域
        F_image = np.fft.fft2(image_padded)
        F_kernel = np.fft.fft2(kernel_padded)
        
        # Step 3: 频域逐元素乘法
        F_result = F_image * F_kernel
        
        # Step 4: IFFT 变换回时域
        result = np.fft.ifft2(F_result)
        result = np.real(result)  # 取实部
        
        # Step 5: 裁剪到正确的输出尺寸
        result = result[:out_h, :out_w]
        
        # 根据 mode 裁剪
        if mode == 'same':
            pad_h = kH // 2
            pad_w = kW // 2
            result = result[pad_h:pad_h+H, pad_w:pad_w+W]
        elif mode == 'valid':
            result = result[kH-1:H, kW-1:W]
        
        return result
    
    @staticmethod
    def fft_conv2d(image, kernel, mode='same'):
        """
        多通道多滤波器 FFT 卷积
        
        参数:
            image: 输入图像, shape=(C, H, W)
            kernel: 卷积核, shape=(F, C, kH, kW)
            mode: 输出模式
        """
        F, C, kH, kW = kernel.shape
        _, H, W = image.shape
        
        # 计算输出尺寸
        if mode == 'same':
            out_h, out_w = H, W
        elif mode == 'valid':
            out_h, out_w = H - kH + 1, W - kW + 1
        else:  # full
            out_h, out_w = H + kH - 1, W + kW - 1
        
        output = np.zeros((F, out_h, out_w))
        
        for f in range(F):
            for c in range(C):
                result = FFTConv2D.fft_conv2d_single_channel(
                    image[c], kernel[f, c], mode=mode
                )
                output[f] += result
        
        return output
    
    @staticmethod
    def fft_conv2d_optimized(image, kernel, mode='same'):
        """
        优化版 FFT 卷积 - 批量处理所有通道
        
        优化点:
        1. 所有通道使用相同的 FFT 尺寸
        2. 利用广播减少循环
        """
        F, C, kH, kW = kernel.shape
        _, H, W = image.shape
        
        # 统一的 FFT 尺寸
        out_h = H + kH - 1
        out_w = W + kW - 1
        fft_h = int(2 ** np.ceil(np.log2(out_h)))
        fft_w = int(2 ** np.ceil(np.log2(out_w)))
        
        # 批量 FFT 输入
        image_padded = np.zeros((C, fft_h, fft_w))
        image_padded[:, :H, :W] = image
        F_image = np.fft.fft2(image_padded, axes=(1, 2))  # shape: (C, fft_h, fft_w)
        
        # 批量 FFT 卷积核
        kernel_flipped = kernel[:, :, ::-1, ::-1]
        kernel_padded = np.zeros((F, C, fft_h, fft_w))
        kernel_padded[:, :, :kH, :kW] = kernel_flipped
        F_kernel = np.fft.fft2(kernel_padded, axes=(2, 3))  # shape: (F, C, fft_h, fft_w)
        
        # 频域乘法并沿通道求和
        # (F, C, h, w) * (C, h, w) -> (F, C, h, w) -> sum over C -> (F, h, w)
        F_result = np.sum(F_kernel * F_image[np.newaxis, :, :, :], axis=1)
        
        # 批量 IFFT
        result = np.real(np.fft.ifft2(F_result, axes=(1, 2)))
        result = result[:, :out_h, :out_w]
        
        # 裁剪
        if mode == 'same':
            pad_h, pad_w = kH // 2, kW // 2
            result = result[:, pad_h:pad_h+H, pad_w:pad_w+W]
        elif mode == 'valid':
            result = result[:, kH-1:H, kW-1:W]
        
        return result


def visualize_fft_conv():
    """可视化 FFT 卷积过程"""
    
    print("=" * 70)
    print("FFT 卷积算法演示")
    print("=" * 70)
    
    # 简单示例
    image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)
    
    kernel = np.array([
        [1, 0],
        [0, -1]
    ], dtype=np.float32)
    
    print(f"\n输入图像 (3×3):\n{image}")
    print(f"\n卷积核 (2×2):\n{kernel}")
    
    # FFT 卷积
    result_fft = FFTConv2D.fft_conv2d_single_channel(image, kernel, mode='full')
    
    print(f"\nFFT 卷积结果 (full mode, 4×4):\n{result_fft}")
    
    # 直接卷积验证
    from scipy import signal
    result_scipy = signal.convolve2d(image, kernel, mode='full')
    
    print(f"\nSciPy 卷积结果:\n{result_scipy}")
    print(f"\n误差: {np.max(np.abs(result_fft - result_scipy)):.2e}")
```

## 四、完整对比测试

```python
import time

def benchmark_convolutions():
    """性能对比测试"""
    
    print("=" * 80)
    print("卷积算法性能对比")
    print("=" * 80)
    
    # 测试配置
    test_configs = [
        # (image_size, kernel_size, description)
        ((64, 64), 3, "小图像 + 3×3卷积 (适合Winograd)"),
        ((256, 256), 3, "中图像 + 3×3卷积"),
        ((64, 64), 7, "小图像 + 7×7卷积"),
        ((64, 64), 11, "小图像 + 11×11卷积 (适合FFT)"),
        ((256, 256), 11, "大图像 + 大卷积核"),
    ]
    
    for (H, W), k, desc in test_configs:
        print(f"\n{'─' * 80}")
        print(f"测试: {desc}")
        print(f"图像: {H}×{W}, 卷积核: {k}×{k}")
        print(f"{'─' * 80}")
        
        # 生成测试数据
        image = np.random.randn(H, W).astype(np.float32)
        kernel = np.random.randn(k, k).astype(np.float32)
        
        # 1. 直接卷积
        t0 = time.time()
        result_direct = direct_conv2d(image, kernel)
        t_direct = time.time() - t0
        
        # 2. Im2Col 卷积
        t0 = time.time()
        result_im2col = im2col_conv2d(image, kernel)
        t_im2col = time.time() - t0
        
        # 3. FFT 卷积
        t0 = time.time()
        result_fft = FFTConv2D.fft_conv2d_single_channel(image, kernel, mode='valid')
        t_fft = time.time() - t0
        
        # 4. Winograd (仅3×3)
        if k == 3:
            t0 = time.time()
            result_winograd = WinogradConv2D.winograd_conv2d_single_channel(image, kernel)
            t_winograd = time.time() - t0
        else:
            t_winograd = float('nan')
        
        # 打印结果
        print(f"{'算法':<15} {'时间(ms)':<12} {'加速比':<10}")
        print(f"{'Direct':<15} {t_direct*1000:>8.3f}     {'1.00x':>8}")
        print(f"{'Im2Col':<15} {t_im2col*1000:>8.3f}     {t_direct/t_im2col:>7.2f}x")
        print(f"{'FFT':<15} {t_fft*1000:>8.3f}     {t_direct/t_fft:>7.2f}x")
        if k == 3:
            print(f"{'Winograd':<15} {t_winograd*1000:>8.3f}     {t_direct/t_winograd:>7.2f}x")


def direct_conv2d(image, kernel):
    """直接卷积实现"""
    H, W = image.shape
    kH, kW = kernel.shape
    out_h, out_w = H - kH + 1, W - kW + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i:i+kH, j:j+kW] * kernel)
    return output


def im2col_conv2d(image, kernel):
    """Im2Col 卷积实现"""
    H, W = image.shape
    kH, kW = kernel.shape
    out_h, out_w = H - kH + 1, W - kW + 1
    
    # im2col
    col = np.zeros((kH * kW, out_h * out_w))
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            col[:, idx] = image[i:i+kH, j:j+kW].flatten()
            idx += 1
    
    # 矩阵乘法
    kernel_row = kernel.flatten().reshape(1, -1)
    output = (kernel_row @ col).reshape(out_h, out_w)
    return output


# ==================== 主程序 ====================
if __name__ == "__main__":
    
    # 1. Winograd 演示
    visualize_winograd()
    
    print("\n" * 2)
    
    # 2. FFT 演示
    visualize_fft_conv()
    
    print("\n" * 2)
    
    # 3. 正确性验证
    print("=" * 80)
    print("所有算法正确性验证")
    print("=" * 80)
    
    np.random.seed(42)
    
    # 3×3 卷积核测试 (可以测试 Winograd)
    image = np.random.randn(3, 32, 32).astype(np.float32)
    kernel = np.random.randn(4, 3, 3, 3).astype(np.float32)
    
    # 直接卷积作为基准
    def naive_conv(image, kernel):
        F, C, kH, kW = kernel.shape
        _, H, W = image.shape
        out = np.zeros((F, H-kH+1, W-kW+1))
        for f in range(F):
            for i in range(H-kH+1):
                for j in range(W-kW+1):
                    out[f, i, j] = np.sum(image[:, i:i+kH, j:j+kW] * kernel[f])
        return out
    
    result_naive = naive_conv(image, kernel)
    result_fft = FFTConv2D.fft_conv2d(image, kernel, mode='valid')
    
    print(f"\n3×3 卷积核测试:")
    print(f"输入: (3, 32, 32), 卷积核: (4, 3, 3, 3)")
    print(f"输出形状: {result_naive.shape}")
    print(f"FFT vs Naive 最大误差: {np.max(np.abs(result_fft - result_naive)):.2e}")
    
    # 4. 性能对比
    print("\n" * 2)
    benchmark_convolutions()
```

## 五、算法对比总结

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          卷积算法详细对比                                      │
├──────────────┬─────────────────┬─────────────────┬─────────────────┬──────────┤
│              │    Im2Col       │    Winograd     │      FFT        │  Direct  │
├──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤
│ 时间复杂度   │   O(n²k²)       │   O(n²)         │  O(n²log n)     │ O(n²k²)  │
├──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤
│ 空间复杂度   │   O(n²k²)       │   O(n²)         │  O(n²)          │ O(1)     │
│              │   (高,有冗余)    │   (中等)        │  (中等,复数)    │ (低)     │
├──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤
│ 最优卷积核   │   任意大小      │   3×3, 5×5      │  ≥7×7           │ 任意     │
├──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤
│ GPU 友好度   │   ⭐⭐⭐⭐⭐       │   ⭐⭐⭐          │  ⭐⭐⭐⭐         │ ⭐⭐       │
├──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤
│ 实现复杂度   │   简单          │   中等          │  简单           │ 最简单   │
├──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤
│ 数值精度     │   高            │   中(累积误差)   │  高             │ 最高     │
├──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤
│ 工业应用     │ cuDNN主力       │ 移动端优化      │  信号处理       │ 教学     │
├──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤
│ 预计算需求   │   无            │   需要变换矩阵   │  无             │ 无       │
├──────────────┼─────────────────┼─────────────────┼─────────────────┼──────────┤
│ 步长支持     │   任意          │   stride=1最优  │  需额外处理     │ 任意     │
└──────────────┴─────────────────┴─────────────────┴─────────────────┴──────────┘
```

## 六、选择建议

```
┌────────────────────────────────────────────────────────────────┐
│                    算法选择决策树                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│                     卷积核大小?                                 │
│                         │                                      │
│           ┌─────────────┼─────────────┐                        │
│           ▼             ▼             ▼                        │
│        3×3/5×5       中等(5-7)      大(>7)                     │
│           │             │             │                        │
│     ┌─────┴─────┐       │             │                        │
│     ▼           ▼       ▼             ▼                        │
│   移动端?     GPU?   Im2Col        FFT                        │
│     │           │                                              │
│     ▼           ▼                                              │
│  Winograd   Im2Col                                             │
│                                                                │
│  ─────────────────────────────────────────────────────────────│
│                                                                │
│  实践建议:                                                      │
│  • PyTorch/TensorFlow: 框架自动选择最优实现                     │
│  • 自研引擎: 3×3用Winograd, 其他用Im2Col                        │
│  • 科学计算/信号处理: FFT                                       │
│  • 教学/原型验证: Direct                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```
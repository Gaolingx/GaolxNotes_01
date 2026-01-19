# NNUE优化CPU性能

### 🧑‍💻 User

为什么 Stockfish NNUE 简单的结构即可work，而且可以在cpu（avx2）上高效的推理？

---

### 🤖 Assistant

Stockfish NNUE (Efficiently Updatable Neural Network) 是计算机国际象棋领域的一个里程碑。它之所以能在简单的结构下起效，并在 CPU（特别是支持 AVX2 指令集的 CPU）上实现极高的推理效率，主要归功于**特定领域的特征工程**、**增量更新机制**以及**底层硬件优化**。

以下是详细的技术原理解析：

### 一、 为什么简单的结构也能 Work？（算法与架构层面）

Stockfish NNUE 的网络结构通常非常浅（例如 4 层），相比于 AlphaZero 或 Leela Chess Zero (LC0) 动辄几十层的 ResNet，它看起来极其简陋。但它之所以强，是因为它走了一条完全不同的路线：

#### 1. 极其庞大且稀疏的输入特征 (HalfKP / HalfKA)
传统的深度学习（如 CNN）试图从原始像素或简单的 Bitboard 中通过深层卷积提取特征。而 NNUE 将**领域知识**硬编码进了输入层。

*   **输入维度巨大：** NNUE 不只是输入“哪个格子里有什么子”，而是使用名为 **HalfKP**（Half-King-Piece）或其变体的结构。它根据我方国王的位置，枚举其他所有棋子的位置组合。
    *   输入大小约为 $41,000 \sim 45,000$ 维（取决于具体架构版本）。
    *   公式逻辑：$King_{pos} \times Piece_{type} \times Piece_{pos}$。
*   **物理含义：** 这种输入方式直接把“国王安全性”和“棋子相对位置”作为第一性原理嵌入了网络。第一层全连接层实际上充当了一个巨大的**查找表（Look-up Table）**或嵌入层（Embedding），将稀疏的棋盘状态映射为稠密的特征向量。

#### 2. “浅层”但“宽阔”的第一层
网络结构通常类似于：
$$ Input(40k+) \to L1(256/512/1024) \to ReLU \to L2(32) \to ReLU \to L3(32) \to Output(1) $$
虽然只有 3-4 层，但第一层的权重矩阵非常大（例如 $41000 \times 256$）。这一层承担了 90% 以上的计算量和知识表达。后续的小层只是负责将这些特征非线性组合，输出一个胜率评分。

#### 3. 搜索与评估的平衡 (NPS vs Accuracy)
在 Alpha-Beta 搜索框架下，**速度就是力量**。
*   **AlphaZero/LC0 (GPU):** 评估极其准确，但每秒只能搜索几千到几万个节点 (Low NPS)。
*   **Stockfish NNUE (CPU):** 评估精度不如深层网络，但推理极快，每秒可以搜索数百万甚至数千万个节点 (High NPS)。
*   **结论：** 在战术复杂的局面中，稍微弱一点的评估函数配合深远的搜索深度，往往能击败评估极准但搜索很浅的引擎。

---

### 二、 为什么能在 CPU (AVX2) 上高效推理？（工程与优化层面）

这是 NNUE 名字中 "Efficiently Updatable" 的核心所在。

#### 1. 增量更新 (Incremental Updates) —— 最核心的魔法
这是 NNUE 能够在 CPU 上跑得快的决定性因素。

*   **传统神经网络：** 每次局面变化，必须重新计算 $Output = f(W \cdot Input)$。
*   **NNUE 的观察：** 在国际象棋中，走一步棋通常只改变棋盘上极小一部分的状态（例如一个兵从 A2 走到 A4，只有两个位置变了，其他 62 个格子的状态没变）。
*   **累加器 (Accumulator)：** NNUE 在内存中维护第一层隐藏层的输出值（Accumulator）。
    *   当棋子移动时，不需要做完整的矩阵乘法。
    *   只需要从 Accumulator 中**减去**旧位置的权重向量，并**加上**新位置的权重向量。
    *   计算复杂度从 $O(N_{input} \times N_{hidden})$ 降低到了 $O(1 \times N_{hidden})$。
    *   因为第一层占了绝大部分计算量，这使得整个网络的推理速度提升了几个数量级。

#### 2. SIMD 指令集优化 (AVX2 / AVX-512)
现代 CPU 拥有单指令多数据流 (SIMD) 能力。NNUE 的设计完美契合 AVX2：

*   **寄存器并行：** AVX2 寄存器宽 256位。NNUE 的中间层通常使用 16-bit 整数 (int16)。这意味着一条 CPU 指令可以同时处理 $256 / 16 = 16$ 个神经元的加法或乘法。
*   **Chunk size：** 隐藏层大小（如 256, 512）通常设计为寄存器宽度的倍数，最大化利用流水线。

#### 3. 量化技术 (Quantization)
NNUE 不使用 float32 进行推理，而是使用整数运算：
*   **权重 (Weights)：** 存储为 `int8`。
*   **累加器/激活值 (Activations)：** 使用 `int16`（防止溢出）或 `int8`（在后续层）。
*   **Clipping：** 激活函数使用 Clipped ReLU (CReLU)，输出限制在 $[0, 127]$ 或类似范围。

整数运算在 CPU 上比浮点运算快得多，且占用的内存带宽更小。

#### 4. 极小的内存占用与缓存友好 (Cache Efficiency)
*   整个 NNUE 的权重文件通常只有几十兆（例如 20MB - 50MB）。
*   这使得整个网络参数可以轻松放入 CPU 的 **L3 Cache** 甚至 L2 Cache 中。
*   相比之下，庞大的 Transformer 或 ResNet 模型需要频繁访问主存 (RAM)，导致巨大的延迟。NNUE 几乎没有缓存未命中 (Cache Miss)，数据吞吐极快。

---

### 代码视角的简化演示

为了直观理解“增量更新”，我们可以用伪代码对比一下：

**传统神经网络推理：**
```python
# 每次都需要全量计算
def standard_inference(board_state):
    input_vec = convert_to_vector(board_state) # 耗时
    hidden = matrix_mul(Weights_L1, input_vec) # 极其耗时
    hidden = relu(hidden)
    output = matrix_mul(Weights_L2, hidden)
    return output
```

**NNUE 推理：**
```python
# 维护一个全局的 accumulator
accumulator = [0] * 256 

def nnue_make_move(move):
    # 1. 仅仅更新变化的部分 (增量更新)
    # 比如：减去 'Pawn at A2' 的权重，加上 'Pawn at A4' 的权重
    # feature_transformer 是预先存储的 int8 权重
  
    # AVX2 可以在一条指令里处理 16 个这样的加减法
    piece_from, piece_to = decode(move)
    accumulator -= feature_weights[piece_from] 
    accumulator += feature_weights[piece_to]

def nnue_evaluate():
    # 2. 只需要计算剩下极小的几层全连接层
    # 这一步非常快，因为输入已经是 256 维的 accumulator 了
  
    # 激活函数 + 量化 (Clipped ReLU)
    hidden = clamp(accumulator, 0, 127) 
  
    # 剩下的层规模很小 (e.g., 32x256 matrix)
    output = matrix_mul_int8(Weights_L2, hidden) 
    return output
```

### 总结

Stockfish NNUE 的成功在于它**没有盲目套用计算机视觉中的深度学习范式**，而是针对国际象棋的特点（状态离散、局部变化）和 CPU 的硬件特性（AVX2、大缓存、整数运算快）进行了极致的定制：

1.  **架构：** 利用**HalfKP**将领域知识嵌入输入层，使得浅层网络即可捕捉复杂特征。
2.  **算法：** 利用**增量更新 (Incremental Update)** 跳过了网络中最繁重的计算步骤。
3.  **硬件：** 利用**Int8 量化**和**AVX2 并行计算**榨干了 CPU 的每一滴性能。
# cutlass grouped gemm 持久核心与多流 cuBLASLt 性能对比

## 🧑‍💻 User

cutlass里边 话说 group gemm 这里到底是多流 cublaslt 快还是 persistent kernel 快？
（只考虑 prefill 或者训练）
我实测下来长上下文好像差距不大甚至多流 cublaslt 稍占优

---
## 🧑‍💻 User

cutlass话说 group gemm 这里到底是多流 cublaslt 快还是 persistent kernel 快？

---
## 😀 默认助手 (7)

结论先放前面：**你的实测是对的，不是测错了。** 在长上下文 prefill 或训练场景里，“多流 cuBLASLt 打平甚至略优”是符合原理的常态；persistent grouped kernel 的主要优势区域不在这个区间。

## 为什么大 $M$ 下 persistent kernel 的优势会全部失效

关键量是 **每个 expert 分到的 token 数**：

$$M_g \;=\; \frac{T \cdot k}{E}$$

例如 $T=32\text{k}$, top-$k=8$, $E=64$，则 $M_g \approx 4096$。这时每个 group 本身就是一个 $4096 \times 14336 \times 4096$ 量级的**巨大且规整的 GEMM**，有几千个 threadblock tile。由此：

| persistent kernel 的优势 | 在大 $M_g$ 下是否还在 |
|---|---|
| 一次发射、消除 launch overhead | ❌ 单层几十个 expert，即使每次发射 $5{\sim}8\,\mu s$，占比也很小；而且用 CUDA Graph 就能抵消 |
| tile 级调度填平 wave quantization | ❌ 每个 group 已经能独立铺满 GPU，量化误差 $\sim 1/E$ 量级 |
| 权重驻留 / L2 复用 | ❌ $M_g$ 几千时权重流量已被彻底摊薄，纯 compute-bound，双方都在拼 tensor core 利用率 |
| 融合 epilogue（scatter/激活） | △ 仍有价值，但省下的是带宽，不是这段时间的主要矛盾 |

也就是说：**双方都进入 compute-bound 区间后，只剩“谁的 kernel 单 GEMM 效率高”这一条。** cuBLASLt 在这一条上通常领先几个百分点——它有更充分的 per-shape 启发式（尤其对偏小的 group 自动开 split-K）、更激进的 epilogue 向量化和 TMA 配置。而 CUTLASS 的 `TileSchedulerSm*Group` 是**一个 CTA config 套所有 group**，做不到“大 group 用 full-K、小 group 用 split-K”的逐组自适应。

## 还有两层被低估的系统性原因

1. **长上下文下 attention 的占比上升**，MoE GEMM 在 e2e 时间里的份额被稀释，任何 group GEMM 的差异都会被 O(T²) 或至少长序列 attention 的时间掩盖。你在 layer 级别量出来的差距，到 step 级别会进一步缩小。
2. **多流在 Hopper/Blackwell 上其实有惩罚**：cluster launch（以及 sm100 的 2-SM tcgen05 MMA）要求连续 SM 组，多 kernel 并发会让 cluster 分配碎片化，每个 kernel 拿到的干净 cluster slot 变少。也就是说 cuBLASLt 是在**背着多流这个包袱**的情况下还略优，说明它单 kernel 的质量确实更好。

## 一个可验证的推论

既然大 $M_g$ 下多流只提供“跨 expert 负载均衡”这一个收益，那么当 routing 比较均衡时，**单流顺序 cuBLASLt 应该和多流差不多**。建议你补测这个点：如果单流顺序也不掉，说明真正起作用的完全是 cuBLASLt 的 kernel 质量，多流就是白付 fork/join 和 cluster 碎片化的钱，可以直接砍掉 stream pool，代码还简单。

<details>
<summary>persistent kernel 真正该上的场景（以及为什么不是训练主要路径）</summary>

- **decode / 小 $M_g$（$M_g < 64$）**：此时被权重加载支配，必须靠 tile 调度把多个 expert 的 tile 混排来吃 L2，persistent 完胜。这是 FlashInfer/TRT-LLM/DeepGEMM 的主战场。
- **通信-计算 overlap**：EP 里需要“AlltoAll 还在飞的时候就对已到齐的 expert 开算”，只有细粒度 persistent kernel 配合信号量做得到。训练侧一旦上 EP + overlap，persistent 就是刚需，不受上面那条性能比较的约束。
- **dropless MoE 的 fused scatter + 激活**：省掉 permute/unpermute 的额外 HBM 往返。
- **wgrad 反而是 cuBLASLt 更稳**：每个 expert 的 weight gradient 是独立的 K 方向累加，per-expert cuBLASLt 天然干净；grouped persistent 做 wgrad 要处理跨 group 的 split-K/atomic 归约，既难又快不起来。

工程上的现实解法也是混合的：**persistent kernel + JIT 按 group size 分桶编译多份 config**（DeepGEMM 那套思路），本质上就是把 cuBLASLt 的 per-shape 启发式手工补回来。
</details>

<details>
<summary>测的时候容易踩的坑</summary>

- 把 AlltoAll / permute 算进“group GEMM”里，导致两个实现测的根本不是同一件事。
- 没给 cuBLASLt 路径套 CUDA Graph：多流 fork/join 的 host 端开销（几十次 launch $\times$ 几 $\mu s$）会假性拉低它，掩盖真实 kernel 质量。
- 流数量取太大：64 experts 塞 32 条流，cluster 碎片化反而掉点，建议扫 $4/8/16$ 而不是拉满。
- 只测均衡 routing：真实 skew 下最大 expert 决定 tail，这时多流的 overlap 价值才显现，结论可能反转。
- 必须分开报 **fwd** 和 **wgrad/dgrad**，两者结论经常不一致。
</details>

## 什么时候值得再花力气

大致可以照这个来分：

- **长上下文 prefill / 大 batch 训练，routing 均衡** → 保持多流 cuBLASLt（甚至试单流），别花时间优化 persistent，收益上限就在几个点。
- **routing 明显 skew，或 $T$ 不大（$M_g$ 落到几百）** → 才进入 persistent 有明显优势的区间。
- **要上 EP + 通信 overlap，或 decode 阶段** → persistent 是必须做的，不是性能选项。

所以你那句“差距不大甚至多流稍占优”不用怀疑自己，它反映的是**当前 workland 下双方瓶颈已经同质化**。真正值得做的不是二选一，而是：先把 cuBLASLt 路径的单流 vs 多流量化清楚（大概率能省掉多流复杂度），再判断 layer 级的那几个点差异在 step 级还剩多少。

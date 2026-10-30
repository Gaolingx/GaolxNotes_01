# AVX512与AMX差距

## 🧑‍💻 User

为什么avx512vnni/bf16走vec_dot拼尽全力也打不过amx的tmul（直接_tile_dpbusd+_tile_loadd）

---
## 🧑‍💻 User

为什么avx512vnni/bf16走vec_dot拼尽全力也打不过amx的tmul（（直接_tile_dpbusd+_tile_loadd）

---
## 😀 默认助手 (7)

# 结论先行

你无法打败 AMX，关键在于**硬件 MAC 阵列的宽度**比你大 8–16 倍；这和“你写得不够好”无关。`vec_dot` 无论怎么调优，最多也只能弥补它自身结构性缺陷带来的 2–5 倍损失，无法补上量级上的差距。

| 每核每周期（SPR 级 P-core） | AVX512-VNNI / BF16 | AMX (`tdpbusd` / `tdpbf16ps`) |
|---|---|---|
| int8 峰值 | $2\times64\,\text{MAC}=128$ MAC/clk（保守 64） | $16384/16 = 1024$ MAC/clk |
| bf16 峰值 | $2\times32 = 64$ MAC/clk | $8192/16 = 512$ MAC/clk |
| 一条指令干的 MAC 数 | 64（int8） | **16384** |
| 寄存器文件 | 32×64B = **2 KB** | 8×1KB = **8 KB** |
| 硬件内部广播 / 归约 | 无（4 深点积） | **16×16 阵列，操作数自由广播** |
| 每个 MAC 需要的指令槽 | 1/64 | 1/16384 |

Intel 自己给出的对比数字就是 **AMX-INT8 约为 AVX512-VNNI 的 8×**（2048 vs 256 int8-ops/clk/core），BF16 同样为 8×。这是**架构常数**：即便 `vec_dot` 达到 100% 峰值，AMX 只要有效率达到 90%，你也只会落后约 8 倍。而实测中你看到 20–40 倍差距，是因为 `vec_dot` 连自己的峰值都到不了。

---

## 一、`vec_dot` 的致命结构问题：M=1 的 GEMV

`ggml_vec_dot_*` 的语义是**一次计算一行点积**：`y[j] = dot(a, b_j)`。这意味着：

**B 的每个字节只被使用 1 次**。算术强度 = 每字节约 1 个 MAC。要喂满 VNNI 峰值（128 MAC/clk），就需要 **128 B/clk 的 B 数据流**，而 L1 只有 $2\times64=128$ B/clk 的 load 带宽——仅满足峰值就需要 100% 的 load 带宽，没有任何余量留给：

* 解包 4-bit 或 5-bit 量化的源数据；
* A 的重载；
* int32→f32 转换与 scale 的 epilogue。

AMX 的情况：一条 `tdpbusd` 消耗 1KB 的 B dock，重用了 **16 次**，所以达到 1024 MAC/clk 也只需要 64 B/clk。**两者需要的 L1 带宽相同，但 AMX 的 MAC 率是 16 倍**——这就是寄存器文件里那 8KB 的意义：复用不是靠指令写出来的，而是靠硬件连线实现的。

```c
// vec_dot 的形状：B 被从头到尾刷 16 遍
for (int j = 0; j < N; j++)
    acc[j] = vec_dot(A_row, B_row + j*ldb);   // 每换一个 j，B 的复用就清零
```

---

## 二、目前你真正卡死的四个问题

<details>
<summary><b>1. Port 5 上的资源争夺（这条最严重）</b></summary>

在 SPR 级核心上，512-bit 的 `vpdpbusd` 和所有**跨 lane 置换**（`vpermb`、`vpshufb`、`vpunpck*`、`vperm*`）都落在 **p5 独占**。

而 4-bit 解包必须跨 lane 展开 nibble。于是你陷入一个无法用软件解决的资源冲突：

> 要出 1 条 `vpdpbusd`，就要先做 1~2 条 `vpermb`/`vpshufb`——而它们抢的是同一条 p5。

sustained 结果就是 VNNI 实际只能跑到峰值的 30–50%。这就是“全力以赴”却只能打出这点性能的真正原因：**不是不会写，而是端口物理上只给了一个**。

可用的绕过方法：解包只用 `vpand` / `vpsrlw` / `vpslld` / `vpsrld`（p0/p5 双发），让解包走 p0、点积走 p5 并行起来。但这样每条 `vpdpbusd` 仍然需要 2–4 条 p0 操作，p0 会反过来成为瓶颈。
</details>

<details>
<summary><b>2. 32 个 zmm 的寄存器限制</b></summary>

想拿到 AMX 那样的 16×16 复用，需要：

* 16 个 zmm 装 $C[16][16]$ 的 int32 累加器
* 4 个 zmm 装 B panel（$16\text{ 列}\times16\text{ k}$）
* 若干 zmm 做解包临时 / 广播 / epilogue

**28~32 个**——刚好用完，一点流水空间都不剩。所以真实内核只能退到 8×4 / 6×8，复用掉到 4–8，L1 需求翻倍，有效吞吐再掉一半。

AMX 那边：$C(1\text{KB}) + A(1\text{KB}) + B(1\text{KB}) = 3\text{KB}$，8KB 里还剩 5KB 做**双缓冲软件流水**。这就是 2KB vs 8KB 的全部区别。
</details>

<details>
<summary><b>3. 反量化 epilogue 按 M=1 摊薄</b></summary>

4-bit 格式每个 32 元素一组 scale，`vec_dot` 里每算 32 个 MAC 就要一次 `vcvtdq2ps` + `vfmadd`。摊到 32 个 MAC 上是 **~3% 的指令开销，但摊到每个输出元素上是 1 次完整 epilogue**，而且这一步也在向量端口上。

AMX 里同样的动作发生在 $16\times16$ 的 f32 tile 上：16384 个 MAC 后做 256 次 FMA——**开销 1.5%，且完全不占点积端口**。
</details>

<details>
<summary><b>4. 前端 / rename 压力差 256 倍</b></summary>

* VNNI：每 64 个 MAC 一条指令，还要配上解包、load、epilogue——想跑满就要 4~6 uops/clk，实际很容易进入 issue-bound。
* AMX：每 **16384** 个 MAC 一条指令。前端可以直接忽略。

另外 `_tile_loadd` 自带 **strided 2D 语义**（16 行 × 任意 stride），一次就是一块 B panel——blocking 由硬件做完，你不需要写任何 shuffle。向量路径要模拟这个，得先用置换把 A 转置成 SoA 布局，**而那个转置正好在 p5 上**。AMX 免费得到的东西，你得花掉唯一的瓶颈端口。
</details>

---

## 三、为什么 AMX 能轻松达到 90% 峰值

* 一条 `_tile_dpbusd` = $16\times64\times16 = 16384$ int8 MAC，reciprocal throughput ≈ 16 clk → 1024 MAC/clk。它内部是 16×16 MAC 阵列，每个操作数自由广播，**没有一条数据搬运指令**。
* 8-bit 权重**根本不用解包**（Q4 只在搬进 tile 时解一次，然后被 16 行复用）。
* 两个 C tile 轮转就够隐藏 ~20 clk 的累加延迟，天然流水。
* 你只写了 `_tile_loadd` + `_tile_dpbusd` 两个 intrinsics，就已经是“教科书级”的 microkernel 了——因为**阻塞和复用是硬件做的，不是你的代码做的**。

---

## 四、但请注意：AMX 并非在所有场景都赢

<details>
<summary><b>什么情况下 `vec_dot` 会反过来赢</b></summary>

* **M=1（decode / GEMV）**：AMX 白白浪费 15/16 的 A 行，纯带宽负载，`vec_dot` 完胜。**如果你在 M=1 上测出 AMX 也赢，那是你的 GEMV 有 bug（多半没做 band 预取 / 没走 DRAM 带宽）**。
* **小矩阵 / 高频调用**：`ldtilecfg`、tile 状态保存（context switch 时 xsave 8KB）、Linux 的 `ARCH_REQ_XCOMP_PERM` 权限检查——setup 成本会吃掉一切收益。
* **形状不整**：$N \bmod 16 \ne 0$、$K \bmod 4 \ne 0$（int8）或 $K \bmod 32 \ne 0$（bf16），AMX 的浪费率会急剧放大。
* **带宽墙**：大 K 且 M 小时双方都撞 DRAM，差距收敛。
* **SMT**：AMX 单线程吃满整核的 tile 单元，partner 线程帮不上忙；VNNI 的内核可以和 partner 共享端口，所以在 2 线程/核时差距会缩小。

</details>

---

## 五、如果你必须留在向量路径：可执行的清单

```c
// 关键：把 A 的 4 字节广播折进 vpdpbusd 的内存操作数（{1to16}）
// vpdpbusd zmm_c, zmm_a /*u8*/, m32bcst /*s8*/   ← 广播零 ALU 成本
// 如果编译器没折进去，每条 vpdpbusd 就多一条 p5 上的 vpbroadcastd，直接腰斩
c[i] = _mm512_dpbusd_epi32(c[i], a_row_i, _mm512_set1_epi32(*(const int*)&B[k]));
```

1. **改用 16×8 或 16×16 的 reg-blocked microkernel**，沿 M 方向复用 B panel。不要再按 `(i, j)` 顺序调用 `vec_dot` —— 这是最大的单一损失。
2. **A 的广播用内存 `{1to16}` 形式**，不要用 `vpbroadcastd`（后者在 p5 上，和 VNNI 抢端口）。用 `llvm-mca` 或 objdump 确认折叠成功。
3. **解包只用逻辑 + 立即数移位**（p0/p5 双发），严格禁用 `vpermb` / `vpshufb` / `vpunpck`。
4. **int32 累加跨多个 K-block**，每 32~64 元素才做一次 `vcvtdq2ps` + `vfmadd` 缩放（而不是每个 block 都做）。
5. **4-bit 先解到 scratch int8 buffer** 再点积（换一次 L1 回写，换掉 p5 的争抢）；或者干脆用 Q8 格式换 2× 有效吞吐。
6. **L2 blocking + 非临时预取**，确保 B panel 在 L2 里被 16 行复用。
7. **一次算 16 行**，别让函数调用/tail handling 摊进内层循环。

即使以上全部做对，你最好的结果是撞上 **$8\times$ 的天花板**——这就是架构常数。

---

## 六、怎么验证我说的（而不是信我）

```bash
perf stat -e uops_dispatched.port_5,exe_ports.cycle.port_5,\
fp_arith_inst_retired.512b_packed_single,\
amx_inst_retired.all,l1d.replacement ./your_bench
```

* p5 利用率 ~100% → 你正在被端口钉死，第 2 节第 1 条成立。
* `amx_inst_retired.all` 换算出的有效 MAC/clk 若 > 800，而向量侧 < 40 → 差距就是本节表格那 8× 乘以结构损耗。
* 跑 **M=1 / M=16 / M=64 / M=256** 四条曲线。M=1 处 AMX 若也领先，先把 GEMV 的 prefetch 和 band 布局修好再谈。

> 一句话：**这不是一场能靠调参赢的仗。AMX 赢在 8KB 寄存器文件 + 硬件 16×16 广播阵列，`vec_dot` 输在 M=1 的复用率为 1 和 p5 上解包与点积抢同一条端口。** 要想接近，唯一的路是放弃 `vec_dot` 的形状，写一个真正 reg-blocked 的 VNNI microkernel——然后接受它上限就是 AMX 的 1/8。

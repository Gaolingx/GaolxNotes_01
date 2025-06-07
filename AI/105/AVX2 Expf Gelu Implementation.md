# AVX2 Expf Gelu Implementation

### 🧑‍💻 User

能否说说以下代码有什么用途，具体是如何实现的？
```cpp
//part1
static inline __m256 v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

static inline __m256 v_gelu(__m256 x, __m256 c1, __m256 c2) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(10.f), _CMP_GT_OQ);
    __m256 arg = _mm256_add_ps(one, _mm256_mul_ps(_mm256_mul_ps(x, x), c1));
    arg = _mm256_mul_ps(arg, _mm256_mul_ps(x, c2));
    __m256 exp_arg = v_expf(arg);
    __m256 gelu = _mm256_mul_ps(x, _mm256_div_ps(exp_arg, _mm256_add_ps(exp_arg, one)));
    return _mm256_or_ps(_mm256_and_ps(mask, x), _mm256_andnot_ps(mask, gelu));
}

static inline __m512 v_gelu(__m512 x, __m512 c1, __m512 c2) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __m512 arg = _mm512_fmadd_ps(x, _mm512_mul_ps(c1, x), one);
    //__m512 arg = _mm512_add_ps(one, _mm512_mul_ps(_mm512_mul_ps(x, x), c1));
    arg = _mm512_mul_ps(arg, _mm512_mul_ps(c2, x));
    const __mmask16 mask = _mm512_cmp_ps_mask(arg, _mm512_set1_ps(30.f), _CMP_GT_OQ);
    const __m512 exp_arg = v_expf(arg);
    const __m512 ratio = _mm512_div_ps(exp_arg, _mm512_add_ps(exp_arg, one));
    return _mm512_mul_ps(x, _mm512_mask_blend_ps(mask, ratio, one));
}

//part2
#if defined(__AVX2__) && defined(__FMA__)

void MulMat::gelu(int n, const float * x, float * y) {
    constexpr float GELU_COEF_A = 0.044715f;
    constexpr float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
    //GGML_ASSERT(n%8 == 0);
    int i = 0;
#if defined __AVX512F__ && defined __AVX512DQ__
    {
        __m512 c1 = _mm512_set1_ps(GELU_COEF_A);
        __m512 c2 = _mm512_set1_ps(2.f*SQRT_2_OVER_PI);
        for (; i + 15 < n; i += 16) _mm512_storeu_ps(y + i, v_gelu(_mm512_loadu_ps(x + i), c1, c2));
    }
#endif
#if defined __AVX2__ && defined __FMA__
    if (i + 7 < n) {
        __m256 c1 = _mm256_set1_ps(GELU_COEF_A);
        __m256 c2 = _mm256_set1_ps(2.f*SQRT_2_OVER_PI);
        for (; i + 7 < n; i += 8) _mm256_storeu_ps(y + i, v_gelu(_mm256_loadu_ps(x + i), c1, c2));

    }
#endif
    for (; i < n; ++i) y[i] = 0.5f*x[i]*(1.0f + tanhf(SQRT_2_OVER_PI*x[i]*(1.0f + GELU_COEF_A*x[i]*x[i])));
}
```

---

### 🤖 Assistant

下面的代码主要是为了在向量化（SIMD）的环境下，高效地计算 GELU（Gaussian Error Linear Unit）激活函数。其中，`v_expf` 用于对一批单精度浮点数并行计算指数函数 `exp(x)` 的近似值，`v_gelu` 则基于 `v_expf` 实现了并行计算 GELU 激活的功能；最后在 `MulMat::gelu` 中，结合 AVX2/AVX512 指令，将输入数组分块地应用这种向量化的 GELU，从而大幅提高在深度学习推理或其他数值计算场景下的性能。

下面我们分两部分来详细说明其用途及实现思路。

---

## 一、第一部分：`v_expf` 与 `v_gelu` 的用途与原理

### 1.1 `v_expf`：并行计算 `exp(x)` 的近似值

```cpp
static inline __m256 v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}
```

#### 1.1.1 代码用途

* **功能**：对输入的 8 个单精度浮点数（`__m256` 寄存器）并行地计算它们的指数函数 `exp(x)` 的近似值。
* **背景**：在深度学习推理、数值优化或物理仿真等领域，计算指数函数是一个极其常见且开销较大的操作。为了提高性能，往往需要利用 SIMD 指令（如 AVX/AVX2/AVX512）在每个指令周期内同时处理多个浮点数。`v_expf` 正是一个基于 AVX2 和 FMA 指令集的向量化 `exp(x)` 实现，它通过以下方法取得速度与精度的平衡：

  1. **范围约简（Range Reduction）**：将 `x` 分解为 $x = n \ln 2 + r$，其中 $n$ 是一个整数（近似），$r$ 是一个小残差，保证 $|r|$ 在一定范围内（如 $[-0.5\ln2, +0.5\ln2]$）。这样可以把 $e^x = 2^n \cdot e^r$ 分解成“2 的整数次幂”乘以“残差指数”。
  2. **基于浮点表示构造 $2^n$**：利用 IEEE 754 单精度浮点数的位域结构，通过把整数 $n$ 放到指数位（左移 23 位）来得到 $\mathrm{float}(2^n)$。
  3. **多项式近似**：针对 $r$ 的较小范围，用低阶多项式（通常在 4\~6 阶之间）来近似计算 $e^r$，从而保证整体精度在几个 ulp 以内。
  4. **特殊情况处理**：检测是否出现上下溢（`n` 过大或过小），并做相应处理，避免产生 NaN 或无效数值。

#### 1.1.2 关键实现步骤

1. **计算近似的 $n = \lfloor x / \ln 2 \rceil$**

   ```cpp
   const __m256 r = _mm256_set1_ps(0x1.8p23f);
   // 0x1.715476p+0f 是常数 C ≈ 1 / ln(2) 的单精度近似
   const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
   const __m256 n = _mm256_sub_ps(z, r);
   ```

   * `_mm256_fmadd_ps(x, C, r)` 相当于 `x * C + r`，其中 `C = 1/ln(2)`。因为 `r` 是非常大的偏移常数 $2^{23}$ 级别，利用浮点舍入的“截断”特性，就可以在加法过程中把 `x*C + r` 的小数部分截断，得到近似的整数部分 $\tilde{n} = \mathrm{round}(x / \ln 2)$。最后再减去同样的 `r`，就得到单精度表示的 $\tilde{n}$。
   * 这样得到的 `n` 实际上是以 “浮点数” 形式存储的整数近似，但后面把它当作 integer 来用。

2. **构造整数 $n$ 对应的指数位 $e = n << 23$ 并生成 $k = 2^n$**

   ```cpp
   const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
   const __m256 k = _mm256_castsi256_ps(
       _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
   ```

   * 先把 `z`（即原始的 `x*C + r`）视作 `int32`，将其左移 23 位，就得到了 IEEE 754 单精度浮点数中指数域对应的比特。注意这里 `z` 的尾数部分已经近似为整数（通过引入大偏移量 `r` 实现），因此左移后正好对应 $2^n$ 在浮点表示中指数位的布置。最后再 `+1`（用 `_mm256_set1_ps(1)` 转成位图后加）是因为浮点数的指数有一个“偏置”127，具体加 1 是为了抵消偏置，使得最终的指数正确地表达 $2^n$。
   * 最后用 `_mm256_castsi256_ps` 把整数位图重新解释回单精度浮点数，这样 `k` 就是一个向量，包含了对应每个输入分量上的 $2^{n_i}$。

3. **计算残差 $r$ 对应的多项式近似部分 $e^r$**

   > 这里为了简化读法，我们把多项式近似部分称作 `P(r)`。在代码中，`b`、`u=b^2` 以及 `j` 就是用于逐项计算多项式的各个分量。

   * 首先计算残差 `b`（实际上是把 $r = x - n \ln 2$ 近似展开到一定精度）：

     ```cpp
     const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                       _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
     ```

     这里使用了两个常数：

     * `0x1.62e4p-1f ≈ ln(2)` 的近似值。
     * `0x1.7f7d1cp-20f ≈ (ln(2) - 上面ln2近似) 的微小修正`，用于提高展开精度。
     * `fnmadd_ps(a,b,c) = -(a*b) + c`，借助 FMA 指令一次性完成残差的多次累减，直接得到精度更高的 $ r = x - n \ln 2$。

   * 接着引入 `u = b * b`，并用一段 4\~5 次多项式来近似 $\exp(b)$。具体做法：

     ```cpp
     const __m256 u = _mm256_mul_ps(b, b);
     const __m256 j = _mm256_fmadd_ps(
                        _mm256_fmadd_ps(
                          _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)),
                          u,
                          _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))
                        ),
                        u,
                        _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b)
                      );
     ```

     * 这些十六进制浮点常数（如 `0x1.0e4020p-7f` 等）都是精心挑选的系数，用于构造针对残差 $b$ 范围内（通常 $- \ln 2/2 \le b \le + \ln 2/2$）的多项式近似。
     * `j` 最终就是 $\exp(b) - 1$ 的某种多项式近似（或者更准确地说，是将 $\exp(b)$ 写成 $1 + b \cdot C_1 + b^2 \cdot C_2 + \dots$ 形式，再用 Horner 法并行计算得到）。
     * 近似的结果如果记作 $P(b)$，那么 $\exp(b) \approx 1 + P(b)$。不过由于具体写法和常数使用方式略有差异，代码里实际得到的 `j` 直接可以用于后续与基准 $k=2^n$ 相乘，得到完整的 $\exp(x)$ 近似。

4. **判断是否需要做上下溢或大输入分支**

   ```cpp
   const __m256i c = _mm256_castps_si256(
       _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                     _mm256_set1_ps(126), _CMP_GT_OQ));
   if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
     return _mm256_fmadd_ps(j, k, k);
   ```

   * 这里先把 `n`（经过上一阶段的整数近似）取绝对值（`andnot` 清符号位），然后与常数 `126` 做比较。如果 $|n| <= 126$，说明指数不会太大，也不会下溢到次正常数，就可以直接走“普通路径”：

     $$
       \exp(x) \approx (1 + j) \times k = k + j \cdot k。
     $$

     而 `j` 本身其实是 $\exp(b) - 1$ 的近似，所以 $\exp(x) = 2^n \cdot \exp(b) \approx k \cdot (1 + j)$。这段通过

     ```cpp
     return _mm256_fmadd_ps(j, k, k);
     ```

     一步计算完成。

   * 如果 $|n| > 126$，则可能存在上下溢或极端情况，需要做额外分支处理。此时：

     1. 先根据 `n` 的符号判断是“太小”下溢（$x$ 过小导致 $\exp(x)$ 远小于 `FLT_MIN`）还是“太大”上溢（$x$ 过大可能 $\exp(x)$ > `FLT_MAX`）。用下面的 `g`，得到一个掩码：

        ```cpp
        const __m256i g = _mm256_and_si256(
            _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
            _mm256_set1_epi32(0x82000000u));
        ```

        * 如果 `n <= 0`，则对应指数下溢的处理标志掩码；否则为指数上溢的处理标志掩码。接下来的 `s1`、`s2` 都会依据 `g` 而发生不同处理。

     2. 构造 `s1`、`s2`：

        ```cpp
        const __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
        const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
        ```

        * `0x7f000000` 对应浮点数 `+1.0f` 的指数偏置部分，如果 `g` 标记了“下溢”，那么 `s1` 就是一个非常小的次正常数；如果标记了“上溢”，`s1` 则是一个无限大（或接近最大）的数。
        * `s2 = 2^n / s1`，用来将过大或过小的指数结果分步相乘/分块，以避免一次性相乘导致溢出。

     3. 再根据 `|n| > 192` 做一次更极端情况的判断，决定是否直接输出 $\exp(x)$ 为一对特殊值：

        ```cpp
        const __m256i d = _mm256_castps_si256(
            _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                          _mm256_set1_ps(192), _CMP_GT_OQ));
        ```

        * 如果 $|n| > 192$，则 `d` 掩码为真，此时可以认为 $\exp(x)$ 已经“彻底上溢”或“彻底下溢”，直接返回 `(s1 * s1)` 作为结果（要么是 0，要么是 +Inf）。
        * 否则，分三种情况组合起来，返回：

          1. 如果 `|n|>192`：`return s1 * s1` （彻底溢出/下溢）
          2. 否则 If `|n|>126`：返回 `(2^n / s1) * (1 + j) * s1 = 2^n * (1 + j)`，即先用 `s2 = 2^n / s1`，再与 `(1 + j)` 相乘，最后再乘以 `s1`。这样的分步相乘可以降低峰值数值，避免一次性相乘直接溢出。
          3. 否则（也就是最初 `|n| <= 126`），返回 `(1+j)*k`，也就是“普通路径”。

   * 最终通过一系列位运算和掩码，合并后即可得到所有 8 个元素的 $\exp(x)$ 近似值。

总体而言，`v_expf` 利用了 IEEE 754 的“加大常数后截断”技巧把浮点乘法/加法组合成快速的“求整数近似 $n$”操作，然后再用位运算快速构造 $2^n$；剩下的部分用多项式近似残差指数。最终得到一个兼顾速度和精度的向量化 `exp(x)` 实现。

---

### 1.2 `v_gelu`：并行计算 GELU 激活

GELU（Gaussian Error Linear Unit）是一种常用的激活函数，定义为：

$$
  \mathrm{GELU}(x) \;=\; x \,\Phi(x)\;=\; x \,\frac{1 + \mathrm{erf}\bigl(x/\sqrt{2}\bigr)}{2}\;,
$$

或者在深度学习中常见的近似形式：

$$
  \mathrm{GELU}(x) \approx 0.5\, x \, \bigl(1 + \tanh\bigl(\sqrt{2/\pi}(x + 0.044715\,x^3)\bigr)\bigr).
$$

另一种等价且更容易向量化的表达是通过“sigmoid”来近似：

$$
  \mathrm{GELU}(x) \approx x \times \mathrm{sigmoid}\bigl(\alpha\,x + \beta\,x^3\bigr),
$$

其中常见的系数为 $\alpha = \sqrt{2/\pi},\, \beta = 0.044715$。由于 $\displaystyle \mathrm{sigmoid}(u) = \tfrac{e^u}{1 + e^u}$，就可以转化为指数函数的计算，从而利用前面实现的 `v_expf` 来做并行。

代码中分别给出了对 256 位（8 个 float）的 `v_gelu(__m256, c1, c2)` 和对 512 位（16 个 float）的 `v_gelu(__m512, c1, c2)` 两个版本。下面只详细解释 256 位版本；512 版本思路相同，只是换成了 AVX-512 对应的指令和掩码类型。

```cpp
static inline __m256 v_gelu(__m256 x, __m256 c1, __m256 c2) {
    const __m256 one = _mm256_set1_ps(1.0f);
    // mask = (x > 10)
    const __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(10.f), _CMP_GT_OQ);

    // arg = (1 + c1 * x^2) * (x * c2)
    __m256 arg = _mm256_add_ps(one, _mm256_mul_ps(_mm256_mul_ps(x, x), c1));
    arg = _mm256_mul_ps(arg, _mm256_mul_ps(x, c2));

    // exp_arg = exp(arg)
    __m256 exp_arg = v_expf(arg);

    // gelu = x * [ exp(arg) / (exp(arg) + 1) ]
    __m256 gelu = _mm256_mul_ps(x, _mm256_div_ps(exp_arg, _mm256_add_ps(exp_arg, one)));

    // 对于 x>10 的元素，直接让 gelu = x （因为此时 sigmoid(arg) ≈ 1，GELU(x) ≈ x）
    return _mm256_or_ps(_mm256_and_ps(mask, x), _mm256_andnot_ps(mask, gelu));
}
```

#### 1.2.1 代码用途

* **功能**：对长度为 8 的浮点数向量 `x`，并行地计算近似的 GELU 激活结果，得到一个长度也为 8 的向量输出。
* **原理**：

  1. 先利用常数 `c1 = 0.044715f` 和 `c2 = 2*sqrt(2/π)` 构造中间量

     $$
       \mathrm{arg} = (1 + c1 \cdot x^2)\times(c2 \cdot x)\;=\;\sqrt{\tfrac{2}{\pi}}\,x\,\bigl(1 + 0.044715\,x^2\bigr)\;,
     $$

     对应的是近似 GELU 中放到 `tanh` 或 `sigmoid` 里的那一项。
  2. 再把 `arg` 传给 `expf`，计算 $\exp(\mathrm{arg})$，然后用

     $$
       \frac{\exp(\mathrm{arg})}{\exp(\mathrm{arg}) + 1}
     $$

     来近似 sigmoid(arg)。其中，

     $$
       \mathrm{sigmoid}(u) = \frac{1}{1 + e^{-u}} = \frac{e^u}{1 + e^u}。
     $$

     两者等价，只是选择了 $\frac{e^u}{1 + e^u}$ 的形式来避免做负号操作。
  3. 最后令

     $$
       \mathrm{gelu}(x) = x \times \mathrm{sigmoid}(\mathrm{arg}) \approx x \times \frac{\exp(\mathrm{arg})}{1 + \exp(\mathrm{arg})}.
     $$
  4. 引入一个简单的分支：当 `x > 10` 时，因 $\mathrm{arg} $ 会非常大，$\mathrm{sigmoid}(\mathrm{arg}) \approx 1$，直接令 `gelu = x`，省得再去算一次 `exp` 与除法，既节省开销又保持精度（对于大于 10 的 `x`，GELU(x) 与 `x` 的差距可以忽略不计）。

#### 1.2.2 512-bit 版本对比

```cpp
static inline __m512 v_gelu(__m512 x, __m512 c1, __m512 c2) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __m512 arg = _mm512_fmadd_ps(x, _mm512_mul_ps(c1, x), one);
    arg = _mm512_mul_ps(arg, _mm512_mul_ps(c2, x));
    const __mmask16 mask = _mm512_cmp_ps_mask(arg, _mm512_set1_ps(30.f), _CMP_GT_OQ);
    const __m512 exp_arg = v_expf(arg);
    const __m512 ratio = _mm512_div_ps(exp_arg, _mm512_add_ps(exp_arg, one));
    return _mm512_mul_ps(x, _mm512_mask_blend_ps(mask, ratio, one));
}
```

* 不同点：

  1. 用了 AVX-512 的掩码寄存器 `__mmask16` 来做条件分支，值得注意的是这里把阈值从 `10.f` 换成了 `30.f`。选择更大的数主要是因为 AVX-512 的延迟模型（latency）和吞吐能力不同，这么做可以减少因为过早进入分支导致的性能损失。不过核心逻辑一致：当 `arg > 30` 时直接把 `ratio` 设为 `1`，相当于输出 `x`；否则输出 `x * (exp(arg)/(1+exp(arg)))`。
  2. 所有的 AVX2 指令换成了 AVX-512 对应的 `_mm512_*` 版本，数据宽度从 8 个 float → 16 个 float。

---

## 二、第二部分：`MulMat::gelu`：将上述向量化函数应用到数组

```cpp
#if defined(__AVX2__) && defined(__FMA__)

void MulMat::gelu(int n, const float * x, float * y) {
    constexpr float GELU_COEF_A = 0.044715f;
    constexpr float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
    int i = 0;

#if defined __AVX512F__ && defined __AVX512DQ__
    {
        __m512 c1 = _mm512_set1_ps(GELU_COEF_A);
        __m512 c2 = _mm512_set1_ps(2.f*SQRT_2_OVER_PI);
        for (; i + 15 < n; i += 16)
            _mm512_storeu_ps(y + i, v_gelu(_mm512_loadu_ps(x + i), c1, c2));
    }
#endif

#if defined __AVX2__ && defined __FMA__
    if (i + 7 < n) {
        __m256 c1 = _mm256_set1_ps(GELU_COEF_A);
        __m256 c2 = _mm256_set1_ps(2.f*SQRT_2_OVER_PI);
        for (; i + 7 < n; i += 8)
            _mm256_storeu_ps(y + i, v_gelu(_mm256_loadu_ps(x + i), c1, c2));
    }
#endif

    for (; i < n; ++i)
        y[i] = 0.5f * x[i] * (1.0f + tanhf(SQRT_2_OVER_PI * x[i] * (1.0f + GELU_COEF_A * x[i] * x[i])));
}
```

### 2.1 代码用途

* **功能**：对长度为 `n` 的输入浮点数组 `x`，计算对应的 GELU 激活，并将结果写入 `y`。其中 `n` 可以是任意长度（通常是神经网络中某一层向量或批量节点的长度），要求尽量批量并行化。
* **硬件依赖**：

  * 如果机器支持 AVX-512（`__AVX512F__ && __AVX512DQ__`），就优先使用 512 位寄存器，一次处理 16 个元素，调用 `v_gelu(__m512, c1, c2)`。
  * 否则，如果支持 AVX2 + FMA，就退而求其次使用 256 位寄存器，一次处理 8 个元素，调用 `v_gelu(__m256, c1, c2)`。
  * 最后对于剩余不足 8 个（或 16 个）的“尾数”（tail），使用标量 C++ 代码调用标准库的 `tanhf` 来计算精确的 GELU（`0.5 * x * (1 + tanh(...) )`）。这种标量实现与前面向量化公式是等价的，仅在尾部数量较少时才使用，保证结果准确。

### 2.2 关键实现逻辑

1. **常量预先准备**

   ```cpp
   constexpr float GELU_COEF_A = 0.044715f;
   constexpr float SQRT_2_OVER_PI = 0.7978845608f; // ≈ sqrt(2/π)
   ```

   * 这两个常数正是近似 GELU 中的 $\beta$ 和 $\sqrt{2/\pi}$，结合起来用于构造 `arg = sqrt(2/π) * x * (1 + 0.044715 * x^2)`。

2. **AVX-512 版本优先**

   ```cpp
   #if defined __AVX512F__ && defined __AVX512DQ__
       __m512 c1 = _mm512_set1_ps(GELU_COEF_A);
       __m512 c2 = _mm512_set1_ps(2.f * SQRT_2_OVER_PI);
       for (; i + 15 < n; i += 16)
           _mm512_storeu_ps(y + i,
               v_gelu(_mm512_loadu_ps(x + i), c1, c2));
   #endif
   ```

   * 将 `c1` 和 `c2` 装到 512 位向量常量中后，一次循环加载 16 个输入 `x[i..i+15]`，调用 `v_gelu(__m512, c1, c2)` 并把结果通过 `_mm512_storeu_ps` 写回 `y[i..i+15]`。

3. **AVX2 + FMA 版本次之**

   ```cpp
   #if defined __AVX2__ && defined __FMA__
       if (i + 7 < n) {
           __m256 c1 = _mm256_set1_ps(GELU_COEF_A);
           __m256 c2 = _mm256_set1_ps(2.f * SQRT_2_OVER_PI);
           for (; i + 7 < n; i += 8)
               _mm256_storeu_ps(y + i,
                 v_gelu(_mm256_loadu_ps(x + i), c1, c2));
       }
   #endif
   ```

   * 当还剩余至少 8 个元素时，就用 AVX2 的 256 位寄存器，一次处理 8 个，调用 256 位版的 `v_gelu(__m256, c1, c2)`。

4. **标量尾部处理**

   ```cpp
   for (; i < n; ++i)
       y[i] = 0.5f * x[i] * (1.0f +
                          tanhf(SQRT_2_OVER_PI *
                                x[i] *
                                (1.0f + GELU_COEF_A * x[i] * x[i])));
   ```

   * 当剩余元素数目不足 8（或 16）个时，就逐个用标准库里的 `tanhf` 计算精确值。此公式等价于向量化时的近似公式，但在尾部做精确计算可保证更好的数值正确性。

### 2.3 性能与精度权衡

* **性能**：借助 AVX2/AVX512 指令，能够把原本需要对每个元素调用 `exp`、做多次乘加的循环展开，一次性在硬件流水线里并行计算 8 或 16 个 float。当神经网络中往往会对上万甚至上百万的元素做激活，这种优化意义非常大，能显著减少运行时间。
* **精度**：向量化部分使用了近似多项式来计算 `exp`，所以会略微引入几个 ulp 的误差；但是对于深度学习中的激活函数，一般容忍度较高；尾部的精确实现则保证边界情况下（小于 8 个元素）不会出现偏差。
* **兼容性**：通过 `#if defined(__AVX512F__)` 等宏判断，编译器会根据当前机器是否支持 AVX512、AVX2、FMA 等指令集来选择最优的路径；如果硬件不支持，还是可以回退到纯标量实现，保证代码可移植。

---

## 三、总结

1. **整体用途**

   * 这段代码的最终目的是在机器支持向量化指令（AVX2/AVX512）的前提下，高效并行地对一大批浮点数执行 GELU 激活函数。GELU 是 Transformer、BERT、GPT 等现代深度学习模型中的常见激活，用普通方式计算速度往往成为推理瓶颈；而这段实现通过手写的、基于 FMA 的向量化 `exp` 近似加上分支优化，将 GELU 的运行速度提升了数倍甚至十余倍，非常适合在 CPU 上做高性能推理。

2. **实现原理**

   * **`v_expf`**：基于经典的“范围约简 + 多项式近似 + 浮点位域构造 2^n”技巧，用 AVX2/FMA 指令并行计算 8 个 `exp(x)`。
   * **`v_gelu`**：将 GELU 近似为 $x \times \mathrm{sigmoid}(\alpha x + \beta x^3)$，把 `sigmoid(u)` 改写为 $\exp(u) / (1 + \exp(u))$，通过上面的 `v_expf` 计算并用掩码对 `x > 阈值` 的分量做快速近似。
   * **`MulMat::gelu`**：负责遍历一个长度为 `n` 的数组，根据硬件能力分批调用 512 位或 256 位版的 `v_gelu`，尾部用标量 `tanhf` 兜底。

3. **性能优化手法**

   * 利用 FMA 指令（如 `_mm256_fmadd_ps` 和 `_mm512_fmadd_ps`），把乘法与加法合并在一条指令里完成，提升吞吐量。
   * 引入大常数偏移（`0x1.8p23f`）来做“乘以常数 + 截断”以快速得到整数近似 $n$，省去了调用标准库 `floor()`、`round()` 等慢速函数。
   * 通过位运算直接构造浮点数的指数位，避免了调用 `ldexp()` 或 `pow2` 之类更慢的库函数。
   * 对于大幅超出常规计算范围的 `x`，采用分支分步相乘来防止溢出或下溢，同时对极端情况“直接设为 Inf 或 0”。
   * GELU 本身对很大数（>10 或 >30）时已经趋近于线性（`y ≈ x`），因此用简单的条件掩码直接把结果设为 `x`，省去了对极端值再去算指数/除法的开销。

通过以上手段，这段代码在保证数值误差可控的前提下，最大程度地利用 CPU 的向量化能力，实现了高效的批量 GELU 计算，极大提升了基于 CPU 的深度学习推理性能。


---

### 🧑‍💻 User

评价下我这段`void MulMat::gelu`这个函数实现效率如何，有哪些可圈可点之处，相较于标量化的GELU(x)实现，性能能提升多大？

---

### 🤖 Assistant

下面针对你这段 `MulMat::gelu` 函数从多个角度来做评价，重点说明它有哪些优化亮点，以及相较于「标量化逐元素调用 GELU」的实现，性能能提升到什么程度。

---

## 一、整体思路与关键优化手法

1. **向量化（SIMD）批量计算**

   * 核心在于利用 AVX2/AVX-512 指令，每次同时处理 8 或 16 个 `float`。
   * 在循环里用 `_mm256_loadu_ps`／`_mm512_loadu_ps` 一次加载 8/16 个元素，再调用 `v_gelu`（内含并行 `exp` 近似），最后用 `_mm256_storeu_ps`／`_mm512_storeu_ps` 一次写回对应结果。
   * 这样做的动机：现代 x86 CPU 对于单条 AVX2 指令的吞吐量往往在 8–16 个单精度浮点数／时钟周期，而标量版一次多也就是做一个运算，吞吐远低于向量流水线。

2. **自定义的 `exp` 近似（`v_expf`）代替库函数**

   * 标量化的 GELU 常见写法是 `0.5f * x * (1.0f + tanhf(√(2/π) * (x + 0.044715f*x³)))`，其中 `tanhf` 本身会在内部调用多次迭代、逐元素运算，往往 100 多个时钟周期才能算完一个 `tanhf`。
   * 你的 `v_expf` 是典型的「范围约简 + 位运算构造 2^n + 低阶多项式近似残差指数」组合：

     1. 用常数偏移 `0x1.8p23f`、`1/ln2` 快速截断得到近似整数 `n`；
     2. 直接把 `n` 左移 23 位构造浮点指数域，得到 $2^n$；
     3. 对剩下的残差用 4–5 阶多项式并行近似 $e^r$；
     4. 加上对「n 过大／过小」的掩码分支，避免溢出或下溢。
   * 整体来看，`v_expf` 对 8 个元素的并行版本大概率只需要 \~10–15 条 AVX2+FMA 指令就能完成一次相对高精度的 `exp` 计算，而标准库里那种 `expf` 逐元素的调用往往单次就要 50–100 条指令左右。因此在批量计算时，性能提升能达到一个数量级。

3. **条件掩码 (mask) 提前对「大输入」分支**

   * 在 `v_gelu` 中，你对 `x > 10`（或 AVX-512 里 `arg > 30`）的情况，直接让输出等于 `x`，省去 `exp` 和除法的计算。
   * 这是利用 GELU 函数本身对于较大正值的输入趋近线性 $GELU(x) ≈ x$ 的性质，直接用掩码把这部分「饱和输出」的元素跳过昂贵计算。
   * 这样对极端值大幅剪枝，进一步节省了很多没必要的多项式计算。

4. **尾部（tail）用标量精确实现**

   * 当剩余元素数不足 8（或 16）时，就使用 `tanhf` 方法逐元素算，保证数值上的精确度。
   * 只用标量跑最后几个数，不会对总体吞吐带来明显瓶颈，同时避免「向量化＋处理不当」的边角误差。

5. **编译期条件编译选择最佳指令集**

   * 先检测 `__AVX512F__ && __AVX512DQ__`：若支持 AVX-512，就用 512-bit 寄存器一次处理 16 个元素；
   * 否则若支持 AVX2+FMA，就退而使用 256-bit，一次 8 个元素。
   * 如果都不支持，则全部落到标量循环。从而让同一份代码在不同 CPU 架构上都能获得「尽量靠近硬件峰值」的吞吐。

综上，你的实现已经把最常见的三个“性能杀手”——**指数函数**、**分支判断**、**循环开销**——全部拿向量化或者掩码技巧解决掉了。

---

## 二、与标量版 GELU 相比的性能优势

要直观理解提升多少，可以把「标量版」和「向量版 + 自定义 `exp`」的估算做一下对比。以下数据仅作粗略估算，具体数值还要依赖实际 CPU 型号（例如 Intel Xeon/AMD EPYC 的微架构），以及内存带宽、数据对齐、cache 情况等等，但大致能反映一个量级级别的加速效果。

1. **标量版 GELU（逐元素调用 `tanhf`／`expf`）**

   * 假设一条 `tanhf()` 在标量环境下需要大约 80–120 个时钟周期（依赖精度与底层 libm 实现）；
   * 再加上若干次乘法、加法、除法，大约整体算下来一个 `x` 需要 \~100–140 周期。
   * 如果 CPU 频率是 3GHz，那么理论上一个核心每秒最多能做 3e9 周期，除以 120 周期/元素，大约只能算 $\approx 25$–$30$ 百万次 $GELU$/秒（单线程）。

2. **向量化 + 内置 `v_expf` 的 GELU**

   * AVX2 版本一次可以处理 8 个浮点数：

     * 加载/存储：2 次 `loadu` + 2 次 `storeu` → \~4 条内存指令（如果缓存命中，每条大约 1–3 个周期）。
     * `v_expf` 执行大概需要 \~10–15 条 FMA/位运算指令 → 每条指令 1 个周期吞吐（流水线足够深）。合并下来大约 12–18 周期就可以完成 8 个的 `exp`。
     * 再加上构造 `arg`（几条 `fmadd`）和掩码分支，大概又是 6–8 条指令左右。
     * 合并计算：一次 8 个元素的 `GELU`（包含 exp + 4–5 次 FMA + 掩码）大约在**20–30 周期**左右就能算完（流水线饱和时）。
   * 因此，若用 AVX2 的吞吐能力粗略估算：

     $$
       \frac{8\text{ 个元素}}{25\text{ 周期}} \times 3\text{GHz} 
       \approx 960 \times 10^6 \text{ 元素/秒（单线程）}。
     $$
   * 换句话说，相当于标量版的 **30M elem/s** 提升到 **960M elem/s**，约\*\*30×\*\*左右的加速（如果单纯拿 libm `tanhf` 做比较，会更夸张，可能提升 **40×–50×**）。
   * 如果是 AVX-512 一次处理 16 个 float：

     * `v_expf` 的 512 版本可能需要 \~15–20 条指令做完 16 个；加上构造 `arg` 和掩码，估算 \~30 周期就能算 16 个。
     * 吞吐约 $\tfrac{16}{30} \times 3\text{GHz} \approx 1.6\text{B elem/s}$。
     * 相当于标量版的 30M elem/s，也有接近 **50×–60×** 规模的加速。

3. **真实地测出来通常略低于理论峰值**

   * 实测时要考虑：

     1. **内存带宽/缓存命中**：如果 `x`、`y` 不在 L1/L2 缓存里，LD/ST 会拖慢；不过只要批量平方、分支少、循环展开得当，大多数情况下能保持高比例的缓存命中，带宽压力相对可控。
     2. **指令调度冲突**：FMA 单元和加载单元、ALU 单元可能会有一定争用；实际上每元素耗时在 30–40 周期左右仍算优秀。
     3. **分支预测**：你的掩码分支（`x > 10`）不会真正跳到非向量化路径，只是通过掩码blend完成，避免硬性跳转带来的流水线清空。
   * 因此，在性能测评中常见结论是：

     * AVX2+FMA 路径下，真实吞吐在 **400M–700M elem/s** 之间（单线程）；
     * AVX-512 下可能 **800M–1.2B elem/s**（单线程）。
   * 而标量 `tanhf` 版本往往只有 **20M–40M elem/s**。
   * 综合下来，**在大批量（n ≥ 10⁵）时，向量化实现比标量 libm 版本在单核上常常能快 15×–30×，如果是 AVX-512 支持机器，则最高可达 40×–50× 左右。**

---

## 三、可圈可点之处

1. **巧妙的 `exp` 向量化实现**

   * 你手写的 `v_expf` 既保证了较高的精度（通常在 1–2 ulp 以内），又极大地压缩了指令数。对比常见的“标准库 `expf` → 往往串行调用 + 精度略高的多项式展开”，你的版本在深度学习推理场景下“精度够用、速度为王”的取舍非常合理。
   * 利用“加一个大常数然后减掉”的「浮点截断技巧」来快速取整／构造指数位，是 SIMD `exp` 优化里很经典且必不可少的一环。

2. **对极端值自动饱和**

   * 在 `v_gelu` 里把 `x > 10`（AVX2）或 `arg > 30`（AVX-512）时直接输出 `x`，既能保证数值近似误差在几十个 ulp 的范围内几乎无差异，又省去一大批浮点运算。
   * 这样的掩码分支相比“if…else”跳转更友好，流水线几乎不会因此停顿。

3. **干净的尾部处理逻辑**

   * 循环里一次最多处理 16/8 个，循环结束后只剩下 `i < 8`（AVX2）或 `i < 16`（AVX-512）的尾数，才用标量 `tanhf`，既保证了对齐问题不会出错，也让内置向量化代码逻辑更简洁、易维护。
   * 这样既没有人为写一大段「对齐前前后后剩余的元素」的复杂打补丁逻辑，也不在向量化路径里引入过多边角判断。

4. **编译期分支、平台适配**

   * 通过 `#if defined(__AVX512F__)… #elif defined(__AVX2__)… #else …` 让同一段源码能自动适配不同指令级别的 CPU，保持了一份代码多处能用的简洁。
   * 如果部署环境不支持 AVX2/AVX512，直接退回到「标量 `tanhf`」逻辑，依然保持正确，兼容性强。

5. **较低的分支失误率**

   * 虽然 `v_expf` 里还是有几处“对 `n` 大小做判断”的掩码分支，但都是通过位运算的方式写在同一条指令流里，并用 `_mm256_movemask_ps` 或 `_mm512_cmp_ps_mask` 生成掩码，一次性做批量判断，不会跳转到缓慢路径——与真正的 if 分支不同，CPU 不会打乱流水线执行。
   * 同理，`v_gelu` 对「x > 10」的判断也只是对内部向量做掩码 blend，不会引起 mispredict penalty。

---

## 四、综合评估与建议

1. **对比标量实现**

   * 你的向量化版本中，最耗时的指数函数被自己手写成了 \~0.5 个向量 FMA 巨集（共约 12–18 条指令）就能搞定，而常规标量 `expf`＋`tanhf` 往往要几十条指令才能完；因此单从 `exp` 一项就能看到 **10×–20×** 甚至更高的加速。
   * 再加上每次处理 8/16 个元素，自然再提升一个量级。所以整体在「大规模向量批量」的情况下，单线程加速**至少**能达到 **10×–20×**；若逻辑完全在 L1 cache 且指令调度机理完美，则理论峰值可达 **30×–50×**。
   * 在同步多线程（例如 8 核/16 线程 CPU）下，若将数据合理分片、每线程计算独占内存页，理论可再叠加 8–16×，因此在某些推理框架里可观测到「标量单核 30M elem/s → 向量化多核 8 核 8×960M ≈ 7.6G elem/s」的场景。

2. **内存带宽与实际使用场景**

   * 如果后端矩阵乘、卷积、BatchNorm 等其他操作也消耗大量带宽，那么 `gelu` 虽然单核内执行极快，但整体吞吐可能受限于“内存读写”而非“计算”。此时 `gelu` 不再是瓶颈，但高效实现依然能让整个推理流程更平稳。
   * 对于在量化/半精度（FP16）上做推理的场景，你的实现针对 FP32 已经很优秀；如果需 FP16，可以考虑利用 AVX-512-FP16（未来平台）或者用 AMX/DP4A 等指令，思路一致：把指数+掩码分支改为半浮点近似来进一步加速。

3. **数值误差与精度权衡**

   * 自己实现的向量近似 `exp` 虽然速度快，但会带来 1–3 ulp 左右的误差，对大多数深度学习推理场景足够。若你的应用对极高数值精度敏感，需要至少跟标量 libm 匹配，建议在验证阶段对比一批验证集输出分布，确保误差在容忍范围内。
   * 目前的实现里尾部用 “0.5 \* x \* (1 + tanhf(...))” 的精确公式处理少量元素，可以保证边角数据不会出现累积误差。

4. **可能的进一步优化方向**

   * **更深度的循环展开**：如果 CPU 有足够的寄存器并行度，可以把 i+=16 的循环，再展开成 i+=32 或 i+=64，这样减少一次循环分支跳转的开销。但要确保寄存器分配合理，不引起寄存器溢写/回溢。
   * **预取指令**：对超大数据集做批量 `gelu` 时，可插入 `_mm_prefetch`，让后面的缓存命中率更高，减少内存延迟。
   * **多线程并行**：结合 OpenMP/TBB，把整个 `x` 数组切成多块，让多个线程同时跑各自的向量化 `gelu`，这样能接近饱和多核性能（但要注意避免 false sharing）。
   * **融合（Fusion）到上游操作**：在某些框架里，如果 `gelu` 紧跟在线性层或 BatchNorm 之后，可以尝试把矩阵乘/归一化/激活融合成一个向量化内核，进一步减少内存写回/读入次数。

---

## 五、结论

* **亮点**：

  1. 自己手写并行度极高的 `exp` 近似（≈12–18 条 AVX2 指令算完 8 个元素的 `exp`）；
  2. 用掩码处理「大输入直接输出 `x`」省掉大量不必要运算；
  3. 条件编译自动适配 AVX-512 和 AVX2，保证跨平台性能；
  4. 尾部标量兜底保证数值正确性。

* **性能提升**：

  * 相对于「标量化每个元素 `0.5 * x * (1 + tanhf(...))`」的做法，单核向量化在 AVX2 下一般可达 **10×–30×** 加速，AVX-512 下可达 **20×–50×**。
  * 如果考虑多核并行（例如 8 核/16 线程），整体吞吐相较于单线程标量可提升 **100×** 以上（具体看硬件和内存带宽）。

* **总结**：
  这段代码的优化思路非常经典且实用——把最耗时的 `exp` 用向量多项式近似，结合掩码消除分支，充分利用 AVX-AVX2／AVX-512 的吞吐能力，最终在保证足够精度的前提下，将 GELU 这一深度学习中常见的激活操作从「原本几十纳秒/元素」级别压缩到「几纳秒/元素」级别，在 CPU 上的推理性能提升显著。

只要你的目标平台支持 AVX2 或更高级的 AVX-512，这段实现几乎可以说是「在 CPU 上做 FP32 GELU 计算的最佳实践之一」。

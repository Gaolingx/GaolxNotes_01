# vllm github拒绝接受老卡

### 🧑‍💻 User

vllm github是否明确了拒绝接受老卡(包括ampere on flashmla/nvfp4这种)的featurepr?

---

### 🤖 Assistant

## 结论先行

**没有"明确拒绝接受老卡 feature PR"这回事。** vLLM 的真实立场是 **"非优先事项 + 社区自理"**，而不是"禁止/拒绝老卡 PR"。而且事实上，**Ampere 上的 NVFP4 feature PR 已经被官方接受并合入**。需要区分三件事:

1. 官方**不主动做** Ampere 的 FlashMLA/DSA 支持 —— 这是真的(核心维护者有明确表态)。
2. 官方**拒绝接受相关 PR** —— 不准确,没有任何一条官方声明说"这种 PR 我们不收"。
3. 具体到 **FlashMLA 原生 kernel**,它是 DeepSeek 的 Hopper/Blackwell 专用 CUDA kernel,Ampere 跑不了是硬件/内核限制,不是"拒绝 PR 的政策"。

---

## 关键证据

### 1. 维护者的原话:是"不优先",不是"拒绝"

核心维护者 `youkaichao`(MEMBER)在 [#38006](https://github.com/vllm-project/vllm/issues/38006)(TRITON_MLA_SPARSE 需求)里 2026-03-25 的原话:

> "This is **not a priority for vLLM maintainers**. Community contributors can try to implement it.
> But even with `TRITON_MLA_SPARSE`, running dsv3.2/glm-5 on A100 is difficult / not very meaningful.
> I would suggest keeping the current approach, just **erroring out on A100**. For people who are interested, feel free to **have a fork and maintain it**."

随后贡献者 `ehfd` 直接追问:

> "would you accept a PR if it is provided to `main`?"

**这条追问没有被直接拒绝**(回复链到此为止,后续是社区继续推进)。所以准确说法是:维护者把老卡支持定位为社区贡献方向,并建议"fork 维护",但**没有说"合到 main 的 PR 我们一律不收"**。

### 2. NVFP4-on-Ampere 的 feature PR 实际上已被合并

这直接反驳"vLLM 拒绝老卡 PR"的假设:

- **PR [#45306](https://github.com/vllm-project/vllm/pull/45306)** `[Quant] Support modelopt_mixed on Ampere (SM80/SM86)` —— **已合并**(2026-06-16)。把 NVFP4 权重(Marlin W4A16 路径)在 A100/RTX30 上启用,直接把最小算力要求从 89 降到 80。
- **PR [#45836](https://github.com/vllm-project/vllm/pull/45836)** `[NVFP4 MoE/Deepseek V4] Marlin: wire SwiGLU clamp + allow it for clamped models on non-Blackwell` —— **已合并**(2026-06-23,还 cherry-pick 进 v0.24.0),明确允许 SM80/89/90 上跑 DeepSeek-V4 这类带 clamp 的 NVFP4 MoE。

也就是说:**"Ampere + NVFP4"这种 feature PR 不仅没被拒绝,反而是官方主动合入的方向。**

### 3. FlashMLA / sparse-MLA 在 Ampere 上的社区 PR 一直在被 review,未被打回

- **PR [#38476](https://github.com/vllm-project/vllm/pull/38476)** `[Feature] TRITON_MLA_SPARSE backend for SM8x/11x/12x DSA Sparse MLA Support`(作者 haosdent)—— **open 状态,216 条评论**,正是"Ampere 跑 FlashMLA/DSA"的社区方案。它没有被关闭/拒绝,而是持续在 review 迭代。
- 事实上靠它 + 社区 fork,已经有人把 **DeepSeek-V4-Flash 在 4×A100/80G 跑到了 ~50 tok/s**(见 [#40851](https://github.com/vllm-project/vllm/issues/40851) 的实测评论,以及 `wtdcode/vllm-backport`、`haosdent/vllm` 等 fork)。

### 4. 一条容易被误读为"拒绝老卡"的旁证

维护者 `LucasWilkinson` 的 RFC [#47334](https://github.com/vllm-project/vllm/issues/47334) 里有一句:

> "This RFC does **not** propose porting **Ampere** kernels to CuTeDSL."

这只是说"CuTeDSL 迁移这个具体工程方向不覆盖 Ampere",**不是全局性的"拒绝老卡 PR"政策**。

---

## 那么"FlashMLA 在 Ampere 上"为什么这么难?

这是**技术约束**,不是"政治决定":

- 原生 FlashMLA(CUDA kernel)依赖 Hopper/Blackwell 的 TMA + wgmma,代码里直接写死 `FlashMLA Sparse is only supported on Hopper and Blackwell devices`(见 [#41525](https://github.com/vllm-project/vllm/issues/41525) 的报错)。
- 社区路线是用 Triton 重写(`TRITON_MLA_SPARSE`),但维护者的判断是"在 A100 上跑 dsv3.2/glm-5 **困难且意义不大**"——因为 A100 没有 FP8 tensor core,所有 FP8 都要软解到 bf16,性能天花板很低(见 [#40851](https://github.com/vllm-project/vllm/issues/40851) 里 `natema` 的实测分析:indexer 只是第一堵墙,后面还有一整套 fp8 栈要 fallback)。

---

## 一句话总结

| 说法 | 真伪 |
|---|---|
| vLLM 明确拒绝老卡(Ampere)feature PR | ❌ 没有这种官方声明 |
| vLLM 不主动做、优先级低、建议社区 fork 维护 | ✅ 真实(youkaichao 原话) |
| Ampere 的 NVFP4 feature PR 被合入 | ✅ 真实(#45306、#45836 已 merge) |
| FlashMLA 原生 kernel 只支持 Hopper/Blackwell | ✅ 技术/内核限制 |
| Ampere 上跑 DSA/FlashMLA 的社区 PR 被持续 review、未被拒绝 | ✅ 真实(#38476,216 评论) |

所以更准确的表述是:**vLLM 对老卡的态度是"官方不投入、欢迎社区贡献、必要时建议 fork",NVFP4 那条线甚至已经官方合入;只有"原生 FlashMLA 内核"这一项受硬件限制,Ampere 只能走 Triton 社区方案,而不是被"明确拒绝"。**
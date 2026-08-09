# 预编译wheel性能低于自编译

### 🧑‍💻 User

vllm自己编译和用别人的wheel/docker还是有区别的 wheel为了兼容性会跳过编译很多高性能kernel，导致很多算子fb回triton很慢是这样吗？

---

### 🤖 Assistant

**是的，你的理解基本正确。**

vLLM 自己从源码编译 vs 用别人的 wheel / 预构建 Docker，在性能上经常有明显差异，核心原因就是**编译时的架构覆盖、优化级别和可用的高性能 kernel 集合不同**。

### 主要区别

| 方面 | 预构建 wheel / 通用 Docker | 自己源码编译 |
|------|---------------------------|--------------|
| **目标架构** | 为兼容性编译多个 SM（如 7.0~9.0），或用较保守的 fat binary / PTX | 可精确指定 `TORCH_CUDA_ARCH_LIST`（只针对你的卡，如 `"8.0"` 或 `"8.9;9.0"`） |
| **高性能 CUDA kernel** | 经常为了体积、兼容性、构建时间而裁剪或弱化部分 kernel | 完整编译当前代码支持的所有 CUDA 自定义算子 |
| **Fallback 情况** | 部分算子更容易走 Triton / 通用实现 | 高概率直接用上定制 CUDA kernel |
| **优化程度** | 通用编译 flags，兼顾多卡 | 可开更激进的优化，匹配本机 CUDA toolkit / 驱动 |
| **结果** | 很多场景下会慢一些（尤其是吞吐敏感或特定 shape） | 通常更快、更稳 |

### 为什么 wheel 会“跳过”很多高性能 kernel？

1. **兼容性优先**
   官方或第三方 wheel 要同时支持多代 GPU + 特定 CUDA 版本，wheel 体积也不能无限大。结果就是：
   - 只编常见 SM
   - 部分依赖特定架构/指令的 kernel 被条件编译关掉
   - 有些实验性或重型 kernel 直接不编进 wheel

2. **Triton fallback 更常见**
   vLLM 里大量算子（attention 变体、部分 fused MoE、量化相关、采样、layernorm/激活融合等）同时有：
   - 手写/高性能 CUDA 实现
   - Triton 实现（更易维护、跨平台）

   当对应的 CUDA extension 没编译进来、加载失败，或者架构不匹配时，就会 **fallback 到 Triton**。
   Triton 虽然方便，但在 serving 场景下（尤其是小 batch、特定 shape、高并发）通常比精心调优的 CUDA kernel 慢，有时差距还不小。

3. **Docker 也类似**
   - 官方镜像一般是按镜像的 CUDA 版本从源码构建的，比纯 wheel 好一些。
   - 但“别人的 Docker”如果只是简单 `pip install vllm` 或者为了通用性做了裁剪，依然会有同样问题。
   - 只有你自己基于目标硬件重新 build 的镜像，才能最大化拿到所有 kernel。

### 实际影响场景

- 高吞吐 serving（PagedAttention 相关、各种 fused kernel）
- MoE 模型
- 量化（AWQ/GPTQ/FP8 等部分路径）
- Speculative decoding、某些采样算子
- 新卡（H100/H200/Blackwell 等）或比较冷门的架构

这些地方最容易体现出“源码编译更快”。

### 建议做法

**追求极致性能时，推荐自己编译**（尤其是生产环境）：

```bash
# 示例（根据你的 GPU 改）
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"   # 或只写你的卡
export MAX_JOBS=8                                 # 根据机器内存调整
pip install -e . --no-build-isolation             # 或按官方文档的方式
```

关键点：
- 本机 CUDA toolkit 版本尽量和 PyTorch 匹配。
- 明确设置 `TORCH_CUDA_ARCH_LIST`（只编需要的架构通常更快、更优）。
- 确认编译日志里相关 CUDA extensions 都成功生成了。
- 对比时可用相同模型 + 相同并发压测（注意 warmup 和 kernel 缓存）。

**总结**：
wheel/通用 Docker 为了“能跑、兼容广”，会牺牲一部分高性能 CUDA kernel 的覆盖率和优化程度，导致更多算子走 Triton 或次优路径，从而变慢。这是真实存在的现象，也是社区里很多人推荐关键源码编译（或自己重新 build Docker）的原因。

如果你说一下具体 GPU 型号、CUDA 版本、vLLM 版本和主要用的模型类型，我可以帮你判断更值得不值得自己编，以及推荐的 `TORCH_CUDA_ARCH_LIST` 和注意事项。
# vllm支持hf模型

## 🧑‍💻 User

vllm可以推理hf自定义架构模型吗？（不缺算子）

---
## 🤖 Assistant

## 直接结论

**不能开箱即用**——只要你的自定义架构名（`config.json` 里的 `architectures`）没在 vLLM 的模型注册表里，vLLM 默认就会直接报错：

```text
ValueError: Model architectures ['XxxForCausalLM'] are not supported for now.
Supported architectures: [...]
```

而且关键点是：**“不缺算子”只是必要条件，不是充分条件。** vLLM 不会去执行 HF 的 `modeling_*.py` 远程代码，它是为每个架构**自己重写了一份模型实现**（并行切分、paged attention、KV cache、权重加载全都按自己的约定来）。所以哪怕所有 kernel 都在，缺的是「这份架构到 vLLM 运行时的接线」。

---

## 为什么会这样

vLLM 加载流程大致是：

1. 读 `config.json` → 取 `architectures`（如 `["LlamaForCausalLM"]`）
2. 去 `vllm/model_executor/models/registry.py` 查找实现
3. **命中** → 用 vLLM 自己的类；**未命中** → 报错

注意几个容易误解的点：

| 误解 | 实际 |
|------|------|
| `--trust-remote-code` 能跑自定义模型代码 | 它只让自定义 **config / tokenizer** 被加载，**不会**执行自定义 `modeling_*.py`（除非走 transformers 后端） |
| 算子都在就能跑 | 还需要模型按 vLLM 的接口组织：`ColumnParallel/RowParallel`、`Attention(paged)`、`load_weights`、`compute_logits` 等 |
| 改了权重名就能用 | 权重映射（`load_weights` 里的 name 转换）也得对得上 |

---

## 三种可行路径

<details>
<summary><b>方法 1：架构别名注册（最省事，适合“其实是某个已支持架构的变体”）</b></summary>

如果你的自定义架构结构/权重和某个已支持架构**完全一致**（比如只是把 `LlamaForCausalLM` 换了个名字），直接注册映射即可：

```python
from vllm import ModelRegistry

ModelRegistry.register_model(
    "MyCustomForCausalLM",              # 你 config.json 里的 architectures 名字
    "vllm.model_executor.models.llama:LlamaForCausalLM",  # 复用已有实现
)
```

把这段代码放在能被导入的位置（或做成插件入口，见方法 3），再正常 `vllm serve` 即可。
</details>

<details>
<summary><b>方法 2：transformers 后端（能跑远程代码，但有代价）</b></summary>

vLLM 提供了 `--model-impl` 选项：

```bash
vllm serve your-org/your-custom-model \
    --model-impl transformers \
    --trust-remote-code
```

- `auto`（默认）：优先 vLLM 实现，没有则回退
- `vllm`：只用自己的实现
- `transformers`：**直接调用 HF 的 `AutoModelForCausalLM`**，因此可以执行 `trust_remote_code` 拉下来的自定义建模代码

**代价（务必评估）：**
- 性能明显更差，用不上 vLLM 针对性的融合 kernel / 高度优化的执行路径
- 张量并行、chunked prefill、prefix caching、部分量化等高级特性支持不完整或不可用
- 更多是「能跑起来 / 做验证」的选项，不适合生产高吞吐

</details>

<details>
<summary><b>方法 3：写 vLLM 模型实现 + out-of-tree 注册（生产推荐）</b></summary>

参照最接近的已有模型（如 `llama.py`、`qwen2.py`）写一个符合 vLLM 接口的类：

```python
class MyCustomForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        # 用 VocabParallelEmbedding / ColumnParallelLinear / Attention 等
        ...

    def forward(self, input_ids, positions, ...):
        # 返回 hidden_states
        ...

    def compute_logits(self, hidden_states, ...):
        ...

    def load_weights(self, weights, ...):
        # 处理你权重文件里的命名 → vLLM 参数名映射
        ...
```

然后通过插件/入口点注册（不用 fork vLLM）：

```python
# 在你的包里 register.py
from vllm import ModelRegistry

def register():
    ModelRegistry.register_model(
        "MyCustomForCausalLM",
        "my_pkg.models:MyCustomForCausalLM",
    )

# setup.py / pyproject.toml 里声明 entry point
# [project.entry-points."vllm.general_plugins"]
# my_plugin = "my_pkg.register:register"
```

这样安装你的包后，vLLM 启动时会自动加载插件完成注册。

</details>

---

## 注意事项

- **`--trust-remote-code` 的语义要分清**：它让 vLLM 用 `AutoConfig`/`AutoTokenizer` 加载远程自定义的 config 和 tokenizer，默认**不包括** `modeling_*.py`。只有方法 2（transformers 后端）会真正执行远程建模代码。
- **方法 2 ≠ 万能**：有些架构即便 transformers 能跑（比如带有复杂自定义 attention/位置的模型），在 vLLM 的 transformers 包裹层下也可能因为 KV cache 组织方式不匹配而失败。
- **算子是必要但不充分**：即便你的架构只用标准算子，只要 forward 里的并行切分、attention 调用、KV cache 布局不符合 vLLM 约定，依然跑不起来。
- 若只是想**快速验证正确性**，可以先用方法 2；想上生产吞吐，就上方法 3。

---

## 一句话总结

> 自定义 HF 架构 + 算子齐全，**仍然默认跑不了**；vLLM 要求架构在注册表里有对应实现。可选方案是：**复用已有实现注册别名** → **transformers 后端兜底** → **自己写 vLLM 模型类并插件注册**（生产首选）。

如果你告诉我具体的架构名（`config.json` 的 `architectures`）和它和哪个已支持模型最接近，我可以帮你判断走哪条路、以及需要改哪些接口。

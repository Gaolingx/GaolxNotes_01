# vLLM如何集成自定义HF transformer模型

## 🧑‍💻 User

vllm已经编写hf transformer modeling代码情况下如何添加自定义模型支持

---
## 😀 默认助手 (7)

I'll pull the current vLLM docs so the answer matches the latest APIs.

Fetched. Let me read the actual doc content.

The fetched pages are mostly nav chrome. Let me read the content portion.

The docs URL redirected. Let me grab the actual source markdown from GitHub.

Got the Transformers-backend content. Now the native "Adding a New Model" guide and plugin system.

Now the two core pages: Basic Model and Registering a Model.

Now the "Basic Model" content.

Let me confirm the exact class API surface from the generated API reference.

下面是基于 vLLM 最新文档（`contributing/model/*`、plugin system、registry 源码）整理的完整方案。核心结论：**你已经写好的 HF modeling 代码，绝大多数情况下不需要移植成 vLLM 原生实现，直接走 "Transformers modeling backend" 即可**；只有当模型含 vLLM 不支持的新算子（如新型 attention、encoder-decoder）时才需要写原生实现 + 注册。

---

## 先做决策：两条路线

| | 路线 A：Transformers modeling backend | 路线 B：vLLM 原生实现 |
|---|---|---|
| 需要写的 vLLM 代码 | 几乎为零（只微调 HF 代码几处） | 需要重写为 vLLM 层 |
| 适用 | embedding / LM / VLM / ALM；encoder-only、decoder-only、MoE；full / sliding attention | 新 attention 算子、encoder-decoder、Mamba-like、自定义并行 |
| 性能 | 官方称与原生实现**相同** | 相同 |
| TP/EP/DP/PP | 自动支持（需补 `base_model_tp_plan` / `base_model_pp_plan`） | 需自己用并行层实现 |

> 官方原话：*"Many decoder language models can now be automatically loaded using the Transformers modeling backend without having to implement them in vLLM. See if `vllm serve <model>` works first!"*

建议顺序：**先试路线 A（10 分钟）→ 不行再上路线 B。**

---

## 路线 A：直接在 vLLM 里跑你的 HF 模型

### 1. 触发方式

vLLM 的 registry 里有 `_try_resolve_transformers()` 回退逻辑：当 `architectures[0]` 既不在 `_VLLM_MODELS` 里、transformers 也没有该类时，它会去读 `config.json` 的 `auto_map`，用 `try_get_class_from_dynamic_module` 动态加载你的类，然后检查：

```python
transformers_model_cls.is_backend_compatible()
or transformers_model_cls._can_set_attn_implementation()
```

通过后，用 `model_config._get_transformers_backend_cls()`（即 `vllm/model_executor/models/transformers/*` 下的包装类）实例化，并把 `self.config._attn_implementation = "vllm"` 设上，让 attention 走 vLLM 的 kernel。

所以使用条件是：

- `config.json` 存在且结构正确；
- `config.json` 里有 **`auto_map.AutoModel`**（自定义/remote code 模型必须）；
- 然后：

```bash
# 在线服务
vllm serve <MODEL_DIR> --trust-remote-code
# 或强制指定后端
vllm serve <MODEL_ID> --model-impl transformers --trust-remote-code
```

```python
# 离线
from vllm import LLM
llm = LLM(model="<MODEL_DIR>", model_impl="transformers", trust_remote_code=True)
```

**注意**：自定义必须写在 **base model**（`MyModel`）里，**不要**写在 `MyModelForCausalLM` 里。

### 2. 要对你已写好的 HF 代码做的 4 处改动

<details>
<summary><b>① Attention 必须走 <code>ALL_ATTENTION_FUNCTIONS</code> 接口（最关键）</b></summary>

- `MyAttention` 必须通过 `ALL_ATTENTION_FUNCTIONS.get_interface(...)` 调用 attention，且**恰好调用一次** —— vLLM 会为每个 `MyAttention` 挂一个 `vllm.model_executor.layers.attention.attention.Attention` 层。
- `MyAttention` 必须有唯一的 `layer_idx` —— vLLM 用它作为 KV cache 的 key。
- 如果 scale 不是 `head_size**-0.5`，必须在调用时传 `scaling=`，vLLM 从这次调用里读取。

```python
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

class MyAttention(nn.Module):
    is_causal = False  # 仅 encoder-only 模型需要

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx          # 必须唯一
        self.scaling = self.head_dim ** -0.5

    def forward(self, hidden_states, **kwargs):
        ...
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states,
            scaling=self.scaling, **kwargs,
        )
        ...
```

**不再需要** `_supports_attention_backend = True`，也**不再需要**把 `kwargs` 从 `MyModel` 一路透传到 `MyAttention`。

</details>

<details>
<summary><b>② MoE 模型的三个约定</b></summary>

1. sparse MoE block 必须有名为 `experts` 的属性；
2. `MyExperts` 要么继承 `nn.ModuleList`（naive），要么包含全部 3D `nn.Parameter`（packed）；
3. `MyExperts.forward` 必须接受 `hidden_states, top_k_index, top_k_weights`。

```python
class MyExperts(nn.ModuleList):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        ...

class MySparseMoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = MyExperts(config)   # 属性名必须是 experts
    def forward(self, hidden_states):
        ...
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
```

</details>

<details>
<summary><b>③ 要 TP/PP 就在 config 里补 <code>base_model_tp_plan</code> / <code>base_model_pp_plan</code></b></summary>

```python
from transformers import PreTrainedConfig

class MyConfig(PreTrainedConfig):
    base_model_tp_plan = {
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj":    "colwise",
        "layers.*.mlp.up_proj":      "colwise",
        "layers.*.mlp.down_proj":    "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers":       (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm":         (["hidden_states"], ["hidden_states"]),
    }
```

- `tp_plan` 只有 `"colwise"` / `"rowwise"`；vLLM 能自动推断标准 attention（q/k/v/o）和 gated-MLP/experts（gate/up/down）的切分，**没被 fuse 也没列进 plan 的 Linear 会按 replicated 处理**，所以只列"不符合常规模式"的层。
- `pp_plan` 只对"不是所有 pipeline stage 都存在的层"才需要；`nn.ModuleList` 假定只有一个并跨 stage 切分。

</details>

<details>
<summary><b>④ 已知限制（不支持时只能走路线 B）</b></summary>

Transformers backend 目前覆盖：modalities = embedding / LM / VLM / ALM；architectures = encoder-only / decoder-only / MoE；attention = full 和/或 sliding。**不覆盖** encoder-decoder（T5/BART 类）、自定义新 attention 算子、Mamba-like 混合层。

另外：VLM 用 `dtype="auto"` 时，vLLM 用 config 的整体 dtype，而原生 transformers 会尊重各 backbone 的 `dtype`，可能带来细微差异。

</details>

### 3. 验证走的是哪个后端

```python
from vllm import LLM
llm = LLM(model="...")
llm.apply_model(lambda model: print(type(model)))
# 打印以 Transformers... 开头 => 走的是 Transformers modeling backend
```

---

## 路线 B：写 vLLM 原生实现 + 注册

### 1. 改造要点（对照官方 "Basic Model" 五步）

关键差异：vLLM 期望的是**扁平化、单 batch 维度**的张量，且 `prefix` 贯穿所有子模块构造器。

**Initialization** —— 每个模块构造器都要带 `prefix`：

```python
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention

class MyAttention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        # prefix 用于: ① attention op 注册唯一层名 ② 非均匀量化按层名匹配
        self.attn = Attention(prefix=f"{prefix}.attn")

class MyDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.self_attn = MyAttention(vllm_config, prefix=f"{prefix}.self_attn")

class MyModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.layers = nn.ModuleList([
            MyDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}")
            for i in range(vllm_config.model_config.hf_config.num_hidden_layers)
        ])
```

**Computation** —— 扁平化输入、提供 `embed_input_ids`：

```python
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,   # PP 需要
    inputs_embeds: torch.Tensor | None = None,                 # 多模态/embedding 输入
) -> torch.Tensor:
    ...

def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    # 统一接口，便于被多模态复合模型复用
    ...
```

**并行层替换**（可选，模型放不下单卡时才需要）：

| 原层 | 替换为 |
|---|---|
| `nn.Embedding` | `VocabParallelEmbedding` |
| LM head | `ParallelLMHead` |
| 复制型 Linear | `ReplicatedLinear` |
| FFN 第二层 / attn 输出 | `RowParallelLinear`（行切分 + all-reduce） |
| FFN 第一层 / 分离的 QKV | `ColumnParallelLinear` |
| gated-FFN 第一层（SiLU 等） | `MergedColumnParallelLinear` |
| QKV 合一投影 | `QKVParallelLinear`（KV head < world size 时自动复制 KV） |

**顶层的 `*ForCausalLM`**：

```python
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

class MyModelForCausalLM(nn.Module):
    # 权重名 fuse 映射：checkpoint 里是分离的，模型里是合并的
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model = MyModel(vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(..., prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(...)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):          # V1 把 lm_head 算 logits 拆到这里
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
```

> 版本差异提示：V1 架构里 `forward` 只返回 hidden states，logits 由 `compute_logits` 产生；较早的版本是在 `forward` 里直接出 logits、并用 `sample()`（已废弃）。请以你安装的 vLLM 版本里 `vllm/model_executor/models/llama.py` 的实际签名为准。

**可选 protocol**（按需继承，registry 会据此判定能力）：

- `SupportsPP` —— 需同时实现 `make_empty_intermediate_tensors`
- `SupportsLoRA`、`SupportsQuant`
- `SupportsMultiModal` —— **多模态模型必须实现**（配 `get_multimodal_embeddings` / `get_input_embeddings`）
- `IsAttentionFree`（纯 Mamba-1/2）、`IsHybrid`（Mamba + attention 混合）
- `SupportsTranscription`（语音转写）

### 2. 注册

**in-tree**（改了 vLLM 源码）：

1. 把 `modeling_my_model.py` 放进 `vllm/model_executor/models/`；
2. 在 `vllm/model_executor/models/registry.py` 的 `_VLLM_MODELS` dict 里加上架构名 → 模块（**按字母序维护**）；
3. 更新 `tests/models/registry.py` 的示例 HF 模型；
4. 更新 `docs/models/supported_models.md`。

**out-of-tree**（推荐，不改源码）—— 通过 plugin 的 entry point 自动执行：

```toml
# pyproject.toml
[project.entry-points."vllm.general_plugins"]
register_my_model = "my_pkg:register"
```

```python
# my_pkg/__init__.py
def register():
    from vllm import ModelRegistry
    # 用 "<module>:<class>" 字符串形式惰性导入，避免 fork 子进程里初始化 CUDA 报错
    if "MyModelForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MyModelForCausalLM",
            "my_pkg.modeling_my_model:MyModelForCausalLM",
        )
```

`register_model` 的签名（源码确认）：

```python
def register_model(self, model_arch: str, model_cls: type[nn.Module] | str) -> None
```

- `model_arch` 必须与模型 `config.json` 里 `architectures[0]` **完全一致**（大小写敏感）；
- 传 class 对象 → `_RegisteredModel`（主进程已 import）；传 `"<module>:<class>"` 字符串 → `_LazyRegisteredModel`（惰性导入，**推荐**）。

插件入口函数必须是 **re-entrant**（可被多次调用），因为每个子进程都会加载它。用 `VLLM_PLUGINS=<plugin_name>` 可只加载指定插件。

---

## 常见坑

<details>
<summary><b>展开查看排查清单</b></summary>

| 现象 | 原因 / 解法 |
|---|---|
| `RuntimeError: Cannot re-initialize CUDA in forked subprocess` | 注册时直接传了 class 对象，import 阶段初始化了 CUDA。改用 `"<module>:<class>"` 字符串惰性导入 |
| `Model architectures ['X'] are not supported for now` | `model_arch` 与 `config.json` 的 `architectures[0]` 不匹配；或插件没被加载（检查 entry point group 是否为 `vllm.general_plugins`） |
| `_can_set_attn_implementation()` 检查失败 | `MyAttention` 没走 `ALL_ATTENTION_FUNCTIONS` 接口，或调用了多次 attention |
| KV cache 错乱 / 结果乱码 | `MyAttention.layer_idx` 重复或不唯一 |
| 走成了原生 HF eager attention（慢） | 别忘了 `--model-impl transformers`，或 `config._attn_implementation` 没被设成 `"vllm"` |
| 权重 loading 报 unexpected keys | 缺少 `packed_modules_mapping`（qkv / gate_up 的 fuse 映射） |
| 量化层没被量化 / 报 prefix 相关错 | 子模块构造器没传 `prefix`，非均匀量化按层名匹配失败 |
| 多模态加载报错 | `*ForCausalLM` 没实现 `SupportsMultiModal` |
| 交错 sliding window 不生效 | `config.json` 需含 `layer_types`，且每层要把对应的 `per_layer_sliding_window` 传给 `Attention` |
| PP 报错 | 继承 `SupportsPP` 并实现 `make_empty_intermediate_tensors`；`forward` 要接受 `intermediate_tensors` |

**interleaved sliding window 示例**（对齐 llama.py 的做法）：

```python
sliding_window = config.sliding_window if config.layer_types[i] == "sliding_attention" else None
self.attn = Attention(..., per_layer_sliding_window=sliding_window, prefix=f"{prefix}.attn")
```

</details>

<details>
<summary><b>进阶：新 attention 算子 / Mamba-like 层 / 纯 encoder-decoder</b></summary>

- **纯 Mamba（无 attention）**：参考 `MambaForCausalLM` / `Mamba2ForCausalLM`，继承 `IsAttentionFree`，实现类方法 `get_mamba_state_dtype_from_config` 和 `get_mamba_state_shape_from_config`，用 `MambaMixer` / `MambaMixer2`，并把模型加进 `vllm/model_executor/models/config.py` 的 `MODELS_CONFIG_MAP`。
- **Mamba + attention 混合**：参考 `JambaForCausalLM` / `NemotronHForCausalLM`，继承 `IsHybrid`（不必加进 `MODELS_CONFIG_MAP`）。
- **Mamba-like（Linear Attention / ShortConv）**：参考 `Lfm2ForCausalLM`；自定义层继承 `MambaBase` 并实现 `get_state_dtype` / `get_state_shape` / `mamba_type` / `get_attn_backend`，同时实现对应的 attention metadata 类，并在 `vllm/v1/attention/backends/registry.py` 的 `MambaAttentionBackendEnum` 注册。要支持 CUDA graph / torch.compile，需用 `direct_register_custom_op` 包一层，并把该 op 加进 `vllm/config/compilation.py` 的 `_attention_ops`。
- **Encoder-Decoder（BART 类）**：官方建议走 **plugin** 路线，参考 `bart-plugin`（`BartForConditionalGeneration`）和 `vllm-nemotron-asr-plugin`。

</details>

---

## 最后：验证清单

```bash
# 1. 架构名是否被识别
python -c "from vllm import ModelRegistry; print('MyModelForCausalLM' in ModelRegistry.get_supported_archs())"

# 2. 端到端跑通
```

```python
from vllm import LLM

# generative
llm = LLM(model="<MODEL_DIR>", runner="generate")
print(llm.generate("Hello, my name is"))

# pooling / embedding
llm = LLM(model="<MODEL_DIR>", runner="pooling")
print(llm.encode("Hello, my name is"))

# 确认实际用的是哪个后端
llm.apply_model(lambda m: print(type(m)))
```

成功返回文本（生成模型）或 hidden states（pooling 模型）即表示接入成功。

> 上述 API 细节来自 vLLM 的 latest/dev 文档，vLLM 的 model interface 演进较快（例如 `Attention` 的 import 路径、`compute_logits` 的引入、`sample()` 的移除）。落地前建议先看一眼你本地版本里 `vllm/model_executor/models/llama.py` 的 `LlamaForCausalLM`，以它作为该版本的"标准答案"。

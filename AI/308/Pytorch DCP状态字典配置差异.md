# Pytorch DCP状态字典配置差异

### 🧑‍💻 User

pytorch dcp中，StateDictOptions是专门给fsdp2用的吗，这个和StateDictType有什么区别？

---

### 🤖 Assistant

## 结论先说

**`StateDictOptions` 并不是专门给 FSDP2 用的。** 它是 `torch.distributed.checkpoint`（DCP）里 state_dict 工具函数（`get_model_state_dict` / `set_model_state_dict` / `get_optimizer_state_dict` / `set_optimizer_state_dict`）统一的配置对象，同时支持 FSDP1、FSDP2、DDP、HSDP 以及非并行模型。

而 **`StateDictType` 是 FSDP1 专用的枚举**，用于 `FSDP.state_dict_type()` 上下文管理器。FSDP2 里没有这个概念。

---

## 1. `StateDictOptions`

定义在 `torch.distributed.checkpoint.state_dict`，是一个 dataclass，主要字段：

| 字段 | 含义 |
|------|------|
| `full_state_dict` | 是否返回**非分片**（完整）的 state dict。`True` 时每个 rank 都拿到完整权重，适合推理/评估；`False` 时返回分片形式（配合 DCP save/load 做分布式保存） |
| `cpu_offload` | 是否把 state dict offload 到 CPU |
| `ignore_frozen_params` | 是否跳过冻结参数 |
| `keep_submodule_prefixes` | 是否保留子模块前缀 |
| `strict` | 加载时是否严格匹配 key |
| `broadcast_from_rank0` | 从 rank 0 广播完整 state dict |
| `flatten_optimizer_state_dict` | 是否展平优化器状态（FSDP 用） |
| `use_dtensor` | 是否用 **DTensor** 表达（FSDP2 和 TP 会用到） |
| `async_state_dict` | 是否异步生成 |

典型用法：

```python
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)

# 每个 rank 拿到完整权重
opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
full_sd = get_model_state_dict(model, options=opts)

# 加载（分片状态）
set_model_state_dict(model, sharded_sd, options=StateDictOptions())
```

它本身**跟并行后端无关**，内部会根据模型是 FSDP1 / FSDP2 / DDP / 普通 nn.Module 来自动选择对应的提取逻辑。

---

## 2. `StateDictType`

定义在 `torch.distributed.fsdp`，是一个枚举，**只属于 FSDP1**：

```python
class StateDictType(Enum):
    FULL_STATE_DICT     # 完整（非分片）
    LOCAL_STATE_DICT    # 本地（rank 上已有分片的本地视图）
    SHARDED_STATE_DICT  # 分片
```

使用方式是通过 FSDP1 的上下文管理器切换 `model.state_dict()` 的行为：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    sharded_sd = model.state_dict()   # 分片结果，可交给 DCP 保存

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    full_sd = model.state_dict()      # 完整结果
```

注意：FSDP1 的 `FSDP.get_state_dict()` / `set_state_dict()` 这套 API 在新版本里已经被标记弃用，官方推荐迁移到上面的 DCP state_dict API。

---

## 3. 两者区别对比

| 维度 | `StateDictOptions` | `StateDictType` |
|------|--------------------|-----------------|
| 所在模块 | `torch.distributed.checkpoint.state_dict` | `torch.distributed.fsdp` |
| 类型 | dataclass（配置对象） | 枚举（Enum） |
| 适用对象 | FSDP1、FSDP2、DDP、HSDP、普通模型 | 仅 FSDP1 |
| 粒度 | 细粒度，多个布尔开关组合 | 粗粒度，只选三种「类型」 |
| 配合 API | `get_model_state_dict` 等 DCP state_dict 函数 | `FSDP.state_dict_type()` 上下文管理器 |
| 是否支持 DTensor | 支持（`use_dtensor=True`） | 不支持 |

---

## 4. FSDP2 的情况

FSDP2（`fully_shard`）**没有 `StateDictType` 和 `state_dict_type()` 上下文管理器**。它的 state dict 提取/加载统一走 DCP state_dict API，即用 `StateDictOptions` 控制行为。

FSDP2 的 state dict 本质上是用 `DTensor` 表达的分片张量，所以在 DCP 场景里通常配合 `use_dtensor=True`：

```python
from torch.distributed.checkpoint.state_dict import StateDictOptions

# FSDP2 + TP 场景，保留 DTensor 表示做分片保存/加载
opts = StateDictOptions(use_dtensor=True)
```

所以一句话总结：**`StateDictOptions` 是「新一代」统一的 state dict 配置，FSDP2 也用它；`StateDictType` 是「旧一代」FSDP1 专用的枚举。**
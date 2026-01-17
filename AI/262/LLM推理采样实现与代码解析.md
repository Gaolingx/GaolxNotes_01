# LLM推理采样实现与代码解析

### 🧑‍💻 User

llm推理采样阶段，结合python代码说说 `top_k_top_p_sampling_from_logits` 一般是如何实现的？

---

### 🤖 Assistant

# LLM 推理采样：Top-K + Top-P 采样实现详解

## 📊 采样流程概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Logits (原始输出)                         │
│                    [batch_size, vocab_size]                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Temperature Scaling  │  ← 控制分布平滑度
                    │   logits / temp       │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     Top-K Filter      │  ← 保留概率最高的K个
                    │  过滤掉低概率token      │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     Top-P Filter      │  ← 保留累积概率达P的
                    │   (Nucleus Sampling)   │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Softmax + Sample    │  ← 多项式采样
                    │    multinomial        │
                    └───────────────────────┘
                                │
                                ▼
                        Token ID 输出
```

## 🔧 完整实现代码

```python
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def top_k_top_p_sampling_from_logits(
    logits: torch.Tensor,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """
    从logits中进行Top-K和Top-P联合采样
    
    Args:
        logits: 模型输出 [batch_size, vocab_size]
        top_k: 保留概率最高的K个token (0表示不使用)
        top_p: 累积概率阈值 (1.0表示不使用)
        temperature: 温度参数，越大分布越平滑
        min_tokens_to_keep: 最少保留的token数
    
    Returns:
        采样的token ids [batch_size]
    """
    # 保存原始形状
    batch_size, vocab_size = logits.shape
    
    # ═══════════════════════════════════════════════════
    # Step 1: Temperature Scaling (温度缩放)
    # ═══════════════════════════════════════════════════
    # 温度越高 → logits差异越小 → 分布越平滑 → 采样越随机
    # 温度越低 → logits差异越大 → 分布越尖锐 → 采样越确定
    if temperature != 1.0:
        logits = logits / temperature
    
    # ═══════════════════════════════════════════════════
    # Step 2: Top-K Filtering (Top-K过滤)
    # ═══════════════════════════════════════════════════
    if top_k > 0:
        # 确保top_k不超过词表大小
        top_k = min(top_k, vocab_size)
        
        # 方法1: 使用torch.topk找阈值
        # topk返回 (values, indices), 取最后一个值作为阈值
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        threshold = top_k_values[:, -1:]  # [batch, 1]
        
        # 将低于阈值的位置设为-inf
        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float('-inf')),
            logits
        )
    
    # ═══════════════════════════════════════════════════
    # Step 3: Top-P Filtering (Nucleus Sampling)
    # ═══════════════════════════════════════════════════
    if top_p < 1.0:
        # 3.1 按概率降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # 3.2 计算累积概率
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 3.3 找到需要移除的位置（累积概率超过top_p的）
        # 注意：我们要保留第一个使累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 右移一位：确保刚好超过top_p的那个token被保留
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False  # 始终保留最高概率的token
        
        # 3.4 将mask映射回原始顺序
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(
            dim=-1, 
            index=sorted_indices, 
            src=sorted_indices_to_remove
        )
        
        # 3.5 应用mask
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # ═══════════════════════════════════════════════════
    # Step 4: Sampling (采样)
    # ═══════════════════════════════════════════════════
    # 4.1 转换为概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 4.2 多项式采样
    next_tokens = torch.multinomial(probs, num_samples=1)
    
    return next_tokens.squeeze(-1)  # [batch_size]
```

## 📈 图解 Top-P 过滤过程

```
原始概率分布 (已排序):
┌────────────────────────────────────────────────────────┐
│ Token:    A      B      C      D      E      F    ... │
│ Prob:   0.35   0.25   0.15   0.10   0.08   0.04   ... │
│ CumP:   0.35   0.60   0.75   0.85   0.93   0.97   ... │
│                                      ↑                 │
│                              top_p=0.9 截断点          │
└────────────────────────────────────────────────────────┘

保留 token: A, B, C, D, E (累积概率 ≤ 0.9)
过滤 token: F, ... (累积概率 > 0.9)
```

## 🚀 优化版本（支持批处理 + 性能优化）

```python
def top_k_top_p_sampling_optimized(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    优化版采样函数，支持重复惩罚
    
    Returns:
        (sampled_tokens, probs) - 采样的token和对应概率
    """
    logits = logits.clone()
    
    # ═══════════════════════════════════════════════════
    # 可选: 重复惩罚 (Repetition Penalty)
    # ═══════════════════════════════════════════════════
    if repetition_penalty != 1.0 and generated_tokens is not None:
        for batch_idx in range(logits.size(0)):
            for token_id in set(generated_tokens[batch_idx].tolist()):
                if logits[batch_idx, token_id] > 0:
                    logits[batch_idx, token_id] /= repetition_penalty
                else:
                    logits[batch_idx, token_id] *= repetition_penalty
    
    # Temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Top-K (向量化优化)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = logits.topk(top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, float('-inf'), logits)
    
    # Top-P (向量化优化)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumsum_probs = sorted_probs.cumsum(dim=-1)
        
        # 创建mask
        mask = cumsum_probs - sorted_probs > top_p
        sorted_logits[mask] = float('-inf')
        
        # 恢复原始顺序
        logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))
    
    # 采样
    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    # 返回token和对应的概率
    token_probs = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    
    return tokens, token_probs
```

## 🎯 使用示例

```python
# 模拟场景
batch_size = 2
vocab_size = 50000

# 模拟模型输出的logits
logits = torch.randn(batch_size, vocab_size)

# 采样参数
sampling_params = {
    'top_k': 50,           # 只考虑top 50个token
    'top_p': 0.9,          # 累积概率90%截断
    'temperature': 0.8,    # 稍微降低随机性
}

# 执行采样
next_tokens = top_k_top_p_sampling_from_logits(logits, **sampling_params)
print(f"Sampled tokens: {next_tokens}")
# 输出: Sampled tokens: tensor([12345, 6789])
```

## ⚖️ 参数效果对比

| 参数 | 低值效果 | 高值效果 |
|------|---------|---------|
| **temperature** | 更确定性、更保守 | 更随机、更多样 |
| **top_k** | 候选更少、更聚焦 | 候选更多、更多样 |
| **top_p** | 只选高概率token | 允许更多低概率token |

```
temperature=0.1        temperature=1.0        temperature=2.0
    ▲                      ▲                      ▲
    │█                     │▓                     │░░
    │█                     │▓▓                    │░░░
    │█                     │▓▓▓                   │░░░░░
    └────────────          └────────────          └────────────
    (尖锐分布)              (正常分布)              (平滑分布)
```

这就是 LLM 推理中 Top-K/Top-P 采样的核心实现！
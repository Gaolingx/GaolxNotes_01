# LLM推理采样实现与代码解析

### 🧑‍💻 User

llm推理采样阶段，结合python代码说说 `top_k_top_p_sampling_from_logits` 一般是如何实现的？

---

### 🤖 Assistant

我来详细解释一下LLM推理中top-k和top-p采样的实现方法，并提供完整的Python代码。

## 基本概念

- **top-k采样**：只从概率最高的k个token中采样
- **top-p采样（核采样）**：从累积概率达到p的最小token集合中采样

## 完整实现代码

```python
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

def top_k_top_p_sampling_from_logits(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    return_probs: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Top-k和top-p采样实现
    
    Args:
        logits: 原始logits张量 [batch_size, vocab_size] 或 [vocab_size]
        top_k: 保留的最高概率token数量
        top_p: 累积概率阈值 (0-1)
        temperature: 温度参数，控制随机性
        return_probs: 是否返回处理后的概率分布
    
    Returns:
        sampled_tokens: 采样的token [batch_size] 或 scalar
        probs: 处理后的概率分布 (可选)
    """
    
    # 确保输入是2D的 [batch_size, vocab_size]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    # 应用温度参数
    if temperature != 1.0:
        logits = logits / temperature
    
    # 转换为概率
    probs = F.softmax(logits, dim=-1)
    
    # 应用top-k过滤
    if top_k is not None and top_k > 0:
        # 获取top-k的值和索引
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        
        # 创建新的概率分布，非top-k的位置设为0
        new_probs = torch.zeros_like(probs)
        new_probs.scatter_(-1, top_k_indices, top_k_probs)
        probs = new_probs
    
    # 应用top-p过滤
    if top_p is not None and top_p < 1.0:
        # 对概率进行排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过top_p的第一个位置
        # 我们需要移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 确保至少保留一个token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 将需要移除的位置设为0
        for i in range(probs.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            probs[i].scatter_(-1, indices_to_remove, 0.0)
    
    # 重新归一化概率分布
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    
    # 从处理后的分布中采样
    if probs.size(0) == 1:
        # 单样本情况
        sampled_token = torch.multinomial(probs[0], num_samples=1).item()
        result = sampled_token
    else:
        # 批量采样
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        result = sampled_tokens
    
    if return_probs:
        return result, probs
    else:
        return result

# 更简洁的版本（单样本）
def sample_from_logits_simple(
    logits: torch.Tensor,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0
) -> int:
    """简化版的top-k top-p采样"""
    
    # 应用温度
    logits = logits / temperature
    
    # top-k过滤
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
    
    # 转换为概率
    probs = F.softmax(logits, dim=-1)
    
    # top-p过滤
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 移除累积概率超过p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
        
        # 重新归一化
        if torch.sum(probs) > 0:
            probs = probs / torch.sum(probs)
    
    # 采样
    return torch.multinomial(probs, num_samples=1).item()
```

## 使用示例

```python
# 示例1：基本使用
def demonstrate_sampling():
    # 模拟LLM输出的logits
    vocab_size = 10000
    logits = torch.randn(vocab_size)
    
    print("=== 不同采样策略对比 ===")
    
    # 1. 贪婪采样（temperature=0）
    greedy_token = top_k_top_p_sampling_from_logits(
        logits, temperature=0.0
    )
    print(f"贪婪采样: {greedy_token}")
    
    # 2. 仅使用temperature
    temp_token = top_k_top_p_sampling_from_logits(
        logits, temperature=0.8
    )
    print(f"温度采样(0.8): {temp_token}")
    
    # 3. top-k采样
    topk_token = top_k_top_p_sampling_from_logits(
        logits, top_k=50, temperature=1.0
    )
    print(f"Top-k采样(k=50): {topk_token}")
    
    # 4. top-p采样
    topp_token = top_k_top_p_sampling_from_logits(
        logits, top_p=0.9, temperature=1.0
    )
    print(f"Top-p采样(p=0.9): {topp_token}")
    
    # 5. 结合使用
    combined_token = top_k_top_p_sampling_from_logits(
        logits, top_k=50, top_p=0.9, temperature=0.8
    )
    print(f"组合采样: {combined_token}")

# 示例2：批量采样
def batch_sampling_example():
    batch_size = 3
    vocab_size = 1000
    batch_logits = torch.randn(batch_size, vocab_size)
    
    print("\n=== 批量采样 ===")
    sampled_tokens = top_k_top_p_sampling_from_logits(
        batch_logits, top_k=40, top_p=0.95, temperature=0.7
    )
    print(f"批量采样结果: {sampled_tokens}")

# 示例3：查看概率分布
def probability_analysis():
    logits = torch.tensor([3.0, 1.0, 0.5, 0.1, -1.0, -2.0])
    
    print("\n=== 概率分布分析 ===")
    sampled_token, processed_probs = top_k_top_p_sampling_from_logits(
        logits, top_k=3, top_p=0.9, return_probs=True
    )
    
    original_probs = F.softmax(logits, dim=-1)
    
    print("原始概率:", [f"{p:.3f}" for p in original_probs.tolist()])
    print("处理后概率:", [f"{p:.3f}" for p in processed_probs[0].tolist()])
    print(f"采样token: {sampled_token}")

if __name__ == "__main__":
    demonstrate_sampling()
    batch_sampling_example()
    probability_analysis()
```

## 实际应用中的注意事项

```python
class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0
    ) -> str:
        """完整的文本生成函数"""
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        for _ in range(max_length):
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
            
            # 重复惩罚
            if repetition_penalty != 1.0:
                self.apply_repetition_penalty(next_token_logits, input_ids, repetition_penalty)
            
            # 采样下一个token
            next_token = top_k_top_p_sampling_from_logits(
                next_token_logits[0],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            
            # 添加到序列中
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=-1)
            
            # 检查结束条件
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(input_ids[0])
    
    def apply_repetition_penalty(self, logits, input_ids, penalty):
        """应用重复惩罚"""
        for token_id in set(input_ids[0].tolist()):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
```

## 关键要点

1. **执行顺序**：通常先应用temperature，然后top-k，最后top-p
2. **数值稳定性**：注意处理概率为0的情况
3. **批量处理**：支持批量采样提高效率
4. **参数调优**：
   - temperature↑ → 随机性↑
   - top-k↓ → 多样性↓
   - top-p↓ → 确定性↑

这种采样策略在保持生成质量的同时，提供了对输出多样性的精细控制。
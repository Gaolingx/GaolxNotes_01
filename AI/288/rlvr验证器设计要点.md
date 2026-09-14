# rlvr验证器设计要点

### 🧑‍💻 User

rlvr 的验证器怎么设计？

---

### 🤖 Assistant

# RLVR 验证器设计指南

RLVR（Reinforcement Learning with Verifiable Rewards）的核心是：**用程序化、可验证的信号替代人工/模型打分**。验证器（Verifier）的设计直接决定训练质量。

## 1. 验证器的基本架构

```
模型生成 → 答案提取 (Extractor) → 验证 (Verifier) → 标量奖励 → RL 算法 (GRPO/PPO)
```

## 2. 按任务类型设计

### 数学类
- **提取**：正则解析 `\boxed{}`、`final answer` 等格式标记
- **验证**：符号等价判断而非字符串匹配
  - 用 `sympy` 做等价化简：`simplify(pred - gold) == 0`
  - 处理数值近似（容差 $10^{-6}$）、单位换算、分数/小数等价
- **奖励**：通常二值 $r \in \{0, 1\}$，可加格式奖励

```python
def math_reward(completion, gold):
    pred = extract_boxed(completion)
    if pred is None:
        return -0.1  # 格式惩罚
    try:
        return 1.0 if sympy_equiv(pred, gold) else 0.0
    except Exception:
        # fallback：规范化字符串比较
        return 1.0 if normalize(pred) == normalize(gold) else 0.0
```

### 代码类
- **验证**：沙箱执行单元测试
- **关键工程点**：
  - 超时限制、资源隔离（如 `resource.setrlimit`、容器）
  - 部分测试通过给部分分：$r = \frac{\text{pass count}}{\text{total}}$
  - 防作弊：隐藏测试集、禁止硬编码输出（混淆输入）

### 逻辑/推理类
- 谜题（Sudoku、Countdown）可写规则校验器
- 多选题：精确匹配选项字母

## 3. 奖励设计原则

| 原则 | 说明 |
|------|------|
| **抗 hack** | 不要用可被绕过的代理（如"包含关键词"），警惕 reward hacking |
| **粒度权衡** | 二值奖励信号最干净但稀疏；部分分加速学习但可能被利用 |
| **格式约束** | 小幅惩罚不遵守输出格式（如 0.1 量级），但主奖励仍靠正确性 |
| **避免长度偏见** | 不直接奖励长度，可用 KL 正则或对过长输出截断/惩罚 |
| **去重与降权** | 对纯重复内容（重复 token 循环）惩罚 |

## 4. 常见工程细节

- **组内归一化**：GRPO 中同一 prompt 采 $G$ 个回答，优势为 $A_i = \frac{r_i - \bar{r}}{\sigma_r}$。注意：组内全对/全错时梯度为零 → 用 **动态采样**（跳过零方差 prompt，重采样保持 batch 多样性）
- **难度课程**：按 pass rate 分层采样，优先中等难度（pass rate 20%–80%）的样本
- **验证器可靠性**：数学等价判定错误率应 <1%，否则错误标签会毁掉训练；大规模前用人工抽查验证器准确率
- **延迟**：验证在 CPU 上与 GPU 生成流水线并行，代码执行类用进程池

## 5. 常见坑

1. **字符串精确匹配**：`"1/2"` vs `"0.5"` 被判错 → 必须用符号/数值等价
2. **Reward hacking**：模型学会输出测试输入本身、或 exploit 解析器空答案得分
3. **过度格式化惩罚**：早期模型没学会格式时被双重惩罚（格式+错误），导致训练崩塌 → 早期放宽格式奖励
4. **验证器泄漏**：gold answer 出现在 prompt 或上下文中

---

**一句话总结**：验证器要"严格而等价宽容"——判定逻辑必须正确且抗 hack，但对正确答案的多种表达形式要宽容；配合动态采样和合理的奖励整形，才能稳定驱动 RL 训练。需要针对某个具体任务（数学/代码/Agent）展开细节可以告诉我。
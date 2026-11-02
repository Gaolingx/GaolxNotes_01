# SFT预分词磁盘占用大原因及优化建议

## 🧑‍💻 User

为什么sft预分词很吃磁盘空间

---
## 😀 默认助手 (7)

# 为什么 SFT 预分词这么吃磁盘

一句话结论：**预分词存的是 token id，不是文本。而“一个 token 的二进制表示”往往比“它对应的原始字符”更占地方**，再叠上 padding、labels 副本、多轮展开和缓存层级，几十 MB 的 JSONL 变成几 GB 是很正常的。

---

## 一、基本换算：token id 本身就比文本大

原始文本是 UTF-8 压缩得比较好的字节流，而 token id 是定长整数：

| 数据类型 | 每 token 字节 | 中文（1 字≈3B≈1~1.5 token） | 英文（1 token≈4B） |
|---|---|---|---|
| `int64` / `torch.long` | 8 B | 膨胀 **2.7×** | 膨胀 **2×** |
| `int32` | 4 B | 膨胀 1.3× | 基本持平 1× |
| `uint16`（vocab < 65536） | 2 B | 缩到 0.7× | 缩到 0.5× |

问题在于：**大多数框架默认 `torch.long` / `int64`**。你原本 3 字节的汉字，tokenize 完变成 8 字节，从第一步就开始亏损。

---

## 二、一条样本被存成多份张量

预分词产物通常长这样：

```python
{
  "input_ids":      [...],  # int64, 8B/token
  "labels":         [...],  # int64, 8B/token —— 和 input_ids 几乎完全相同
  "loss_mask":      [...],  # 标记哪些位置算 loss
  "attention_mask": [...],  # int64, 8B/token —— 完全可由长度推出
  "position_ids":   [...],  # int64, 8B/token —— 完全可由长度推出
}
```

- `labels` 是 `input_ids` 的错位副本 → **2×**
- `attention_mask` / `position_ids` 是纯冗余，却按 int64 存 → 再 **2×**
- `loss_mask` 用 `bool`/`int8` 还好，用 int64 就又翻倍

**光这一层就是 3~5 倍的放大**，而这部分信息熵几乎为零。

---

## 三、padding：最大的单项放大来源（常被忽略）

预分词为了能 batch，经常按 `max_length` 做成稠密矩形：

$$
\text{磁盘} = N_{\text{sample}} \times \text{max\_length} \times \text{bytes} \times \text{份数}
$$

而不是

$$
\text{磁盘} = N_{\text{token}} \times \text{bytes} \times \text{份数}
$$

实际放大的倍数是 $\dfrac{\text{max\_length}}{\text{avg\_length}}$。

举实例：`max_length=4096`，平均长度 600 token，利用率只有 15% → **6.7 倍浪费**。这在按长度分桶或不 packing 的 pipeline 里极其常见。

---

## 四、多轮对话的“展开”会让 token 数超线性增长

这是很多人踩过的坑：多轮 SFT 常把一段 $L$ 轮的对话展开成 $L$ 条训练样本，**每条都带完整对话历史**：

| 轮次 | 该样本 token 数 |
|---|---|
| 1 | $t_1$ |
| 2 | $t_1+t_2$ |
| 3 | $t_1+t_2+t_3$ |
| … | … |

总 token 数从 $T$ 变成约 $T \cdot L/2$。一段 10 轮的对话，磁盘占用能到原来的 **5 倍以上**，样本数也从 1 条变成 10 条。同理，长 `system prompt` 在每条样本里都被完整复制一遍，文本层面可去重/可共享的部分，tokenize 后被彻底「物化」。

---

## 五、多阶段产物叠加 + 缓存从不清理

典型 pipeline 里，每一步 `dataset.map()` 都会落盘一份完整副本：

```
raw.jsonl
  └─> apply_chat_template + tokenize      → cache/xxx.arrow   (1份)
        └─> truncate / mask labels        → cache/yyy.arrow   (2份)
              └─> pack / group_texts      → cache/zzz.arrow   (3份)
                    └─> 按 rank 切 shard  → cache/rank_xx/   (N份)
```

- `num_proc>1` 时还有中间分片文件
- 换 chat template、换 `max_length`、换 tokenizer 版本 → **缓存指纹变了，整份重写**，旧的不删
- 多机训练时如果每个节点本地盘各存一份 → ×节点数

磁盘上是 raw + 3~4 份中间态 + 多个历史版本同时在。

---

## 六、序列化格式的二次放大

如果产物是 JSONL 而不是二进制：

```json
{"input_ids": [1234, 5678, 9012, ...]}
```

每个 token 变成 `"1234, "` 约 6 字节 —— 比 int32 的 4 字节还大 50%，比 uint16 大 **3 倍**。再加上 `labels` 再复制一遍，轻松到 12B/token。用 `pickle` 存 Python list 更糟（每个 int 对象 28 字节 + 指针）。

---

## 七、合起来算一笔账

假设：5 万条样本、平均 800 token、中文、展开不带多轮

| 项目 | 计算 | 大小 |
|---|---|---|
| 原始 JSONL | 文本 | ~70 MB |
| input_ids (int64) | 40M × 8B | 320 MB |
| labels (int64) | ×1 | 320 MB |
| attention_mask + position_ids | ×2 | 640 MB |
| loss_mask (int64) | ×1 | 320 MB |
| **小计** | | **≈ 1.6 GB** |
| padding 到 4096（×5） | | **≈ 8 GB** |
| 多阶段缓存留 3 份 | | **≈ 24 GB** |

**70 MB 的输入 → 24 GB 的磁盘**，膨胀 300 倍以上。感觉「反直觉」是因为直觉里只有「文本→数字」这一层，实际有 5 层乘数叠在一起。

---

<details>
<summary><b>怎么把空间压回去（点击展开）</b></summary>

按收益排序：

1. **不 padding，改 packing**：把样本拼成连续 token 流，利用率接近 100%。收益最大，通常 3~8×。
2. **降 dtype**：`int64 → uint16`（vocab<65536 时）或至少 `int32`。收益 2~4×。
3. **只存 `input_ids`**：`labels` 训练时错位生成，`attention_mask` / `position_ids` 按长度现算。收益 3~4×。
4. **用二进制格式**：`.npy` / `.arrow` / `.parquet` / Megatron 的 `.bin`+`.idx`+`memmap`，别用 JSONL 存整数数组。收益 1.5~3×。
5. **避免多轮展开重复存历史**：存「整段对话」一次，训练时用 sample packing + 自定义 mask，而不是存 $L$ 份前缀。
6. **缓存治理**：给 `cache_dir` 设版本前缀，`max_length`/template/tokenizer 变更时换目录并删旧目录；训练完清中间态。
7. **评估 on-the-fly**：如果 CPU 余量够，验证集甚至训练集可以流式 tokenize（配 collator 里的 padding），省掉整份落盘。代价是可复现性和吞吐，需要权衡。
8. **共享只读缓存**：多机时挂同一个只读缓存目录或对象存储，别每节点一份。

</details>

---

**核心记忆点**：预分词吃磁盘 ≈ $N_{\text{token}} \times \text{bytes\_per\_token} \times \text{份数} \times \dfrac{\text{max\_length}}{\text{avg\_length}}$。上面任何一个乘数没管好，都能单独贡献一个数量级。

# Llama2模型Attention Map绘制方法

### 🧑‍💻 User

以llama2架构为例，如何绘制decoder-only transformer模型不同layer以及不同head的attentionmap？

---

### 🤖 Assistant

好的，这是一个非常深入且实践性很强的问题。我们将以 Llama 2 架构为例，详细讲解如何绘制 Decoder-Only Transformer 模型不同层（Layer）和不同头（Head）的 Attention Map。

### 核心概念理解

1.  **Decoder-Only Transformer**： 像 Llama 2、GPT 系列这样的模型属于此类。它们使用**因果掩码（Causal Mask）**，确保每个位置只能关注到它之前（包括自身）的位置，而不能“窥见未来”。这反映在 Attention Map 上就是一个上三角矩阵（右上角为 `-inf` 或非常小的值，经过 softmax 后变为 0）。
2.  **Attention Map**： 本质上就是经过 Softmax 归一化后的 Attention Weight 矩阵。对于一个给定的头，其大小为 `[target_seq_len, source_seq_len]`。在自回归解码中，`source` 和 `target` 是相同的序列，所以矩阵是方阵。
3.  **获取方式**： 需要在前向传播过程中，从模型的特定层和特定头中“钩取”（hook）或直接返回这个权重矩阵。

---

### 步骤详解

我们将过程分为三个主要步骤：**模型准备**、**数据前向传播与权重抓取**、**可视化绘图**。

#### 第 1 步：模型准备

首先，你需要加载模型和分词器。由于原始 Llama 2 模型的前向传播默认不会返回 Attention Weights，我们需要采取一些方法将其“钩”出来。

**方法一：使用模型的 `output_attentions=True` 参数（推荐且简单）**

Hugging Face Transformers 库中的 Llama 2 实现已经支持在调用模型时直接返回注意力权重。

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
model_name = "meta-llama/Llama-2-7b-chat-hf" # 以 7B-chat 版本为例
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, 
                                         torch_dtype=torch.float16, # 半精度以节省显存
                                         device_map="auto")

# 非常重要：如果tokenizer没有pad_token，将其设置为eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval() # 设置为评估模式
```

**方法二：使用 PyTorch Hook（更底层，控制更灵活）**

如果出于某种原因你需要更底层的控制，可以使用 Hook。这种方法更复杂，但可以访问到中间层的所有变量。

```python
# 定义一个字典来存储抓取到的attention maps
attention_maps = {}

# 定义hook函数
def get_attention_hook(layer_idx, head_idx):
    # 这个hook会在该模块的前向传播完成后被调用
    # module: 该层的Attention模块
    # input: 输入元组
    # output: 输出元组 (通常包含hidden_states, attention_weights, ...)
    def hook(module, input, output):
        # output 通常是一个元组，第二个元素是attention weights
        # attention_weights 的形状: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = output[1]
        # 我们取batch中的第一个样本，和指定的head
        # 使用detach()将其从计算图中分离并移到CPU
        attention_maps[f"layer_{layer_idx}_head_{head_idx}"] = attn_weights[0, head_idx].detach().cpu()
    return hook

# 注册hook。我们需要遍历模型找到所有的attention层。
# LlamaModel的结构是 model.model -> LlamaModel
# 然后 model.model.layers 是一个由 LlamaDecoderLayer 组成的ModuleList
for layer_idx, layer in enumerate(model.model.layers):
    # 每一层的self-attention模块是 layer.self_attn
    # 我们需要钩住它的输出
    hook = get_attention_hook(layer_idx, head_idx=0) # 这里以抓取每个层的第0头为例
    handle = layer.self_attn.register_forward_hook(hook)
    # 如果你想之后移除hook，可以保存handle
```

#### 第 2 步：数据前向传播与权重抓取

现在我们处理输入文本，并将其馈送到模型中。

**使用方法一 (`output_attentions`)：**

```python
# 准备输入
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 前向传播，关键参数：output_attentions=True
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# outputs现在包含一个元组 ‘attentions’
# attentions 是一个元组，长度为 num_hidden_layers
# 每个元素的形状: [batch_size, num_heads, seq_len, seq_len]
all_attentions = outputs.attentions

# 获取第0层的所有头的attention map
# all_attentions[0] 的形状: [1, 32, seq_len, seq_len] (对于Llama-2-7B，有32个头)
layer_0_attentions = all_attentions[0].squeeze(0) # 移除batch维度 -> [32, seq_len, seq_len]

# 获取第5层，第10个头的attention map
layer_idx = 5
head_idx = 10
attention_map_5_10 = all_attentions[layer_idx][0, head_idx].cpu().numpy()
```

**使用方法二 (Hook)：**
在运行前向传播之前，确保你已经注册好了 Hook。

```python
# 清空之前存储的attention maps
attention_maps.clear()

# 准备输入（同上）
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 前向传播
with torch.no_grad():
    outputs = model(**inputs) # 不需要output_attentions=True

# 现在，attention_maps 字典中已经存储了所有注册hook的层的attention数据
# 例如，访问第0层第0头的map
attention_map_0_0 = attention_maps[“layer_0_head_0”].numpy()
```

#### 第 3 步：可视化绘图

使用 `matplotlib` 的 `imshow` 函数来绘制热力图。

```python
def plot_attention_map(attention_matrix, layer_idx, head_idx, tokens=None):
    """
    绘制指定头和层的attention map。
    
    参数:
        attention_matrix: 2D numpy数组 [seq_len, seq_len]
        layer_idx: 层索引，用于标题
        head_idx: 头索引，用于标题
        tokens: 可选的token列表，作为x和y轴的刻度标签
    """
    plt.figure(figsize=(10, 8))
    # 使用imshow显示矩阵，颜色映射用viridis看起来不错
    cax = plt.imshow(attention_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(cax) # 添加颜色条

    # 设置标题和轴标签
    plt.title(f"Attention Map - Layer {layer_idx}, Head {head_idx}")
    plt.xlabel("Source Token Position")
    plt.ylabel("Target Token Position")

    # 如果提供了tokens，设置刻度标签
    if tokens is not None:
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)

    plt.tight_layout()
    plt.show()

# 获取tokenized的文本，用于轴标签
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 绘制我们之前提取的第5层第10头的图
plot_attention_map(attention_map_5_10, layer_idx=5, head_idx=10, tokens=tokens)
```

**结果解读：**
生成的图像是一个方阵。Y 轴是**目标 token**（正在计算注意力的 token），X 轴是**源 token**（被关注的 token）。由于因果掩码，你会发现矩阵的右上三角部分（对应“展望未来”）的值几乎为 0（深色）。对角线及其左下部分则显示出不同的关注模式。明亮的黄色点表示该目标 token 对源 token 有很强的关注。

---

### 完整代码示例（使用方法一）

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import matplotlib.pyplot as plt

# 1. 模型准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, 
                                         torch_dtype=torch.float16,
                                         device_map="auto")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# 2. 数据处理与前向传播
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

all_attentions = outputs.attentions # Tuple of layers

# 3. 可视化特定层和头
layer_to_plot = 0  # 查看第一层
head_to_plot = 0   # 查看第一个头

# 提取attention matrix
# all_attentions[layer_to_plot].shape = [1, num_heads, seq_len, seq_len]
attn_matrix = all_attentions[layer_to_plot][0, head_to_plot].cpu().numpy()

# 4. 绘图
plt.figure(figsize=(10, 8))
plt.imshow(attn_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title(f"Layer {layer_to_plot}, Head {head_to_plot}")
plt.xlabel("Source Token")
plt.ylabel("Target Token")
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.tight_layout()
plt.show()
```

### 进阶技巧

*   **批量绘制多个头/层**： 你可以使用 `plt.subplots` 在一个大图中创建多个子图（subplot），来同时比较同一层的不同头，或者不同层的同一个头。
*   **平均注意力**： 你可以计算某一层所有头的平均注意力 `torch.stack(all_attentions[layer_idx]).mean(dim=1)`，来观察该层的整体关注模式。
*   **分析特定模式**： 观察是否有头专门关注[CLS] token、下一个token、标点符号或长距离依赖。Decoder-only 模型中的头通常有更专门的角色，如“前缀头”、“重复头”等。

通过以上步骤，你就可以有效地可视化和分析 Llama 2 这类 Decoder-Only 模型内部的神秘 Attention 机制了。
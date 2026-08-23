# Jinja2聊天模板

### 🧑‍💻 User

jinja2模板常见操作符有哪些，llm聊天模板chat template如何编写？

---

### 🤖 Assistant

下面分两大部分为你梳理：**Jinja2 模板常用操作符** 和 **LLM 聊天模板（Chat Template）的编写方法**。

---

## 一、Jinja2 模板常见操作符

Jinja2 的表达式语法与 Python 高度相似，以下是常用操作符分类：

### 1. 比较运算符
`==`、`!=`、`<`、`>`、`<=`、`>=`

```jinja2
{% if user.role == "admin" %}
  欢迎管理员
{% endif %}
```

### 2. 逻辑运算符
`and`、`or`、`not`

```jinja2
{% if age > 18 and age < 60 %}
  有效年龄
{% endif %}
```

### 3. 数学运算符
`+`、`-`、`*`、`/`、`//`（整除）、`%`（取模）、`**`（幂）

```jinja2
{{ total + 10 }}
{{ 2**3 }}  {# 输出 8 #}
```

### 4. 字符串连接符
`~`：将操作数转为字符串后拼接

```jinja2
{{ "Hello " ~ name ~ "!" }}
```

### 5. 成员与身份运算符
- `in`：成员测试（可用于字符串、列表、字典等）
- `is`：身份/类型测试（结合测试器使用，如 `is defined`、`is none`）
- `is not`：否定测试

```jinja2
{% if "error" in message %}
  报错
{% endif %}
{% if var is defined %}
  已定义
{% endif %}
```

### 6. 过滤器管道 `|`
严格说是过滤器应用符，而非运算符，但极其重要。

```jinja2
{{ name | upper }}
{{ list | join(', ') }}
{{ value | default("默认值") }}
```

### 7. 索引与属性访问
- `[ ]`：下标访问
- `.`：属性/字典键访问

```jinja2
{{ messages[0] }}
{{ user.name }}
```

### 8. 三元表达式（内联 if）
```jinja2
{{ "成人" if age >= 18 else "未成年" }}
```

### 9. 赋值（不是运算符，但在模板中常用）
使用 `{% set %}` 创建变量。
```jinja2
{% set system_message = messages[0].content %}
```

### 10. 空白控制语法标记
- `{{- ... }}` 或 `{%- ... %}`：去除该侧空白（不是运算符，但影响输出格式，在聊天模板中大量使用）。

---

## 二、LLM 聊天模板（Chat Template）编写指南

### 1. 什么是 Chat Template？
在 Hugging Face `transformers` / `tokenizers` 中，`chat_template` 是一个 **Jinja2 模板字符串**，用于将标准的对话结构（`messages` 列表）转换为模型训练时使用的 **原始文本格式**。
通过 `tokenizer.apply_chat_template(messages, tokenize=False)` 即可得到格式化后的字符串。

### 2. 模板中可用的上下文变量
模板运行时，会自动注入以下变量（取决于库的实现，通常至少包含）：

| 变量名 | 说明 |
|--------|------|
| `messages` | 对话列表，每个元素是字典：`{"role": "...", "content": "..."}` |
| `add_generation_prompt` | 布尔值，是否需要添加表示“模型开始生成”的提示，如 `<|assistant|>` |
| `bos_token` | 开始 token（可在模板中直接使用，但常用 `{{ bos_token }}` 输出） |
| `eos_token` | 结束 token |
| `raise_exception` | Jinja2 过滤器，用于抛出错误，常用于验证消息格式 |

此外，`messages` 中的字典还可能包含 `tool_calls`、`name` 等额外字段。

### 3. 编写模板的常见模式

一个典型的聊天模板会做三件事：
1. **提取并可能单独处理系统消息**（放在循环外或特殊前缀）
2. **循环遍历消息**，根据 `role` 输出对应的格式头和内容
3. **若 `add_generation_prompt` 为真，追加生成提示**

#### 示例：一个类 ChatML 格式的模板
```jinja2
{%- if messages[0]['role'] == 'system' -%}
    {{- '<|system|>\n' + messages[0]['content'] + '<|end|>\n' -}}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set loop_messages = messages -%}
{%- endif -%}

{%- for message in loop_messages -%}
    {%- if message['role'] == 'user' -%}
        {{- '<|user|>\n' + message['content'] + '<|end|>\n' -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- '<|assistant|>\n' + message['content'] + '<|end|>\n' -}}
    {%- else -%}
        {{ raise_exception('Unknown role: ' + message['role']) -}}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {{- '<|assistant|>\n' -}}
{%- endif -%}
```

#### 输出效果（假设输入）
```python
messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"}
]
```
模板渲染后生成：
```
<|system|>
你是一个助手<|end|>
<|user|>
你好<|end|>
<|assistant|>
```

### 4. 关键编写技巧

- **空白控制**：大量使用 `-%}` 和 `{{- ...` 来避免多余的换行和空格，保证生成的 token 序列与训练时一致。
- **系统消息**：许多模型要求系统消息在对话开头以特殊 token 包裹，模板需要灵活处理第一条消息为 `system` 的情况。
- **特殊 token 处理**：可以直接通过 `{{ bos_token }}` 和 `{{ eos_token }}` 引用 tokenizer 中定义的 token，无需硬编码。但有时训练时直接使用了固定的特殊标记，需要查阅模型文档。
- **错误校验**：`{{ raise_exception('...') }}` 可在遇到不支持的角色时抛出异常，避免静默错误。
- **工具调用（Function Calling）**：如果模型支持工具调用，模板中可能需要判断 `'tool_calls' in message`，并循环输出 `tool_calls` 内容。例如：
  ```jinja2
  {%- if 'tool_calls' in message -%}
      {{- '<|tool_calls|>\n' -}}
      {%- for tool in message['tool_calls'] -%}
          {{- tool['function']['name'] + '\n' + tool['function']['arguments'] + '\n' -}}
      {%- endfor -%}
      {{- '<|end_tool_calls|>\n' -}}
  {%- endif -%}
  ```

### 5. 如何使用和注册 Chat Template

- **查看/设置模板**：`tokenizer.chat_template` 是一个字符串属性，可以直接赋值。
  ```python
  tokenizer.chat_template = """
  {%- for message in messages %}
      {%- if message['role'] == 'user' %}{{ '<|user|>' + message['content'] + '<|end|>' }}
      {%- elif message['role'] == 'assistant' %}{{ '<|assistant|>' + message['content'] + '<|end|>' }}
      {%- endif %}
  {%- endfor %}
  {%- if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}
  """
  ```

- **调用模板**：
  ```python
  # 返回字符串
  prompt = tokenizer.apply_chat_template(messages, tokenize=False)
  # 返回 token ids
  input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
  ```

### 6. 常见预定义模板参考
很多开源模型已经自带模板，可直接查看其 `tokenizer_config.json` 中的 `chat_template` 字段，例如：
- **Llama 2**：使用 `[INST] ... [/INST]` 包裹用户消息，`<<SYS>>` 包裹系统消息
- **Mistral**：使用 `<s>[INST] ... [/INST]</s>` 结构
- **ChatML**：`<|im_start|>role\n content <|im_end|>`

编写自定义模板时，可以仿照这些结构，**最重要的原则是保持与模型训练时完全一致的文本格式和特殊 token 顺序**。
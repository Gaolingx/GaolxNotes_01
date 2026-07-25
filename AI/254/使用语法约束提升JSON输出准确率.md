# 使用语法约束提升JSON输出准确率

### 🧑‍💻 User

为了提高json输出准确率，对于结构化字段输出内容可以提示模型走<tool_call>输出的json保证语法对吗？（激活inference语法约束采样）

---

### 🤖 Assistant

简单来说：**不能仅靠提示词来激活底层的语法约束采样**。

虽然提示模型“保证JSON语法正确”或“输出合法JSON”能在一定程度上降低错误率，但这属于**概率性引导**，无法实现100%的结构化保证。模型本质上仍在进行概率采样，存在输出非法token的可能性。

要真正激活“推理时语法约束采样”，必须在**API参数**或**推理引擎层面**进行配置，通过强制掩码来限制输出空间。

以下是具体的原理分析与实现方案：

### 1. 提示词 vs 语法约束采样

| 机制 | 原理 | 准确率 | 实现方式 |
| :--- | :--- | :--- | :--- |
| **提示词引导** | 通过指令让模型学习到需要输出JSON，在高概率空间内生成结构。 | ~90% - 99% (不稳定) | 在Prompt中加入“请输出合法JSON”。 |
| **语法约束采样**<br>(Constrained Decoding) | 在每一步生成时，将不符合JSON语法的token logits设为$-\infty$，强制只采样合法token。 | **100%** | 设置API参数或配置推理引擎的Grammar。 |

### 2. 如何真正激活语法约束采样

要实现100%准确的JSON输出，必须根据你使用的大模型类型采取以下特定方法：

#### 方案 A：使用商业 API (如 OpenAI / Anthropic)
商业模型通常封装了底层的约束采样逻辑，必须通过特定的请求参数来激活，而不是通过提示词。

*   **OpenAI (GPT-4o等)**:
    使用 `response_format` 参数。这会激活模型内置的JSON模式或结构化输出功能。
    ```python
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
      model="gpt-4o-2024-08-06",
      messages=[{"role": "user", "content": "提取信息..."}],
      response_format={
          "type": "json_schema",
          "json_schema": {
              "name": "info_extraction",
              "schema": {
                  "type": "object",
                  "properties": {
                      "name": {"type": "string"}
                  },
                  "required": ["name"]
              }
          }
      }
    )
    ```

*   **Anthropic (Claude 3.5 Sonnet等)**:
    Claude 官方推荐使用 **Tool Use (工具调用)** 来保证结构化输出，或者通过预填充（Prompt Pre-filling）技术强制模型以 `{` 开头。
    ```python
    # 预填充示例
    messages=[
        {"role": "user", "content": "返回用户信息JSON"},
        {"role": "assistant", "content": "{"} # 强制模型接着写JSON
    ]
    ```

#### 方案 B：使用开源模型 (如 Llama-3, Qwen, GLM)
如果使用开源模型，需要推理引擎支持。以下以常用的 `vLLM` 和 `llama.cpp` 为例：

*   **vLLM 框架**:
    vLLM支持通过 `guided_json` 参数激活语法约束。这底层使用了 `outlines` 或 `xgrammar` 库进行logits掩码。
    ```python
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
  
    # 这才是激活语法约束的正确姿势
    output = llm.chat(
        messages=[{"role": "user", "content": "提取信息"}],
        sampling_params=sampling_params,
        chat_template_kwargs={"add_generation_prompt": True},
        guided_json={
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
    )
    ```

*   **llama.cpp / Ollama**:
    llama.cpp 使用 **GBNF** (GGML BNF) 语法定义约束。Ollama 则支持在请求时传入 `format: "json"` 或具体的 JSON Schema。
    ```json
    {
      "model": "llama3",
      "prompt": "...",
      "format": {
        "type": "object",
        "properties": {
          "name": { "type": "string" }
        }
      }
    }
    ```

<details>
<summary><b>3. 推荐的最佳实践：提示词 + 引擎约束</b></summary>

虽然引擎层能保证语法正确，但结合良好的提示词能确保**语义正确**（即字段内容符合预期）。推荐组合使用：

1.  **系统提示词**：明确任务目标和字段定义。
2.  **引擎约束**：保证输出的是合法JSON。

**提示词模板示例：**
```text
你是一个信息提取专家。请从用户输入中提取结构化信息。

要求：
1. 必须输出合法的JSON格式。
2. 不要输出任何解释性文字或Markdown标记。
3. 字段定义：
   - `name`: 字符串，人名
   - `age`: 整数，年龄

用户输入：{input}
```
配合引擎层的 `guided_json` 或 `response_format` 使用，即可达到最高准确率。

</details>

**总结：** 提示模型“保证语法对”是有效的**辅助手段**，但若要追求工业级的稳定性，必须依赖 API 的 `response_format` 或推理引擎的 `guided_json` 参数来激活底层的语法约束采样。
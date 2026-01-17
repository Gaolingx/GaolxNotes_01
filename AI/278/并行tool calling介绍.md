# 并行tool calling介绍

### 🧑‍💻 User

什么是并行tool calling？在coding agent场景，这可以将延迟降低多少？

---

### 🤖 Assistant

**并行 Tool Calling (Parallel Tool Calling)** 是一种大语言模型（LLM）的功能特性，它允许模型在**单次**响应中生成多个独立的工具调用（Function Calls），而不是一次只生成一个。

随后，Agent 运行时（Runtime）可以同时执行这些工具，并将所有结果一次性返回给模型。

### 1. 什么是并行 Tool Calling？

为了直观理解，我们对比一下**串行（Sequential）**和**并行（Parallel）**的流程：

#### 🚫 串行模式 (Sequential)
模型必须通过多次对话轮次（Round Trips）来完成多个任务。

1.  **User:** "查询 A 公司和 B 公司的股价。"
2.  **LLM:** 调用 `get_stock_price("A")`
3.  **System:** 执行函数 -> 返回 "100"
4.  **LLM:** (接收结果，思考) -> 调用 `get_stock_price("B")`
5.  **System:** 执行函数 -> 返回 "200"
6.  **LLM:** "A公司股价100，B公司股价200。"

*耗时 = 2次 LLM 推理 + 2次 网络请求 + 2次 工具执行时间*

#### ✅ 并行模式 (Parallel)
模型在一次思考中决定同时做两件事。

1.  **User:** "查询 A 公司和 B 公司的股价。"
2.  **LLM:** 同时调用 `get_stock_price("A")` 和 `get_stock_price("B")`
3.  **System:** **并发**执行这两个函数 -> 同时返回 "100" 和 "200"
4.  **LLM:** "A公司股价100，B公司股价200。"

*耗时 = 1次 LLM 推理 + 1次 网络请求 + Max(工具执行时间)*

---

### 2. Coding Agent 场景中的应用

在编程助手（Coding Agent）场景中，并行 Tool Calling 极其关键，因为编程任务通常包含大量的**信息收集（Information Retrieval）**步骤，这些步骤往往是相互独立的。

#### 典型场景：
1.  **多文件读取**：用户问 "理解一下 `auth/` 目录下的逻辑"，Agent 需要读取 `login.py`, `register.py`, `token.py`。
    *   *并行：* 一次性发出 3 个 `read_file` 请求。
2.  **跨文件搜索**：用户问 "哪里用到了 `User` 类？"，Agent 需要在多个目录执行 `grep` 或语义搜索。
3.  **运行独立测试**：修改代码后，同时运行 Lint 检查（`run_linter`）和单元测试（`run_tests`）。

---

### 3. 延迟可以降低多少？(定量与定性分析)

延迟降低的幅度取决于任务的性质，但在**信息收集阶段**，提升通常是巨大的。

设：
*   $T_{LLM}$ = LLM 生成一次 Token 的耗时（通常 1~3秒）
*   $T_{Tool}$ = 工具执行耗时（IO操作，如读文件通常很快，网络搜索较慢）
*   $N$ = 需要调用的工具数量

#### 数学模型对比

**串行总耗时 ($T_{seq}$):**
$$ T_{seq} \approx N \times (T_{LLM} + T_{Tool}) $$
(注：这里忽略了 System Prompt 的处理开销，实际上随着 Context 变长，串行模式后期的 $T_{LLM}$ 会增加)

**并行总耗时 ($T_{par}$):**
$$ T_{par} \approx T_{LLM} + \max(T_{Tool\_1}, T_{Tool\_2}, ..., T_{Tool\_N}) $$

#### 具体案例分析：Coding Agent 读取 5 个文件

假设 Agent 需要读取 5 个代码文件来回答问题。
*   $T_{LLM}$ (思考调用哪个文件) ≈ 1.5秒
*   $T_{read}$ (读取文件IO) ≈ 0.1秒 (本地) 或 0.5秒 (云端API)

**串行模式：**
需要 5 个对话轮次。
$$ \text{Time} \approx 5 \times (1.5s + 0.5s) = 10 \text{秒} $$

**并行模式：**
1 个对话轮次，同时发出 5 个调用。
$$ \text{Time} \approx 1.5s + \max(0.5s) = 2 \text{秒} $$

#### 结论：延迟降低幅度

1.  **减少 LLM 推理轮次 (Round Trips)**：这是最大的收益来源。LLM 生成是昂贵的，减少 $N-1$ 次推理可以**降低 50% 到 80% 的延迟**。
2.  **I/O 并发**：如果是网络请求（如搜索文档）或耗时较长的测试，并发执行可以将 $Sum(T_{exec})$ 变为 $Max(T_{exec})$。
3.  **实际体感**：
    *   对于**信息收集（Gathering）**阶段：通常可以获得 **3倍 ~ 5倍** 的速度提升。
    *   对于**复杂调试（Sequential Logic）**阶段（即步骤 B 依赖步骤 A 的结果）：无法并行，**0% 提升**。

### 总结

在 Coding Agent 中，**并行 Tool Calling 可以将信息检索类任务的延迟降低 60%-80%**。

它将原本线性的 "$O(N)$ 复杂度" 交互变成了接近 "$O(1)$ 复杂度" 的交互（就 LLM 轮次而言）。对于用户来说，这意味着在询问 "请帮我看看项目里的所有配置文件" 时，等待时间从 20 秒缩短到了 5 秒左右。
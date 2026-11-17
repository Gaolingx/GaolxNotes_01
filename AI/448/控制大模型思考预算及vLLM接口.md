# 控制大模型思考预算及vLLM接口

## 🧑‍💻 User

最近发现了一个可以控制llm思考预算的办法
llm的decode本质不就是ar吗
你可以让模型提前输出话甚至可以直接截断当前cot补上</think>然后输出答案
之前gemini网页端那个立即回答估计也是这么干的
对于自部署模型到底能不能这么干
vllm有没有提供接口给前端控制

---
## 😀 默认助手 (7)

`. They mention Gemini web's "立即回答" (answer now) button. They ask: can this be done on self-deployed models, and does vLLM provide an interface for frontend control?

This is a technical discussion question. I should answer substantively. Let me think about what I know about vLLM's APIs.

Key points:

1. Truncation + appending `` — this is "hard" injection. Whether it works depends on the model's chat template and how the reasoning parser handles it. For self-deployed, you control the tokenizer/chat template, so you can literally inject tokens.

2. vLLM specifics:
   - vLLM has `--reasoning-parser` option (added ~v0.10.x), with parsers like `deepseek_r1`, `deepseek_v3`, `qwen3`, `glm45`, `hunyuan_a13b`, `granite`, `holo2`, `granite`, etc. This parses reasoning_content out of the output.
   - The reasoning parser has a `force_reasoning` flag? Actually, in vLLM's reasoning parser implementation, there is `ReasoningParser` base class with `is_reasoning_end`, and some parsers support `force_reasoning` mode.
   - vLLM OpenAI server supports `chat_template_kwargs` in the request — e.g. `{"enable_thinking": false}` for Qwen3, `{"thinking": {"type": "disabled"}}` for some models. That's the "soft" control.
   - vLLM supports `min_tokens`? There's `min_tokens` in SamplingParams — but that's a floor, not ceiling.
   - For hard control, vLLM does NOT have a built-in "inject tokens mid-generation" API. But:
     - There is `--enable-prompt-embeds` / prompt embeds? Supports embedding input.
     - There is the `TokensPrompt` with lists of token IDs — you can append the `` token IDs directly to the prompt.
     - vLLM v1 has `structured_outputs` / `guided_decoding` (xgrammar, guidance, outlines) — you could use a grammar to force the model to emit `` after a certain number of tokens? Not directly length-based.
   
3. Actual approaches to token-budget-limited thinking:
   - **Token-level max_tokens with continuations**: Not exactly.
   - **vLLM `SamplingParams` has `max_tokens`** — but that truncates the whole output including the answer.
   - **Two-call approach**: call 1 with `max_tokens=N` (thinking budget), then call 2 with the truncated CoT as prefix + `` appended to prompt, and ask for the answer. This is the "budget forcing" approach from s1 paper ("budget forcing" — s1: Simple test-time scaling). Indeed the s1 paper introduced "budget forcing": either force the model to stop thinking by appending "End of thinking" or truncate and inject ``.
   - **Prefix continuation / return logits**: vLLM has `echo`, and `prompt_logprobs`. Also `vllm` supports `--enable-prefix-caching`. And there's the `completion` API with `prompt` as token IDs, plus `add_special_tokens=False`.
   - vLLM has "prompt adapter"? No.
   - **`--enable-auto-tool-choice`** irrelevant.

4. Actually vLLM does have something relevant: **`ignore_eos`** and **`min_tokens`**. And `stop` strings. And `include_stop_str_in_output`. So if the model emits ``, you can set `stop=[""]`. But you can't force the model to emit it.

5. There's also **vLLM's `--reasoning-parser` + `chat_template_kwargs`** which is the cleanest "thinking budget" control.

6. Newer: vLLM supports **`max_thinking_tokens`**? Hmm. Let me think. Actually I recall in vLLM there was discussion / PR for "reasoning budget". SGLang has `think_end_id` and... Actually SGLang has a feature: `--reasoning-parser` and for DeepSeek it supports forcing. SGLang's `Engine` has `force_reasoning`? I'm not 100% sure.

Actually — I recall **SGLang** added `think_end_id` support, and there's a "reasoning budget" / "adaptive thinking". Hmm, I should be careful about claiming specifics I'm not sure of.

Let me be honest about uncertainty and structure the answer around:

**A. 原理层面 — 截断 + 补 `` 到底行不行？**
- AR 模型下，答案是 prompt 的函数。所以"截断 CoT + 补 ``"等价于构造了一个新 prompt = 原始 prompt + 部分 CoT + ``。这完全合法，模型会继续生成。
- 但要注意几点:
  1. **chat template 决定一切**。`` 必须和模板里训练的终止符一致（token id 层面）。比如 DeepSeek-R1 用 ``，Qwen3 用 ``，但有的模型用 `<|/think|>`、`<thought>`、`</thought>` 等。补错 token 会让模型输出垃圾。
  2. **强行截断会让 CoT 中断在句子中间**，模型可能"以为"自己还没想完，继续"想"而不是"答"——表现为在答案里继续反思。s1 论文的 budget forcing 正是观察到了这点，所以他们的做法是**先插入一个"过渡短语"**（如 "\n\nConsidering the limited time by the user, I have to give the solution based on the simplest and most direct approach being attempted above.\n\n"）再补 ``。这是个很实用的 trick。
  3. **答案质量损失**：CoT 被砍，答案可能不完整。s1 的结论是：budget forcing 效果随预算单调上升，直到某个点，说明"多想想"确实有用。
  4. **不能反向**（"多想想"）除了强行续写 "Wait"。
- 这就是 **budget forcing**（s1, 2025）。另外还有 **"Wait" injection** 来延长思考。

**B. 自部署能不能这么干？能，而且比 API 更自由。**
关键是自己掌控 tokenizer 和 chat template，可以直接构造 token 序列。

**C. vLLM 提供什么接口？**

分几层：

1. **软控制（官方支持，最省事）**: `chat_template_kwargs`
   - Qwen3: `{"enable_thinking": false}` / true
   - 很多模型有类似开关
   - vLLM 会把这个 dict 透传给 chat template 的 Jinja 渲染
   - 局限：只能全开或全关，没有"预算"

2. **max_tokens + stop**：
   - `max_tokens=N` 只会把整段输出截断，不会自动收尾
   - `stop=[""]` 让生成到 think 结束就停 —— 但这不是你要的

3. **Prompt 层面塞 token（真正实现截断+补 ``）**：
   - vLLM 的 `/v1/completions` 支持 `prompt` 直接传 **token id 列表**（`TokensPrompt`），配合 `add_special_tokens=False`
   - 或者用 `tokenizer.encode(..., add_special_tokens=False)` 得到 CoT 部分 token，拼上 `` 的 id，再作为 prompt 传
   - 注意：`/v1/chat/completions` 走 chat template 渲染，没法直接拼 token；要绕到 completions 端点，或自己渲染完整 prompt 字符串后传（这时用 completions + raw string 也行，但要保证 tokenizer 不重新加 BOS）
   - vLLM 还支持 `--enable-prompt-embeds`，可以传 `prompt_embeds`，但对文本注入没必要

4. **多轮/续写**：
   - 把上一轮的 assistant 输出（截断版 + ``）塞回 messages 当 assistant 消息 —— 但 chat template 通常会把 assistant 消息包在 `<|assistant|>` 里，导致重复；正确做法是构造 prompt = 原 prefix + assistant 开头 + 截断 CoT + `` + `\n\n`，然后从 completions 续写
   - 这就是 "prefill"/"prefix continuation"

5. **结构化输出（强制 token）**：
   - `guided_decoding` / `structured_outputs`（xgrammar/guidance/outlines）
   - 可以构造 grammar 强制输出以 `` 开头？技术上可以让生成的第一个 token 序列是固定的 —— 用 `guided_regex` 或 `guided_choice`，但只能约束**生成**的内容，不能帮你"截断前文再续"。不过可以配合上面的 prompt 注入。
   - 更 hack 的：用 `guided_regex` 让模型必须在 N 个 token 内输出 ``？regex 不好做长度约束。可以用 `guided_json` 不适用。

6. **vLLM 的 reasoning parser**（`--reasoning-parser qwen3` / `deepseek_r1`）
   - 输出里把 `reasoning_content` 和 `content` 分开
   - 部分 parser 有 `force_reasoning` 选项
   - 这只是**解析**，不是**控制**

7. **有没有原生的 "thinking budget" 参数？**
   - 截至我了解的版本（vLLM 0.10/0.11 附近），**OpenAI 兼容层没有** `thinking_budget` 这种参数
   - 但有 `SamplingParams` 的 `max_tokens`、`min_tokens`（vLLM 有 `min_tokens`？我记得是有的，用于避免过早 EOS）。是的，vLLM `SamplingParams` 有 `min_tokens`。
   - 社区里有 PR / 讨论在做 reasoning budget（如 `reasoning_max_tokens`），但不能保证已合并 —— 我应该标注不确定

8. **更上层框架的现成实现**：
   - **SGLang**：有 reasoning parser + 对 `think_end_id` 的处理；我记得 SGLang 支持 `custom_params` 或 `forced thinking`。不确定具体 API 名。
   - **llama.cpp / Ollama**：可以用 GBNF grammar 强制，或直接构造 prompt
   - **Transformers**：`StoppingCriteria`、`LogitsProcessor`、`PrefixLM` 风格的 `assistant_prefix`；最灵活
   - **s1 官方实现**：就是 transformers + vLLM，用 prompt 拼接

**D. 工程上的坑**
1. tokenizer 边界：不能对文本做 `str += ""` 然后重新 encode —— 需要保证 token 化后与原始 CoT 的 token 完全一致（否则 prefix 变了）。正确：`ids = tok(prompt).input_ids + tok(cot_part).input_ids + tok("").input_ids`，或者用 `add_special_tokens=False` 分段编码。BPE 边界问题会导致最后一个 token 合并，所以**必须按 token 拼接而不是按字符串拼接**。
2. 模型不认识 `` 的语义位置（比如 CoT 里本来就有），可能出现"我还没输出 ``"的自我矛盾
3. 注入后模型的 **attention 会看到不自然的边界**，可能出现格式崩坏（直接输出 `` 后又重复 ``）
4. `max_tokens` 之后必须留够答案空间，否则答案是空的
5. **beam/采样**：vLLM 的 `temperature`、`top_p` 在续写时照常
6. **缓存**：prefix caching 会让第二次调用几乎免费（`--enable-prefix-caching` 默认开），这是实现 "budget forcing 两段式" 的性能关键
7. **`continue_final_message`**：vLLM 的 chat API 有个参数 `continue_final_message`（配合 `add_generation_prompt=False`）——这个正好可以用于"续写已有 assistant 消息"。这个我记得存在，是从 OpenAI/vLLM 对齐的。

Actually yes — vLLM's `ChatCompletionRequest` does have `continue_final_message` and `add_generation_prompt`. That's directly useful: you can put the truncated CoT + `` as the final assistant message and set `continue_final_message=True`, `add_generation_prompt=False`. That's the cleanest chat-API-level way.

Hmm, but does the chat template then work? `continue_final_message=True` means the template doesn't add the assistant prefix/suffix for the final message — it just renders it and continues. That's exactly what we want.

**E. 更优雅的替代：让模型自己学会"预算"**
- RL 训练 / 系统提示 "think no more than N tokens"（效果不稳定）
- **adaptive thinking**（Claude 的 thinking budget）—— 训练时就限制了
- Qwen3 的 `enable_thinking=False` 走的是 chat template 里注入 "empty thinking block"（`<|im_start|>assistant\n\n\n` 之类），本质也是 prompt 注入
- 有些模型支持 `/no_think` 软开关

**F. 具体代码示例**

I should give a concrete vLLM example. Let me write:

```python
# 两段式 budget forcing with vLLM
import vllm, transformers

llm = vllm.LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tok = llm.get_tokenizer()

THINK_END = tok.encode("", add_special_tokens=False)  # critical: no specials
```

Wait — need to be careful: `` might be a special token in DeepSeek's vocab. In DeepSeek-R1 tokenizer, `` is a regular string (actually DeepSeek uses `<|end_of_sentence|>` etc; `` is plain text). For Qwen3, `` is also plain text. OK.

Better to mention checking `tok.convert_tokens_to_ids` / whether special.

```python
# Stage 1: 给定思考预算
prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
out = llm.generate(vllm.TokensPrompt(prompt_token_ids=prompt_ids),
                   vllm.SamplingParams(max_tokens=BUDGET, temperature=0.6,
                                       ignore_eos=True))  # 强制用满预算
cot_ids = out[0].outputs[0].token_ids
```

Hmm, `ignore_eos=True` so it doesn't stop early. But then it may run past `` naturally. Actually with budget you want it to stop at budget. Use `max_tokens=BUDGET` and `ignore_eos=False`; if model finishes thinking early it's fine.

Actually for budget forcing you want `ignore_eos=True` only in the "force longer" direction. For "force shorter", you truncate at N tokens.

```python
# Stage 2: 截断 + 注入  + 强制过渡语
TRANSITION = "\n\nOkay, I've thought enough. Final answer:\n\n"
inject_ids = tok.encode(TRANSITION + "\n\n", add_special_tokens=False)
forced = prompt_ids + cot_ids + inject_ids
```

Then generate with `max_tokens` for the answer.

Note the important subtlety: `tok.encode(cot_text)` vs `cot_ids` — must use the raw token ids from stage 1, because re-encoding the decoded text may produce a different tokenization (especially at the boundary). Good point to emphasize.

Also mention: for chat API path:

```bash
curl localhost:8000/v1/chat/completions -d '{
  "model": "...",
  "messages": [
    {"role":"user","content":"..."},
    {"role":"assistant","content":"<truncated cot>\n\n"}
  ],
  "continue_final_message": true,
  "add_generation_prompt": false,
  "max_tokens": 1024
}'
```

Hmm, actually with `continue_final_message=true`, vLLM sets `add_generation_prompt=false` automatically I think. Let me just present it.

Also `chat_template_kwargs`:
```json
{"model":"qwen3", "messages":[...], "chat_template_kwargs": {"enable_thinking": false}}
```

Also mention `--reasoning-parser`:
```
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3
```

**G. 关于 Gemini 网页端的"立即回答"**
- 大概率是 Google 自己的 serving 做了类似东西，但更可能是训练时就有的能力（"skip thinking" 的 control token），或者简单地切换到 non-thinking 模式重新生成
- Gemini 的 thinking budget 是 API 参数 `thinkingConfig.thinkingBudget`（-1 dynamic, 0 disable）；网页端按钮可能只是切到 0 预算
- 实际上 Gemini 的 thinking budget 是产品化训练出来的，不是纯 prompt hack

**H. 关于"提前输出话"**
- 用户说的"让模型提前输出话"可能指 few-shot / 提示它 "早点收尾"。这个方法效果一般，因为模型对自己 token 计数不敏感。

Let me also mention the **"Wait" 反向操作** and the **answer-first-then-verify** pattern.

Also mention limitations: **截断+注入会破坏 KV cache 之外的语义连续性**，以及 **对所有 CoT 都一样强，不能自适应**。

And mention there are papers: s1 (budget forcing), "Chain of Draft", "Token-budget-aware reasoning", "Think Prune", L1 (length-controlled policy optimization), etc.

Let me also address whether there's an official vLLM API: I should say **no native `thinking_budget` param** as of my knowledge, but there are three composable primitives. I'll flag version uncertainty.

I should also mention **`min_tokens`** and **`stop`** and **`include_stop_str_in_output`**.

Also — the SGLang angle: SGLang recently (v0.4.x/0.5.x) does have reasoning parser + a `think_end_id` skipping feature, and I believe there's an API for `forced reasoning`? Hmm. I'll mention SGLang has `--reasoning-parser` and that some forks implement forced think-end, but flag uncertainty.

Let me also think about whether vLLM has `bad_words`/`logit_bias` — yes! `SamplingParams` has `bad_words` and `logit_bias`. You could use **`bad_words`** to ban the `` token? That's the opposite. Or... interesting: you can't force a token, only ban. But there's `guided_regex` to force.

Actually a neat trick: **grammar-forced ``**: use `guided_regex` with a pattern that starts with the literal ``? No, that constrains from token 0. Hmm, actually you could do: the prompt already ends mid-CoT, and use `guided_choice` won't work.

But there IS a clean trick: since we control the prompt, injecting the token is simpler than grammar. Grammar is useful for the *reverse*: forcing the model to *emit* `` when it hasn't — no.

OK. Another important primitive: **vLLM's `--enable-prefix-caching`** makes the two-stage approach cost ~nothing extra because stage 2 reuses stage 1's KV cache prefix. Emphasize this — it makes budget forcing cheap in production.

Also: **`logprobs` and `prompt_logprobs`** can tell you the probability of `` at each step, useful for adaptive decisions.

Let me now also mention a cleaner "single-call" approach: **stop token trick with a capped max_tokens, then a cheap "continuation" call**. Or: run with `max_tokens=BUDGET` and `stop=[""]`; if it hit the stop, great; if it hit max_tokens, do stage 2.

Also note: some models already have "thinking budget" via the chat template — e.g., Qwen3's template accepts `enable_thinking`; DeepSeek-V3.1 accepts `thinking` mode; GLM-4.5 has modes. vLLM passes `chat_template_kwargs`.

Let me write an answer in Chinese, well-structured with Markdown, code fences, and collapsible sections for less critical detail.

Structure:
1. 结论先行 (TL;DR)
2. 原理：为什么可行 / 有哪些坑（这是 budget forcing）
3. 自部署：完全可行，而且比 API 自由 — 关键在于 token 级拼接
4. vLLM 到底提供什么（分层表格）
   - 原生 interface 清单
   - 没有的
5. 具体实现（三段代码：chat_template_kwargs / token 拼接两段式 / chat API continue_final_message）
6. Gemini 网页端那个按钮到底是什么
7. 工程坑清单
8. 相关论文/参考

I need to be careful about accuracy and flag uncertainty where I have it. Let me not over-claim exact version numbers.

Let me think about `continue_final_message` in vLLM more. In vLLM's `ChatCompletionRequest`, there are fields `add_generation_prompt: bool = True` and `continue_final_message: bool = False`. And in the chat template application, vLLM's `apply_hf_chat_template` / `resolve_chat_template_content_format`... Actually vLLM has its own `chat_utils.py` with `load_chat_template` and it calls `tokenizer.apply_chat_template(conversation, add_generation_prompt=..., continue_final_message=...)`. Jinja chat templates support `continue_final_message` via `tokenizer.apply_chat_template` in transformers >= 4.44ish. Yes, transformers added `continue_final_message` support. Good.

Caveat: not all chat templates support it — transformers implements it generically by trimming. Actually transformers' `apply_chat_template` with `continue_final_message=True` requires the template to render the final message; it works by... hmm, it's implemented in the template rendering loop, but I believe transformers does it generically by rendering with `add_generation_prompt=False` and then the template needs to not add extra tokens. Let me not overclaim — I'll say "取决于 chat template 是否支持，transformers 侧有通用实现，但部分模板会加额外 token 导致重复".

Actually, in vLLM there's a known caveat: `continue_final_message` with reasoning models can produce a duplicated `` if the template appends one. So better to just strip it.

Let me also give a robust "raw completions + token ids" recipe as the most reliable.

One more: vLLM `/v1/completions` with `prompt` as `list[int]` — supported. And `add_special_tokens` param? vLLM's `CompletionRequest` has `add_special_tokens: bool = True`. Hmm, actually for token-id prompts, special tokens are already in the ids. I'll mention setting it False to be safe.

Also `truncate_prompt_tokens` — not needed.

Let me also mention **`skip_special_tokens`** for output decoding.

Also I should mention there's a totally different and arguably better approach: **让模型输出一个"预算感知"的 CoT**，或者用 **draft-then-verify**。And mention **"Think Prune"/"Chain of Draft"**.

Let me also mention **vLLM 的 `--max-model-len` / `max_num_batched_tokens`** irrelevant.

One more real thing: vLLM has **`SamplingParams.repetition_penalty`** irrelevant.

And there's **`--speculative-config`** irrelevant.

OK, one genuinely relevant thing: **`vllm.LLMEngine.add_request` with `prompt_token_ids`** — same as above.

Also **vLLM `structured_outputs` with `guided_regex`** — could force the *answer* format after injection, preventing the model from re-entering thinking. E.g., `guided_regex = r"(?s).*"` no. But you could ban `<` in the first tokens? Use `bad_words=["<"]`? Hmm, you could set `bad_words` to ban `` so it can't re-emit it. That's a nice practical safeguard! Yes — `bad_words=[""]` prevents the model from emitting another think-close token. Good tip.

Hmm, `bad_words` in vLLM is a list of strings that get tokenized and banned. Works.

Let me write the answer now. Chinese, technical, well-organized. Not too long but thorough. Use tables and collapsible sections.

I should also directly answer: "vllm有没有提供接口给前端控制" → 没有一键的 `thinking_budget` 参数；但有三层可组合的原语，加上 token-level prompt 拼接就能实现。并指出前端要做的其实就是两段式编排。

Let me write it.

Let me make sure about a few facts I'll state:
- vLLM `--reasoning-parser` accepts values like `deepseek_r1`, `deepseek_v3`, `qwen3`, `glm45`, `hunyuan_a13b`, `granite`, `holo2`, `mistral`, `olmo3`, `ernie45`, `gptoss`, `nemotron_v3`, `granite`, `minimax_m2`, `kimi_k2`... I'll say "各家 parser" without exhaustive list to avoid error.
- `chat_template_kwargs` exists in vLLM's OpenAI server: yes, `ChatCompletionRequest.chat_template_kwargs: Optional[dict]`. Confident.
- `continue_final_message` exists: confident-ish (vLLM added it). I'll present with a "确认一下版本" caveat.
- `min_tokens`: vLLM SamplingParams has `min_tokens`. Confident.
- `bad_words`: confident.
- `ignore_eos`: confident.
- prefix caching default on in vLLM v1: yes, `enable_prefix_caching` defaults True in V1.
- `TokensPrompt` in vLLM: confident.
- Gemini API `thinkingConfig.thinkingBudget`: confident.
- s1 "budget forcing": confident, paper "s1: Simple test-time scaling" (Muennighoff et al., 2025).

Good. Write.

Length: this deserves a thorough answer. Maybe ~1200-1800 words equivalent. Use details/summary for the "论文参考" and "更多坑" sections.

Let me draft.

Also add a note on 一个更省事的"单次调用"做法: 用 `max_tokens=BUDGET` 先跑，如果没在预算内收尾，就 stage 2 —— 这叫 "lazy budget forcing"，只在需要时付第二段的钱，且 prefix cache 让第二段几乎免费。

Also mention streaming 场景下前端可以直接做"截断重连"：读到 N 个 reasoning token 就断开，发第二个请求带 prefix。这样前端就是控制点。

Also mention: 打断请求 vLLM 会浪费一些计算，但 prefix cache 还在（其实 abort 后 prefix cache 可能没写入... hmm, vLLM 的 prefix cache 只缓存已完成的 block，abort 的请求可能部分缓存。不确定，我含糊点说"已完成的 KV block 通常仍可复用"）。

Actually to be safe: "abort 的请求其已完成部分的 KV 是否进 prefix cache 取决于实现，比较稳的做法是让第一段自己以 max_tokens 正常结束。" Good, honest.

Write it.你想到的是对的，而且这个办法有正式名字：**budget forcing**（出自 s1 那篇 *Simple test-time scaling*）。你们讨论的两条路（让模型自己提前收尾、外部截断补 ``）在自部署场景下都成立，而且比调 API 更可控。关键在于区分“软控制”和“硬注入”：前者靠模板或参数，后者靠 token 级别的拼接。

## TL;DR

- **原理上完全成立**：decode 是自回归，`prompt + 半截 CoT + ` 只是同一个条件分布的另一个前缀，模型会接着生成。
- **自部署不但能这么干，还比闭源 API 更自由**，因为你能拿到 tokenizer 和 chat template，能做 token 级拼接。
- **vLLM 没有 `thinking_budget` 这种一键参数**（至少目前 OpenAI 兼容层没有），但它给了三层可组合的原语，组合起来就够用：`chat_template_kwargs`、`continue_final_message`/completions 传 token ids、`bad_words`/prefix cache。
- **最容易踩的坑**：不要用字符串拼接后再重新 encode，必须按 token 拼接。

## 为什么可行，以及它的真实失效模式

把 `` 硬插进去，模型确实会转入答案模式，但你会遇到三个可预期的退化：

| 现象 | 原因 | 缓解 |
|---|---|---|
| 答案前半段废话，模型“重新开始想” | CoT 被砍断在半句，分布上不自然 | 注入过渡短语再补 ``（s1 的做法） |
| 直接输出 `` 或之后又补一个 `` | 模板/训练里这个 token 的邻接结构被破坏 | `bad_words=[""]` 禁用重复 |
| 答案质量掉 | 信息确实被砍掉了 | 预算曲线通常是单调的，掉是应该的 |

s1 里那个过渡短语大概长这样，很土但很有效：

```
\n\nConsidering the limited time by the user, I have to give the solution
based on the simplest and most direct approach being attempted above.\n\n\n\n
```

还有一个前提：**`` 必须是该模型训练时用的那个终止符**，且 token 化后是同一个 token id。DeepSeek-R1、Qwen3 用 ``；GLM、MiniMax、Kimi 各家不一样（`<|/think|>`、`<thought>` 等）。补错就是纯噪声。

<details>
<summary>反向操作：把思考拉长</summary>

同一套机制反向用就是“延长思考”，即在 CoT 里注入 `Wait,` 或 `Let me reconsider...` 再让它续写。这是 s1 的另一半，也是现在很多 “test-time scaling” 工作的基础。对你来说就是：不给 ``，而是给一个鼓励继续思考的短 seed。
</details>

## vLLM 到底提供了什么

按“从软到硬”排：

### 1. 软控制：`chat_template_kwargs`（官方支持，但只有开关没有预算）

vLLM 的 `/v1/chat/completions` 接受 `chat_template_kwargs`，它会透传给 Jinja 模板：

```json
{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Qwen3 的 `enable_thinking=False`、DeepSeek-V3.1 的 `thinking` 模式、GLM-4.5 的模式切换，本质都是**模板里注入了一个空的思考块**，属于 prompt 注入的一种特例。局限很明确：全开或全关，没有“给 500 token 想想”这种中间态。

配套的还有 `--reasoning-parser`：

```bash
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3
```

它把 `reasoning_content` 和 `content` 分开返回。注意这是**解析**，不是**控制**，很多人会搞混。

### 2. 硬注入 A：`continue_final_message`

这是最贴近你需求的 chat 层接口。把截断后的 CoT 加 `` 作为最后一条 assistant 消息发回去：

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

r = c.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": q},
        {"role": "assistant", "content": truncated_cot + "\n\n\n\n"},
    ],
    continue_final_message=True,
    add_generation_prompt=False,
    max_tokens=1024,
)
```

底层是 transformers `apply_chat_template` 的 `continue_final_message`。**坑**：模板如果会在 assistant 段末尾自动补 ``，你会得到一个重复的；所以内容里别自己带，或者反过来靠模板带。

### 3. 硬注入 B：completions + 直接传 token ids（最可靠）

这条路绕开模板，你自己拼 token，最不容易出幺蛾子：

```python
import vllm

llm = vllm.LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tok = llm.get_tokenizer()

BUDGET = 512
THINK_END = tok.encode("", add_special_tokens=False)
assert len(THINK_END) >= 1, "检查一下这个模型用的是什么终止符"

# ---- 第一段：只给预算，让它想 ----
prefix_ids = tok.apply_chat_template(
    [{"role": "user", "content": q}],
    add_generation_prompt=True,
    tokenize=True,
)

stage1 = llm.generate(
    vllm.TokensPrompt(prompt_token_ids=prefix_ids),
    vllm.SamplingParams(max_tokens=BUDGET, temperature=0.6, skip_special_tokens=False),
)
cot_ids = stage1[0].outputs[0].token_ids   # 注意：拿 token_ids，不要拿 text

# ---- 第二段：截断 + 过渡语 + 强制收尾 ----
TRANSITION = "\n\nOkay, I've thought enough. Final answer:\n\n"
inject = tok.encode(TRANSITION, add_special_tokens=False) + THINK_END + tok.encode("\n\n", add_special_tokens=False)

forced_ids = prefix_ids + cot_ids + inject

stage2 = llm.generate(
    vllm.TokensPrompt(prompt_token_ids=forced_ids),
    vllm.SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        bad_words=[""],      # 防止它又开一个思考块
    ),
)
print(tok.decode(stage2[0].outputs[0].token_ids, skip_special_tokens=True))
```

三个关键点：

- **`cot_ids` 必须来自 `outputs[0].token_ids`，不能把 text 解码后重新 encode**。BPE 在边界处会合并，重新编码得到的 id 序列和你实际生成的可能不一样，prefix 就悄悄变了。
- **`--enable-prefix-caching`（vLLM V1 默认开）让第二段几乎免费**，因为它的 prefix 就是第一段的长前缀，KV 直接命中。这是这套方案在生产上能跑的核心原因。
- `bad_words` 是 vLLM `SamplingParams` 的原生字段，用它做“禁止二次收尾”的保险，比 grammar 便宜得多。

### 4. 其他可用但不解决核心问题的原语

| 原语 | 能干什么 | 不能干什么 |
|---|---|---|
| `max_tokens` | 卡死总长度 | 不会自动收尾，答案是空的 |
| `stop=[""]` | 想到就停 | 不能**强制**它发 |
| `min_tokens` | 防止过早 EOS | 与预算无关 |
| `ignore_eos` | 强行用满预算 | 会越过 `` 继续“想” |
| `guided_regex` / `guided_json` | 约束输出形态 | 不擅长长度约束，且是**生成侧**约束 |
| `prompt_logprobs` | 每步看 `` 的概率 | 只观测，不干预 |
| `--enable-prompt-embeds` | 传 embedding | 文本场景没必要 |

目前**没有** `reasoning_max_tokens` / `thinking_budget` 这类原生字段。社区里有过 PR 和讨论，但不能指望它在你的版本里。

<details>
<summary>关于中止请求能不能复用 KV cache</summary>

流式场景下前端可以在读到 N 个 reasoning token 时 abort，再发第二个请求。省了带宽，但**已 abort 请求的 KV 是否进 prefix cache 取决于实现细节**，不保证。更稳的做法是让第一段以 `max_tokens` 自然结束，别依赖 abort 的副作用。
</details>

## 关于 Gemini 网页端那个“立即回答”

大概率不完全是纯 prompt hack。Gemini API 有 `thinkingConfig.thinkingBudget`（`0` 就是关思考，`-1` 是动态），网页端的按钮很可能就是切到 0 预算**重新生成**，或者训练时就有 “skip thinking” 的控制能力。产品化路径和你的 hack 路径不同，但**在 AR 语义上它们是同一件事**。

## 工程坑清单

1. **按 token 拼，不要按字符串拼**（最重要，重复三遍都不为过）。
2. 各家的 think 终止符不同，先去 tokenizer 里确认，必要时 `tok.convert_tokens_to_ids` 验证。
3. 过渡短语不要省。不加的话模型经常“假装还在想”，在答案里继续 reframe。
4. 第二段的 `max_tokens` 要留够，否则你花了两次调用拿到空答案。
5. 推理时 `temperature` 在两段之间尽量一致，不一致会放大分布漂移。
6. 如果你开了 `--reasoning-parser`，第二段的输出会被正常切开；但注入的 `` 在前缀里，parser 只看生成部分，不受影响。
7. **预算要自适应**：不要所有 query 用同一个 N。可以用第一段前若干 token 的 `prompt_logprobs` 或者答案置信度做门限，简单题早早收，难题给满。

## 一个更省钱的编排

不必每单都两段式。先跑 `max_tokens=BUDGET, stop=[""]`：

- 如果它自己收尾了，结束，只花一次调用。
- 如果撞到 `max_tokens` 还没收，说明这叫“该截断”，再走第二段。此时 prefix cache 命中，边际成本极低。

这就是 “lazy budget forcing”，实际线上大部分请求走不到第二步。

<details>
<summary>可参考的工作</summary>

- **s1: Simple test-time scaling** — budget forcing 的原始出处，`` 截断 + 过渡语就是这篇给的。
- **Chain of Draft** — 从训练/prompt 侧压缩 CoT，和你这个正交，可以叠加。
- **L1 / length-controlled policy optimization** — 用 RL 训出“预算感知”，比外挂注入更优雅，但需要训练。
- **Think Prune / 各种 CoT pruning** — 生成后再裁，属于后处理路线。

要真正产品化，长期看是“训练时给模型一个 budget token”，短期最实用的就是你现在想的这套注入。自部署的最大优势正是这个：**你能直接改 token 流，而 API 用户只能求模型配合。**
</details>

需要的话，我可以把上面那段 demo 整理成一个可以直接跑的 `BudgetForcingEngine` 封装，带上自适应预算和 vLLM 的 async streaming 版本。

---
## 😀 默认助手 (7)

` then outputting answer. They ask: for self-deployed models, can you do this? Does vLLM provide an interface for the frontend to control this?

This is a technical question. I should answer accurately. Let me think about what I know about vLLM's APIs for controlling reasoning/thinking.

Key facts I know:

1. Yes, LLM decode is AR. Truncating the CoT and forcing closing think tag is a real technique. Gemini's "instant answer" probably resamples from the point before thinking, or uses trained control tokens/special modes.

2. For self-deployed models:
   - Qwen3 has `/think` and `/no_think` soft switches (in the chat template, it appends `/no_think` to the user message), which the model is trained to obey. Qwen3 also supports `enable_thinking=False` in chat template which pre-fills `

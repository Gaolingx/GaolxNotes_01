jinja 模板 set 语句查询：示例 {%- set enable_thinking = false %}

| 序号 | set 语句 / 变量 | 所在位置 | 含义 |
|---:|---|---|---|
| 1 | `image_count = namespace(value=0)` | 模板开头 | 创建图片计数器，用于多模态内容中统计图片数量。 |
| 2 | `video_count = namespace(value=0)` | 模板开头 | 创建视频计数器，用于多模态内容中统计视频数量。 |
| 3 | `enable_thinking = false` | 默认参数区 | 若外部未传入 `enable_thinking`，默认关闭 thinking。 |
| 4 | `strict_live_evidence = true` | 默认参数区 | 若外部未传入，默认开启实时证据严格门。 |
| 5 | `tool_response_char_limit = 4000` | 默认参数区 | 设置工具返回内容的默认最大回灌字符数，防止上下文爆炸。 |
| 6 | `preserve_reasoning = false` | 默认参数区 | 默认不回灌 `reasoning_content`。 |
| 7 | `result_state_replay = true` | 默认参数区 | 默认开启 `<RESULT_STATE>` 状态摘要回灌机制。 |
| 8 | `result_state_only_history = true` | 默认参数区 | 默认历史消息只回灌状态摘要，不完整回灌历史内容。 |
| 9 | `replay_only_latest_result_state = true` | 默认参数区 | 默认只回灌最近一个历史 `<RESULT_STATE>`。 |
| 10 | `result_state_start = '<RESULT_STATE>'` | 默认参数区 | 设置结果状态摘要的开始标记。 |
| 11 | `result_state_end = '</RESULT_STATE>'` | 默认参数区 | 设置结果状态摘要的结束标记。 |
| 12 | `tools_available = tools and tools is iterable and tools is not mapping` | 工具检测区 | 判断工具列表是否存在且格式有效。 |
| 13 | `tools_available = false` | 工具检测区 | 若没有传入 `tools`，则工具不可用。 |
| 14 | `ns_flags = namespace(enable_thinking=enable_thinking)` | 参数区后 | 创建可变命名空间，用于在循环内动态更新 thinking 开关。 |
| 15 | `image_count.value = image_count.value + 1` | `render_content` 宏 | 识别到图片内容时，图片计数加 1。 |
| 16 | `video_count.value = video_count.value + 1` | `render_content` 宏 | 识别到视频内容时，视频计数加 1。 |
| 17 | `tool_call = tool_call.function` | `render_tool_call` 宏 | 若工具调用对象中存在 `function` 字段，则取其内部函数对象。 |
| 18 | `args_value = tool_call.arguments[args_name]` | `render_tool_call` 宏 | 取出当前工具参数值。 |
| 19 | `args_value = args_value \| tojson if ... else args_value \| string` | `render_tool_call` 宏 | 若参数值是对象或数组，转为 JSON；否则转为字符串。 |
| 20 | `after_start = content.split(result_state_start)[-1]` | `extract_result_state` 宏 | 从内容中截取 `<RESULT_STATE>` 之后的部分。 |
| 21 | `state_body = after_start.split(result_state_end)[0]` | `extract_result_state` 宏 | 从状态开始标记后截取到状态结束标记前，得到状态正文。 |
| 22 | `ctrl = namespace(initial_count=0, closed=false, content='')` | 系统/开发者消息聚合区 | 创建控制对象，用于收集开头连续的 system/developer 消息。 |
| 23 | `raw_content = render_content(message.content, false, true)\|trim` | system/developer 聚合循环 | 渲染当前 system/developer 消息正文。 |
| 24 | `ns_flags.enable_thinking = false` | system/developer 聚合循环 | 如果系统消息含 `<\|think_off\|>`，关闭 thinking。 |
| 25 | `raw_content = raw_content.replace('<\|think_off\|>', '').strip()` | system/developer 聚合循环 | 删除 `<\|think_off\|>` 控制标记。 |
| 26 | `ns_flags.enable_thinking = true` | system/developer 聚合循环 | 如果系统消息含 `<\|think_on\|>`，开启 thinking。 |
| 27 | `raw_content = raw_content.replace('<\|think_on\|>', '').strip()` | system/developer 聚合循环 | 删除 `<\|think_on\|>` 控制标记。 |
| 28 | `ctrl.content = ctrl.content + '\n\n' + raw_content` | system/developer 聚合循环 | 若已有系统内容，则追加当前内容。 |
| 29 | `ctrl.content = raw_content` | system/developer 聚合循环 | 若尚无系统内容，则初始化系统内容。 |
| 30 | `ctrl.initial_count = ctrl.initial_count + 1` | system/developer 聚合循环 | 记录开头连续 system/developer 消息数量。 |
| 31 | `ctrl.closed = true` | system/developer 聚合循环 | 一旦遇到非 system/developer 消息，关闭系统消息聚合。 |
| 32 | `ns = namespace(multi_step_tool=true, last_query_index=messages\|length - 1)` | 最近用户查询定位区 | 创建状态对象，用于定位最后一个真实用户查询。 |
| 33 | `index = (messages\|length - 1) - loop.index0` | 反向扫描消息 | 将反向循环下标换算为原始消息下标。 |
| 34 | `content = render_content(message.content, false)\|trim` | 反向扫描用户消息 | 渲染用户消息内容，用于判断是否为工具响应包装。 |
| 35 | `ns.multi_step_tool = false` | 反向扫描用户消息 | 找到最后一个非 `<tool_response>` 用户消息后，结束工具链扫描。 |
| 36 | `ns.last_query_index = index` | 反向扫描用户消息 | 记录最后一个真实用户查询的位置。 |
| 37 | `rs = namespace(latest_state_assistant_index=-1, historical_message_count=0)` | 历史状态扫描区 | 创建状态对象，用于寻找最近一个历史 `<RESULT_STATE>`。 |
| 38 | `rs.historical_message_count = rs.historical_message_count + 1` | 历史状态扫描区 | 统计历史消息数量。 |
| 39 | `c = render_content(message.content, false)\|trim` | 历史状态扫描区 | 渲染历史 assistant 消息内容，用于检测状态摘要。 |
| 40 | `rs.latest_state_assistant_index = loop.index0` | 历史状态扫描区 | 若历史 assistant 消息含 `<RESULT_STATE>`，记录其位置；循环结束后即最近一个。 |
| 41 | `content = render_content(message.content, true)\|trim` | 当前 user 分支 | 渲染用户消息正文，并统计视觉内容。 |
| 42 | `content = content.replace('<\|think_off\|>', '').strip()` | 当前 user 分支 | 删除用户消息中的 thinking 关闭标记。 |
| 43 | `content = content.replace('<\|think_on\|>', '').strip()` | 当前 user 分支 | 删除用户消息中的 thinking 开启标记。 |
| 44 | `content = render_content(message.content, true)\|trim` | assistant 分支 | 渲染 assistant 消息正文，并统计视觉内容。 |
| 45 | `content = content.replace('<\|think_off\|>', '').strip()` | assistant 分支 | 删除 assistant 消息中的 thinking 关闭标记。 |
| 46 | `content = content.replace('<\|think_on\|>', '').strip()` | assistant 分支 | 删除 assistant 消息中的 thinking 开启标记。 |
| 47 | `result_state = extract_result_state(content)\|trim` | assistant 分支 | 从 assistant 正文中提取 `<RESULT_STATE>`。 |
| 48 | `reasoning_content = ''` | assistant 普通消息分支 | 初始化 reasoning 内容为空。 |
| 49 | `reasoning_content = message.reasoning_content` | assistant 普通消息分支 | 若开启 `preserve_reasoning` 且消息中有 `reasoning_content`，读取它。 |
| 50 | `reasoning_content = content.split('[/think]')[0].rstrip('\n').split('[think]')[-1].lstrip('\n')` | assistant 普通消息分支 | 若正文中包含 `[think]...[/think]`，提取其中的 reasoning。 |
| 51 | `content = content.split('[/think]')[-1].lstrip('\n')` | assistant 普通消息分支 | 从 assistant 正文中移除 thinking 部分，只保留可见内容。 |
| 52 | `reasoning_content = reasoning_content\|trim` | assistant 普通消息分支 | 清理 reasoning 前后空白。 |
| 53 | `content = render_content(message.content, false)\|trim` | tool 分支 | 渲染工具返回内容。 |
| 54 | `original_len = content\|length` | tool 分支 | 记录工具返回原始字符长度。 |
| 55 | `content = content[:tool_response_char_limit] + ...` | tool 分支 | 若工具返回过长，则截断并追加 `[Tool_Result_Truncated]` 标记。 |

## 按功能归类总结

| 类别 | 相关 set 语句 | 作用 |
|---|---|---|
| 多模态计数 | `image_count`、`video_count`、`.value += 1` | 统计图片、视频输入数量。 |
| 默认参数 | `enable_thinking`、`strict_live_evidence`、`tool_response_char_limit` 等 | 为模板运行提供默认配置。 |
| 工具检测 | `tools_available` | 判断是否存在可调用工具。 |
| reasoning 控制 | `preserve_reasoning`、`reasoning_content` | 控制是否保留推理内容，默认关闭。 |
| 结果状态回灌 | `result_state_replay`、`result_state_only_history`、`replay_only_latest_result_state`、`result_state_start/end` | 控制 `<RESULT_STATE>` 摘要回灌机制。 |
| 系统消息聚合 | `ctrl.*`、`raw_content` | 合并开头连续 system/developer 消息。 |
| 最后用户查询定位 | `ns.*`、`index`、`content` | 找到最后一个真实用户查询，区分工具链和新问题。 |
| 最近历史状态定位 | `rs.*`、`c` | 只定位并回灌最近一个历史 `<RESULT_STATE>`。 |
| 工具调用渲染 | `tool_call`、`args_value` | 将工具调用对象转为 XML 格式。 |
| 工具结果截断 | `original_len`、`content = content[:limit]...` | 限制工具返回进入上下文的长度。 |
| thinking 标记清理 | `content.replace(...)` | 移除 `<\|think_on\|>` / `<\|think_off\|>` 控制符。 |

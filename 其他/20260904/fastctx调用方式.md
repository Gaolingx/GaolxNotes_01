# fastctx调用方式

## 🧑‍💻 User

https://github.com/yc-duan/fastctx agent在实际调用fastctx时是怎么样的

---
## 🤖 Assistant

I'll explore this repository to understand how the agent actually calls fastctx. Let me start by discovering available tools.

Let me fetch the repository's README to understand the project.

I have the README. Now let me dig into the source to see the actual tool registration, naming, and the AGENTS.md instructions the agent receives. Let me explore the repository tree.

The README describes the capabilities, but the actual agent-facing calling convention lives in the source. Let me fetch the key files: the AGENTS.md guidance, tool schemas, and server manifest.

Let me examine the server registration and the Codex config wiring to see the exact MCP namespace and how tools are exposed to the agent.

Let me confirm the exact AGENTS.md marker strings (HTML comments were stripped in the text fetch).

我阅读了 `fastctx` 的源码（不只是 README），重点看了 `src/server.rs`、`src/server_manifest.rs`、`src/model_guidance.rs`、`src/control/agents.rs`（AGENTS.md 生成）和 `src/control/codex_config.rs`（Codex 配置写入）。结论是：**agent 并不是把 fastctx 当命令行去执行，而是把它注册成一个标准 MCP server，然后以"结构化工具调用"的方式调用它。** 下面是完整链路。

## 一、接入方式：注册为 MCP server，而非 CLI

`fastctx` 命令打开的是控制终端（TUI）。选 **Connect to Codex** 后，`src/control/codex_config.rs` 会把下面这段写进 `~/.codex/config.toml`：

```toml
[mcp_servers.fastctx]
command = "C:/Users/<you>/.fastctx/bin/fastctx.exe"
args = ["serve"]                     # 开启 Bash 工具时为 ["serve", "--enable-shell"]
startup_timeout_sec = 120
tool_timeout_sec = 300
env = { FASTCTX_TOKEN_BUDGET = "...", FASTCTX_READ_TOKEN_BUDGET = "...",
        FASTCTX_GREP_TOKEN_BUDGET = "...", FASTCTX_GLOB_TOKEN_BUDGET = "...",
        FASTCTX_RUN_TOKEN_BUDGET = "...", FASTCTX_JOB_OUTPUT_TOKEN_BUDGET = "..." }

[features.code_mode]
direct_only_tool_namespaces = ["mcp__fastctx"]
```

所以 agent 侧看到的是一个名为 `fastctx` 的 MCP server。在 Codex 里，实际可调用的工具名按 Codex 的 MCP 命名规范拼成 `mcp__fastctx__<tool>`（例如 `mcp__fastctx__grep`）。`serve` 进程本身只是一个 stdio 代理，真正干活的是 per-user 的共享控制中心。

> 源码里有一段注释解释了为什么指导文本改用"服务器 + 裸工具名"（如 `inspect_local_file`）而不是固定的 `mcp__fastctx__xxx`：不同 host 对 MCP 工具的拼接方式不同，某些 host 会把 server 当命名空间、成员是裸名。用 server+裸名的写法两边都能对上。

## 二、Agent 能看到的工具面：共 9 个

| 工具 | 分组 | readOnly | 作用 |
|------|------|:--------:|------|
| `inspect_local_file` | File（默认） | ✅ | 读单个文件（文本/图片/PDF/hex），或批量 1–32 个文本文件 |
| `grep` | File（默认） | ✅ | 正则内容搜索（ripgrep 引擎） |
| `glob` | File（默认） | ✅ | 按路径模式找文件 |
| `replace` | File（默认） | ❌ | 机械式批量替换 |
| `run` | Shell（`--enable-shell`） | ❌ | 前台执行 bash |
| `run_background` | Shell | ❌ | 后台任务 |
| `job_output` | Shell | ✅ | 查后台任务输出 |
| `job_kill` | Shell | ❌ | 杀后台任务进程树 |
| `job_list` | Shell | ✅ | 列出后台任务 |

默认只发布 4 个文件工具；后 5 个必须由用户在控制终端里开启 Bash 终端（即 `--enable-shell`）才会出现。所有工具的注解都是 `destructiveHint=false`、`openWorldHint=false`。工具定义受 `ToolManifest::validate` 强约束（名字、顺序、注解都必须与 manifest 一致）。

## 三、Agent 收到的行为指导：写进 `~/.codex/AGENTS.md`

除了工具 schema，fastctx 还会在 `~/.codex/AGENTS.md` 里维护一段由标记包围的托管块（当前契约版本 `guidance-v4`）：

```markdown
<!-- fastctx:begin -->
## Local file inspection

For reading, searching, and finding local files, prefer the FastCtx MCP
server's own tools — `inspect_local_file`, `grep`, and `glob` — over shell
equivalents such as `cat`/`Get-Content`, `rg`/`findstr`/`Select-String`,
and `dir`/`ls -R`.
Use FastCtx file tools directly for local-file operations, including when a
local reference is URI-shaped; pass the equivalent plain absolute filesystem path.
Read only what the task needs. When you need several files, pass them to
one `inspect_local_file` call as files=[{"path": ...}, ...] instead of one
call per file. The last line of every result says `Complete` or
`Partial` — continue only with the exact parameters a `Partial` note
provides.

### Batch replacement

Use FastCtx's `replace` for mechanical find-and-replace across files.
It preserves each file's encoding and line endings, supports dry-run previews,
and rejects concurrent changes before writing. Use apply_patch for generated
content, semantic rewrites, or small local edits.

### Shell commands            # 仅在 Bash 终端开启时写入

Prefer FastCtx's `run` over the built-in shell for terminal work: it
executes with bash (Git Bash on Windows), so always write POSIX bash —
never PowerShell syntax.

Never pass `apply_patch` to FastCtx's `run`: it is not a program and
no shell can run it. Reach it through Codex itself ...
<!-- fastctx:end -->
```

此外，MCP `initialize` 时还会带一句 server instructions（一行短 blurb），由 `model_guidance::server_instructions()` 动态生成，例如：

> `Local-file tools: inspect_local_file, grep, glob, and replace. Use FastCtx file tools directly for local-file operations, including when a local reference is URI-shaped; pass the equivalent plain absolute filesystem path.`

## 四、实际调用长什么样（结构化 JSON，不是 shell）

这是 agent 真正发出的调用形态：

**读文件**（批量一次调用替代每文件一次 round trip）：
```json
{ "files": [
    { "path": "V:/repo/src/main.rs", "offset": 120, "limit": 40 },
    { "path": "V:/repo/src/config.rs" },
    { "path": "V:/repo/docs/legacy.txt", "encoding": "gbk" } ],
  "limit": 400 }
```
返回带 1-based 行号，末尾是 `(Partial: lines 120-159 of 512 shown. Continue with offset=160.)` 或 `(Complete: ...)`。

**搜索**：`{ "pattern": "fn \\w+_lock", "path": "V:/repo/src", "output_mode": "content", "context": 1 }`
**找文件**：`{ "pattern": "**/*.toml", "path": "V:/repo", "sort": "modified", "output_mode": "details" }`
**批量替换（先 dry-run）**：`{ "pattern": "old_name\\(", "replacement": "new_name(", "path": "V:/repo/src", "glob": ["**/*.rs"], "dry_run": true }`
**执行命令**：`{ "command": "cargo test --quiet 2>&1 | tail -n 40", "timeout_ms": 180000 }`
**后台任务**：`run_background` 拿 job id → `job_output {job_id, wait_ms}` → `job_kill {job_id}`

一个典型的 agent 调用序列会是：

```text
mcp__fastctx__glob         { "pattern": "**/*.rs" }
mcp__fastctx__grep         { "pattern": "fn main", "output_mode": "files_with_matches" }
mcp__fastctx__inspect_local_file  { "files": [ {"path": ".../main.rs"}, {"path": ".../config.rs"} ] }
mcp__fastctx__replace      { ..., "dry_run": true }   →  再去掉 dry_run 真正写入
mcp__fastctx__run          { "command": "cargo build" }
```

## 五、关键的运行时契约（决定 agent 怎么用）

- **结果状态行**：每个成功结果的最后一行一定是 `Complete` 或 `Partial`；`Partial` 会给出续读所需的**精确参数**（offset / files 数组 / seq），agent 应当直接照抄继续，而不是自己猜。
- **结构化输出**：`grep` 有 `files_with_matches` / `content` / `count` / `summary` 四种 `output_mode`；`glob` 有 `paths` / `details`。
- **schema 归一化**：`src/tool_schema.rs` 把发布的 JSON Schema 压到各 LLM provider 的公共子集（去掉 `$ref`、`oneOf`/`anyOf`/`allOf`、`const`、`format`、`type:["x","null"]` 等），避免某个 provider 因 schema 报 400 导致整轮工具全消失。
- **超时/预算**：`tool_timeout_sec=300`；`run` 前台最长约 4 分钟，更久用 `run_background`；`job_output.wait_ms` 上限 240000；各工具受 `FASTCTX_*_TOKEN_BUDGET` 控制输出预算。
- **编码**：读文件自动探测，不肯定时返回候选编码而不是猜；单文件用 `encoding`，目录搜索用 `fallback_encoding`。
- **权限边界**：`replace` 默认就发布（写入用 dry-run/CAS/原子替换保护）；Bash 工具默认关闭，需用户在 TUI 开启；`default_tools_approval_mode = "writes"` 可让 `replace` 与 shell 走审批。
- **旁注**：任务在跑时，每个成功文本结果都会附带一行后台任务状态读数（只在下次调用时刷新，不会主动推送）。

---

**一句话总结**：agent 调用 fastctx 的方式是——通过 Codex 的 MCP 通道，以 `mcp__fastctx__<tool>` 的工具调用形式传入**结构化 JSON 参数**（`file_path`/`files`、`pattern`、`output_mode`、`dry_run`、`command` 等），拿回带行号和 `Complete`/`Partial` 状态行的结构化文本；行为规则由写进 `~/.codex/AGENTS.md` 的 `guidance-v4` 托管块约束，而 `~/.codex/config.toml` 负责注册 server、命名空间、超时与 token 预算。README 里那些 `fastctx apply/status/jobs` 命令行只是**用户侧/运维侧**的非交互入口，agent 本身并不走 CLI。

如果你想进一步确认某个细节（比如某个工具的完整参数 schema、或 `Partial` 续读数组的确切格式），我可以继续深入对应源码。

---
## 🧑‍💻 User

我需要这个给agent暴露的tools，tools的schema，你能帮我找找吗？

---
## 🤖 Assistant

I'll dig into the source to extract the exact tool definitions and schemas. Let me first see what tools are available in the hub.

I have fetch tools. Let me pull the repository tree and the key schema source files.

GitHub API is blocked. Let me fetch the raw source files directly from `raw.githubusercontent.com`.

All four files are accessible. Let me read `server.rs` fully (it contains the tool registration and schemas) plus `model_guidance.rs`.

Now I see the pattern: schemas come from `Parameters<ReadRequest>`, `GrepRequest`, `GlobRequest`, and the shell/edit param structs. Let me get the file list and the parameter-struct definitions.

I now have the full file tree. The schemas come from `Parameters<XYZRequest>` structs. Let me pull the manifest and all the request-struct definitions.

I have the file tools and shell tool routes. Now let me extract the remaining request structs: `GrepRequest`, `ReplaceRequest`, and the shell request types.

Now the `ReplaceRequest` and the shell request structs.

Let me pull the remaining shell request structs from `src/shell/mod.rs`.

Let me grab the guidance module (dynamic descriptions) and confirm the namespace string in the Codex config writer.

Let me read `tool_schema.rs` so I can describe exactly how the published wire schema is normalized.

我已经把源码读透了。fastctx 的工具 schema **不是手写 JSON**，而是由 Rust 参数结构体（`#[derive(JsonSchema)]`）经 `schemars` 生成，再经 `src/tool_schema.rs` 归一化成"可移植子集"后发布。下面是我从源码逆推出的**发布态 schema**（也就是 agent 在 `tools/list` 里真正看到的东西）。

## 总览：9 个工具

服务器名 `fastctx`，命名空间常量 `FASTCTX_NAMESPACE = "mcp__fastctx"`（`src/control/codex_config.rs`）。工具名不含命名空间，在 Codex 中拼成 `mcp__fastctx__<tool>`。顺序由 `src/server_manifest.rs` 的 `TOOL_ENTRIES` 冻结：

| # | 工具名 | 组 | title | readOnly | 参数结构体 |
|:-:|--------|----|-------|:--------:|-----------|
| 1 | `inspect_local_file` | File | Inspect local file | ✅ | `ReadRequest` (`read_tool/mod.rs`) |
| 2 | `grep` | File | Search file contents | ✅ | `GrepRequest` (`grep_tool.rs`) |
| 3 | `glob` | File | Match file paths | ✅ | `GlobRequest` (`glob_tool.rs`) |
| 4 | `replace` | File | Batch replace file contents | ❌ | `ReplaceRequest` (`edit/mod.rs`) |
| 5 | `run` | Shell | Run bash command | ❌ | `RunRequest` (`shell/mod.rs`) |
| 6 | `run_background` | Shell | Start background bash job | ❌ | `RunBackgroundRequest` |
| 7 | `job_output` | Shell | Check background job output | ✅ | `JobOutputRequest` |
| 8 | `job_kill` | Shell | Kill background job | ❌ | `JobKillRequest` |
| 9 | `job_list` | Shell | List background jobs | ✅ | `JobListRequest` |

所有工具的 `destructiveHint=false`、`openWorldHint=false`（`ToolManifest::validate` 强制校验）。

<details>
<summary><b>发布子集的归一化规则（为什么 schema 长这样）</b></summary>

`tool_schema.rs` 的注释给出了硬约束——**每个已发布 schema 只允许出现这些关键字**：`type`、`description`、`properties`、`required`、`items`、`enum`、`default`、`minimum`、`maximum`、`minItems`、`maxItems`。

具体转换：
- **剥离** `$schema`、`additionalProperties`、`format`（`format` 带的 `uint64/int64` 是各 provider 不认的值，边界改由 `minimum/maximum` 表达）。
- **内联 `$ref`**：`#/$defs/x` 被替换为定义体，引用节点自身的键优先（所以参数的 `description` 覆盖类型的描述）。深度上限 32。
- **折叠可空联合**：`anyOf:[T,{type:null}]` → `T`；可选性由 `required` 列表单独表达。多于一个非 null 分支则保留（宁可报错也不擅自收窄）。
- **折叠 const 枚举**：字段枚举的 `oneOf`/`const` → `{"type":"string","enum":[...]}`（因此下文很多参数字段带 `enum`）。
- **`schemars(with = "...")`** 会改写**发布类型**（例如 `Option<Vec<String>>` 发布成 `array`，联合类型发布成纯 `string`），但 serde 反序列化仍用原 Rust 类型——即"schema 声明的永远比实际接受的范围更窄或相等"。

该子集由 `server_contract::published_tool_schemas_stay_inside_the_portable_subset` 全量守护。
</details>

---

## 1. `inspect_local_file`

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Plain absolute local filesystem path. When the source reference is URI-shaped, use its equivalent local absolute path. File to inspect; both / and \\ are accepted. Mutually exclusive with files."
    },
    "files": {
      "type": "array",
      "minItems": 1,
      "maxItems": 32,
      "description": "Batch form: an array of {\"path\", \"offset\"?, \"limit\"?, \"encoding\"?} objects for 1-32 text ranges in one call. Repeat a path for distinct ranges, each with its own offset/limit, and freely mix ranges from multiple files. Each entry behaves like its own single-file text request; results stay in request order. Mutually exclusive with file_path and with the top-level offset/encoding/pages/pdf_mode/view parameters.",
      "items": {
        "type": "object",
        "properties": {
          "path": { "type": "string", "description": "Plain absolute local filesystem path. When the source reference is URI-shaped, use its equivalent local absolute path. Text file to inspect; both / and \\ are accepted." },
          "offset": { "type": "integer", "minimum": 1 },
          "limit": { "type": "integer", "minimum": 1 },
          "encoding": { "type": "string" }
        },
        "required": ["path"]
      }
    },
    "offset": { "type": "integer", "minimum": 1, "description": "The 1-based line number to start reading from. Use for paging through large files." },
    "limit": { "type": "integer", "minimum": 1, "description": "The number of lines to return..." },
    "pages": { "type": "string", "description": "Page range for PDF files, e.g. \"1-5\", \"3\", \"10-20\". Max 20 pages per call. Required in text mode for PDFs with more than 10 pages." },
    "pdf_mode": { "type": "string", "description": "PDF only: \"text\" (default) returns the selected pages' text layer; \"image\" returns each selected page rendered as a PNG image." },
    "encoding": { "type": "string", "description": "Text files only. Known source encoding as a WHATWG label, e.g. \"gbk\", \"shift_jis\", \"big5\", \"euc-kr\", \"windows-1252\", \"utf-16le\", plus \"utf-32le\"/\"utf-32be\". ..." },
    "view": { "type": "string", "description": "\"auto\" (default) picks the channel by file type; \"hex\" returns a paged hex dump of the raw bytes of any file — the way to inspect binary files." }
  },
  "required": []
}
```

> `pdf_mode` 与 `view` 因 `#[schemars(with = "Option<String>")]` **不发布 enum**，合法值只写在 `description` 里。`files` 与 `file_path` 互斥由运行时校验（schema 层无法表达）。

**工具描述**（`model_guidance::inspect_tool_description()` 动态拼装）：
> Inspect the contents of local filesystem paths: one file (text, image, or PDF), a batch of text files, or any file's raw bytes. Use FastCtx file tools directly for local-file operations, including when a local reference is URI-shaped; pass the equivalent plain absolute filesystem path. Text returns 1-based `Ncontent` lines... Text, PDF, and hex responses end with a Complete or Partial status — continue only with the exact parameters a Partial note provides.

---

## 2. `grep`

```json
{
  "type": "object",
  "properties": {
    "pattern": { "type": "string", "description": "The regular expression to search for (Rust regex syntax; escape literal braces like `interface\\{\\}`)." },
    "path": { "type": "string", "description": "Plain absolute local filesystem path. When the source reference is URI-shaped, use its equivalent local absolute path. File or directory to search. Omit for the session working directory." },
    "glob": { "type": "array", "items": { "type": "string" }, "description": "Globs to filter files, e.g. [\"**/*.rs\", \"!tests/**\"]. A leading `!` excludes and always wins; negative-only lists include every other file." },
    "type": { "type": "string", "description": "File type filter, e.g. \"js\", \"py\", \"rust\" (equivalent to rg --type; more efficient than glob for standard types)." },
    "output_mode": { "type": "string", "enum": ["content", "files_with_matches", "count", "summary"], "description": "\"content\" = matching lines with optional context; \"files_with_matches\" (default) = matching paths only; \"count\" = per-file counts plus their total; \"summary\" = global totals from a full scan (ignores head_limit/offset)." },
    "case_insensitive": { "type": "boolean" },
    "line_numbers": { "type": "boolean" },
    "only_matching": { "type": "boolean" },
    "before_context": { "type": "integer" },
    "after_context": { "type": "integer" },
    "context": { "type": "integer" },
    "multiline": { "type": "boolean" },
    "head_limit": { "type": "integer" },
    "offset": { "type": "integer" },
    "encoding": { "type": "string" },
    "fallback_encoding": { "type": "string" }
  },
  "required": ["pattern"]
}
```

> 注意 `file_type` 字段被 `#[serde(rename="type")]` + `#[schemars(rename="type")]` 发布成了 **`type`**（对 agent 可见的键名是 `type`）。

---

## 3. `glob`

```json
{
  "type": "object",
  "properties": {
    "pattern": { "type": "array", "items": { "type": "string" }, "description": "Globs to match files, e.g. [\"**/*.rs\", \"!tests/**\"]. A leading `!` excludes and always wins; negative-only patterns list every other file." },
    "path": { "type": "string", "description": "Plain absolute local filesystem path. ... Directory to search. Omit for the session working directory; when provided, it must name an existing directory." },
    "filter_mode": { "type": "string", "enum": ["ignore", "all"], "description": "\"ignore\" respects only plain .ignore files; \"all\" disables that filtering. Both include hidden files and .git, and neither reads any Git ignore source. The legacy value \"project\" is accepted as \"ignore\" but is not published." },
    "sort": { "type": "string", "enum": ["path", "modified"], "description": "\"path\" = byte-order path sort. \"modified\" = most recently modified first." },
    "output_mode": { "type": "string", "enum": ["paths", "details"], "description": "\"paths\" (default) returns one path per line. \"details\" returns one compact JSON object per line as {\"path\":\"...\",\"bytes\":123,\"modified\":\"YYYY-MM-DDTHH:MM:SS.NNNNNNNNNZ\"}." },
    "offset": { "type": "integer" },
    "limit": { "type": "integer", "minimum": 1, "maximum": 1000 }
  },
  "required": ["pattern"]
}
```

> `pattern` 虽然 Rust 类型接受"字符串或数组"，但按注释 `#[schemars(with = "Vec<String>")]` **只发布纯数组**（联合类型是唯一没有 provider 接受的构造）。

---

## 4. `replace`

```json
{
  "type": "object",
  "properties": {
    "pattern": { "type": "string", "description": "The regular expression to replace (Rust regex; escape literal braces)." },
    "replacement": { "type": "string", "description": "Replacement text; $1/${name} reference groups, $$ is a literal $, empty deletes the match." },
    "path": { "type": "string", "description": "Plain absolute local filesystem path. When the source reference is URI-shaped, use its equivalent local absolute path. File or directory to edit." },
    "glob": { "type": "array", "items": { "type": "string" }, "description": "Globs for directory targets, e.g. [\"**/*.rs\", \"!tests/**\"]..." },
    "literal": { "type": "boolean", "description": "Treat pattern as a literal string, not a regex." },
    "case_insensitive": { "type": "boolean", "description": "Case-insensitive matching." },
    "dot_all": { "type": "boolean", "description": "`.` also matches newlines (spanning-line matches); `\\n` also matches `\\r\\n`." },
    "max_replacements": { "type": "integer", "description": "Refuse to write if the total match count exceeds this guard." },
    "dry_run": { "type": "boolean", "description": "Preview matches and counts without writing anything." },
    "encoding": { "type": "string", "description": "Single-file target only: decode with this WHATWG label." },
    "fallback_encoding": { "type": "string", "description": "Directory target: fallback encoding for otherwise unresolved files." }
  },
  "required": ["pattern", "replacement", "path"]
}
```

---

## 5–9. Shell 组（需 `serve --enable-shell`）

```json
// run  ── RunRequest  (deny_unknown_fields)
{
  "type": "object",
  "properties": {
    "command": { "type": "string", "description": "The bash command line to run (passed to bash)." },
    "cwd": { "type": "string", "description": "Absolute path of the working directory. Omit for the session working directory." },
    "timeout_ms": { "type": "integer", "minimum": 1, "maximum": 240000, "description": "Kill the command (whole process tree) after this many milliseconds." },
    "login_shell": { "type": "boolean", "default": true, "description": "Run with a login shell (bash -lc)... Set false for a clean non-login shell (--noprofile --norc)..." },
    "encoding": { "type": "string", "description": "Known source encoding of the command's output, as a WHATWG label like \"gbk\"..." }
  },
  "required": ["command"]
}

// run_background  ── RunBackgroundRequest
{
  "type": "object",
  "properties": {
    "command": { "type": "string", "description": "The bash command line to run (passed to bash)." },
    "cwd": { "type": "string", "description": "Absolute path of the working directory. Omit for the session working directory." },
    "login_shell": { "type": "boolean", "default": true, "description": "Same as run: login shell (bash -lc) by default; false for a clean non-login shell." },
    "encoding": { "type": "string", "description": "Default source encoding for this job's output when shown by job_output (WHATWG label like \"gbk\")." }
  },
  "required": ["command"]
}

// job_output  ── JobOutputRequest
{
  "type": "object",
  "properties": {
    "job_id": { "type": "string", "description": "The job id returned by run_background." },
    "wait_ms": { "type": "integer", "default": 30000, "minimum": 0, "maximum": 240000, "description": "How long this query may take, in milliseconds. It returns earlier only when the job ends. Use 0 for an immediate snapshot." },
    "after_seq": { "type": "integer", "minimum": 0, "description": "Return output after this line number of the job's log..." },
    "encoding": { "type": "string", "description": "Decode this job's stored output with this source encoding for this call (WHATWG label like \"gbk\")..." }
  },
  "required": ["job_id"]
}

// job_kill  ── JobKillRequest
{
  "type": "object",
  "properties": {
    "job_id": { "type": "string", "description": "The job id returned by run_background." }
  },
  "required": ["job_id"]
}

// job_list  ── JobListRequest
{
  "type": "object",
  "properties": {
    "status": { "type": "string", "enum": ["running", "finished", "all"], "default": "running", "description": "Lifecycle subset to list: \"running\" (default, still-alive process trees), \"finished\" (retained exited and interrupted records), or \"all\". Omit for currently running jobs." },
    "limit": { "type": "integer", "minimum": 1, "maximum": 100, "description": "Maximum records in this page. Omit to use `fastshell.job_list_limit` (default 20)." },
    "offset": { "type": "integer", "minimum": 0, "description": "Skip this many entries of the sorted list (from a prior Partial note's offset)." }
  },
  "required": []
}
```

---

## 一个 agent 实际会收到的 `tools/list` 片段

```json
{
  "name": "inspect_local_file",
  "description": "Inspect the contents of local filesystem paths: one file ...",
  "inputSchema": { "type": "object", "properties": { "...": {} }, "required": [] },
  "annotations": { "title": "Inspect local file", "readOnlyHint": true, "destructiveHint": false, "openWorldHint": false }
}
```

## 几点容易踩坑的结论

- **键名可能与 Rust 字段名不同**：`grep` 的 `file_type` → 发布为 `type`；`glob`/`grep`/`replace` 的 `pattern` 发布为纯数组。
- **可选参数绝不带 `"null"`**：`Option<T>` 的 null 分支被折叠，可选性只体现在 `required` 数组里。
- **enum 会真的发布出来**（`output_mode`、`filter_mode`、`sort`、`status` 等），因为归一化把字段枚举的 `oneOf/const` 转成了 `string+enum`；而 `view`、`pdf_mode` 因 `schemars(with=String)` **不发布 enum**。
- **`additionalProperties` 被剥离**，但运行时仍有 `#[serde(deny_unknown_fields)]` 拒绝未知键（`grep`、`glob`、`replace` 未加该属性，而 `inspect_local_file` 的 `ReadRequest`/`BatchReadEntry` 有）。
- **schema 与反序列化可不对称**：schema 是"各 provider 子集交集"，实际 serde 可能接受更多——设计上保证"schema 描述的输入一定被接受"。
- 契约哈希 `ToolContract.hash` 是 name+description+inputSchema+annotations+group 的 SHA-256，`fastctx doctor` 用它比对跨进程一致性。

需要的话，我可以把上面 9 个 schema 直接汇总成一个可导入的 JSON 文件（例如用于给别的 MCP 客户端做 mock，或写 contract 测试）。

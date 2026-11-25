# RenderDoc接入方案

## 🧑‍💻 User

有什么办法可以为主流harness（dsh、claudecode、codex）接入renderdoc mcp辅助unity editor的frame debugger对渲染管线调试

---
## 😀 默认助手 (7)

# RenderDoc MCP × Unity Frame Debugger：接入 dsh、Claude Code、Codex 的完整方案

先给结论：**不要让 RenderDoc MCP 直接“替代”Frame Debugger，而要让两者分工、由 harness 作为协调者。** Unity Frame Debugger 提供的是 Unity 抽象层（SRP Pass、材质属性、Shader Keyword）的“语义视图”；RenderDoc 提供的是真实 API 调用、descriptor、cbuffer 数值、像素历史的“GPU 实际视图”。MCP 的价值在于把这两个视图变成 agent 可自动查询、可反复核对的数据源。

---

## 1. 先定架构：两条数据链 + 一个核对流程

```text
                    ┌─────────────────── dsh / Claude Code / Codex ───────────────────┐
                    │  读代码、改 shader、跑命令、看截图、写报告                          │
                    └───────┬───────────────────────────────────┬─────────────────────┘
                            │ MCP(A)                            │ MCP(B)
                   ┌────────▼─────────┐              ┌──────────▼──────────┐
                   │ RenderDoc MCP    │              │ Unity Editor MCP    │
                   │ .rdc 重放/取证    │              │ Play/相机/开关/Frame │
                   └────────┬─────────┘              │ Debugger 事件导出    │
                            │ renderdoc.pyd / dll    └──────────┬──────────┘
                   ┌────────▼─────────┐              ┌──────────▼──────────┐
                   │ RenderDoc replay │              │ Unity Editor (Frame │
                   │ (headless OK)    │              │ Debugger/Profiler)  │
                   └──────────────────┘              └─────────────────────┘
```

**分工原则（写进 `AGENTS.md` 或 `CLAUDE.md`）**

| 问题 | 该问谁 |
|---|---|
| 这个物件为什么被剔除？关键字是否泄漏？ | Unity 侧（Frame Debugger / 材质属性） |
| 这个 pass 实际下发了几次 draw？RT 绑错没有？ | RenderDoc |
| cbuffer 里的 `_MainLightPosition` 到底是多少？ | RenderDoc（可读 float 数值） |
| 半透明排序、Overdraw、带宽 | RenderDoc 的 pass 分析 + 自建统计 |
| Compute / 后处理中间 target | 只能看 RenderDoc |
| 像素为什么是黑的 | RenderDoc `pixel_history` + `debug_shader_at_pixel`；Unity 的 shader 逐行调试只有 D3D11/Vulkan 且受限 |

---

## 2. MCP server 选型（2026-11 现状）

按“能否被 agent 无人工干预地驱动”排序：

| 项目 | 形态 | 优点 | 代价 / 注意 |
|---|---|---|---|
| **JiaboLi-GitHub/renderdoc-mcp**（C++，约 380★，59 tools） | 独立 exe + 自带 `renderdoc.dll`；**同时有 CLI**；发布包内含 **Codex skill + `install-codex.ps1`** | 唯一明确提供“live-capture 截帧 / 断言 / diff（CI 用）/ 导出取证”的，agent 自动化程度最高 | 需 CMake 从源码构建（或直接用 Release zip） |
| **Linkingooo/renderdoc-mcp**（Python，约 180★，42 tools） | headless，走 `renderdoc.pyd` Python API，`pip install`，只需 `RENDERDOC_MODULE_PATH` | 纯 Python 零编译；高层工具齐全（`get_frame_overview`、`analyze_render_passes`、`diff_draw_calls`、`analyze_bandwidth`、移动端 GPU quirk 检测） | 作者自述 vibe-coding 项目，稳定性自负；Windows 路径需绝对化 |
| **halby24/RenderDocMCP**（Python，约 520★） | **RenderDoc GUI 扩展 + 文件 IPC**（`%TEMP%/renderdoc_mcp/`） | 最易上手；你在 RenderDoc 里手动定位 event，AI 在旁边读 | 必须开着 qrenderdoc 并已加载 capture；工具粒度粗；实测仅 Win+D3D11 |
| **Hengle/RenderDocMCP2**（约 45★，37 tools） | GUI 扩展 + IPC，`launch_renderdoc` 可一键起 RenderDoc 并加载 rdc | 独有 **`export_to_unity`**（OBJ/FBX + 三套材质属性映射 + `_UnityImport.cs`），项目自述“配合 unityMCP 效果更佳” | 强 Unity 逆向/还原场景专用；依赖 GUI |
| **SleepEaaarly/RenderDocMCP**（halby24 派生） | 同上 | 补了 **remote-device-replay**（Android 真机）与其他后端 buffer 读取 | 小众 fork |
| **EdenLabs/agentic-renderdoc** | MCP + RenderDoc 扩展，面向 live capture 交互 | 交互式 live 调试 | 较新，生态小 |

**推荐组合（Unity 渲染管线调试）**

- 主线：**JiaboLi（headless + CLI + 截帧能力）** 或 **Linkingooo（Python，装起来最快）** —— 选一个当“GPU 事实源”。
- 辅线：**Hengle/RenderDocMCP2 或 halby24** —— 当你要“我人在 RenderDoc 里指了某个 event，让 AI 接着分析”时用。
- Unity 侧：任意 Editor MCP（自建最省心，见 §4.3）；若目标是“把截帧里的材质/模型还原进 Unity”，直接用 Hengle 的 `export_to_unity`。

> ⚠️ 两个 MCP 同开时**命名要区分**（`renderdoc` / `renderdoc-gui`），否则 `mcp__renderdoc__*` 工具名撞车，agent 会选错。

---

## 3. 三个 harness 的具体接线

<details>
<summary><b>Claude Code</b>（project scope `.mcp.json` / user scope）</summary>

```bash
# 推荐用 CLI 加，写到项目 .mcp.json（团队共享）
claude mcp add renderdoc --scope project \
  --env RENDERDOC_MODULE_PATH="C:\\Program Files\\RenderDoc" \
  -- python -m renderdoc_mcp

# headless exe 版本（JiaboLi）
claude mcp add renderdoc --scope project -- "C:\\tools\\renderdoc-mcp\\bin\\renderdoc-mcp.exe"

claude mcp list        # 验证
```

`.mcp.json` 等价写法：

```json
{
  "mcpServers": {
    "renderdoc": {
      "command": "python",
      "args": ["-m", "renderdoc_mcp"],
      "env": { "RENDERDOC_MODULE_PATH": "C:\\Program Files\\RenderDoc" }
    }
  }
}
```

两个关键点：
1. **工具注册名是 `mcp__renderdoc__<tool>`**，把权限写进 `.claude/settings.json` 免得每次点允许：
   ```json
   { "permissions": { "allow": ["mcp__renderdoc__*", "Bash(renderdoccmd:*)"] } }
   ```
2. 大 capture 容易触发工具超时/上下文爆炸 —— 在 `.claude/CLAUDE.md` 里明确“先用 `get_frame_overview` / `analyze_render_passes`，禁止直接 dump `get_pipeline_state`，RT 一律存 PNG 再看图”。
</details>

<details>
<summary><b>Codex CLI / Codex Desktop</b>（`~/.codex/config.toml`）</summary>

```toml
[mcp_servers.renderdoc]
command = "C:\\tools\\renderdoc-mcp\\bin\\renderdoc-mcp.exe"
args = []
startup_timeout_sec = 30
tool_timeout_sec = 180

[mcp_servers.renderdoc.env]
RENDERDOC_MODULE_PATH = "C:\\Program Files\\RenderDoc"
```

也可用 `codex mcp add renderdoc -- "C:\\...\\renderdoc-mcp.exe"`（以你本地 `codex mcp --help` 为准）。JiaboLi 的发行包直接带 `install-codex.ps1`，会自动写 `~/.codex/config.toml` **并把 `skills/renderdoc-mcp/` 装上**——这是目前唯一“开箱给 Agent 一套 RenderDoc 工作流”的方案。

Codex 侧的两个实务建议：
- Codex 沙盒默认禁网/限写，`renderdoc` 需要读写 rdc 与临时目录 → 在 profile 里放开读取路径，或让 capture 目录落在 workspace 内。
- 用 `AGENTS.md` 固化取证流程（见 §6），比在 prompt 里重复描述省 token。
</details>

<details>
<summary><b>dsh（DeepSeek Harness）</b>（`@deepseek-ai/dsh-mcp-client` 插件实例）</summary>

dsh 是 Cordis 驱动的“everything is a plugin”，MCP **不是内建 Standard 预设里的一行**，而是**每个 server 加一个插件实例**。官方包：`@deepseek-ai/dsh-mcp-client`（源码 `packages/mcp/mcp-client/README.md`），支持 stdio 与 Streamable HTTP，工具注册为 `mcp__<serverName>__<tool>`。

在**自定义 Cordis 组合**里加一段（YAML 形态示意，字段与官方 README 一致）：

```yaml
- id: mcp-renderdoc
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: renderdoc
    transport: stdio
    command: 'C:\tools\renderdoc-mcp\bin\renderdoc-mcp.exe'
    args: []
    env:
      RENDERDOC_MODULE_PATH: 'C:\Program Files\RenderDoc'

- id: mcp-unity
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: unity
    transport: stdio
    command: 'node'
    args: ['C:\tools\unity-mcp\build\index.js']
```

要点：
- `serverName` 决定工具前缀 → `mcp__renderdoc__open_capture`、`mcp__unity__enter_playmode`，**语义化命名很关键**（别叫 `rd`）。
- 只 `install` 依赖不等于挂载，必须**重载组合**；校验方式就是看工具列表里有没有 `mcp__renderdoc__` 前缀。
- dsh 是 developer preview，明确会 breaking change —— 把 MCP 接线与 prompt 规范**放在仓库里版本化**，别只靠全局配置。
</details>

---

## 4. Unity 侧怎么让 agent “自己截帧”

这是整条链最容易卡住的一环：**没有 programmatic capture，就没有自动化核对流程**。

### 4.1 三条截帧路径

| 场景 | 做法 |
|---|---|
| Editor / Windows Player（D3D11、D3D12、Vulkan） | `renderdoccmd.exe capture --delay 90 --frame-count 1 -d C:\caps -- "C:\...\Unity.exe" -projectPath ...`（参数以 `renderdoccmd capture --help` 为准）；或 GUI 里勾 “Auto Start Capture” + 热键 |
| 已有进程 / 想按需触发 | 进程内 API：注入 `renderdoc_app.h` 拿 `RENDERDOC_API_1_6_0*`，调 `TriggerCapture()`；Unity 里用现成的 C# 封装（或 P/Invoke 自写），绑到 Editor 菜单 / MCP 工具“抓下一帧” |
| Android / Quest | RenderDoc GUI 的 remote target + `renderdoccmd android`，注意 GLES 需 layer、Vulkan 需 `VK_LAYER_PATH`；Unity 侧打包 `-force-vulkan` 更可控 |

**关键：把“截帧”也做成 MCP 工具**（`unity.capture_frame(delayFrames=2)`），让 agent 在改完 shader 后自己重截一次再 diff —— 这就是“自动核对流程”的起点。JiaboLi 版有 live-capture 工具，可直接用；Linkingooo 版没有，就得靠 Unity MCP 或 Bash 调 `renderdoccmd`。

### 4.2 用 RenderDoc 复现“Frame Debugger 的逐步视图”

Frame Debugger 的核心体验是“点一个事件 → 看当前 RT”。headless 也可以完全模拟：

```text
open_capture(frame.rdc)
  → get_frame_overview()          # 先看 pass 结构，别一头扎进 draw 列表
  → analyze_render_passes()       # 自动按 Clear/RT 切换切分 pass（这就是"渲染管线图"）
  → find_draws(min_vertices=..., blend=True)   # 按渲染状态反查 draw
  → set_event(142); get_draw_call_state(142)   # 等价于 Frame Debugger 的单事件面板
  → save_render_target(142, "./rt/142.png")    # 等价于 Frame Debugger 的 RT 预览
  → export_draw_textures(142, "./tex/")        # 输入附着一并落地
```

**这条链比 Unity Frame Debugger 强的地方**：能吃 compute pass、能看真实 descriptor 绑定、能读 cbuffer 的 float 值、能 `pixel_history` 追溯“哪一 draw 把这个像素写坏”、能 `diff_draw_calls(140,142)` 对比两个 draw 的状态差异（找“为什么这个物件和那个不一样”）。

### 4.3 Unity MCP 侧最小自建（推荐，比装第三方可控）

只需 6~8 个工具，Editor 脚本即可：

```csharp
// Assets/Editor/RenderDocMcpBridge.cs  —— 仅示例骨架
// 1) enter_playmode / exit_playmode
// 2) set_camera(transform, fov)          // 让 agent 把相机对到问题区域
// 3) set_feature(string keyword, bool)   // 开关某个 shader keyword / Renderer Feature
// 4) dump_frame_debugger_events()        // 导出事件树 JSON（反射 FrameDebuggerUtility，版本间不稳定，需 try/catch + 版本兜底）
// 5) dump_render_graph()                 // URP/HDRP: RenderGraph Viewer 的 pass/resource 依赖，比 Frame Debugger 更接近"管线结构"
// 6) capture_frame()                     // 调 renderdoccmd / in-app TriggerCapture
// 7) get_material_info(renderer)          // 关键字、变体、pass 列表
```

> `dump_frame_debugger_events` 是唯一“没有稳定公开 API”的部分：`FrameDebuggerUtility` 是 internal，只能反射或退化成“截图 + 让模型看”。**不要把核心流程压在这一步上** —— 用 RenderDoc 当事实源，Frame Debugger 只用来回答“Unity 认为它在哪个 pass”。

---

## 5. 各管线注意点（URP / HDRP / Built-in）

- **Built-in**：draw 数少、可读性最好，最适合作为“先跑通闭环”的验证项目。
- **URP**：RenderDoc 里会出现大量 `RenderLoop.Draw`/`DrawSRPBatcher` marker；先 `analyze_render_passes` 把 blit/中间 RT 摸清，再定位业务 draw。SRP Batcher 会合并 cbuffer —— 想看单个物件的材质值，**优先读 cbuffer 而不要依赖 draw 计数**。
- **HDRP**：pass 数量爆炸（几十到上百），**务必禁止 agent 全量 dump**；用 `find_draws(filter=...)` + `analyze_state_changes` 找 batching 机会，用 `diagnose_precision_issues` 查 R11G11B10 / 浅深度 / sRGB 问题。
- **移动端**：`diagnose_mobile_risks` / GPU quirk（Adreno/Mali/PowerVR）这类工具只有 Linkingooo 版有，值是有的；但真机截帧路径（remote replay）走 SleepEaaarly 或官方 GUI remote。
- **Metal（macOS）**：RenderDoc 支持有限，别把它当成 macOS 上的唯一依据，另配 Xcode GPU Capture。

---

## 6. 上下文工程：这才是成败关键

一次 HDRP 截帧的全量 pipeline state 可以是 **100KB+ JSON**，3 次调用就把上下文占满。必须约束：

1. **禁止调用清单**写进 `CLAUDE.md` / `AGENTS.md`：`get_resources`、`get_textures`（全量）、裸 `get_pipeline_state`。改用“先 overview → 再按 event 精确取”。
2. **图片落地，不要 base64**：一律 `save_render_target` / `save_texture` 存 PNG，让 harness 读文件（Claude Code / Codex 都支持看图和引用路径），比像素数组省 10~100 倍 token。
3. **迭代式缩小范围**：`get_frame_overview`（框架）→ `find_draws`（候选集 3~5 个）→ 单 event 详情 → RT 截图。任何一步超过 N 个候选就先过滤。
4. **缓存不变的东西**：资源列表在同一 capture 内恒定，第二次调用直接复用。
5. **超时**：`pixel_history`、`export_drawcall`、`identify_drawcalls` 是 30~120s 级别，MCP 客户端 timeout 要给到 180s（Codex 的 `tool_timeout_sec`、Claude Code 的 MCP 默认值记得改）。

---

## 7. 建议的 agent 工作流（可直接抄进规范）

```text
1. 复现：unity.enter_playmode → 相机/参数定位 → unity.capture_frame(delay=2)
2. 建模：renderdoc.get_frame_overview → analyze_render_passes
        → 输出"本帧 pass 树 + 每 pass RT/预算"，先给结论再展开
3. 定位：按现象选分支
   - 画面错色/发黑 → pixel_history + debug_shader_at_pixel + cbuffer 数值
   - 数量不对/漏画  → find_draws 按 blend/RT/纹理名反查；对照 Unity 侧剔除日志
   - 性能          → get_pass_timing / analyze_overdraw / analyze_bandwidth
4. 假设：diff_draw_calls(异常, 正常) → 明确指出"差异项 X 导致 Y"
5. 改代码：在 workspace 内改 shader / Renderer Feature / 关键字
6. 验证：重截 → save_render_target 前后对比（或 assert_pixel / diff capture）
7. 报告：截图 + 关键数值 + 变更点，写进 PR 描述
```

第 6 步的“机器可验证”是这套方案相对“人肉看 Frame Debugger”的唯一硬优势 —— 也让 **CI 里能跑渲染回归**（JiaboLi 版有 assert/diff 工具，直接把抓帧断言塞进流水线）。

---

## 8. 常见坑清单

| 症状 | 原因 / 处理 |
|---|---|
| `ping` / 工具调用超时 | GUI 扩展版必须 qrenderdoc 已开且已加载 rdc；headless 版检查 `RENDERDOC_MODULE_PATH` 指向**目录**而非文件 |
| 工具数不对 / “Unknown method” | server 与扩展版本不匹配（GUI 扩展 vs Python 包要成对更新） |
| Unity Editor 截帧重启/崩溃 | 关掉 “Auto Start Capture” 的热键冲突；Editor 建议 D3D11 先验；大场景 capture 上百 MB，给足 delay |
| Vulkan/ANGLE 下纹理 resourceId=0 | 已知 SWIG 绑定差异，用 `debug_vulkan_bindings` 或换 headless 版 |
| D3D12 上 Unity 帧抓不全 | RenderDoc 与 Unity 的 D3D12 组合问题较多，先降 D3D11 验证结论，再回 D3D12 复核 |
| agent 反复全量 dump | 没写 `AGENTS.md` 约束；见 §6 |
| 两个 MCP 工具名冲突 | `serverName` 必须区分 |

---

## 9. 落地顺序（建议照这个顺序做，别一次性全上）

1. **Day 1**：一个 Built-in/URP 小场景 + **Linkingooo**（`pip install` 最快）+ Claude Code 或 Codex，跑通 `open_capture → overview → RT 截图`。
2. **Day 2**：让 `capture_frame` 可被 agent 触发（`renderdoccmd capture` 一行 Bash 就够），形成“改 → 抓 → 看”循环。
3. **Day 3**：写 `AGENTS.md` 约束 + 工具白名单，把 token 压下来。
4. **Day 4**：接 Unity Editor MCP（自建 6 个工具），把 Frame Debugger 的“pass 归属”信息补上。
5. **Day 5**：换到 JiaboLi 版（或加 Hengle 版做 GUI 交互 / Unity 资源导出），把 diff/assert 接进 CI。

---

### 补充说明

- 关于 dsh：`@deepseek-ai/dsh-mcp-client` 的配置字段取自其官方包 README/文档；**组合文件的确切落盘路径随版本变化**（developer preview），以 `packages/mcp/mcp-client/README.md` 为准，我未在本机验证你的组合文件位置。
- 关于 Codex：`config.toml` 的 `[mcp_servers.*]` 与 `startup_timeout_sec` / `tool_timeout_sec` 字段按其官方文档；`codex mcp add` 子命令形态请以你本地 `--help` 确认。
- 各 RenderDoc MCP 的工具清单与局限（如 `find_by_shader` 未完全实现、`get_texture_data` 截断 10000 字符）来自各自 README，实际行为可能有出入。

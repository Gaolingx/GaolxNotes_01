# 浏览超大JSONL

## 🧑‍💻 User

帮我搜下在vscode中浏览超大jsonl的优雅方式

---
## 🤖 Assistant

I'll help you search for elegant ways to browse large JSONL files in VSCode. Let me first discover what tools are available.

I have browser and fetch tools. Let me inspect the browser tools to search effectively.

Now let me search for elegant ways to browse large JSONL files in VSCode.

Excellent! DuckDuckGo returned great results. Let me fetch the most relevant sources for details.

搜到不少好方案。先说结论:**VSCode 原生打开大 JSONL 必卡**,原因是它把整个文件读进 text model 并全量分词;真正优雅的做法是用**采用「流式读取 + 虚拟滚动(按需渲染)」**的专用 viewer 扩展。以下是对比与推荐。

## 核心推荐对比

| 扩展 | 特点 | 最适合 | 安装 |
|---|---|---|---|
| **Big JSONL Viewer** | 表格视图、`tail -f` 追实时日志、可就地编辑、支持 Remote-SSH/容器 | 正在写的**日志流**、远程服务器 | `code --install-extension rootfily.big-jsonl-viewer` |
| **JSON Vision** | 从磁盘流式读取、表格/卡片/树/Schema 图、搜索筛选排序、导出 CSV/SQL | 想要**可视化分析**(日志、LLM 对话、REST 数据) | `code --install-extension aimyerdaulet.json-vision` |
| **Quick JSONL Viewer** | 只读、Pretty print/Raw 模式、索引式虚拟渲染 | 只想**快速看几行 + 查找** | 市场搜 `Quick JSONL Viewer` |
| **Fast JSON & JSONL Viewer** | 懒加载树、虚拟表格、搜索过滤 | 树形浏览 + 小文件编辑 | 市场搜 `Fast JSON & JSONL Viewer` |

---

<details>
<summary><b>1. Big JSONL Viewer — 为「超大 + 还在增长」的文件而生(点击展开)</b></summary>

官方页面:[jsonl.co/vscode](https://jsonl.co/vscode),MIT 免费。核心卖点:

- **字节偏移行索引**:只建一次并缓存,之后只读你当前可见的那几行,**内存占用基本恒定**,几个 GB 也是秒开。
- **`tail -f` 模式**:监视磁盘文件,新行写入就增量扩展索引并自动跟到底部 —— 浏览器版永远只能拿到「快照」,这是扩展独有的优势。
- **搜索/筛选**:支持原始行子串匹配,也支持解析后对象的 `key:value` 匹配。
- **解析错误就地高亮**:不用再靠二分法找坏行。
- **就地编辑**:展开一行改原始文本,`Cmd+S` 存回,文件以流式重写,行长变化也安全。
- **Remote-SSH / Dev Containers / WSL 可用**,扩展跑在文件所在机器上,无需把 4GB 日志拷回本地。

```bash
code --install-extension rootfily.big-jsonl-viewer
```

> 注:虚拟文件系统不支持(行索引需要对真实文件做字节偏移访问)。
</details>

<details>
<summary><b>2. JSON Vision — 想把 JSONL「当数据看」时的首选(点击展开)</b></summary>

仓库:[yerdaulet-damir/json-vision](https://github.com/yerdaulet-damir/json-vision) (MIT)。

**设计理念一句话:「never hold the file in memory, never render what isn't on screen」**

| 阶段 | 做法 |
|---|---|
| 索引 | 流式扫描一次,记录每行**字节偏移**,字节级找换行,不做全量 `JSON.parse` |
| 读取 | 只 seek 读取屏幕可见行(web 用 `File.slice`,扩展用 `fs`) |
| 搜索/筛选/排序 | 一次流式遍历生成显示顺序,再随机访问取行 |
| 渲染 | 虚拟化,DOM 只持有屏幕上约 30 行 |
| 内存 | 常驻页 LRU 限制,RAM 恒定 |

实测:84 MB / 72,143 行的 JSONL **索引约 80ms**、滚动 60fps、搜索/排序 < 0.5s。

亮点功能:
- **自动识别记录形态**并切换视图:LLM 对话(`role`/`content`)→ 聊天气泡;服务器日志(`level`+`message`)→ 日志徽章;推文/REST 实体 → 卡片;其他 → 树图。
- 表格视图把嵌套对象**展平成点号列**(`a.b.c`)。
- **Schema 图**:推断每个字段的类型和填充率。
- **一键导出** CSV / SQL / Markdown,以及把选中数据发给 IDE 的 AI Chat(生成 pandas 脚本/SQL Schema)。
- Hover 任意 `JWT` 看解码 payload、Hover epoch 时间戳看可读日期。
- 支持 **VS Code / Cursor / Windsurf / Antigravity / VSCodium**(发布在 Open VSX)。

```bash
code --install-extension aimyerdaulet.json-vision
```

> 对比:`JSON Crack` 适合小文件图形化但大文件会炸;`Dadroit` 是独立桌面程序;原生 VS Code 直接冻死。**
</details>

<details>
<summary><b>3. Quick JSONL Viewer — 轻量只读(点击展开)</b></summary>

仓库:[jsh9/quick-jsonl-viewer](https://github.com/jsh9/quick-jsonl-viewer),可装市场或 [Open VSX](https://open-vsx.org/extension/jsh9/quick-jsonl-viewer)(Cursor 等可用)。

- 只读自定义编辑器,支持 **Pretty print / Raw(wrapped)/ Raw(unwrapped)** 三种模式。
- 顶部信息栏显示文件大小、总行数、最后修改时间。
- 关键设置 `quickJsonlViewer.maxLines: 0` —— **索引整个文件并按需渲染可见行**,DOM 大小有界。
- 可设置从**某一行开始**预览(`Start at line`),`Ctrl/Cmd+F` 搜索已渲染行。
- Git 的 staged/unstaged diff 仍走 VSCode 原生 diff 编辑器。
</details>

---

## 为什么 VSCode 原生打开大文件会卡

- 编辑器会把**整个文件**加载进 text model 并**预先分词(tokenize)**,超过约 10–50 MB(或某个超长单行)就会耗尽主线程,弹「Window no longer responding」。
- 大多数 JSON 扩展沿用同一模型(先全量 load 再 parse),`JSON Crack`、`JSON Grid Viewer` 等通常在 10–50 MB 之间就崩。

**优雅的本质 = 不把文件装进内存 + 只渲染可见部分。**

---

<details>
<summary><b>不加扩展的临时技巧(命令行 / 调试器)</b></summary>

**① 用 Node 调试器把 JSON 变树(经典 trick)**:新建一个 JS 文件,内容只需两行,然后按 `F5` 启动 Node.js 调试会话,左侧变量面板就会把 JSON 渲染成可折叠树:

```javascript
const data = require('./your-file.json')
debugger
```

> 局限:只适合 `.json` 而非 `.jsonl`;且文件太大(社区反馈 >700MB)时仍会挂。

**② 命令行快速查看(零依赖)**:

```bash
wc -l data.jsonl                 # 统计行数(不解析)
head -n 10 data.jsonl            # 看前 10 条
tail -n 5 data.jsonl             # 看最后 5 条
head -n 1 data.jsonl | jq .      # 美化第一条
jq -c 'select(.level=="error")' data.jsonl   # 过滤
jq -s '.' data.jsonl > data.json # 转成 JSON 数组
```

**③ 太大就用 `split` 切块再分别打开**:

```bash
split -l 100000 data.jsonl chunk_        # 每 10 万行一块
```

</details>

---

## 选型建议

- **正在被写入的生产日志 / 远程服务器上的大文件** → **Big JSONL Viewer**(`tail -f` + Remote-SSH 是杀手锏)。
- **想当数据来探索**(搜索、筛选、排序、看 Schema、导出 CSV) → **JSON Vision**。
- **只想快速扫一眼几行并查找** → **Quick JSONL Viewer**。
- **只是偶尔看一眼** → 直接用 `head` / `jq` 命令行,不必装扩展。

需要的话,我可以帮你写一份针对你实际 JSONL 结构(比如 OpenAI 微调数据、应用日志)的 `jq` 查询速查表。

**参考来源:** [jsonl.co/vscode](https://jsonl.co/vscode)、[json-vision (GitHub)](https://github.com/yerdaulet-damir/json-vision)、[quick-jsonl-viewer (GitHub)](https://github.com/jsh9/quick-jsonl-viewer)、[dev.to 大 JSON 查看技巧](https://dev.to/dtinth/viewing-and-navigating-huge-json-files-with-ease-using-visual-studio-code-246m)。

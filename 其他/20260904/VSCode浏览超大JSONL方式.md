# VSCode浏览超大JSONL方式

### 🧑‍💻 User

帮我搜下在vscode中浏览超大jsonl的优雅方式

---

### 🤖 Assistant

搜到了，**如果是 GB 级 JSONL，建议优先试 `Fast JSON & JSONL Viewer`**。它的实现方式比较符合“超大文件浏览”：流式读取、磁盘索引、虚拟表格，点选一条记录再展开 JSON 树。

这几款最值得看：

| 扩展                                                                                                              | 浏览体验                      | 超大文件处理及限制                                         |
| --------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------- |
| [**Fast JSON & JSONL Viewer**](https://marketplace.visualstudio.com/items?itemName=daucloud.vscode-json-viewer) | 表格浏览、单条树形展开、全文搜索、字段过滤     | 默认先读 200 条，后台建立磁盘索引；大文件只读，不整体载入内存。**最贴近你的需求**     |
| [**Quick JSONL Viewer**](https://marketplace.visualstudio.com/items?itemName=jsh9.quick-jsonl-viewer)           | 格式化逐条阅读，可指定起始行，也能查看原始文本   | 限量预览、索引与虚拟渲染；`Ctrl+F` 只搜索已渲染内容，不能当作全文件搜索          |
| [**JSONL Gazelle**](https://marketplace.visualstudio.com/items?itemName=gabor.jsonl-gazelle)                    | 表格、嵌套字段展开、Pretty Print、编辑 | 功能比较丰富；提供 100 MB+ 文件拆分命令，但文档不足以证明其能低内存浏览任意 GB 级文件 |

**推荐你这样用：**

1. 在 VS Code 扩展页搜索 **Fast JSON & JSONL Viewer**，认准扩展 ID：`daucloud.vscode-json-viewer`。也可以在终端安装：

   ```bash
   code --install-extension daucloud.vscode-json-viewer
   ```
2. 在资源管理器里右键 JSONL → **Open With… / 打开方式…** → **Fast JSON Viewer**。
3. 用表格快速扫记录，选中某条后在 JSON 树里展开；查看长文本时将选中记录全屏显示，`Esc` 返回。
4. 找特定数据用扩展内置的全文搜索或字段过滤。首次索引仍需扫描文件，速度取决于文件大小和磁盘。它也支持 VS Code 的 Remote Workspace，适合在远端浏览数据。[扩展说明](https://marketplace.visualstudio.com/items?itemName=daucloud.vscode-json-viewer)

**如果你主要是逐条读长文本，而不需要表格和全文件查询，可以先试 Quick JSONL Viewer，界面更简单。**以上是基于项目文档筛选，我没有拿你的实际文件做性能实测。

你的“超大”大概是 **几百 MB、几 GB，还是几十 GB**？以及是本地文件还是 Remote SSH 上的文件？这会影响最终选择。

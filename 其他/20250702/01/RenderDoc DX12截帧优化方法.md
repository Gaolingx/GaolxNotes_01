# RenderDoc DX12截帧优化方法

### 🧑‍💻 User

RenderDoc里面DX12截帧一堆Copy Constant Buffer的消息，有啥办法能简化或者分类显示吗？

---

### 🤖 Assistant

在 RenderDoc 中处理 DX12 截帧时出现大量 **Copy Constant Buffer** 消息是常见现象（通常由频繁的小缓冲区更新引起）。以下是简化或分类显示的解决方案：

---

### 方法 1：使用 **事件过滤器**（推荐）
1. **打开过滤器面板**  
   - 在事件浏览器（Event Browser）顶部，点击 **漏斗图标**（Filter Events）。
   - 或按快捷键 `Ctrl + F`。

2. **排除 Copy 类操作**  
   输入过滤规则（支持正则表达式）：
   ```regex
   -"Copy.*Buffer"   // 排除所有 Copy Buffer 操作
   ```
   - 保留绘制事件：添加 `+Draw` 确保绘制调用可见：
     ```regex
     -"Copy.*Buffer" +Draw
     ```

3. **按类别折叠**  
   - 在过滤框下方启用 **"Collapse"** 复选框。
   - 勾选 **"By API Call"** → 相同类型的复制操作会被折叠为一行。

---

### 方法 2：**分组显示**
- 在事件浏览器表头右键 → **Grouping** → 选择分组方式：
  - **By API Call**：所有 `CopyBufferRegion` 合并为同一组。
  - **By Drawcall**：将复制操作关联到最近的绘制调用下（需展开查看）。

---

### 方法 3：**着色器调试时忽略副本**
- 调试像素/顶点着色器时：
  1. 在 **Pipeline State** 选项卡中选择着色器。
  2. 切换到 **Constants** 子选项卡。
  3. RenderDoc **自动显示关联的常量数据**（无需手动追踪 Copy 消息）。

---

### 方法 4：**配置捕获选项（减少冗余副本）**
下次截帧时优化：
1. **开启捕获设置**：  
   `Capture → Edit Capture Settings`
2. 启用 **"Ref All Resources"** 和 **"Capture All Cmd Lists"**  
   （避免驱动生成冗余副本）。
3. 禁用 **"Save All Initials"**  
   （减少初始资源备份）。

---

### 方法 5：**Python 脚本扩展**
通过脚本自动标记/隐藏副本操作：
1. 打开 **Python Console** (`Window → Python Console`)。
2. 运行脚本：
   ```python
   def filter_constant_copies():
       ctx = renderdoc.GetEventBrowser()
       for ev in ctx.GetEvents():
           if "CopyBufferRegion" in ev.name and "Constant" in ev.name:
               ctx.SetEventBookmark(ev.eventId, "Constant Copy")  # 添加书签
               ctx.HideEvent(ev.eventId, True)  # 隐藏事件
   filter_constant_copies()
   ```
   - 效果：隐藏所有常量缓冲副本并添加书签，可通过书签导航栏快速定位。

---

### 额外建议
- **书签关键操作**：  
  右键重要事件 → **Add Bookmark**，后续通过书签栏快速跳转。
- **资源别名（Aliasing）**：  
  在 DX12 代码中使用 `ID3D12Resource::CreatePlacedResource` 减少显存副本。
- **批量更新常量**：  
  合并小常量更新为大缓冲区，减少 `CopyBufferRegion` 调用次数。

---

通过上述方法，可大幅简化事件列表，聚焦于 **Draw/Dispatch** 等核心操作。优先推荐 **过滤 + 分组组合**（方法1+2），无需修改代码即可快速清理视图。
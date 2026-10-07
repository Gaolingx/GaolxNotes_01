# tmux日志导出

## 🧑‍💻 User

tmux如何读取/导出历史日志

---
## 😀 默认助手 (7)

# tmux 历史日志的读取与导出

tmux 里“历史日志”其实分两类，处理方式不同：

| 类型 | 说明 | 用途 |
|------|------|------|
| **scrollback（回滚缓冲区）** | 当前 pane 已经显示过的内容，存在内存里 | 查看、临时导出 |
| **pipe-pane（实时日志）** | 把 pane 输出持续写入文件 | 长期、实时记录 |

---

## 一、读取历史：进入 Copy Mode

默认前缀键是 `Ctrl-b`：

```text
Ctrl-b  [          # 进入 copy mode
```

进入后常用按键（emacs 风格，默认）：

```text
PageUp / PageDown        翻页
↑ ↓ ← → / Ctrl-u Ctrl-d  逐行滚动
g / G                    跳到历史开头 / 末尾
/ 关键字                 向前搜索
? 关键字                 向后搜索
n / N                    重复搜索 / 反向重复
空格                      开始选择
Enter                     复制选区并退出
q 或 Esc                  退出 copy mode
```

若习惯 vi 键位，可在配置里改：

```text
# ~/.tmux.conf
setw -g mode-keys vi
```

> 注意：默认 `history-limit` 只有 2000 行，想要更多历史（比如 5 万行）需在配置里加大，**且新开的 pane 才生效**：
> ```text
> set-option -g history-limit 50000
> ```

---

## 二、导出历史到文件

### 方法 1：`capture-pane`（最推荐，可脚本化）

这是导出 scrollback 最灵活的命令：

```bash
# 打印当前 pane 可见内容到 stdout
tmux capture-pane -p

# 从历史开头(-S -)到当前，导出全部 scrollback
tmux capture-pane -pS - > ~/history.txt

# 导出某 pane（-t 指定目标）
tmux capture-pane -pS - -t 0:1.0 > ~/pane.txt

# 保留颜色/转义序列（-e），并合并软换行（-J）
tmux capture-pane -pS - -e -J > ~/history_colored.txt
```

常用参数：

| 参数 | 作用 |
|------|------|
| `-p` | 输出到 stdout（而非存进 buffer） |
| `-S -` | 起始行，`-` 表示历史最开头 |
| `-E -` | 结束行，`-` 表示当前末尾 |
| `-e` | 保留 ANSI 转义（颜色） |
| `-J` | 合并被自动折行的长行 |
| `-t <target>` | 指定 pane / window / session |

### 方法 2：复制到 tmux buffer 再保存

```text
Ctrl-b  [        # 进入 copy mode，选中内容（空格开始 / Enter 复制）
Ctrl-b  :        # 进入命令模式
save-buffer ~/history.txt
```

或直接一行命令：

```bash
tmux save-buffer ~/history.txt
```

查看缓冲区：`tmux show-buffer`

### 方法 3：`pipe-pane` —— 实时持续记录

适合需要**长期、实时**留档的场景（如日志、长任务）：

```bash
# 把 pane 的输出实时追加到文件（-o 表示仅开启一次时执行）
tmux pipe-pane -o 'cat >> ~/pane.log'

# 指定目标 pane
tmux pipe-pane -t 0:1.0 -o 'cat >> ~/pane.log'

# 关闭 pipe
tmux pipe-pane -t 0:1.0
```

带时间戳的记录更实用：

```bash
tmux pipe-pane -o 'while IFS= read -r line; do printf "%s %s\n" "$(date +%F\ %T)" "$line"; done >> ~/pane.log'
```

---

## 三、综合示例

**导出某个会话所有窗口的历史：**

```bash
for w in $(tmux list-windows -F '#{window_index}'); do
  tmux capture-pane -pS - -t "$w" > "~/window_${w}.log"
done
```

**日常推荐配置**（`~/.tmux.conf`）：

```text
set-option -g history-limit 100000
setw -g mode-keys vi

# 快捷键 r 直接导出当前 pane 全部历史
bind r capture-pane -S - \; save-buffer ~/tmux-history.txt

# 快捷键 L 开关实时日志
bind L pipe-pane -o 'cat >> ~/tmux-$(date +%F).log'
```

---

## 小结

- **看历史** → `Ctrl-b [` 进 copy mode，翻页/搜索。
- **导出一次性历史** → `tmux capture-pane -pS - > file`。
- **持续记录** → `tmux pipe-pane -o 'cat >> file'`。
- **默认历史太短** → 提前设大 `history-limit`（对新 pane 生效）。

如果你告诉我具体场景（比如“想导出带颜色的完整历史”或“想记录某个持续运行的进程日志”），我可以给出更精准的命令。

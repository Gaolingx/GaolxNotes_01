# tmux终端滚动无法

## 🧑‍💻 User

vscode ssh 终端开的tmux无法翻页是怎么回事？

---
## 😀 默认助手 (7)

这是一个很常见的 tmux + VSCode 集成终端问题，根源在于**「谁来负责滚动」这件事被 tmux 抢走了**。下面分原因和解决方案说明。

## 原因

1. **tmux 使用了备用屏幕（alternate screen）**
   tmux 启动后会接管整个终端屏幕，VSCode 集成终端自己的 scrollback 缓冲区就失效了。你在 VSCode 里往上滚，看到的是 tmux 重绘的当前画面，而不是历史输出。

2. **滚动历史现在归 tmux 管**
   tmux 有自己的 `history-limit`，输出的历史存在 tmux 内部。想回看只能进入 tmux 的 **copy mode（复制模式）**，而不是靠 VSCode 的滚动条。

3. **VSCode 终端和 tmux 对鼠标滚轮事件的争夺**
   - 如果 tmux **没开鼠标模式**（`mouse off`，默认），滚轮事件会被 VSCode 终端吞掉，但 tmux 在备用屏里，VSCode 滚的只是它自己的空缓冲 → 看起来「滚不动」。
   - 如果 tmux **开了鼠标模式**（`mouse on`），只有当 VSCode 正确把鼠标上报事件转发给 tmux 时才能滚，某些 VSCode / 终端组合下转发不完整，也会失效。

4. 少量情况是 `$TERM` 或 GPU 渲染相关的小 bug。

<details>
<summary><strong>方法 1：开启 tmux 鼠标模式（推荐，最省事）</strong></summary>

编辑 `~/.tmux.conf`：

```tmux
set -g mouse on
set -g history-limit 50000
```

然后让配置生效：

```bash
tmux source-file ~/.tmux.conf
# 或者杀掉重开
```

这样在 VSCode 里直接用**鼠标滚轮**就能滚动 tmux 各 pane 的历史了。

如果开了鼠标还是滚不动，可能是 VSCode 没把滚轮事件转发进去，试试在 VSCode 设置里检查：

```json
"terminal.integrated.mouseWheelScrollSensitivity": 1
```

或临时按住 `Shift` 再滚（部分终端下 `Shift+滚轮` 会强制走终端自身滚动）。

</details>

<details>
<summary><strong>方法 2：用 copy mode 手动翻页（不依赖鼠标）</strong></summary>

即使鼠标有问题，键盘永远可用：

| 操作 | 按键 |
|------|------|
| 进入 copy mode | `Ctrl-b` 然后 `[` |
| 向上/下翻页 | `PageUp` / `PageDown` |
| 逐行滚动 | 方向键，或 `Ctrl-u` / `Ctrl-d` |
| 跳到顶部/底部 | `g` / `G`（vi 模式） |
| 退出 copy mode | `q` 或 `Esc` |

想要 vi 风格按键：

```tmux
setw -g mode-keys vi
```

</details>

<details>
<summary><strong>方法 3：彻底把滚动权交回 VSCode</strong></summary>

如果你更习惯用 VSCode 右侧滚动条 / 触控板滚动，可以让 tmux 完全退出「全屏接管」的干扰路线：

- 确保 tmux 里 **不要** 开鼠标模式：

```tmux
set -g mouse off
```

- 但注意：只要 tmux 在跑，VSCode 的 scrollback 基本仍看不到 tmux 内部历史。所以这条路通常只在「你想滚 VSCode 自己的输出」时有用，**对于 tmux 内部历史还是得靠方法 1 或 2**。

</details>

<details>
<summary><strong>常见坑 & 补充</strong></summary>

- **`Ctrl-b` 被 VSCode 占用？** 一般不会，但如果按 `Ctrl-b [` 没反应，检查是否 tmux 前缀键被改过，或键盘焦点在别处。
- **分屏后只在某个 pane 滚动**：copy mode 只作用于当前 pane，鼠标模式也是。
- **滚轮滚的是 shell 而不是 tmux**：确认你真的在 tmux 会话里（看底部绿条）。
- **WSL/远程 SSH**：原理相同，问题主要出在 tmux 层，不在 SSH 层。
- **history-limit 太小**：默认只有 2000 行，回看不到更早的输出，记得调大。

</details>

## 一句话总结

不是 VSCode 坏了，而是 **tmux 在备用屏里自己管滚动**。最实用的做法是 `set -g mouse on` 让鼠标直接滚 tmux 历史；没有鼠标就用 `Ctrl-b [` 进 copy mode，`q` 退出。

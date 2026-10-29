# PowerShell 启动 崩溃 更新 通知 中文 导致 PadRight 负值

## 🧑‍💻 User

https://github.com/PowerShell/PowerShell/issues/28131
这个issue讲了什么

---
## 😀 默认助手 (7)

## PowerShell Issue #28131 概览

**标题**：*PowerShell can crash on startup when displaying the update notification under zh-Hans due to negative PadRight width*（在 zh-Hans 下显示更新通知时，负的 `PadRight` 宽度会导致 PowerShell 启动即崩溃）

| 项目 | 内容 |
|---|---|
| 状态 | **Open**（已指派给 `daxian-dbw`，标签 `Needs-Triage`） |
| 作者 | 高氙酸 (`Perxenic-Acid`)，2026-09-30 创建 |
| 关联修复 | PR **#28236** *Fix startup crash when formatting localized update notifications* |

---

### 一句话总结

更新通知横幅的格式化代码**假定“第一行一定是最长的”**。中文、日文、韩文、俄文等本地化文案并不满足这个假设，于是算出一个**负数**的填充宽度并传给 `String.PadRight()`，抛出 `ArgumentOutOfRangeException`；该异常发生在控制台宿主启动阶段，无人捕获，直接触发 `Environment.FailFast()`，结果是 **`pwsh` 完全无法启动**。

```
Process terminated.
totalWidth ('-17') must be a non-negative value. (Parameter 'totalWidth')
   at System.String.PadRight(Int32 totalWidth, Char paddingChar)
   at Microsoft.PowerShell.UpdatesNotification.ShowUpdateNotification(PSHostUserInterface hostUI)
   at Microsoft.PowerShell.ConsoleHost.Start(...)
```

### 根因代码

`src/Microsoft.PowerShell.ConsoleHost/host/msh/UpdatesNotification.cs`（约第 136 行）：

```csharp
// The first line is longest, if the message changes, this needs to be updated
line3Padding = line3Padding.PadRight(
    line1Length - line3Length + 3   // ← 可能为负
);
```

以 zh-Hans 的 `PreviewUpdateNotificationMessage` 为例：第一行 37 字符、第三行（URL 行）57 字符，因此

$$37 - 57 + 3 = -17$$

### 为什么 2026-10-20 突然集体爆发

`ShowUpdateNotification()` 对新版本有 `NotificationDelayDays = 7` 天的延迟。7.6.7 于 2026-10-13 发布，所以**通知第一次真正可显示的瞬间**正是 `2026-10-20T00:00Z`，也就是北京时间 08:00 左右。于是当天 zh-CN、ja-JP、ru-RU 等用户同时报告了相同问题，看起来像是“更新之后才坏”。

<details>
<summary><b>影响范围不止 zh-Hans（点击展开）</b></summary>

- **7.6.6 stable 同样中招**：7.6.6 是第一个带本地化 `ManagedEntranceStrings` 的 7.6 版本；7.6.0–7.6.5 各语言都只有英文模板，而英文模板第一行的确最长。
- 13 种本地化中 **9 种会算出负数**（cs、de、fr、it、ja、ko、ru、zh-Hans、zh-Hant）；报告出现过的实测值：zh-Hans `-17`、zh-Hant `-13`、ja-JP `-10`、ko-KR `-23`、ru-RU `-11`、fr-FR `-19`、de-DE `-1`。
- **行 2 也会负**：cs、de、fr、it、ko、ru 的语言里第二行（“立即升级或访问发布页”那行）比标题长，所以两个 `PadRight` 都需要 clamp。
- 代码在 `v7.6.6`、`v7.6.7`、`v7.7.0-preview.5`、`master` 中**完全没变**。

</details>

<details>
<summary><b>常见误解与正确绕过办法（点击展开）</b></summary>

**❌ 升级到 7.6.7 并不等于已修复**：它只是当前最新版，还没有“可通知”的新版本而已。等 7.6.8 跨过 7 天窗口，同样会崩。

**❌ `POWERSHELL_UPDATECHECK=LTS` 不是绕过办法**：LTS 模板同样会算出负值（zh-Hans 为 $-15$）。

**✅ 唯一有效绕过**：

```powershell
setx POWERSHELL_UPDATECHECK Off
```

然后**必须完全关掉终端窗口再重开**。在 Windows Terminal 里按“Enter 重启”是不够的，因为新 shell 会继承已经运行的 `WindowsTerminal.exe` 进程的环境块，所以仍然会崩。

其他说明：非交互/管道调用不受影响；Windows PowerShell 5.1 完全没有更新通知功能，因此不受影响。

</details>

<details>
<summary><b>建议的修复方式与次要问题（点击展开）</b></summary>

PR #28236 的做法是把两处宽度都 clamp：

```csharp
PadRight(Math.Max(0, line1Length - line3Length + 3))
```

社区还建议：与其假设“第一行最长”，不如**取三行中最长的那一行**来算 padding；并且由于 `line1Length` / `line3Length` 用的是 `String.Length`（按码位计数），而中日韩字符实际占两个显示列，即使不崩溃，中文/日文的横幅**对齐也是歪的**。要真正修好布局，应该测量显示宽度而不是 `Length`。另外需要 backport 到 `release/v7.6`（当前 LTS 线），不能只进 `master`。

</details>

### 一句话建议

如果你正在用 7.6.6 或 7.6.7 且系统语言是中文、日文、韩文、俄文、德文或法文，**现在就设置 `POWERSHELL_UPDATECHECK=Off`**，否则下一个维护版本过 7 天后，`pwsh` 会在启动时静默崩溃。

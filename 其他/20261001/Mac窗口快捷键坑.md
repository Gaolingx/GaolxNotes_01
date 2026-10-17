# Mac窗口快捷键坑

## 🧑‍💻 User

习惯win/linux用户用macos会遇到哪些问题，比如鼠标滚轮方向反了、大写锁要长按等等

---
## 😀 默认助手 (7)

下面按“**上手第一天最容易翻车**”到“**用久了才发现的深坑**”分层整理。核心差异可以浓缩成一句话：**macOS 把「所有应用的菜单」搬到屏幕顶端，把「修饰键主力」从 Ctrl 换成 ⌘，把「关机式思维」换成「不关应用只关窗口」。**

## 🔥 最容易踩的 10 个坑（先看这个）

| 现象 | Windows/Linux 习惯 | macOS 实际行为 | 解决 |
|---|---|---|---|
| 滚轮方向反了 | 向下滚 = 页面下移 | 触控板「自然滚动」默认开，滚轮方向相反 | 系统设置→鼠标，关掉「自然滚动」 |
| 大写锁要按一会儿 | 轻点就锁定 | Caps Lock 有约 0.2–0.5s 防误触延迟 | `hidutil property --set '{"CapsLockDelayOverride":0}'` |
| 找不到窗口按钮 | 右上角 ✕ − □ | 红黄绿三个灯在**左上角** | 习惯即可，`⌘W` 关窗、`⌘Q` 退出 |
| 关掉窗口程序还在 | 关窗=退出 | 关窗后 App 仍在 Dock 运行 | 用 `⌘Q` 真正退出；`⌘⌥Esc` 强退 |
| 最大化不是最大化 | □ 铺满屏幕 | 绿灯 = 全屏（独占），不是放大窗口 | **按住 ⌥ 点绿灯** = 真正的最大化(zoom) |
| Ctrl+C 复制没反应 | Ctrl 系快捷键 | 复制/粘贴是 `⌘C` / `⌘V` | Ctrl 在终端/Emacs 场景才用 |
| Delete 键是退格 | Delete = 向后删 | 标着 delete 的键 = Backspace | `Fn+Delete` 才是向后删除 |
| Home/End 不听话 | 跳到行首/行尾 | 默认滚到文档顶部/底部 | 用 `⌘←` / `⌘→`（或 `Fn+←/→`） |
| 装不了软件 | 双击 .exe | 没有 exe，也没有 apt/yum | 用 `.dmg/.pkg`，包管理用 **Homebrew** |
| 外接 NTFS 硬盘写不进 | 随便读写 | NTFS 默认**只读** | 装 Paragon / Mounty / `ntfs-3g` |

---

## ⌨️ 键盘

<details>
<summary><b>修饰键体系 & 快捷键（点开）</b></summary>

- **⌘ (Command)** 才是 Windows 里的 Ctrl：`⌘C/V/X/A/S/Z/F/Q/W/T/N`。
- **⌃ (Control)** 主要给终端、Emacs 系操作、以及系统级快捷键用。
- **⌥ (Option/Alt)** = 输入特殊字符、显示菜单里的隐藏选项（按住 ⌥ 打开菜单会多出条目）。
- **🌐/Fn** = 各种媒体/功能切换。

**文字框里内置 Emacs 快捷键**（Linux 用户会很爽，Windows 用户可能完全不知道）：

| 快捷键 | 作用 |
|---|---|
| `⌃A` / `⌃E` | 行首 / 行尾 |
| `⌃B` / `⌃F` | 左/右移一格 |
| `⌃K` | 删到行尾 |
| `⌃D` | 删除光标后字符 |
| `⌃H` | 退格 |
| `⌃Y` | 粘贴刚删的内容 |

**常见快捷键对照：**

| 功能 | Windows | macOS |
|---|---|---|
| 切应用 | Alt+Tab | `⌘Tab` |
| 同应用切窗口 | — | `` ⌘` `` |
| 强退 | Ctrl+Shift+Esc | `⌘⌥Esc` |
| 锁屏 | Win+L | `⌃⌘Q` |
| 截图 | PrintScreen | `⌘⇧3` / `⌘⇧4` / `⌘⇧5` |
| 任务管理器 | Ctrl+Shift+Esc | 活动监视器 |
| 全屏 | F11 | `⌃⌘F` |
| 表情符号 | Win+. | `⌃⌘Space` |
| 剪贴板历史 | Win+V | 系统无原生，需第三方（Maccy 等） |

</details>

<details>
<summary><b>其他键盘坑</b></summary>

- **Apple 键盘的 Caps Lock 没有指示灯**，锁没锁全靠屏幕右上角提示，很不直观。
- **功能键默认是媒体键**（亮度/音量），要按 F1–F12 得先按住 `Fn`，或在设置里改成标准功能键。
- **没有独立的 Insert / PrintScreen 键**。
- **特殊字符靠 ⌥ 组合**：`⌥u` 再按 `u` → ü，`⌥e` 再按 `e` → é，`⌥2` → ™。
- **英式键盘布局和 Windows 英式不同**（`@`、`"`、`#` 位置对不上），刚换过去会疯狂打错。
- **Tab 键默认不能在所有控件间跳转**（Full Keyboard Access 默认关），需要去「键盘→键盘导航」打开，否则对话框里 Tab 跳不过去。
- **外接 PC 键盘**：Windows 键 = ⌘，Alt = ⌥。手感和 Mac 原生键盘（⌘ 在空格旁边）不一致，很多人会把 ⌘ 和 ⌃ 对调：`系统设置→键盘→键盘快捷键→修饰键`。

</details>

---

## 🖱️ 鼠标 / 触控板

<details>
<summary><b>点开看细节</b></summary>

- **自然滚动是同一个底层开关**：改鼠标的会连触控板一起变。想「鼠标反向、触控板自然」，用 **Scroll Reverser** 或 **LinearMouse**。
- **鼠标指针加速曲线很“飘”**，没有 Windows/Linux `xset m 1 1` 那种线性档位，要精确手感用 **LinearMouse / SteerMouse / USB Overdrive**。
- **右键重开**：全新的机器触控板默认可能没开右键，需在触控板设置里勾「辅助点按」。Linux X11 那种「中键粘贴」不存在。
- **轻拍拍击 (Tap to Click) 默认关**，三指拖拽也默认关，都要手动开。
- **没有滚轮“平滑滚动/逐行滚动”档位**，滚动是带惯性的，长时间读文档体感不同。
- 滚轮中键点击在浏览器里可用，但系统层面用途很少。
- 触控板手势（四指上滑 Mission Control、三指/四指左右切桌面、双指缩放）是 macOS 的优势，值得花十分钟学。

</details>

---

## 🪟 窗口 & 系统操作

<details>
<summary><b>点开看细节</b></summary>

- **菜单栏在屏幕最顶端且是全局的**，菜单内容跟着当前活动 App 变，不在窗口里。
- **绿灯=全屏**（菜单栏/Dock 都会自动隐藏），`⌥+绿灯`或双击标题栏=放大窗口。**macOS 15 Sequoia 起**才加入拖到屏幕边缘的分屏贴靠；更早版本要装 **Rectangle / Magnet**。
- **红点只关窗口，不退出程序**——这是 Windows 用户最深的肌肉记忆冲突。
- **没有任务栏**，只有 Dock（下面）+ `⌘Tab`（应用切换）+ Mission Control（虚拟桌面）。
- **`⌘H` 隐藏应用**、`⌘M` 最小化、`⌘⌥H` 隐藏其他。
- **强制退出** = `⌘⌥Esc`；看进程用「活动监视器」，不是 Ctrl+Alt+Del。
- **滚动条默认隐藏**，滚动时才出现。
- **双击标题栏**可设置为缩放或最小化（系统设置里选）。
- 卡住时那个彩虹圈叫 **beachball（风火轮）**。

</details>

---

## 🗂️ 文件系统 & Finder

<details>
<summary><b>点开看细节</b></summary>

- **没有盘符** `C: D:`，外接盘挂载在 `/Volumes/xxx`。
- **文件系统默认大小写不敏感**（保留大小写）。这会让 Git 出现「只改大小写的重命名」诡异问题。
- **Finder 里没有“剪切”文件**：`⌘C` 复制后按 `⌘⌥V` 才是移动。
- 常用 Finder 快捷键：`⏎` 重命名、`⌘O`/`⌘↓` 打开、`⌘↑` 上一层、`⌘⇧G` 前往路径、`⌘⇧.` 显示隐藏文件、空格 Quick Look 预览、`⌘⌫` 移到废纸篓、`⌘⇧⌫` 清空废纸篓。
- 每个目录里都有 `.DS_Store`（存储视图设置），会污染 Git 和共享盘，记得加进 `.gitignore`。
- **APFS/HFS+**、扩展属性（xattr）、资源分叉、**quarantine 隔离属性**（Gatekeeper 拦截的根源）。
- **外接 NTFS 默认只读**；exFAT/FAT32 正常。
- 废纸篓是**每个卷一个**（`.Trashes`），未必能跨盘“还原”。
- `/etc`、`/System` 受 **SIP** 保护，很多 Linux 上理所当然的改配置操作会被拒绝。

</details>

---

## 📦 软件安装 & 包管理

<details>
<summary><b>点开看细节</b></summary>

- **没有 apt/yum/dnf/pacman**，事实标准是 **Homebrew**：
  - Apple Silicon 装在 `/opt/homebrew`，Intel 在 `/usr/local`；
  - `brew install xxx`（命令行工具）、`brew install --cask xxx`（GUI 应用）、`brew services`（后台服务）；
  - 记得把 `eval "$(/opt/homebrew/bin/brew shellenv)"` 加进 `~/.zshrc`。
  - 也有 MacPorts。
- **分发格式**：`.dmg`（拖进 Applications）、`.pkg`（安装器）、`.app`（其实是目录，右键「显示包内容」）。
- **Gatekeeper / 公证(Notarization)**：下载的未签名 App 会被拦。
  - 临时放行：右键→打开，或 `xattr -dr com.apple.quarantine /path/app.app`。
- **卸载没有“卸载程序”**：拖到废纸篓，残留配置在 `~/Library/Application Support`、`~/Library/Preferences`、`~/Library/Caches`，可用 **AppCleaner** 清理。
- **装不了 `.exe`**：要跑 Windows 程序得用 Parallels / VMware / CrossOver / Wine / UTM。
- Apple Silicon 跑 x86 程序靠 **Rosetta 2**（`arch -x86_64`），Homebrew 也有 x86 分版（`/usr/local`）。

</details>

---

## 💻 终端 & 开发环境（Linux 用户重点）

<details>
<summary><b>点开看细节</b></summary>

**Shell / 工具链**
- 默认 shell 是 **zsh**（Catalina 起），配置文件 `~/.zshrc`；系统自带 bash 是很旧的 **3.2**（因 GPLv3），要用新版 `brew install bash` 再 `chsh`。
- 编译器是 **clang** 不是 gcc（`xcode-select --install` 装命令行工具）；真 gcc 用 `brew install gcc`。
- 没有原生 Linux 容器，要 **Docker Desktop / Colima / OrbStack**；Apple 也出了 `container` 工具。

**BSD 工具 vs GNU 工具（最高频的踩坑点）**

| 场景 | Linux/GNU | macOS(BSD) |
|---|---|---|
| 原地改文件 | `sed -i 's/a/b/' f` | `sed -i '' 's/a/b/' f` |
| Perl 正则 | `grep -P` | 不支持，用 `ggrep`/`rg` |
| 超时 | `timeout 5 cmd` | 无，用 `gtimeout`（coreutils） |
| 定时执行 | `watch` | 无，用 `brew install watch` |
| 解析路径 | `readlink -f` | 老版本没有，用 `realpath` |
| 日期运算 | `date -d ...` | 不支持，用 `date -v` 或 `gdate` |

> 万能解法：`brew install coreutils gnu-sed gnu-tar findutils grep`，然后用 `g` 前缀版（`gls`、`gsed`、`gfind`、`ggrep`、`gdate`…），或在 PATH 里把 `gnubin` 放前面。

**系统管理差异**
- **没有 systemd**，用 **launchd / launchctl**，服务配置是 plist，放在 `~/Library/LaunchAgents`、`/Library/LaunchDaemons`。
- **没有 `/proc`**，进程/内核信息靠 `sysctl`、`ps`、`top`。
- **网络**：`ifconfig`、`netstat`、`scutil`，没有 `ip` / `nmcli`；Wi-Fi 细节曾用 `airport`（新版改了）。
- macOS 专属好用小工具：`open`（替代 xdg-open）、`pbcopy`/`pbpaste`（替代 xclip）、`mdfind`（Spotlight 命令行）、`diskutil`、`hdiutil`、`caffeinate`（防休眠）、`sips`（图片处理）、`defaults`（改隐藏偏好）、`softwareupdate`。
- **大小写不敏感的 FS** 会让 Git 的「仅大小写重命名」失效：`git mv a b` 前要先 `git mv a tmp && git mv tmp b`。

</details>

---

## 🔐 权限 & 安全（自动化/脚本杀手）

<details>
<summary><b>点开看细节</b></summary>

- **TCC 隐私权限**：摄像头、麦克风、屏幕录制、完全磁盘访问、辅助功能、输入监控、自动化，第一次用都会弹窗要授权。
  - **从终端跑脚本**时尤其容易卡在权限上：抓屏要「屏幕录制」权限、控制其他 App 要「辅助功能」权限、读某些目录要「完全磁盘访问」。
  - 给了权限通常要**重启对应 App/终端**才生效。
- **SIP**（System Integrity Protection）保护 `/System`、`/usr` 等，很多 Linux 上的“直接改系统文件”做法行不通。
- `sudo` 输密码；可开启 **Touch ID for sudo**（改 `/etc/pam.d/sudo` 加 `pam_tid.so`）。
- 新版本对「后台项目」「登录项」和「App 管理其他 App」还有单独的开关，系统更新后偶尔会被重置。

</details>

---

## 🧩 其他零碎但常被吐槽的点

- **Spotlight**（`⌘Space`）很好用，但和输入法/其他软件快捷键易冲突。
- **Quick Look（空格）** 是神器，Windows 用户往往很久才发现。
- **截图默认存到桌面**，可用 `defaults write com.apple.screencapture location <路径>` 改。
- **没有系统级剪贴板历史**，需装 Maccy / Paste。
- **AirDrop / 隔空投送、Handoff、iCloud** 是生态优势，跨平台用户用不到会觉得“多余”。
- **时区/输入法**：`⌃Space` 切输入法，`⌘Space` 是 Spotlight，别混。
- **BSD 版 `ls` 的 `-G` 上色**，不是 `--color`。
- **`~` 展开、路径含空格**在 Finder 和终端之间复制时偶尔要转义。
- **brew 安装的软件不在 `/usr/local/bin` 就是 `/opt/homebrew/bin`**，PATH 忘了加会“命令找不到”。

---

## ✅ 迁移建议（少走弯路）

1. **第一周**：先改三件事——关自然滚动（或装 Scroll Reverser）、把 ⌘ 当 Ctrl 用、学会 `⌘Q` 和 `⌘Tab`。
2. **装好基础工具**：Homebrew → `brew install coreutils gnu-sed gnu-tar findutils grep` → 装 Rectangle（分屏）、Maccy（剪贴板）、Stats / iTerm2 / Raycast。
3. **Linux 用户**：把脚本里的 `sed -i`、`grep -P`、`date -d`、`timeout`、`readlink -f` 全部替换成 GNU 版，能省掉一大半“为什么脚本跑不了”。
4. **Shell 配置**：确认 `~/.zshrc` 里 PATH 包含 `/opt/homebrew/bin`，并 `eval brew shellenv`。
5. **权限预期**：任何自动化/抓屏/控制类工具，第一次都会要 TCC 权限，给完记得重启该 App。

如果你告诉我具体是什么身份（纯日常办公 / 后端开发 / 网工 / 设计），我可以再给一份**针对性的“必改设置 + 必装软件 + 快捷键改键方案”清单**。

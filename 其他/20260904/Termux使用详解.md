# Termux使用详解

### 🧑‍💻 User

介绍termux的技术路线、原理、具体使用方法。条件：中等偏上arm cpu，国产android魔改系统，x64 uefi win10，备选ubuntu

---

### 🤖 Assistant

# Termux 技术路线、原理与使用方法详解

下面从**技术路线（它是什么/怎么实现的）**、**底层原理（为什么能跑）**、**具体使用**、以及**结合你给出的四条环境条件**四部分展开。你的条件（中等偏上 ARM、国产魔改 ROM、x64 UEFI Win10、备选 Ubuntu）我放在最后一节专门对应分析。

---

## 一、先一句话定性

> **Termux 不是虚拟机，也不是模拟器，更不是双系统。它是一个跑在 Android 应用沙箱里的"原生 Linux 用户态环境 + 终端模拟器"。**

Android 内核本身就是 Linux 内核。Termux 做的事，是**编译一套适配 Android 的 ELF 二进制程序**，直接在 Android 内核上以普通 App 用户身份运行，不需要 root、不需要 chroot 容器。你的"中等偏上 ARM"就是给它干活的 CPU。

---

## 二、技术路线（它是怎么搭起来的）

Termux 的完整技术路线可以拆成五层：

| 层 | 内容 | 说明 |
|----|------|------|
| 1. 终端模拟 | Terminal Emulator | 负责键盘输入、ANSI 转义、颜色、快捷键，本质是一个 Android View |
| 2. 运行时环境 | Native ELF binaries | 程序是**针对 Android 编译的原生二进制**，直接由内核 execve 执行 |
| 3. C 库层 | **Bionic libc** | Android 的 libc（BSD 系，非 glibc），Termux 的所有包都针对它编译 |
| 4. 路径层 | Prefix 重定位 | 所有文件放在 `$PREFIX`，通过重写路径让"Unix 世界"正常工作 |
| 5. 包管理层 | `pkg` / `apt` | 自建软件仓库，预编译好 aarch64/arm/x86_64/i686 四架构的包 |

### 2.1 关键点一：原生执行，不是模拟

在 Termux 里跑 `python`、`clang`、`nginx`，和你 PC 上跑程序是同一类事情——**CPU 直接执行机器码**。所以：

- ARM64 手机上跑的是 aarch64 的二进制，性能损失接近零；
- 对比 QEMU 模拟 x86（慢一到两个数量级），Termux 是"满血"的；
- 这也是为什么"中等偏上 ARM"在 Termux 里能跑编译、数据库、甚至机器学习推理。

### 2.2 关键点二：Bionic libc 兼容层

这是 Termux 最大的工程量。Android 用的是 **Bionic**，不是桌面 Linux 的 glibc。绝大多数开源软件默认按 glibc 写/编译，Termux 团队需要：

- 给软件打补丁，适配 Bionic 缺失的接口；
- 交叉编译，指定 `--sysroot` 为 Android NDK 环境；
- 维护一个几千个包的补丁仓库（`termux-packages`）。

所以你能看到 Termux 的包数量和版本进度直接取决于上游补丁情况，这也是它**不能直接装 deb 包、不能直接 pip install 某些带 C 扩展的包**的根源（有些能装，有些要 `pip install --no-cache-dir` + 编译工具链）。

### 2.3 关键点三：Prefix 重定位

Android 的 App 沙箱让 Termux 只能写自己的私有目录：

```
/data/data/com.termux/files/
├── home/          # 你的家目录
├── usr/           # $PREFIX，等价于 Unix 的 /usr
│   ├── bin/
│   ├── lib/
│   ├── etc/
│   └── ...
└── tmp/
```

Termux 把整个"根文件系统"映射到这个目录，用环境变量统一管理：

```bash
PREFIX=/data/data/com.termux/files/usr
HOME=/data/data/com.termux/files/home
TMPDIR=/data/data/com.termux/files/tmp
PATH=$PREFIX/bin:$PREFIX/bin/applets
LD_LIBRARY_PATH=$PREFIX/lib
```

所有二进制在编译时就把硬编码路径（如 `/usr/share`、`/etc`）改指向 `$PREFIX`。配合 `termux-exec` 包，把脚本 shebang（`#!/usr/bin/env python` 这类）在 execve 时重写为 Termux 前缀路径，让脚本也能正常跑。

### 2.4 关键点四：无 root，但有 root 更爽

- **无 root**：一切在沙箱内，写不到 `/system`，也不能 `mount`、`chroot`、开 1024 以下端口（Android 限制非特权端口）。够用，安全。
- **有 root**：可以真正 `chroot` 进完整发行版、挂载、用 `tsu` 提权、访问全盘。但国产魔改系统多数不好 root，且现在主流玩法其实**不需要 root**。

### 2.5 关键点五：proot-distro（想跑"完整 Ubuntu"就在这里）

`proot` 用 **ptrace 系统调用拦截**来模拟 chroot/root 权限，不需要真 root。于是有了 `proot-distro`：

```bash
pkg install proot-distro
proot-distro install ubuntu     # 或 debian / arch / alpine
proot-distro login ubuntu
```

这就是把"备选 Ubuntu"直接装进你手机的一条路——ARM64 的 Ubuntu 用户态，跑在 Termux 之上。**注意**：proot 有 syscall 转译开销，性能大约是原生的 70%~90%，但兼容性（glibc、systemd 之外的大多数东西）远好于裸 Termux。

---

## 三、底层原理（为什么能正常工作）

### 3.1 Android 的组成

```
Android = Linux 内核 + Bionic libc + ART(Java 运行时) + HAL + 框架
```

Termux 只用到**内核 + Bionic + 自己编译的用户态**，完全绕开 Java/ART 层。所以它快。

### 3.2 沙箱模型

每个 App 是一个独立 uid（`u0_aXXX`），内核通过 SELinux + uid 隔离。Termux 的所有文件、进程都属于这个 uid。这也解释了：

- 为什么别的 App 读不到 Termux 的文件（除非用 SAF / `termux-setup-storage`）；
- 为什么 Android 的**杀后台**会直接杀掉 Termux 的子进程（见第四节国产 ROM 对策）。

### 3.3 动态链接

Termux 的 ELF 二进制用 Android 自带的动态链接器 `/system/bin/linker64` 加载，但库搜索路径被 `LD_LIBRARY_PATH` 指到 `$PREFIX/lib`。这是它能"自带一套库、不污染系统"的原理。

### 3.4 包管理

`pkg` 本质是 `apt` 的一层封装脚本（自动 `apt update` 等）。仓库地址在：

```bash
termux-change-repo   # 交互式换源（强烈建议换成国内清华 TUNA 镜像）
```

仓库里的 `.deb` 是 Termux 团队自己编译的、带 Bionic 补丁的包，**不是 Debian/Ubuntu 的 deb**，所以不能混用。

---

## 四、具体使用方法

### 4.1 安装（关键：别用应用商店的旧版）

> Google Play 版已停更。请从 **F-Droid** 或 **GitHub Releases** 安装。

```bash
# 推荐顺序
1. F-Droid 官网下载 F-Droid.apk → 搜 Termux 安装（版本最新）
2. 或 GitHub: termux/termux-app 下载最新 apk
```

同时建议装插件（可选）：`Termux:API`、`Termux:Boot`、`Termux:Widget`、`Termux:Styling`。

### 4.2 首次初始化与换源（国内必须做）

```bash
# 首次进入自动跑 bootstrap，等它完成
# 换清华镜像（解决国内 apt 源连不上/慢）
termux-change-repo
# 选 "Mirrors in China" → 选 TUNA

# 更新
pkg update && pkg upgrade

# 常用基础包
pkg install vim git openssh curl wget python nodejs clang make
```

### 4.3 存储互通

```bash
termux-setup-storage   # 授权后生成 ~/storage 软链
# ~/storage/shared  =  /sdcard (内部存储根)
# ~/storage/downloads = Download 目录
```

> 注意：Android 把 `/sdcard` 挂成 noexec，**不能在 /sdcard 里放可执行文件直接运行**，脚本要放进 `~/` 才能执行。

### 4.4 SSH（与你的 Win10/Ubuntu 打通）

```bash
# Termux 当服务器，PC 来连
pkg install openssh
passwd                       # 设密码
sshd                         # 默认端口 8022（非特权端口）
# 查看手机 IP
ifconfig
# PC 端连接
ssh -p 8022 用户名@手机IP
```

反向场景：Termux 当客户端连 PC 的 Ubuntu（WSL 里也要开 sshd）。

### 4.5 后台保活（核心痛点）

```bash
pkg install termux-services   # runit 服务管理
sv-enable sshd                # 开机自启 sshd

# 屏幕常亮锁（配合后台）
termux-wake-lock              # 需要 Termux:API
termux-wake-unlock
```

---

## 五、针对你的四个条件的具体对策

### 5.1 中等偏上 ARM CPU → 性能结论

以骁龙 8 Gen 系列 / 天玑 9000 系这类 SoC 为例，Termux 里能干的事：

| 用途 | 可行性 | 说明 |
|------|--------|------|
| Python 科学计算 | ✅ | numpy/scipy/pandas 可装，性能接近桌面中端 |
| 编译 C/C++ | ✅ | clang + make，编译中小项目很快 |
| Node.js 服务 | ✅ | 跑 API、爬虫、自动化 |
| 数据库 | ✅ | MariaDB / PostgreSQL 原生可跑 |
| 完整 Ubuntu | ✅（proot） | 兼容性最好，性能略降 |
| 机器学习推理 | 一般 | onnxruntime 可行，大模型训练别指望 |
| 跑 Docker | ❌ | 需 root + 内核特性，基本放弃（用 Ubuntu 实体机/WSL 干这事） |

**结论**：这台手机当"随身 Linux 开发机 / 服务器"完全够用，但**重型容器化、大模型训练**这类活交给你的 Win10/Ubuntu。

### 5.2 国产魔改系统（MIUI/EMUI/ColorOS/OriginOS/鸿蒙）→ 三个必踩的坑

国产 ROM 后台管理极激进，Termux 默认会被杀。必须逐项设置：

1. **电池优化白名单**：设置 → 电池 → Termux 设为"不优化/无限制"。
2. **自启动 + 后台弹出界面**：应用启动管理里允许 Termux 自启动、允许后台。
3. **多任务锁定**：最近任务卡片下拉锁定，防止被一键清理。
4. **Android 12+ 幽灵进程限制**：系统默认最多 32 个"幽灵子进程"，超了直接杀。两条对策：

```bash
# 对策一（推荐，需要连一次电脑 adb，重启失效需重设）
adb shell settings put global settings_enable_monitor_phantom_procs false

# 对策二：所有常驻服务走 termux-services（runit），减少裸子进程
```

5. **鸿蒙（HarmonyOS）特例**：手机版鸿蒙目前仍兼容 Android APK，Termux 能跑，但后台限制和"纯净模式"（非华为应用商店的 App 会被警告）需要注意；建议用 F-Droid 装。

### 5.3 x64 UEFI Win10 → 定位与联动

Termux 和你的 Win10 是**互补关系**，不是替代：

| 方案 | 用途 |
|------|------|
| Termux 开 sshd，Win10 用 `ssh`/`scp` 连 | 手机当便携服务器，PC 当终端 |
| Win10 装 `adb`（platform-tools） | 调试、`adb shell`、传文件、关幽灵进程限制 |
| Win10 用 `sshfs`/WinSCP | 挂载/传输 Termux 文件 |
| 反向：Termux `ssh` 连 Win10 的 WSL/OpenSSH | 手机远程操作 PC |

```bash
# Win10 自带 OpenSSH 客户端，直接连
ssh -p 8022 u0_a123@192.168.x.x
# 传文件
scp -P 8022 本地文件 u0_a123@手机IP:~
```

### 5.4 备选 Ubuntu → 三条路线怎么选

你有 Ubuntu 备选，实际是三种"Linux 环境"的组合拳：

```
路线 A：实体 Ubuntu / WSL2（x86_64，Win10 上）  → 重型任务、Docker、编译大项目
路线 B：Termux 原生（手机 ARM64，Bionic）        → 随身、轻量、性能满血
路线 C：proot-distro Ubuntu（手机 ARM64，glibc） → 手机里跑"标准 Ubuntu 用户态"
```

选择建议：

- **要 glibc 兼容性、要 apt 装标准 Ubuntu 包** → 路线 C（`proot-distro install ubuntu`）；
- **要原生性能、要省电省内存、只装轻量工具** → 路线 B；
- **要 Docker / 重编译 / GUI 桌面** → 回到 Win10 上的 Ubuntu（路线 A），Termux 只做远程终端。

---

## 六、一个最小上手流程（汇总）

```bash
# 1. F-Droid 装 Termux（别用应用商店旧版）
# 2. 进应用等 bootstrap 完成
termux-change-repo          # 选国内镜像
pkg update && pkg upgrade
pkg install vim git openssh python nodejs

# 3. 存储互通
termux-setup-storage

# 4. 想跑完整 Ubuntu（你的备选）
pkg install proot-distro
proot-distro install ubuntu
proot-distro login ubuntu

# 5. 后台保活（国产 ROM 必做）
# 系统设置：电池不优化 + 自启动 + 多任务锁定
pkg install termux-services
sv-enable sshd
termux-wake-lock
```

---

## 一句话总结

Termux 的技术本质是**"用 Android 内核 + Bionic 编译的原生用户态，加上前缀重定位，在 App 沙箱里造一个无需 root 的 Linux 环境"**。你的中等偏上 ARM 手机足够当一台正经的随身 Linux 服务器；国产 ROM 的主要敌人是后台杀进程（按 5.2 节设置即可）；Win10/Ubuntu 负责重型任务，Termux 负责轻量便携 + 远程终端，三者用 SSH 串成一套。
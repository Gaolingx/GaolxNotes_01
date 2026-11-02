# WSL八耻八辱VS原生Linux爽点总结

## 🧑‍💻 User

列出wsl的八耻八辱vs原生linux的爽

---
## 😀 默认助手 (7)

# WSL 八耻八辱 · 原生 Linux 八爽

> 以跨越 `/mnt/c` 为耻，以根在 `/` 为荣。

---

### 一耻：跨越 `/mnt/c` 之耻
**WSL**：在 `/mnt/c/Users/.../project` 里执行 `git status`，需要等 30 秒；`chmod` 改完立刻变回 777，`ln -s` 会变成一个文本文件，inotify 不触发，导致 webpack、nodemon、vite 的热更新完全不工作。9p 与 drvfs 这两个文件系统协议，让你每次保存文件都要跨一次虚拟化边界。
**原生**：`/` 就是磁盘本身。ext4 或 xfs 全速读写，inotify 正常，权限语义完整，而且编译速度能快 2~5 倍。

### 二耻：vmmem 吸血之耻
**WSL**：`vmmem` 的内存只增不减。你在 Linux 里下载了 8G 代码，回到 Windows 打开任务管理器，发现内存被占满了。必须写 `.wslconfig` 里的 `memory=8GB`，并且 `wsl --shutdown` 才会把内存释放出来。
**原生**：`free -h` 显示多少就是多少，OOM killer 执行得干净利落，不会因为“虚拟化层忘了回收内存”而让桌面卡顿。

### 三耻：终端一关全死之耻
**WSL**：早期版本根本没有 systemd，部署 `sshd`、`cron`、`docker` 只能手动启动。现在需要在 `/etc/wsl.conf` 中写 `systemd=true`，但它仍然时不时报 `Failed to mount /boot` 错误；这些后台服务只要窗口关掉就会全部停止。
**原生**：用 `systemctl enable --now` 一条命令就能设置开机自启，用 `journalctl -u xxx -f` 查看日志，timer、socket activation、cgroup 一应俱全。

### 四耻：NAT 里找 localhost 之耻
**WSL**：`localhost` 转发有时可用、有时不可用；一旦开启 VPN，连接就会断开；`/etc/resolv.conf` 每次启动都会被重新生成并覆盖你的 nameserver，迫使你设置 `generateResolvConf=false`；如果想实现镜像网络，还需要 Win11 22H2 及以上版本。
**原生**：`ip a` 显示出来的是什么，就是什么。iptables、nftables、bridge、macvlan、wireguard、tcpdump 都可以直接使用。

### 五耻：用别人的内核之耻
**WSL**：内核由微软编译并签名，模块不能自行编译；`nf_tables` 功能不完整，嵌套 KVM 能否使用取决于 Hyper-V；eBPF 和 `perf` 也无法完整使用。
**原生**：`make menuconfig` 后几分钟就能编译出一版内核；DKMS 可以编译驱动；ZFS 和 btrfs 可以随意选择；从 `CAP_NET_ADMIN` 开始的 capability 体系也完整保留。

### 六耻：休眠醒来时间错乱之耻
**WSL**：笔记本合盖睡一夜后，`date` 显示的时间会差半小时；`git commit` 的时间会凭空穿越；必须手动执行 `hwclock -s`，或者重启 WSL 发行版才能恢复正确时间。
**原生**：`chrony` 和 `systemd-timesyncd` 稳定运行，时区正确，NTP 校时也不会出问题。

### 七耻：vhdx 只增不缩之耻
**WSL**：删掉 50G 文件后，`ext4.vhdx` 仍然是 50G。必须执行 `wsl --shutdown`，然后使用 `diskpart` → `select vdisk` → `compact vdisk`（或者 `Optimize-VHD`）来压缩虚拟磁盘。
**原生**：`rm` 就是真的删除。使用 btrfs 或 LVM 时，还可以执行 `snapshot` 后回滚；在虚拟机上做瘦分配也是自然而然的事。

### 八耻：WSLg 输入法候选框飘走之耻
**WSL**：图形界面依赖 Xwayland/RDP 转发，每个窗口的 DPI 缩放各自独立，fcitx 或 ibus 的候选框会飘到屏幕角落，字体渲染显得发虚，并且跨系统复制粘贴还分“两套剪贴板”。
**原生**：Wayland 或 X11 原生运行，输入法光标在哪里，候选框就在哪里，HiDPI 也能统一缩放。

---

## 原生 Linux 的额外爽点

- **一条命令装全部**：`apt install`、`pacman -S` 一条命令就能安装全部工具链，不用考虑“先安装 WSL，再挑选发行版，再进行 `wsl --set-version`”。
- **编译不再依赖运气**：内核、Chromium、LLVM 从源码编译时，I/O 不再受 9p 限制。
- **不用重启**：`uptime` 显示多久就是多久，不会被 Windows 更新的“正在配置，请勿关闭计算机”中断。
- **声明式可复制**：NixOS、Ansible 配置一次，就能运行在十台机器上，包括你那台“奇怪的双系统机器”。
- **硬件直通**：通过 VFIO 把显卡直通给虚拟机，KVM 在原生环境中才是真正可用的工具。
- **抓包和调试**：`perf record`、`ftrace`、`bpftrace`、`strace` 全套可用，`/proc` 目录也是真实的。

---

<details>
<summary><b>公平地说：WSL 也有值得称道的地方</b></summary>

- 不需要划分磁盘、重启进入 BIOS、折腾引导，就能直接使用 Ubuntu；
- CUDA 直接透传，可以直接运行 PyTorch；这是原生 Linux 双系统难以做到的便利；
- `wsl --import` 就能获得一份干净的发行版，用坏了删掉再 import 一次；
- Windows 中的 Office、微信、Adobe 不必关闭；
- 现在有了 `systemd`、`mirrored` 网络和 `WSLg`，多数痛点已经缓解了一半。

</details>

<details>
<summary><b>把「耻」压到最低的实操建议</b></summary>

1. **项目必须放在 `/` 下**（例如 `~/code`），绝对不能放在 `/mnt/c`。这一条就能解决前三条耻中的大部分问题。
2. `~/.wslconfig` 中写入：
   ```ini
   [wsl2]
   memory=12GB
   processors=8
   swap=2GB
   localhostForwarding=true
   ```
3. `/etc/wsl.conf` 中写入：
   ```ini
   [boot]
   systemd=true
   [network]
   generateResolvConf=false
   ```
4. 在 Windows 计划任务中定期执行 `wsl --shutdown` 和 `compact vdisk`，或者直接编写一个 `.bat` 文件。
5. 如果真正需要性能、驱动、eBPF 和嵌套 KVM 等功能，不要犹豫，安装原生 Linux；WSL 的定位始终是便利性工具，并非完整的运行和开发环境。

</details>

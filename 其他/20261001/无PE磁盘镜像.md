# 无PE磁盘镜像

## 🧑‍💻 User

linux可以做到在无pe情况下提取磁盘镜像吗（包括引导分区

---
## 😀 默认助手 (7)

# 结论

**完全可以，而且这是 Linux 的强项。** 「PE」通常指 Windows PE / WinPE（微软的预安装环境）。提取磁盘镜像本质上是**块设备级的原始读取**，Linux 对此支持得非常完整——你只需从任意 Linux Live 环境（如 Ubuntu Live、SystemRescue、Clonezilla Live）启动，或用现有 Linux 系统即可，**完全不需要 PE**。

而且对于**引导分区（MBR/GPT、EFI 系统分区 ESP、BIOS Boot 分区）**，Linux 的处理能力通常比 PE 更透明、更可控。

---

## 一、整体磁盘镜像：最直接的做法

逐字节读取整块磁盘，会**自动包含**分区表（MBR/GPT）、所有分区，当然也包括引导分区。

```bash
# 最基础：整盘镜像（会连空白区域一起复制，体积=磁盘容量）
sudo dd if=/dev/sda of=/path/disk.img bs=4M status=progress conv=fsync

# 压缩 + 忽略读错误（坏盘场景）
sudo dd if=/dev/sda conv=sync,noerror bs=64K status=progress \
  | gzip -c > /path/disk.img.gz

# 恢复到新盘
sudo gunzip -c /path/disk.img.gz | sudo dd of=/dev/sdb bs=4M status=progress
```

> 优点：1:1 完整，引导区一个字都不少。
> 缺点：连空闲块也复制，未压缩时镜像大小 = 整盘容量。

**坏盘请用 `ddrescue`**（可断点续跑、自动重试）：

```bash
sudo ddrescue -d -r3 /dev/sda /path/disk.img /path/mapfile
```

---

## 二、引导分区到底包含什么？为什么整盘镜像能覆盖它

| 固件/引导方式 | 关键区域 | 位置 |
|---|---|---|
| 传统 BIOS | MBR（含引导代码） | 磁盘第 0 扇区（512 字节） |
| GPT (UEFI) | GPT 主头 + 备份头 + 分区表 | 首尾各一份 |
| BIOS/GPT 混合 | BIOS Boot Partition | 磁盘开头某个小分区 |
| UEFI | **EFI 系统分区 ESP**（FAT32，存 `BOOTX64.EFI` 等） | 通常第一个分区 |
| Windows | MSR + 系统分区 + 恢复分区 | GPT 上的多个分区 |

整盘 `dd` 会把上面**全部**原样复制，所以不存在「引导分区漏掉」的问题。

### 只单独备份/恢复引导部分也很轻松

```bash
# 备份 MBR（前 512 字节）
sudo dd if=/dev/sda of=mbr.bin bs=512 count=1

# 备份 GPT 分区表（sgdisk 会同时存头尾）
sudo sgdisk --backup=gpt.bin /dev/sda
sudo sfdisk -d /dev/sda > partitions.txt   # 纯文本分区表，可移植性最好

# 单独镜像 EFI 系统分区
sudo dd if=/dev/sda1 of=esp.img bs=4M status=progress
```

恢复时：

```bash
sudo dd if=mbr.bin of=/dev/sdb bs=512 count=1 conv=notrunc
sudo sgdisk --load-backup=gpt.bin /dev/sdb
```

> 换盘恢复后，若目标盘更大，用 `sudo sgdisk -e /dev/sdb` 修复 GPT 备份头位置。

---

## 三、更聪明的做法：按文件系统只复制「已用块」

整盘 `dd` 太占空间。生产环境更常用 **Clonezilla 同款底层工具 `partclone`**，它理解文件系统，只按块复制有效数据。

```bash
# 查已安装/可用
partclone.info /dev/sda1

# 克隆分区（支持 ntfs/ext/fat/xfs…）
sudo partclone.ntfs -c -s /dev/sda2 -o win.ntfs.img
sudo partclone.vfat -c -s /dev/sda1 -o esp.vfat.img   # 引导分区也能智能克隆

# NTFS 专用轻量工具
sudo ntfsclone --save-image --output win.ntfs.img /dev/sda2

# ext2/3/4 专用
sudo e2image -ra -p /dev/sda3 ext4.img
```

**图形化/自动化首选 Clonezilla**——它本身就是 Linux Live，能做到整盘、分区、多播、网络部署，且原生处理 MBR/GPT/ESP，无需 PE。

---

## 四、典型「无 PE」实战流程（整盘 + 引导）

```bash
# 1. 识别目标盘
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT

# 2. 先记录分区结构（保险）
sudo sfdisk -d /dev/sda > /root/partitions.txt
sudo sgdisk --backup=/root/gpt.bin /dev/sda

# 3. 整盘压缩镜像
sudo dd if=/dev/sda bs=64K conv=sync,noerror status=progress \
  | zstd -T0 -19 -o /mnt/backup/disk.img.zst

# 4. 校验完整性
zstd -t /mnt/backup/disk.img.zst

# 5.（换盘恢复）
zstd -dc /mnt/backup/disk.img.zst | sudo dd of=/dev/sdb bs=64K status=progress
sudo partprobe /dev/sdb
sudo sgdisk -e /dev/sdb   # 目标盘更大时修复 GPT
```

---

## 五、必须注意的坑

<details>
<summary><strong>展开查看常见注意事项</strong></summary>

- **Windows 休眠/快速启动**：若 Windows 分区处于休眠或 Fast Startup 状态，文件系统是「脏」的。用 `dd` 整盘无碍；但用 `partclone/ntfsclone` 智能克隆前，最好先禁用快速启动并完全关机。
- **热镜像运行中的系统**：直接 `dd` 运行中的 ext4 可能得到不一致的文件系统。用 **LVM 快照** 或先 `remount,ro`：
  ```bash
  sudo lvcreate -L 10G -s -n snap /dev/vg/root
  sudo dd if=/dev/vg/snap of=root.img bs=4M
  ```
- **网络直传**（无需落地中转）：
  ```bash
  sudo dd if=/dev/sda bs=4M | zstd | ssh user@host "zstd -d > /data/disk.img"
  ```
- **对齐/扇区大小**：4Kn 盘镜像恢复到 512e 盘（或反之）需谨慎，尽量避免混用。
- **不要写错盘**：`dd` 的 `of=` 一旦写错就是灾难。建议先 `lsblk` 确认，或用 `ddrescue` 的显式参数。
- **校验**：恢复后建议对关键引导区做哈希比对。
  ```bash
  sudo dd if=/dev/sda bs=512 count=2048 | sha256sum
  sudo dd if=/dev/sdb bs=512 count=2048 | sha256sum
  ```

</details>

---

## 一句话总结

Linux 做磁盘镜像（含引导分区）不仅**不需要 PE**，而且因为有 `dd` / `ddrescue` / `partclone` / `ntfsclone` / Clonezilla 这一整套利器，在**透明度、可控性和脚本化**上通常**优于** PE。整盘 `dd` 保证 1:1 完整（引导区天然覆盖），`partclone`/Clonezilla 则能在保留引导结构的同时只复制有效数据、大幅缩小体积。

如果你告诉我**源盘是什么系统（Windows/Linux）、磁盘有多大、目标是整盘还是单分区、以及打算恢复到哪**，我可以给你一套更精确的命令。

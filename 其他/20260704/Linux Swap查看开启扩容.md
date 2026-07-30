# Linux Swap查看开启扩容

### 🧑‍💻 User

1. linux 如何查看swap状态？
2. 如何开启/关闭swap
2. 如何扩容swap空间

---

### 🤖 Assistant

好的，以下是关于Linux中Swap（交换分区/交换文件）管理的详细指南，涵盖了查看、启停和扩容操作。

## 1. 如何查看 Swap 状态

### 查看总览信息
使用 `swapon` 或 `free` 命令：

```bash
# 查看swap设备的摘要信息（推荐）
swapon --show

# 或使用 free 命令查看内存和swap总览
free -h
```

**示例输出（swapon --show）**：
```
NAME      TYPE      SIZE   USED PRIO
/swapfile file      2G     0B   -2
/dev/sda3 partition 4G     1.2G -1
```

**示例输出（free -h）**：
```
              total        used        free      shared  buff/cache   available
Mem:           7.6G        2.1G        3.2G        256M        2.3G        5.0G
Swap:          6.0G        1.2G        4.8G
```

### 查看详细使用情况
```bash
# 查看每个进程的swap使用量（需要root）
for file in /proc/*/status ; do awk '/VmSwap|Name/{printf $2 " " $3}END{ print ""}' $file 2>/dev/null; done | grep kB | sort -k 2 -n -r | head -10

# 或使用更直观的工具
sudo smem -s swap -r
```

## 2. 如何开启/关闭 Swap

### 临时启用/禁用（重启后失效）

```bash
# 关闭所有swap
sudo swapoff -a

# 启用所有swap
sudo swapon -a
```

### 永久启用/禁用（修改fstab）

**禁用**：注释掉 `/etc/fstab` 中的swap条目
```bash
# 备份fstab
sudo cp /etc/fstab /etc/fstab.backup

# 编辑fstab，在swap行前加#
sudo sed -i '/swap/s/^/#/' /etc/fstab

# 立即关闭（配合上面的swapon -a可重新启用）
sudo swapoff -a
```

**启用**：恢复fstab中的行并执行 `swapon -a`

## 3. 如何扩容 Swap 空间

主要有两种方式：**增加swap文件** 或 **扩展swap分区**。

### 方法一：通过Swap文件扩容（推荐，无需分区）

1. **创建新的swap文件**
```bash
# 创建一个4GB的swap文件
sudo dd if=/dev/zero of=/swapfile2 bs=1M count=4096 status=progress

# 设置正确的权限（仅root可读写）
sudo chmod 600 /swapfile2
```

2. **格式化为swap格式**
```bash
sudo mkswap /swapfile2
```

3. **启用该swap文件**
```bash
sudo swapon /swapfile2
```

4. **永久生效（写入fstab）**
```bash
echo '/swapfile2 none swap sw 0 0' | sudo tee -a /etc/fstab
```

5. **验证**
```bash
swapon --show
```

### 方法二：扩展Swap分区（有风险，需要空闲磁盘空间）

> **警告**：操作分区有数据丢失风险，请先备份重要数据！

1. **确认当前swap分区**
```bash
lsblk | grep swap
# 或
fdisk -l | grep swap
```

2. **使用live CD/USB启动并修改分区表**
   - 启动live环境
   - 使用 `gparted` 或 `fdisk` 删除并重新创建更大的swap分区
   - 重新运行 `mkswap` 和 `swapon`

### 调整Swap优先级

如果同时有多个swap设备，可以设置优先级来影响使用顺序：

```bash
# 设置优先级（值越大越优先，默认-1）
sudo swapon --priority 100 /swapfile2

# 查看优先级
swapon --show
```

## 重要注意事项

1. **内存与swap比例**：
   - 物理内存充足时，swap使用量应很少（< 10%）
   - 如果swap持续大量使用，考虑增加物理内存

2. **swappiness参数**（内核使用swap的倾向性）：
```bash
# 查看当前值（0-100，默认60）
cat /proc/sys/vm/swappiness

# 临时修改
sudo sysctl vm.swappiness=10

# 永久修改
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```
   - 值越低，越倾向于使用物理内存
   - 值越高，越倾向于使用swap

3. **性能影响**：
   - 使用SSD的swap文件比HDD快得多
   - 如果内存足够大（≥16GB），可以考虑完全禁用swap

4. **验证修改**：
```bash
# 重启后检查swap状态
sudo swapon --show
free -h
```

如果需要更多自动化脚本或特定场景的配置建议，请提供更多细节（如发行版、内存大小、磁盘类型等）。
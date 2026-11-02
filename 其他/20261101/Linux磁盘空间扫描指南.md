# Linux磁盘空间扫描指南

## 🧑‍💻 User

怎么扫描linux上的空间占用

---
## 😀 默认助手 (7)

# Linux 磁盘空间扫描：从快速定位到深挖

## 一、快速排查流程（四步定位）

### 第 1 步：确认哪个分区满了

```bash
df -hT          # 人类可读 + 文件系统类型
df -i           # 检查 inode 是否耗尽（文件多但总大小不大时）
```

`Use%` 高的那一行，看 `Mounted on`，即挂载点，例如 `/`、`/var`、`/data`。

### 第 2 步：定位到占用空间最大的目录

先用**不跨文件系统**的写法，避免扫描挂载到 `/mnt`、`/proc` 的内容：

```bash
# 逐层定位：先看根目录一级
du -xh --max-depth=1 / 2>/dev/null | sort -hr | head -20

# 找到大目录后继续往下钻，例如 /var
du -xh --max-depth=1 /var 2>/dev/null | sort -hr | head -20
```

参数说明：

| 参数 | 作用 |
|---|---|
| `-x` | 不跨文件系统（不含其他挂载点） |
| `-h` | 人类可读 |
| `--max-depth=1` | 只看一层 |
| `2>/dev/null` | 忽略权限报错 |
| `sort -hr` | 按大小倒序（`-h` 支持 K/M/G 排序） |

### 第 3 步：用 `ncdu` 交互式查看（强烈推荐）

`du` 只能逐层查看，`ncdu` 可以按键进入每一层目录查看：

```bash
# 安装
apt install ncdu      # Debian/Ubuntu
yum install ncdu      # RHEL/CentOS（需 EPEL）
dnf install ncdu
apk add ncdu          # Alpine

# 使用
ncdu -x /             # 扫描根目录，不跨文件系统
ncdu -x /var          # 或直接扫某个目录
```

常用按键：`↑↓` 移动，`→` 进入，`←` 返回，`d` 删除，`n` 按名字排序，`s` 按大小排序，`g` 显示百分比/图表，`q` 退出。

### 第 4 步：找出具体大文件

```bash
# 找 >500M 的文件，按大小排序
find / -xdev -type f -size +500M -printf '%s\t%p\n' 2>/dev/null \
  | sort -rn | head -20 | awk '{printf "%.1fG\t%s\n", $1/1073741824, $2}'
```

---

## 二、常见“空间去哪了”的坑

<details>
<summary><b>坑 1：文件已删除，但空间没释放（df 和 du 对不上）</b></summary>

进程还持有被删文件的句柄，空间不会释放。

```bash
# 找出被删除但仍被占用的文件
lsof +L1
lsof | grep -i deleted

# 或直接看某个进程
lsof -p <PID> | grep deleted
```

处理方式：

- 重启对应服务（最干净）：`systemctl restart nginx`
- 或者清空句柄（危险，仅对日志类文件）：`: > /proc/<PID>/fd/<FD>`

这也是 `df` 显示 100% 但 `du` 统计不出来 的**最常见原因**。
</details>

<details>
<summary><b>坑 2：inode 耗尽（df -h 有空间但写不进去）</b></summary>

```bash
df -i                       # IUse% 高就是这个问题
# 找哪个目录文件数最多
find / -xdev -type d -printf '%p\n' 2>/dev/null | \
  while read d; do echo "$(ls -A "$d" 2>/dev/null | wc -l) $d"; done | sort -rn | head -20
```

常见元凶：`/var/spool/postfix`、session 文件、`/tmp` 下海量小文件、邮件队列。
</details>

<details>
<summary><b>坑 3：日志文件暴涨</b></summary>

```bash
# systemd journal
journalctl --disk-usage
journalctl --vacuum-size=500M      # 收缩到 500M
journalctl --vacuum-time=7d        # 只保留 7 天

# 普通日志目录
du -xh --max-depth=1 /var/log | sort -hr | head

# 检查 logrotate 配置
cat /etc/logrotate.conf
ls /etc/logrotate.d/
logrotate -d /etc/logrotate.conf   # dry-run 看规则是否生效
```

临时清空某日志但保留文件（不要用 `rm`，否则进程句柄问题）：
```bash
: > /var/log/xxx.log
```
</details>

<details>
<summary><b>坑 4：Docker / 容器占用</b></summary>

```bash
docker system df -v                # 镜像、容器、卷、构建缓存占用
docker system prune -a --volumes   # 危险：清理所有未使用资源，先确认

# 容器日志（单个 json 日志可能几十 G）
du -sh /var/lib/docker/containers/*/*-json.log | sort -hr | head
```

限制日志大小（`/etc/docker/daemon.json`）：
```json
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "100m", "max-file": "3" }
}
```
</details>

<details>
<summary><b>坑 5：包管理器缓存 / snap / 旧内核</b></summary>

```bash
# Debian/Ubuntu
apt clean && apt autoremove --purge
du -sh /var/cache/apt/archives

# RHEL/CentOS
dnf clean all
du -sh /var/cache/dnf

# snap 旧版本
snap list --all | awk '/disabled/{print $1, $3}' | \
  while read name rev; do snap remove "$name" --revision="$rev"; done

# 旧内核（Ubuntu）
dpkg -l 'linux-image-*' | awk '/^ii/{print $2}'
```
</details>

---

## 三、工具速查表

| 工具 | 场景 | 命令示例 |
|---|---|---|
| `df` | 分区级总览 / inode | `df -hT`、`df -i` |
| `du` | 目录级统计，脚本友好 | `du -xh --max-depth=1 / \| sort -hr` |
| `ncdu` | 交互式逐层探索（最推荐） | `ncdu -x /` |
| `find` | 按大小/时间精确定位文件 | `find / -xdev -size +1G` |
| `lsof +L1` | 已删除未释放文件 | `lsof +L1` |
| `dust` | `du` 的现代替代，带柱状图 | `dust -d 2 /` |
| `duf` | `df` 的美化版，支持 JSON | `duf` |
| `gt5` | 与上次扫描对比差异 | `gt5 /var` |
| `baobab` / `filelight` | 图形化（桌面环境） | `baobab` |
| `fdupes` / `rmlint` | 找重复文件 | `fdupes -r /data` |
| `iotop` / `fatrace` | 谁在疯狂写盘 | `fatrace -f W` |

---

## 四、一条实用组合命令

想一次性输出“按大小排序的前 30 个大文件（>100M）”，可直接粘贴：

```bash
find / -xdev -type f -size +100M -printf '%s\t%p\n' 2>/dev/null \
  | sort -rn | head -30 \
  | numfmt --to=iec --field=1 --suffix=B
```

如果怀疑是**日志或容器**拖慢磁盘增长，加上实时监控再观察：

```bash
# 找出正在大量写入的进程与文件
fatrace -f W | head -50
```

---

**总结优先级**：`df -h` 看分区，再用 `ncdu -x` 交互查看，并用 `du -x` 逐层查看 → `find -size` 定位大文件 → `lsof +L1` 查已删除未释放文件 → `df -i` 查 inode。绝大多数“磁盘满了”都能在这五步内定位到具体位置。

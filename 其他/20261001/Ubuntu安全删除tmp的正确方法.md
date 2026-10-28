# Ubuntu安全删除tmp的正确方法

## 🧑‍💻 User

linux ubuntu如何安全删除tmp？

---
## 😀 默认助手 (7)

## 核心结论

**不要直接 `rm -rf /tmp` 或 `rm -rf /tmp/*`。** Ubuntu 上 `/tmp` 的正确清理方式是交给 systemd 的 `systemd-tmpfiles`（按时间策略清理），或者直接重启；如果必须手动删，只删**内容**、保留 `/tmp` 目录本身及其 `1777` 粘滞位权限。

```bash
# 推荐 1：按系统策略清理（安全，尊重保留期）
sudo systemd-tmpfiles --clean

# 推荐 2：直接重启（磁盘型 /tmp 会在启动时清理；tmpfs 型直接清零）
sudo reboot
```

---

## 一、先确认 `/tmp` 是什么类型

Ubuntu 16.04+ 由 `systemd-tmpfiles-clean.timer` 每天自动清理；**Ubuntu 24.04 起 `/tmp` 默认是 tmpfs（内存盘）**，重启即清空。

```bash
mount | grep ' /tmp                 # 有输出 = 独立分区/tmpfs；无输出 = 根分区上的目录
df -h /tmp                           # Type 为 tmpfs 说明在内存里
ls -ld /tmp                          # 必须是 drwxrwxrwt，属主 root
readlink -f /tmp                     # 确认不是软链接
```

- **tmpfs**：删内容等于释放内存，重启也会清空，几乎无风险。
- **独立分区 / 根分区目录**：用 `find` 按时间删，或重启。

> 关键点：`/tmp` **目录本身**永远要保留。缺了它或权限不是 `1777`（粘滞位），X11、`ssh-agent`、许多程序会异常。

---

## 二、手动删除的正确姿势

<details>
<summary><b>展开：按文件年龄删除（最稳）</b></summary>

```bash
# 删除 7 天内没有被访问/修改过的条目，-xdev 防止跨越子挂载点
sudo find /tmp -xdev -mindepth 1 -atime +7 -delete
```

说明：
- `-mindepth 1`：不碰 `/tmp` 自己。
- `-atime +7`：比 `-mtime` 更贴近"是否还在用"；但挂载参数含 `relatime`/`noatime` 时 atime 可能不准，此时用 `-mtime +7`。
- `-xdev`：万一 `/tmp` 下有子挂载，不会误删别的文件系统。

</details>

<details>
<summary><b>展开：想清空全部内容（保留特殊条目）</b></summary>

```bash
sudo find /tmp -mindepth 1 -maxdepth 1 -xdev \
  ! -name '.X11-unix' ! -name '.ICE-unix' ! -name '.font-unix' ! -name '.XIM-unix' \
  ! -name 'systemd-private-*' ! -name 'snap-private-tmp' \
  -exec rm -rf -- {} +
```

- 用 `find ... -exec rm -rf -- {} +` 而不是 shell 通配符：能匹配隐藏文件，也不会触发 `Argument list too long`。
- `--` 防止以 `-` 开头的文件名被当成选项。
- 排除 `systemd-private-*` / `snap-private-tmp` / `.X11-unix`：这些被正在运行的服务或 X 会话占用，删了可能让新进程连不上 X、或影响某服务的 `PrivateTmp`。

</details>

---

## 三、千万别这样写 ⚠️

| 危险写法 | 问题 |
|---|---|
| `rm -rf /tmp/*` | 漏掉隐藏文件；文件极多时报 `Argument list too long`；`/tmp/*` 若因变量为空展开成 `/*` 就是灾难 |
| `cd /tmp && rm -rf *` | `cd` 失败时你还在原目录；通配符同样漏隐藏文件 |
| `rm -rf /tmp/` 或删掉 `/tmp` 目录本身 | 服务/图形会话会因缺少 `1777` 目录而出错 |
| `rm -rf "$DIR"/*`（`$DIR` 可能为空） | 变量为空时变成 `rm -rf /*` —— 最经典的删库事故 |
| 在脚本里不检查就 `sudo rm -rf` | 建议加 `set -u` 并为路径写全、先 `echo` 预览 |

---

## 四、删之前先看一眼谁在用

```bash
sudo lsof +D /tmp | head          # 哪些进程打开了 /tmp 下的文件
sudo fuser -vm /tmp               # 占用 /tmp 的进程
```

Linux 上删除正在被打开的文件本身不会立刻崩溃（inode 会保留到句柄关闭），但**依赖该路径**的程序会出问题（比如 X socket、`ssh-agent` socket、数据库 socket）。因此优先"按时间删旧文件"，而不是"清空一切"。

---

## 五、想调整自动清理的保留期

<details>
<summary><b>展开：修改 tmpfiles.d 策略</b></summary>

系统默认规则在 `/usr/lib/tmpfiles.d/tmp.conf`，其中类似：

```
q /tmp     1777 root root 10d
q /var/tmp 1777 root root 30d
```

要覆盖，**不要改 `/usr/lib`**，而是新建 `/etc/tmpfiles.d/tmp.conf`（同名会覆盖，无需 `override.conf` 语法）：

```bash
sudo tee /etc/tmpfiles.d/tmp.conf >/dev/null <<'EOF'
q /tmp     1777 root root 3d
q /var/tmp 1777 root root 30d
EOF
```

- `q` 表示"设权限 + 到时清理"，`3d` = 3 天。
- 定时器状态：`systemctl status systemd-tmpfiles-clean.timer`（默认开机 15 分钟后首次、之后每天一次）。
- 立即应用：`sudo systemd-tmpfiles --clean`。

</details>

---

### 一句话总结

`/tmp` 该删的是**里面的内容**，方式优先级为：**重启 → `sudo systemd-tmpfiles --clean` → `find` 按年龄删除**；始终保留 `/tmp` 目录与 `1777` 权限，并避开 `systemd-private-*`、`.X11-unix` 等正在使用的条目 —— 永远别用 `rm -rf /tmp/*`。

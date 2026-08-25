# 直接重启不会应用新GRUB设置

### 🧑‍💻 User

update-grub没有运行而是直接重启了会apply新grub设置吗？

---

### 🤖 Assistant

# 不运行 `update-grub` 直接重启，新设置**不会**生效

## 原因

GRUB 启动时读取的配置文件是 `/boot/grub/grub.cfg`（部分发行版是 `/boot/grub2/grub.cfg`），而不是你通常编辑的 `/etc/default/grub`。

整个流程是这样的：

```
/etc/default/grub  ──┐
                      │  update-grub (grub-mkconfig)
/etc/grub.d/*       ──┘──────────────────────────▶  /boot/grub/grub.cfg  ──▶  重启时 GRUB 读取
   (配置源)                                          (实际生效的配置)
```

`update-grub` 的作用就是把 `/etc/default/grub` 和 `/etc/grub.d/` 下的脚本**生成**为最终的 `grub.cfg` 文件。

- 你改了 `/etc/default/grub`（比如改了 `GRUB_TIMEOUT`、`GRUB_CMDLINE_LINUX` 等）
- 但**没有运行** `update-grub`
- → `grub.cfg` 没有被重新生成
- → 重启后 GRUB 读到的还是**旧的 `grub.cfg`**
- → **新设置不会生效**

## 例外情况

| 你改了什么 | 需要运行 `update-grub` 吗 |
|---|---|
| `/etc/default/grub` | ✅ **需要** |
| `/etc/grub.d/` 下的脚本 | ✅ **需要** |
| 直接编辑 `/boot/grub/grub.cfg` | ❌ 不需要，但**不推荐**（会被下次 `update-grub` 覆盖） |

## 解决办法

如果你已经直接重启了，改的设置没生效，不用担心。重新进入系统后运行：

```bash
sudo update-grub
```

然后再重启即可。

> ⚠️ **注意**：某些发行版（如 Fedora/RHEL/openSUSE）使用的是 `grub2-mkconfig`，命令为：
> ```bash
> sudo grub2-mkconfig -o /boot/grub2/grub.cfg
> ```
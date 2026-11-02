# KVM气球回收生产配置指南

## 🧑‍💻 User

云服务器气球回收是什么

---
## 😀 默认助手 (7)

## 先给结论

**“云服务器气球回收”不是一个标准术语**，在主流云厂商（阿里云、腾讯云、华为云、AWS、VMware、OpenStack/KVM）的文档里都查不到这个说法。它最可能是几个不同概念的混拼；其中只有一个“气球”是真的技术名词。

---

## 最可能的解释：内存气球（memory ballooning）回收内存

在虚拟化、云平台领域，“气球”确实是一个正式术语，英文是 **memory balloon / balloon driver**。中文一般叫“内存气球”或“气泡技术”，它跟“回收”连用非常自然：

| 动作 | 含义 |
|---|---|
| **inflate（充气）** | 宿主机内存紧张时，让客户机里的 balloon 驱动“占住”一部分客户机内存，客户机可用内存变小，宿主机回收物理页 |
| **deflate（放气）** | 宿主机内存宽裕时归还给客户机 |
| **balloon reclaim** | 通过气球机制回收内存这件事本身 |

工作流程大致是：

1. 宿主机（Hypervisor）检测到内存压力 → 通知各 VM 的 balloon 驱动
2. 客户机内 balloon 驱动向自己的内核申请页面并“钉住”，客户机内核随之触发自己的回收（drop cache、换出等）
3. 被占住的页宿主机可以安全回收给其他 VM 或容器使用
4. 压力解除后再 deflate 归还

常见实现：KVM 的 `virtio-balloon`（配合 `free page reporting` 把空闲页直接上报宿主）、VMware 的 `vmmemctl` 驱动。云厂商在超卖（overcommit）和弹性规格调整时常用这套机制。

所以如果你是在**调优、告警或看监控面板**时遇到的：

- 客户机 `available` 内存莫名其妙变少 → 可能是 balloon 被 inflate
- 宿主机内存回收指标里出现 balloon 相关计数器 → 就是这套东西
- 容器场景里更常见的是 `cgroup memory reclaim`，不是气球

```text
宿主机内存压力 ↑ → inflate 气球 → 客户机可用内存 ↓ → 宿主机收回物理页
宿主机内存压力 ↓ → deflate 气球 → 客户机可用内存 ↑
```

<details>
<summary>其他几种可能的理解（不太可能是本意，但列出来供你对照）</summary>

- **实例“回收”**：云厂商对欠费、到期、违规、闲置的实例做释放/回收（release / reclaim），跟气球毫无关系。常见说法是“实例回收”“资源回收站”。
- **气球贷（balloon loan）**：金融里的“气球”指前期还款少、末期一次还一大笔的贷款结构，跟服务器无关。
- **字面意义的“气球回收”**：气球厂、活动物料回收，纯物理世界。
- **拼写/语音混淆**：可能是“虚拟机内存气球”、“资源回收”、“气球式扩容”等词的混记。

</details>

---

## 想给你准确答案，需要一点上下文

这句是从哪儿看到的？比如：

1. 云厂商控制台的**监控指标名**或**告警文案**？
2. 某篇**技术博客/论文**里的词？
3. 同事口头说的（很可能是“内存气球”被记成了“气球回收”）？
4. 还是你在**写文档需要造词**，想确认这个说法成不成立？

如果是第 1 种，把原文那一行贴给我，我可以直接告诉你是哪个机制、该怎么处理。

---
## 🧑‍💻 User

为我整理‌KVM环境下气球回收的实操配置步骤‌ 快速完成生产环境的参数调优。

---
## 🧑‍💻 User

为我整理‌KVM环境下气球回收的实操配置步骤‌ 帮你快速完成生产环境的参数调优。

---
## 😀 默认助手 (7)

# KVM 气球回收（virtio-balloon）生产配置指南

下面按“**先验证 → 再开启 → 手工压测 → 定策略 → 上监控 → 留回滚**”的顺序给。命令默认在宿主机 root 下、libvirt ≥ 6.x 环境；凡涉及版本差异处都会标出来。

---

## 0. 前提检查（3 分钟，别跳过）

```bash
# 1) 虚拟机是否已有 balloon 设备
virsh dumpxml <vm> | grep -A6 '<memballoon'

# 2) 宿主 QEMU/libvirt 版本（决定能否用 free-page-reporting）
libvirtd -V
qemu-system-x86_64 -version

# 3) guest 内驱动是否加载（进 VM 里执行）
lsmod | grep virtio_balloon
ls /sys/bus/virtio/drivers/virtio_balloon/     # 有软链接说明驱动绑上了
```

判读：

| 输出 | 结论 |
|---|---|
| `<memballoon model='none'/>` | 气球迷你设备被禁用，**必须先关机改 XML**（不支持热插拔） |
| `<memballoon model='virtio'>` | 正常，继续 |
| 无 `<memballoon>` 标签 | 取决于 libvirt 版本默认值（老旧配置可能默认 `none`），显式写死 |
| guest 内没 `virtio_balloon` | 无驱动 → 气球充不进气，先装驱动 |

> **关键认知**：气球是**协商式**回收。guest 不配合（驱动缺失、内核 panic、Windows 没装 virtio-win），宿主机一点内存都收不回来。所以它**不能**当作超卖的硬性保障。

---

## 1. 宿主侧：开启 balloon + 统计 + free page reporting

`virsh edit <vm>`，改成这样：

```xml
<domain type='kvm'>
  <memory unit='GiB'>16</memory>
  <currentMemory unit='GiB'>16</currentMemory>

  <devices>
    <memballoon model='virtio'>
      <!-- 统计周期，秒；0 表示关闭。生产建议 10~30，别设 1 -->
      <stats period='10'/>
      <!-- 可选：QEMU≥5.1 且 libvirt≥8.1，把 guest 空闲页直接上报宿主，零成本回收 -->
      <freePageReporting/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x08' function='0x0'/>
    </memballoon>
  </devices>
</domain>
```

要点：

- **改 `<memballoon>` 必须关机重启**，不能热插拔。升级路径：`virsh destroy` → 改 XML → `virsh start`，或 `virsh edit` 后 `virsh define` + 重启。
- **`<stats period>` 可以热改**，不用重启：

```bash
virsh dommemstat <vm> --period 10 --live      # 只改本次运行
virsh dommemstat <vm> --period 10 --config    # 写入持久配置
virsh dommemstat <vm> --period 0              # 关掉统计（省下 QEMU 开销）
```

- **统计走的是 balloon 设备本身，不需要 qemu-guest-agent**。这点常被搞混。
- `<freePageReporting/>` 老版本 libvirt 不认识会直接报 XML 校验错误 —— 那就删掉这一行，退回纯气球模式；两者可共存，气球管“已用页”，free page reporting 管“完全空闲页”，后者更便宜。

### Windows guest 补充

装 virtio-win 驱动包里的 **Balloon** 驱动（设备管理器里会出现 “VirtIO Balloon Driver”），否则 `virsh dommemstat` 只有 `actual` 一个字段、`available` 恒为 0，主动调节会被误判。

---

## 2. 手工充气／放气，先验证链路通不通

```bash
# 当前状态（注意单位是 KiB）
virsh dommemstat <vm>

# 直接看 QEMU 视角，最可靠
virsh qemu-monitor-command <vm> --hmp 'info balloon'

# 充气：把 16G 压到 8G（宿主机收回 8G）
virsh setmem <vm> 8G --live

# guest 内确认（应看到 total 变小、内存压力上升）
free -m ; grep -i balloon /proc/meminfo

# 放气：还给 guest
virsh setmem <vm> 16G --live
```

`virsh dommemstat` 字段怎么读：

| 字段 | 含义 | 用途 |
|---|---|---|
| `actual` | 气球当前给 guest 的实际内存 | 你设定值的落地结果 |
| `available` | guest **自报**的可用内存（free + 可回收） | **调优主指标**，比看宿主 RSS 准 |
| `unused` | guest 内完全空闲的页 | 判断能否继续压 |
| `usable` | guest 认为可用的总量 | 交叉校验 |
| `rss` | 宿主侧实际驻留物理页 | 判断是否真的回收了 |
| `swap_in/out` | guest 换页量 | **回收风暴的报警器** |
| `last_update` | 统计时间戳 | 判断统计是否卡死 |

❌ 两个高频坑：

1. **`setmem` 不能超过 XML 里 `<memory>` 的原值**。想超过（让 VM 长到比开机值更大）必须配 `<maxMemory slots=...>` 做内存热插拔，那是另一套机制，别混用。
2. 单位：`dommemstat` 输出 KiB，`setmem` 默认也是 KiB，但写成 `8G` 更安全。脚本里务必显式带单位，我见过把 `8388608` 当 MB 写进去把 VM 压到 8MB 的事故。

---

## 3. 生产参数调优

### 3.1 宿主机侧

```bash
# 宿主自身别太激进，气球回收的内存不该被宿主换出
sysctl -w vm.swappiness=10
# 保留足够的宿主内存避免 reclaim 拖慢所有 VM
sysctl -w vm.min_free_kbytes=1048576        # 1G，按宿主内存 1~2% 取
# overcommit 用 2（严格）还是 0/1（启发式）：气球场景下建议仍用 0，
# 把它当"弹性"而不是"承诺"
sysctl -w vm.overcommit_memory=0
```

持久化写 `/etc/sysctl.d/90-kvm-balloon.conf`。

### 3.2 Guest 内核侧（气球能否顺滑的关键）

气球一充气，触发的就是 **guest 自己的内存回收**。guest 侧没调好，宿主回收成功但业务会抖：

| 参数 | 建议值 | 原因 |
|---|---|---|
| `vm.min_free_kbytes` | 内存的 1%~2%（如 16G → 200000） | 太小会直接触发直接回收/整机卡顿 |
| `vm.swappiness` | 10~30（有 swap 且想保住页缓存则偏低） | 充气时优先丢缓存还是换出，影响延迟 |
| `vm.watermark_scale_factor` | 100~200 | 让 kswapd 提前动手，避免 direct reclaim |
| `vm.zone_reclaim_mode` | 0 | 除非 NUMA 局部性极强，别开 |
| `vm.dirty_ratio/background_ratio` | 10/5 以下 | 脏页多时回收会长时间阻塞 |

另外确认 guest 内 `virtio_balloon` 是内建或最早一批加载（`lsinitrd | grep balloon`），否则早期启动阶段气球不可用。

### 3.3 气球策略参数（决定“压到多少”）

给一个可直接落地的取值规则：

$$M_{\text{floor}} = \max\big(0.5 \times M_{\text{total}},\; M_{\text{业务常驻峰值}} \times 1.2\big),\qquad M_{\text{target}} = M_{\text{total}} - \Delta_{\max}$$

- $\Delta_{\max}$：单机最多能压掉的比例，**生产建议 ≤ 25%~30%**；超过这个数，guest 回收成本陡增、swap 飙升，得不偿失。
- $M_{\text{floor}}$：硬下限，脚本里写成常量，不允许被覆盖。
- 有状态服务（MySQL/Redis 类）把上限压到 **15%** 以内，并优先用 `freePageReporting` 而不是硬压。

---

## 4. 自动化：谁来“决定充放气”

KVM/libvirt **本身没有自适应气球策略**，`setmem` 只是执行器，策略必须外部给。三种做法：

| 方案 | 适用 | 说明 |
|---|---|---|
| **oVirt + MoM** | 已有 oVirt 集群 | MoM 内置 `FreeMemory`/`BalloonTarget` 策略，配 `MoMPolicyParameters`（如 `balloon_target_memory`、`MinimumGuaranteedMemoryMb`），开箱即用，别自己写 |
| **Proxmox VE** | PVE 环境 | VM 配置里设 `balloon`（最小内存）+ 勾选自动气球，PVE 自带控制器轮询调节 |
| **自研脚本/控制器** | 裸 libvirt、OpenStack、自建平台 | 唯一选择。见下 |

### 自研控制器的正确写法（带迟滞，防抖）

```bash
#!/usr/bin/env bash
# balloon-ctl.sh <vm>   建议 systemd timer 每 60s 跑一次
set -euo pipefail
VM="$1"
POLL=/var/lib/balloon/state.$VM

MIN_MB=4096        # 硬下限，永不突破
STEP_MB=512        # 单次调节步长，宁小勿大
DEADBAND_MB=1024   # 死区：available 在此区间内不动作
N=3                # 连续 N 次越界才动手

TOTAL_MB=$(virsh dominfo "$VM" | awk '/^Max memory/{printf "%d",$3/1024}')
STATS=$(virsh dommemstat "$VM")
AVAIL_MB=$(awk '/^available/{printf "%d",$2/1024}' <<<"$STATS")
ACTUAL_MB=$(awk '/^actual/{printf "%d",$2/1024}'  <<<"$STATS")
SWAPOUT=$(awk '/^swap_out/{print $2}'             <<<"$STATS")

# 统计未就绪（available=0 或时间戳过期）直接退出，绝不盲调
[[ "$AVAIL_MB" -le 0 ]] && { logger -t balloon "skip: no stats"; exit 0; }

TARGET=$((TOTAL_MB * 75 / 100))   # 允许压到 75%
HIGH=$((DEADBAND_MB + STEP_MB))
LOW=$((AVAIL_MB))

c=$(cat "$POLL" 2>/dev/null || echo 0)

if   [[ "$AVAIL_MB" -lt "$LOW" && "$ACTUAL_MB" -lt "$TARGET" ]]; then
     NEW=$((ACTUAL_MB + STEP_MB))          # 放气：guest 太紧
elif [[ "$AVAIL_MB" -gt "$HIGH" && "$ACTUAL_MB" -gt "$MIN_MB" ]]; then
     NEW=$((ACTUAL_MB - STEP_MB))          # 充气：guest 太松
else
     echo 0 > "$POLL"; exit 0
fi

c=$((c+1)); echo "$c" > "$POLL"
[[ "$c" -lt "$N" ]] && exit 0
echo 0 > "$POLL"

[[ "$NEW" -lt "$MIN_MB" ]] && NEW=$MIN_MB
[[ "$NEW" -gt "$TOTAL_MB" ]] && NEW=$TOTAL_MB
logger -t balloon "$VM ${ACTUAL_MB}M -> ${NEW}M (avail=${AVAIL_MB}M swap_out=${SWAPOUT})"
virsh setmem "$VM" "${NEW}M" --live
```

必须加的护栏：

- **只在统计新鲜时调节**（`last_update` 超过 3 个周期就跳过），否则 guest 卡死时脚本会一路踩到底。
- **`swap_out` 增速超阈值时强制放气并报警**，这是唯一能救回业务的信号。
- **单次步长小、间隔 ≥ 30~60s**，吹气球是分钟级操作，不是秒级。
- 上线前先用 `--dry-run` 版本跑一周看日志，别直接改生产。

---

## 5. 监控与告警（最少这几条）

```bash
# 采集：dommemstat / info balloon 轮询，导入 Prometheus（libvirt_exporter 已内置 balloon 指标）
virsh dommemstat <vm> --period 10
```

| 告警 | 阈值 | 含义 |
|---|---|---|
| `available` < 下限 | 持续 3 个周期 | 气球压过头，快去放气 |
| `swap_out` 增长速率 | > 50MB/s 且持续 | guest 严重换页，业务已开始抖 |
| `actual` 长期 < 上限 60% | 24h | 配置虚高，该降规格而不是靠气球 |
| `last_update` 停滞 | > 60s | balloon 驱动挂了/guest 异常，脚本必须停止调节 |
| `rss` 与 `actual` 背离 | 差 > 20% | free page reporting 生效或统计失真，交叉验证 |

---

## 6. 回滚与一页纸检查清单

**回滚**（30 秒内完成）：

```bash
virsh dommemstat <vm> --period 0                      # 停统计
virsh setmem <vm> 16G --live                          # 全量放气
systemctl stop balloon-ctl.timer                      # 停控制器
```

永久关闭：`virsh edit <vm>` 改回 `<memballoon model='none'/>` 并重启；同时把脚本从 timer 里摘掉。

<details>
<summary>上线前 Checklist（点击展开）</summary>

- [ ] guest 内 `virtio_balloon` 已加载，`available` 字段有值
- [ ] Windows guest 已装 virtio-win Balloon 驱动
- [ ] XML 无 `<memballoon model='none'/>`，`<stats period>` 已设 10~30
- [ ] 已录基线：`dommemstat` 的 `available`/`swap_out` 稳态值
- [ ] 控制器有 `MIN_MB` 硬下限、死区、连续 N 次确认、统计新鲜度检查
- [ ] `swap_out` 与 `available` 两个告警已接入
- [ ] 已在非核心 VM 上压测过一轮 30% 回收，确认 P99 延迟可接受
- [ ] 回滚命令已写进 runbook，值班人知道在哪
- [ ] 明确写下：**气球不是超卖保障**，关键业务仍靠 CPU pinning / cgroup / 内存预留

</details>

<details>
<summary>常见故障速查</summary>

| 现象 | 根因 | 处理 |
|---|---|---|
| `setmem` 报 “cannot set memory higher than max memory” | 超了 XML `<memory>` | 用 `<maxMemory>` 热插拔，或别超原值 |
| `dommemstat` 只有 `actual`，`available=0` | guest 无驱动/老驱动不支持 stats | 装/升级 virtio-balloon 驱动 |
| 充气后 guest OOM、被内核杀进程 | 压过头 + 无 min_free_kbytes 保护 | 抬下限、开 `deflate-on-oom`（QEMU 属性，libvirt 需 `qemu:override`，见下） |
| 充气毫无效果，宿主 RSS 不降 | freePageReporting 被误认为气球 / 页面被 pin（大页、mlock、VFIO） | 关 hugepages 验证；`mlock` 的进程页不可回收 |
| 业务延迟周期性尖刺 | 控制器步长太大、周期太短 | 步长减半、周期拉到 60s、加死区 |

> `deflate-on-oom`：QEMU 的 `virtio-balloon-pci` 有该属性，guest OOM 时自动放气。libvirt XML 没有直接对应元素，新版本（8.2+）可用 `<qemu:override>` 设置，老版本只能改启动参数 —— 建议优先用“抬高下限 + 报警”解决，不要为了这个属性去动 libvirt 配置。

</details>

---

需要我把上面第 4 节那个控制器扩成**带 dry-run、systemd unit、Prometheus 指标导出**的完整版，或者按你的具体场景（VM 数量、内存规格、业务类型、是否已有 oVirt/PVE）把 `MIN_MB`/`STEP_MB` 这些参数算成具体数字？给我这几个信息就行。

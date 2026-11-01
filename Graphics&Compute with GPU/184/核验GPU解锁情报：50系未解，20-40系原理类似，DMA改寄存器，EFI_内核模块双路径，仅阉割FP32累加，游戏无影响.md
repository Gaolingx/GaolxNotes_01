# 核验GPU解锁情报：50系未解，20-40系原理类似，DMA改寄存器，EFI/内核模块双路径，仅阉割FP32累加，游戏无影响

## 🧑‍💻 User

https://github.com/Zagitta/gpu-unlock 这个repo做了什么，有什么用

---
## 😀 默认助手 (7)

## 一句话概括

`gpu-unlock` 是一个 Linux 内核模块，它把消费级 GeForce RTX 30 系（GA102/GA103/GA104/GA106）被驱动固件**人为限制**的 FP16/BF16 指令发射速率，解锁到专业卡（A 系列 / pro 型号）的水平。本质上是**做 Tensor Core 的 BF16/FP16 算力解锁**，解锁后可翻倍，不是破解 SLI、显存或 NVLink。

---

<details><summary><b>它具体做了什么（技术原理）</b></summary>

作者把两套已知手段组合起来：

1. **CMP unlocker 的 V67/PLM 技术**：往 GPU 的 PLM（性能锁管理）寄存器写入 “speed selector”。
2. **UEFI 方案里的 pre-driver 时机**：在 NVIDIA 驱动绑定显卡**之前**完成解锁。

具体动作是在内核模块里：
- 写入速度选择器 `SS0 = 0x88888888`（选中完整 BF16 速率）和 4 位 `SS1 = 0x8`（IMLA4）；
- 执行 **FLR**（Function Level Reset）；
- 然后**未经修改的官方 NVIDIA 驱动**才去绑定这块卡。

关键定位（README 明确说了）：

| 不是 | 是 |
|---|---|
| UEFI 应用程序 | 普通 Linux kmod |
| 打过补丁的 NVIDIA 驱动 | 官方原版驱动照常使用 |

> 这带来最大的实用优势：**不用每次 NVIDIA 发新驱动就重新打补丁**。以前 UEFI 方案要刷 BIOS，驱动补丁方案要跟版本，这个是“一次装好，长期有效”。

</details>

<details><summary><b>解锁效果（官方 benchmark 数据）</b></summary>

3090 Ti，8192³ 稠密 GEMM，450 W，FP32 累加，五次采样中位数：

| 输入精度 | Before | After | 提升 |
|---|---|---|---|
| FP16 | 87.46 TFLOPS | 118.39 TFLOPS | ~35% |
| BF16 | 87.50 TFLOPS | **157.41 TFLOPS** | **~80%** |
| TF32 | 42.58 TFLOPS | 82.61 TFLOPS | ~94% |

效果表述为：**把 GA102/GA103/GA104/GA106 上 FP/BF16 FMA（累加到 FP32）的指令发射速度翻倍**。

注意 FP16 提升明显小于 BF16，说明瓶颈不完全在同一个发射通路上，所以“翻倍”是理论上限，实际收益看工作负载。

</details>

<details><summary><b>怎么用（安装要点）</b></summary>

```sh
git clone https://github.com/Zagitta/gpu-unlock
cd gpu-unlock
make install          # 自动识别 Arch / Debian-Ubuntu / Fedora-RHEL 并装包
```

然后在内核命令行加参数（**必须显式 opt-in，否则模块什么都不做**）：

```sh
gpu_unlock.targets=all                          # 解锁所有未绑定的 GA10x 卡
gpu_unlock.targets=0000:21:00.0                 # 只解锁指定 PCI 地址
gpu_unlock.targets=0000:21:00.0,0000:22:00.0    # 多张卡
```

- 地址用 `lspci -D` 打印的完整形式，与 GPU 编号、是否接显示器无关。
- 包会把模块和固件放进 **initramfs**，并在 `nvidia` 之前加载（mkinitcpio / initramfs-tools / dracut 都有 drop-in）。
- 显示卡也能解锁，但屏幕会在 reset 到 `nvidia-drm` 接管之间黑一下。
- 装好后验证：

```sh
journalctl -k -b | grep gpu_unlock
lspci -nnk -s 0000:21:00.0
```

- 想关掉：从内核命令行删掉 `gpu_unlock.targets=` 即可，一次性关闭就在 bootloader 编辑界面删。

</details>

<details><summary><b>其他值得知道的细节</b></summary>

- **睡眠/唤醒**：S3 挂起会断电、速度选择器被复位，所以模块带 suspend notifier，在 `nvidia` 恢复显卡**之前**重新解锁（多花约 1 秒）。可用 `gpu_unlock.resume=0` 关掉。**休眠（hibernation）不支持**。
- **多卡并发**：所有目标卡并发解锁，总共约 1 秒。
- **会跳过**：已被驱动绑定的卡、已断电的卡、非 GA10x 的卡（会记日志忽略）。
- **构建可复现**：每次构建从 HTTPS 下载固定版本的 NVIDIA/CachyOS 输入并校验 SHA-256，生成的固件只存在于构建目录和包里；`make dist` / Debian 源码包 / `.nosrc.rpm` 都不含 NVIDIA 归档和生成固件。
- **许可**：内核模块和本地打包是 GPL-2.0-only；生成 payload 源自固定的 CMP 参考实现，**NVIDIA 固件仍受 NVIDIA 自己的条款约束**（见 `firmware/NOTICE.md`、`licenses/`）。
- 热度：约 **1.2k stars / 142 forks**，作者在收 GPU 捐款（“Help me buy more GPUs”）。

</details>

---

## 有什么用 / 适合谁

1. **低成本本地大模型训练/微调**：BF16 是最常用的训练精度，3090/3080 这类卡在 87 TFLOPS 档位经常是瓶颈；解锁到 ~157 TFLOPS 后，单卡吞吐能显著接近上一档专业卡，性价比大幅提升。
2. **学术界 / 个人研究者的算力平权**：本身就是当年 “GeForce 被阉割 FP16/BF16 速率” 这件事的回应，属于解锁被市场分层限制的硬件能力。
3. **多卡工作站**：按 PCI 地址精确选择要解锁哪些卡，接显示器的卡也能一起解锁。
4. **不想折腾驱动补丁的人**：这是相对旧方案最核心的体验改进——官方驱动照常升级。

## 需要注意的风险（README 没强调但我建议你权衡）

- **保修与条款**：直接写 GPU 内部速度选择器寄存器、做 FLR，属于对硬件的非官方操作，大概率违反保修条款和 NVIDIA 驱动/固件许可；是否构成许可规避在部分地区有法律灰区。
- **功耗与散热**：算力翻倍意味着更高功耗密度，笔记本 / 风道差的机箱要留意。
- **只解锁“指令发射速率”**：**显存容量、显存带宽、NVLink、ECC 等一律不变**。如果模型受限于显存带宽（大量 decode、小 batch 推理），收益会远小于 GEMM benchmark 里的数字。
- **稳定性自担**：FLR + initramfs 阶段介入，出问题时可能导致开机黑屏，需要能进恢复模式删内核参数（作者已提供“改 bootloader 一次性启动”的退路，但要有心理准备）。
- **不支持休眠**，挂起/恢复会多约 1 秒。

---

**结论**：这不是“破解显卡”的通用魔法，而是一个针对 GA10x **BF16/FP16→FP32 累加吞吐**的精准解锁器；对做 BF16 训练、算力吃紧的人来说，是目前体验最省心的一个实现（kmod + initramfs，不碰官方驱动）；对只跑推理或用 FP32/FP64 的人来说，收益有限甚至为零。

---
## 🧑‍💻 User

能access裸金属，如何让集群管理员给每台3090/4090服务器配置上，要能开机自启的，或者slurm启动作业时候能保证patch上

---
## 😀 默认助手 (7)

## 0. 先纠正两个前提（很重要，不然方案会跑偏）

<details><summary><b>① 4090 不在支持范围内，只有 3090 系的 GA10x 能上</b></summary>

main 分支 README 明确只覆盖 **GA102 / GA103 / GA104 / GA106**：

| 你的卡 | 芯片 | main 支持 |
|---|---|---|
| 3090 / 3080 / 3090 Ti / 3060 | GA102 / GA103 / GA104 / GA106 | ✅ |
| **4090** | **AD102** | ❌ |
| 4080 / 4080S | AD103 | ❌（只有未合并的 PR）|

我查了这个仓库的 issue/PR，Ada 支持**全部停留在 PR 阶段**，没有进 main：

- `#4`（open）问 AD103 移植可行性，作者列的阻塞项是“没有可独立验证的 AD103 V67 payload、没有 AD103 的 PLM/SS 可写掩码与编码、没有验证过能交回官方驱动”。
- `#24`（open，4080 SUPER）实测能跑通，但作者写了两句话要注意：**AD103 需要冷启动（cold boot）**，**失败一次就只能断电重来**。
- `#41`（已 closed）4080/4080S，同样标注“AD10x 需要冷启动，失败的尝试需要 power cycle”。

也就是说：**在 4090 机器上试，最坏情况是节点卡死到必须 PDU 远程断电**。如果那台 4090 机器没法远程控电，就别碰。

> 顺带一个现实问题：4090 的 BF16→FP32 dense 本来就有 ~165 TFLOPS，比 3090 的 87 高一档；**4090 的边际收益小、风险大，3090 才是这个项目真正甜点**。

</details>

<details><summary><b>② "Slurm 启动作业时 patch 上" 这个模型不成立</b></summary>

这个模块的工作位置是 **initramfs 里、`nvidia` 驱动绑定显卡之前**：

```
boot → gpu_unlock（写 SS0=0x88888888、SS1=0x8、FLR）→ nvidia 绑定 → 正常用
```

README 里有两句关键的：

- “Loading the module by hand after boot does nothing to a card that `nvidia` has bound.”
- “a card some driver already initialised is still refused.”

所以：

- **开机自启 = 唯一正确姿势**，而且不是 systemd service（那一定太晚，`nvidia` 已经 bind 了，模块会静默跳过，你会以为生效了其实没有）。
- **作业级"补解锁"做不到**。Slurm prologue 里 `modprobe gpu_unlock` 只会得到一行 "skipped"，不会有任何效果。
- 解锁状态**持续到断电**；正常情况下每次开机由 initramfs 自动重做，所以作业期是"已经解锁"的稳态。

**结论：Slurm 侧不该做"打补丁"，该做"验证 + 门禁"。** 下面是完整方案。

</details>

---

## 1. 开机自启：整套部署方案（3090 集群）

<details><summary><b>前置检查清单（先过这关，否则白装）</b></summary>

| 检查项 | 说明 | 不满足的后果 |
|---|---|---|
| **Secure Boot** | 自建 out-of-tree 模块未签名，SB 开启时根本不会加载 | 静默失效。要么关 SB，要么用 MOK 给模块签名（集群安全策略大概率否决） |
| **内核 headers** | initramfs-tools/dracut 的 hook 只给"装了 headers 的内核"编译 | 升级内核后模块消失 → **静默掉回半速** |
| **内核 lockdown** | `lockdown=confidentiality` 会拦 BAR 访问 | 解锁失败 |
| **initramfs 工具** | mkinitcpio / initramfs-tools / dracut 三家 drop-in 齐备 | dracut 没带上时手动加 `force_drivers=gpu_unlock` |
| **不可变/镜像式 OS**（Flatcar、Talos、CoreOS） | 没有包管理器和 headers | 必须**烤进镜像**，不能运行时装 |
| **电源/散热** | 算力翻倍 → 功耗密度上升 | 需重算 PSU 与 PDU 余量、重测温升 |
| **远程带外** | 失败时要能断电 | 见 §3 |
| **hibernation** | 不支持 | 集群一般不用，忽略 |
| **S3 suspend** | 支持，会重解锁，多 ~1s | 集群一般不用 |

</details>

<details><summary><b>安装（每台节点，或烤进镜像）</b></summary>

```sh
# 依赖
# Debian/Ubuntu: build-essential dkms? (无 DKMS) + linux-headers-$(uname -r) + 打包工具
# Arch: base-devel
# Fedora/RHEL: gcc make kernel-devel

git clone https://github.com/Zagitta/gpu-unlock
cd gpu-unlock
sudo make install          # 自动按 /etc/os-release 选 deb/rpm/arch
```

`make install` 会做三件事：装模块 → 装 initramfs drop-in → **重建 initramfs**。装完立刻确认模块真的在 initramfs 里：

```sh
# Debian/Ubuntu
lsinitramfs /boot/initrd.img-$(uname -r) | grep -i gpu_unlock
# Arch
lsinitcpio /boot/initramfs-linux.img | grep -i gpu_unlock
# dracut / RHEL
lsinitrd /boot/initramfs-$(uname -r).img | grep -i gpu_unlock
```

**看不到就不要重启**——重启前先修，否则你会以为什么都没发生。

</details>

<details><summary><b>内核命令行（必须显式 opt-in，否则模块什么都不做）</b></summary>

同构集群直接用 `all` 最省事：

```sh
gpu_unlock.targets=all
```

异构节点（部分卡要留给 vfio-pci 直通）才用 PCI 地址：

```sh
gpu_unlock.targets=0000:21:00.0,0000:22:00.0
```

Ansible 里最稳的写法是**统一渲染 `/etc/kernel/cmdline`**（同时覆盖 UKI 和传统 grub），再按发行版重生成：

```yaml
- name: 追加 gpu_unlock opt-in
  ansible.builtin.lineinfile:
    path: /etc/kernel/cmdline
    regexp: 'gpu_unlock\.targets='
    line: '{{ kernel_cmdline }} gpu_unlock.targets=all'
    create: true
  notify: regen boot

# handlers
- name: regen boot
  ansible.builtin.command: "{{ item }}"
  loop: "{{ regen_commands }}"   # 按发行版: update-grub / grub-mkconfig / grubby / mkinitcpio -P / dracut -f
```

发行版对应命令：

| 发行版 | 命令 |
|---|---|
| Debian/Ubuntu + grub | 改 `/etc/default/grub` 的 `GRUB_CMDLINE_LINUX_DEFAULT` → `sudo update-grub` |
| Arch/CachyOS | 同上 → `sudo grub-mkconfig -o /boot/grub/grub.cfg` |
| Fedora/RHEL | `sudo grubby --update-kernel=ALL --args="gpu_unlock.targets=all"` |
| systemd-boot | 改 boot entry 的 `options` 行（`bootctl status` 看路径）|
| UKI / 生成的条目 | 改 `/etc/kernel/cmdline` → 重生成（如 `sudo mkinitcpio -P`）|

**验收（每台都要过）：**

```sh
journalctl -k -b | grep gpu_unlock      # 应有每张卡的成功日志，无 "refused"/"skipped"
cat /proc/cmdline | grep gpu_unlock
lsmod | grep gpu_unlock
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv
```

</details>

<details><summary><b>内核升级的坑（这是最容易踩的）</b></summary>

- Debian/RHEL 包会自动为**已装且带 headers** 的内核编译模块并在**安装/卸载时**重建 initramfs；之后新装的内核由它自己的 install hook 处理。
- **但**：自己编译的内核、vendor 内核、没装 headers 的内核 → 模块缺失，而且**没有任何报错**，卡就是半速跑。这就是为什么 §2 的 Health Check 是必需品而不是可选项。
- 仓库**没有 DKMS 支持**（我查了 README，是 install-hook 路线）。想上 DKMS 得自己写 `dkms.conf`，或者更简单：用配置管理在每次内核变更后跑一次 `make install`。

</details>

---

## 2. Slurm 侧：用 Node Health Check 做门禁，而不是打补丁

<details><summary><b>思路</b></summary>

解锁是**节点级、持久化、开机完成**的状态。所以 Slurm 只需要回答一个问题：

> 这台节点**现在**真的解锁了吗？没有就 drain 掉，别让用户的作业在无人察觉的情况下跑半速。

三个层次：

1. **Prolog（毫秒级）**：`/proc/cmdline` + `lsmod` 粗筛。
2. **HealthCheck（秒级，周期性）**：查本 boot 的内核日志 + 抽样 microbenchmark 断言。
3. **节点标签**：给健康的节点打 `unlocked` feature，用户 `--constraint=unlocked` 显式选。

</details>

<details><summary><b>验证脚本（先在一台 3090 上抓准日志字符串再用）</b></summary>

README 没给出成功日志的**确切措辞**，所以先在一台 canary 上把字符串抓下来，再写死进脚本：

```sh
nvidia-smi -q | head -5
journalctl -k -b | grep -i gpu_unlock
```

然后：

```sh
#!/bin/bash
# /usr/local/sbin/gpu-unlock-check   (exit 0 = healthy, 非 0 = 不健康)
set -uo pipefail
THRESH="${THRESH:-130}"     # 3090 BF16 8192^3 期望 >= 130 TFLOPS；半速约 85-90

fail() { echo "UNHEALTHY: $*"; exit 1; }

# 1. 内核命令行 opt-in
grep -q 'gpu_unlock.targets=' /proc/cmdline || fail "cmdline 无 gpu_unlock.targets"

# 2. 模块已加载
[ -d /sys/module/gpu_unlock ] || fail "模块未加载（内核升级后 initramfs 掉了？）"

# 3. 本 boot 有成功记录、无失败记录
log=$(journalctl -k -b --no-pager | grep -i gpu_unlock || true)
[ -n "$log" ] || fail "内核日志无 gpu_unlock 记录"
echo "$log" | grep -qiE 'refus|skip|fail|timeout' && fail "日志含失败/跳过: $(echo "$log" | grep -iE 'refus|skip|fail|timeout' | head -1)"
echo "$log" | grep -qiE 'success|unlocked'    || fail "日志无成功标记（措辞待校准）"

# 4. 抽样 benchmark（建议只在 idle 节点跑，见下）
if [ "${SKIP_BENCH:-0}" != "1" ]; then
  tf=$(python3 /usr/local/lib/gpu-unlock/mmapeak.py --dtype bf16 --size 8192 --seconds 3 2>/dev/null | tail -1)
  awk -v t="$tf" -v th="$THRESH" 'BEGIN{exit !(t>=th)}' \
    || fail "BF16 实测 ${tf} TFLOPS < ${THRESH}（疑似未解锁）"
fi
echo "OK: unlocked"
```

**说明两点，别照抄踩坑：**

- 日志匹配规则用 canary 的真实输出替换。别指望我猜对字符串。
- **benchmark 只在 idle 节点跑**：`HealthCheckNodeState=IDLE`，否则会因与其他作业争用而误判。

</details>

<details><summary><b>slurm.conf</b></summary>

```ini
# 周期性健康检查
HealthCheckProgram=/usr/local/sbin/gpu-unlock-check
HealthCheckInterval=300          # 秒
HealthCheckNodeState=IDLE        # 只在空闲节点跑 benchmark，避免误判
ReturnToService=2                # 检查通过后自动回池；防止半速节点被永久 drain 或永久回池

# 让脚本自己 drain（比依赖非零退出码更可控）
```

脚本里失败时主动 drain 并写清原因，比让 Slurm 把节点标 DOWN 更好排查：

```sh
scontrol update NodeName="$(hostname -s)" State=DRAIN \
  Reason="gpu_unlock: $(echo "$log" | tail -1)"
```

**用户侧：**

```sh
srun --constraint=unlocked --gres=gpu:1 ...
# 或在 Prolog 里给个醒目 banner
```

<details><summary>Prolog 轻量版（可选，毫秒级）</summary>

```sh
# /etc/slurm/prolog.d/10-gpu-unlock.sh
grep -q 'gpu_unlock.targets=' /proc/cmdline \
  && [ -d /sys/module/gpu_unlock ] \
  || echo "WARNING: 本节点 gpu_unlock 未生效，GPU 吞吐可能只有一半" >&2
```

不要在这里跑 benchmark（每个作业都跑，延迟不可接受）；也不要在这里 `modprobe`（无效果）。

</details>

</details>

---

## 3. Runbook 必须写进去的失败模式

<details><summary><b>已知 / 需要自测的坑</b></summary>

| 场景 | 后果 | 应对 |
|---|---|---|
| **内核升级丢 headers** | 模块缺失，**静默半速** | HealthCheck 抓；配置管理里加内核变更后 `make install` |
| **`nvidia-smi -r` / GPU reset** | 速度选择器被清回默认值（selector 是易失的，S3 都会清）| ⚠️ **推断，需自测**：跑一次 `nvidia-smi -r` 再测 TFLOPS。如果确实会掉，就在集群规范里禁用 `-r`，或 reset 后自动 reboot 节点 |
| **驱动崩溃 / `nvidia-persistenced` 重启** | 卡已 bind，模块无法重做 | 只能 reboot；HealthCheck 会先 drain |
| **固件永不停止**（模块日志会写） | 模块保留 DMA buffer 等下次断电 | **必须断电**，不能软重启 |
| **AD10x（如果以后在 4090 上试）** | 失败需 power cycle | 需要远程 PDU；且要能派人到现场 |
| **Secure Boot 开启** | 模块不加载 | 关 SB 或 MOK 签名 |
| **容器 / enroot / PyTorch 镜像** | 无影响，unlock 在容器之下 | — |

</details>

<details><summary><b>回滚（写清楚，管理员才敢批）</b></summary>

```sh
# 单节点临时关闭：启动界面删掉 gpu_unlock.targets= 参数，一次性生效
# 永久关闭：
#   包管理 / systemd-boot: 从 cmdline 删除 gpu_unlock.targets=*
#   然后卸载：apt remove gpu-unlock / dnf remove / pacman -R gpu-unlock
#   卸载会一并移除模块和 initramfs drop-in
```

**回滚不需要动 NVIDIA 驱动、不需要重装系统、不需要刷 BIOS。** 这一点是它相对 UEFI 方案和驱动补丁方案的最大优势，一定要写进提案。

</details>

---

## 4. 怎么让管理员签字：一页提案模板

<details><summary><b>可以直接贴给管理员的论证</b></summary>

**收益（可量化）**

- 3090 Ti 实测（8192³ 稠密 GEMM，450 W，中位数）：BF16 $87.50 \to 157.41$ TFLOPS（$1.80\times$），TF32 $42.58 \to 82.61$（$1.94\times$），FP16 $87.46 \to 118.39$（$1.35\times$）。
- 3090 预期同档（同 GA102），即**单卡 BF16 训练吞吐接近翻倍**，等效于凭空多出约一半的卡。
- **不需要改驱动**：不像旧方案那样每次 NVIDIA 发新版都要重新打补丁，长期维护成本极低（这是它设计上的核心卖点）。

**风险（如实列）**

- 保修与 NVIDIA 驱动/固件条款：写内部寄存器 + FLR 属于非官方硬件操作，**可能违反保修条款**，法律灰区需法务/采购确认。
- 无 ECC、无 MIG、显存容量与带宽不变 → **对显存带宽瓶颈的负载（decode、小 batch 推理）几乎无收益**。
- 功耗密度上升，需重算 PDU / 制冷余量。
- 上游是 **1 人项目（1.2k star / 142 fork）**，Ada 支持还卡在未合并 PR；**不建议作为生产 SLA 的唯一依赖**。

**落地建议**

1. 先 1 台 canary，跑 48h：吞吐、功耗、温度、job 失败率 vs baseline。
2. 用**用户真实训练脚本**验收，不要只用 8192³ GEMM。
3. 先以 `--constraint=unlocked` 的**可选池**开放，不全网默认开。
4. HealthCheck 门禁 + drain 语义先上，保证"半速节点不吞作业"。
5. 明确回滚 = 删一个内核参数，可 5 分钟内完成。

</details>

---

## 5. 如果 4090 是你们的主力

三条路，按推荐度排：

1. **不用**。4090 本身 BF16 dense 就有 ~165 TFLOPS，比 3090 高不少，边际收益和风险不成比例。
2. **等 AD10x 进 main**。盯 `#4` / `#24`。现在 main 没有 Ada 路径，别拿生产节点试。
3. **自己 port**（`TOTORONG/AD103_win_unlocker` 那条线，PR #41 就是基于它）。这是**研究项目**而不是运维工作，需要：可远程断电的机器、能接受失败即断电、有人能读 Falcon/WPR 寄存器。PR #41 里的收益参考：4080S BF16 $101.2 \to 198.4$、FP16 $101.5 \to 202.7$、FP8 $201.8 \to 376.1$ TFLOPS。

---

**最小可行路线总结（3090）：** `make install` → 内核命令行加 `gpu_unlock.targets=all` → 确认 initramfs 里有模块 → 重启 → `journalctl -k | grep gpu_unlock` 验收 → 上 `HealthCheckProgram` 门禁 → 先一台 canary 跑 48h。**不要**在 Slurm prologue 里试图解锁。

---
## 🧑‍💻 User

所以阉割的原理是部分指令在流水线里面多赖一个周期？现在密度更高应该能看到功耗有一点提高（跑f32累加，比起原来）才对，为什么没变化呢

---
## 😀 默认助手 (7)

## 结论先行

1. **不能把这种限制简单理解为“某条指令在流水线里多拖一个周期”。** 证据显示，它实现的是**按操作类别分档的速率分频表**（`SS0`/`SS1`），并非延迟惩罚。最关键的两点：FP16 累加 FP16 的路径**完全没变**（202.5 → 201.8 TFLOPS），而解锁后 FP32 累加**精确等于** FP16 累加的上限（202.7 vs 202.5）。如果是“延迟 +1 周期”，这个现象解释不通。
2. **功耗“没变化”是报告方式造成的假象，并不说明物理上没变。** 450 W 标的是 power limit，不是实测 draw。在功率封顶的芯片上，解锁改变的是**每瓦产出多少有效 FLOP**，而不是“用多少瓦”——多出来的算力会先被功耗墙以**降频**的形式吃掉。

---

## 1. 限制机制：速率分频器，不是流水线气泡

<details><summary><b>寄存器位域本身就是答案：<code>SS0</code> 是 8 个 nibble 的分档表</b></summary>

PR #41 / #24 里的实测读写值：

| 状态 | `SS0` | `SS1` |
|---|---|---|
| GA102 出厂（推测档位） | `0x25121025` | `0x7` |
| AD103 出厂 | `0x36000134` | `0x4` |
| 解锁后 | `0x88888888` | `0x8` |

`0x88888888` 这个“整齐”的值几乎不可能是巧合——它是一个**8 个 4-bit 字段的表，每个字段对应一个操作类别**，解锁就是把所有制都拉到最大值 8。而出厂值 `0x36000134`、`0x25121025` 的各个 nibble **互不相同**，正说明不同输入精度被设了不同的速率限制。

`SS1` 的 `0x4 → 0x8` 也是同一个含义：一个**数值翻倍**的字段。这迫使你选择“2 倍速率”这个模型，而不是“加一个周期”；后者的编码方式通常是 `1 → 2` 这样的周期计数。

</details>

<details><summary><b>决定性证据：FP16 累加路径完全没被碰</b></summary>

PR #41（4080S）的 `mmapeak`，每核 3 秒：

| Kernel | Before | After | 比值 |
|---|---|---|---|
| BF16 → FP32 | 101.2 | 198.4 | $1.96\times$ |
| FP16 → FP32 | 101.5 | 202.7 | $2.00\times$ |
| TF32 → FP32 | 50.6 | 99.2 | $1.96\times$ |
| FP8 → FP32 | 201.8 | 376.1 | $1.86\times$ |
| **FP16 → FP16** | **202.5** | **201.8** | **$1.00\times$** |

两个数字放在一起看：

- `FP16→FP16` **一动不动**（202.5 → 201.8）。
- `FP16→FP32` 解锁前 101.5 ≈ $202.5/2$，解锁后 202.7 ≈ **精确回到 202.5 的上限**。

所以锁的性质非常干净：

$$
\text{FP32 累加吞吐} = \frac{1}{2}\times\text{（硬件真实上限）}\quad\Longrightarrow\quad\text{解锁 = 去掉这个 } \tfrac{1}{2}
$$

被限制的不是“某条指令慢一点”，而是 **“累加进 FP32”这条数据通路被按 1:2 的占空比分频了**。FP16 累加根本不经过那条被限流的路径，所以完全不受影响——这也顺带解释了 PFLOPS 规格表上为什么只标 “with FP32 accumulate” 会减半。

**这也直接回答了“多延迟一个周期”的说法为什么站不住：** 对 8192³ 这种分块 GEMM，延迟几乎完全能被 ILP 和 tile 级流水藏掉。就算 MMA 的延迟真的 +1 cycle，你也只会损失百分之几的吞吐，绝不会是干净的 $2\times$。能拿掉 $2\times$ 的只能是**发射速率/占空比**，不是延迟。

</details>

<details><summary><b>那物理实现到底像什么？</b></summary>

从外部只能观察到“按类别 × 1/2”，内部两种实现都能等概率地产生这个现象：

| 实现 | 说明 | 能否区分 |
|---|---|---|
| **发射占空比**（前端每 2 拍只放行 1 个 FP32-acc MMA） | 最便宜、最符合“熔断 + PLM 分档表”的设计风格 | — |
| **数据通路半分使能**（复用一半累加器阵列） | 需要额外的门控逻辑，且会让预解锁时功耗明显更低 | 见 §3 的锁频实验 |

从“解锁几乎不花电”这个事实反推，**占空比模型更可能**：占空比的空拍照样要跑时钟树、照样要保持累加器状态、照样要在 warp 间切换上下文，这些能量早就付掉了。

> ⚠️ 注意这只是从仓库证据反推的推断，作者没公开 PLM 位域语义。要确认得做 §3 的锁频测量。

</details>

---

## 2. 为什么功耗看不出变化

<details><summary><b>第一层：450 W 是 cap，不是实测值</b></summary>

README 写的是 “3090 Ti dense 8192³ GEMM **at 450 W**”——这是 `nvidia-smi -pl` 的上限设定。**在功率封顶的条件下，功耗本来就“不可能”变化**，它是自变量。被测量的是吞吐。

正确的换算方式是**每瓦有效算力**：

$$
\text{Before}: \frac{450\ \text{W}}{87.50\ \text{TFLOPS}} = 5.14\ \frac{\text{W}}{\text{TFLOPS}},\qquad
\text{After}: \frac{450\ \text{W}}{157.41\ \text{TFLOPS}} = 2.86\ \frac{\text{W}}{\text{TFLOPS}}
$$

也就是**能效改善 $1.80\times$**。功耗曲线没动，但同样的瓦数现在产出近两倍 FLOP。

</details>

<details><summary><b>第二层：既然密度更高了，为什么瓦数不涨？因为 SM 从来没闲着</b></summary>

这是最容易搞错的地方。`nvidia-smi` 报的 99% utilization 在解锁前后**都是 99%**——功率被封顶的芯片，SM 时钟在跑、warp 在切、寄存器文件在读写、shared memory 在搬数据，功耗满载；区别只是**每秒钟退休了多少个 MMA**。

被分频的那一半空拍，能量花在了：

- 时钟树（占 GPU 动态功耗的大头之一，与是否产出 FLOP 无关）
- 累加器/操作数的保持与重读
- 更多 warp 上下文切换、更多调度开销（因为要在同样的时间里塞进两倍的迭代次数）

所以“功耗不变、吞吐翻倍”不是奇迹，而是说：

$$
\text{预解锁功耗} = \underbrace{P_{\text{有用}}}_{\text{一半}} + \underbrace{P_{\text{空转开销}}}_{\text{另一半}}
$$

解锁后 $P_{\text{空转开销}}$ 被转化成了 $P_{\text{有用}}$，总额不变，产出翻倍。**从 NVIDIA 的角度这也是为什么用“分频”而不是“物理砍单元”做产品分层——后者会让预解锁卡功耗明显更低，产品线功耗表就对不上了。**

</details>

<details><summary><b>第三层：真正的物理反应是“降频”，而你没看那一栏</b></summary>

功率封顶时的反馈回路是：

$$
f \downarrow \;\Longrightarrow\; P = P_{\text{dyn}}(f,V) + P_{\text{leak}} \le 450\ \text{W}
$$

解锁后 SM 利用率真的上去了，功率会先冲过 450 W，然后 **boost 逻辑把时钟压下来**贴住墙。净效果：**功率不动、时钟变低、吞吐变高**。

如果你只 `--query-gpu=power.draw`，当然什么都看不到。应该看：

```sh
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu \
           --format=csv -l 1
```

**预测：解锁后 `clocks.sm` 会掉一档（这是最该看到的信号），`power.draw` 基本贴着 cap。**

</details>

---

## 3. 要看到功耗真的变，必须做这三个实验

<details><summary><b>实验协议 + 预期结果（这才是能证伪模型的）</b></summary>

### 实验 A：锁频，看功耗增量（隔离“密度”这一项）

```sh
# 固定时钟 + 放开功耗上限，让电压/频率不再自适应
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 450
sudo nvidia-smi -lgc 1500          # 锁定 SM clock
python3 mmapeak.py --dtype bf16 --size 8192 --seconds 10
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv
```

| 观察 | 说明 |
|---|---|
| 吞吐 $\approx 2\times$，功耗 **+15~40%** | 占空比模型成立：空拍本来就费电，只多出了数据通路那部分 |
| 吞吐 $\approx 2\times$，功耗 **接近 $2\times$** | 半分使能模型：解锁把原本门控掉的单元全打开了 |
| 吞吐 $\approx 2\times$，功耗**几乎不变** | 说明被限流的单元在预解锁时**根本没上电**，或者测量点不对（看的是整板而不是 GPU 核心） |

### 实验 B：锁功耗，看降频（还原真实机房条件）

```sh
sudo nvidia-smi -rgc                    # 放开时钟
sudo nvidia-smi -pl 300                 # 压低 cap
# 对比解锁前后的 clocks.sm 与 TFLOPS
```

预期：**瓦数相同、频率更低、吞吐更高**。如果你在 3090（350 W，不是 3090 Ti 的 450 W）上做，cap 更容易咬住，**收益会明显小于 $1.8\times$**——这是你评估集群收益时必须按机型分别实测的原因。

### 实验 C：追那个 FP16 异常值

3090 Ti 的数据里 FP16 只涨了 $1.35\times$（$87.46 \to 118.39$），而 BF16 涨了 $1.80\times$，TF32 涨了 $1.94\times$。但 4080S 上 FP16→FP32 是干净的 $2.00\times$。**这两个数据集互相矛盾**，说明问题出在特定 kernel 或特定功率点，而不是机制不同。可能原因：

- 450 W 下 FP16 kernel 的能耗更高，功率墙先咬住（→ 用实验 A 的锁频法排除）
- 那个 FP16 测量用的 kernel/形状不同，或累加精度设置不一致
- 测量噪声

**别把 $1.8\times$ 当常数写进容量规划**：按机型 × 按 dtype 实测，尤其 3090（350 W）和 3080（320 W）会差很多。

</details>

---

## 4. 三个反直觉推论

<details><summary><b>对集群规划的直接影响</b></summary>

1. **低功耗卡收益更小。** 3090 Ti（450 W）能吃下 $1.8\times$；3090（350 W）和 3080（320 W）会被功耗墙更早咬住。**“3090 也能翻倍”是不成立的预期**，必须实测。
2. **散热受限的机器收益更小。** 机房靠风冷 + 高进风温度的节点，解锁后降频更狠。这也解释了为什么作者要做 450 W 的测量——那是他最宽松的条件。
3. **能效提升是真实的，这比吞吐提升更有价值。** $5.14 \to 2.86\ \text{W/TFLOPS}$ 意味着在**总功耗预算固定**的集群里，你等价于凭空多出约 45–80% 的训练吞吐，而不用改供电。这是提案里最该强调的数字，比“翻倍”更站得住。

</details>

---

**一句话回答你的两个问题：** 限制机制是**按操作类别编码在 8×4-bit 分档表里的发射占空比分频**（FP16 累加路径根本不经过它，所以 202.5 → 201.8 一动不动）；功耗看不出变化，是因为**它被 450 W 的 cap 钉死了，而 SM 在分频空拍里从来没停过——那些能量本来就付了**。想看功耗真变，锁频起来测；想看频率真变，放开功耗测。

---
## 🧑‍💻 User

那么原来跑f16累加会稍高吗

---
## 🧑‍💻 User

multiplier

---
## 😀 默认助手 (7)

## 直接回答

两种情况下的结果不一样：

| 你的问题 | 答案 |
|---|---|
| A. 出厂状态下，FP16 累加比 FP16→FP32 累加高多少？ | 不是只“稍高”，而是**整两倍**，因为限流原本就作用在 FP32 累加通路上 |
| B. 出厂状态下的 FP16 累加，对比解锁后的 FP32 累加，谁更高？ | **基本完全一样**（202.5 vs 202.7，两者差异在噪声范围内） |

---

## 数据

<details><summary><b>PR #41（4080S，<code>mmapeak</code>，每核 3 秒）</b></summary>

| 状态 | FP16 → **FP16** 累加 | FP16 → **FP32** 累加 | 比值 |
|---|---|---|---|
| 出厂 | **202.5** | 101.5 | $2.00$ |
| 解锁后 | **201.8** | 202.7 | $1.00$ |

两个结论：

1. **出厂时 FP16 累加 = 出厂时 FP32 累加的两倍**，而且分毫不差：$101.5 \times 2 = 203 \approx 202.5$。这是“锁只把 FP32 累加通路减半”最直接的证据。
2. **解锁后两者汇合**：FP32 累加从 101.5 被抬到 202.7，正好追上 FP16 累加所达到的 202 左右。

所以准确说法是：**出厂状态下，FP16 累加高出一倍；解锁之后，FP32 累加追平了它。**

</details>

<details><summary><b>但 3090 Ti 这边有个异常，值得你单独跑一次</b></summary>

README（issue #1）只给了 FP32 累加的数，**没给 3090 Ti 的 FP16→FP16**：

| 3090 Ti，450 W | Before | After |
|---|---|---|
| FP16 → FP32 | 87.46 | **118.39**（$1.35\times$）|
| BF16 → FP32 | 87.50 | **157.41**（$1.80\times$）|
| TF32 → FP32 | 42.58 | 82.61（$1.94\times$）|

用规格核对一下：3090 Ti 的 FP16 Tensor 稠密峰值约为 $320/2 = 160$ TFLOPS。解锁后的 BF16 跑出 **157.4**，已接近满速；但解锁后的 FP16 只有 **118.4**，为峰值的 74%。

**如果锁的性质就是“FP32 累加 = 1/2 × 上限”，那 FP16 也应该到 ~157，而不是 118。** 这里要么是 FP16 那次测量所用的 kernel 或形状不同，要么是 450 W 下 FP16 先受到功耗限制（但 BF16 没有受到，这不合理），要么就是测量噪声。

**而区分这几种可能性的唯一方法，正好就是你说的那个数：出厂状态下跑 FP16→FP16。**

- 如果出厂 FP16 累加约为 **157**，那么上限就是 157，118 是那次测量的假象，解锁是完整的；
- 如果出厂 FP16 累加只有 **~118**，那么 GA102 上 FP16 的上限本身就和 BF16 不同，是 BF16 在解锁时额外获得了更多速率，而不只是“去掉了 1/2”。

</details>

---

## 那能不能“一直用 FP16 累加”来白拿这 2 倍？

<details><summary><b>不能——精度代价是硬性的，这也是解锁真正的价值所在</b></summary>

FP16 累加的限制来自数值范围，与速度无关：

- **溢出**：FP16 最大值 $65504$。GEMM 的累加和一旦超过这个数，结果就变成 `inf`。
- **吸收误差**：当累加和已经很大时，加上一个小项会被舍入掉（$2^{-11}$ 的相对分辨率）。K 维越长（LLM 训练里 K 动辄几千到几万），误差累积越严重。
- 所以训练里**必须**使用 FP32 累加（`allow_fp16_reduced_precision_reduction = False`），这也是 Ampere 规格表把 “with FP32 accumulate” 单独列成一半的原因。

于是形成了一个畸形的取舍：

| 想要 | 出厂状态 |
|---|---|
| 要精度（FP32 累加） | 只能拿到一半速度 |
| 要速度（FP16 累加） | 牺牲数值正确性 |

**解锁做的事，就是把这个取舍取消了**：FP32 累加现在也能跑满。这才是它有用的根本原因：它并非单纯从数字上“多给一倍算力”，实际效果是把“必须用 FP16 累加才能跑满”这个限制移除了。对深度学习来说，这就是从“不可用”变成“可用”。

> 反过来，如果你的负载本来就是 FP16 累加（纯推理且误差可接受、或者 K 很小的 GEMM），那么解锁对你是**零收益**，因为它本来就没有被限流。

</details>

---

## 怎么在你们的 3090 上把这两组数测准

<details><summary><b>关键开关 + 脚本（注意默认值会误导你）</b></summary>

PyTorch 的默认值是 **允许 FP16 降低精度归约**，如果你不显式关闭它，你以为在测 FP32 累加，实际测的却是 FP16 累加：

```python
import torch, time

def bench(dtype, fp32_acc, size=8192, iters=30):
    # 这一行决定走哪条数据通路，是本实验唯一的自变量
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not fp32_acc
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not fp32_acc
    torch.backends.cuda.matmul.allow_tf32 = (dtype is torch.float32)

    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    for _ in range(5):                      # warmup
        torch.mm(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 2 * size**3 * iters / dt / 1e12
    print(f"{dtype} fp32_acc={fp32_acc}: {tflops:.1f} TFLOPS")
    return tflops

# 四个格子，缺一不可
bench(torch.float16, fp32_acc=True)    # ① 出厂=锁着的 FP32 累加
bench(torch.float16, fp32_acc=False)   # ② 出厂=满速的 FP16 累加（你问的那个数）
for m in (bench(torch.float16, True), bench(torch.float16, False)):
    pass
# 解锁后重复 ①②，看它们是否汇合
```

**顺带三个必须固定的条件，否则数不可比：**

```sh
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1500        # 锁频：否则功耗墙会偷偷吃掉差异
sudo nvidia-smi -pl 450          # 或者锁一个你真实机房用的值
```

`--query-gpu=clocks.sm,power.draw,temperature.gpu` 全程录下来。上次 118 那个数，很可能就是因为没锁频、被 450 W 墙咬住所致。

</details>

---

## 一个诚实的保留

我在上一轮说“解锁后精确回到天花板”，这里需要修正为更弱的表述：**202.5 和 202.7 相等，也可以解释为“两者都被同一个后续瓶颈（功耗）挡住了”**，而不一定是“共享同一个硬件天花板”。PR #15 里作者提到测试功率上限不超过 220 W，而在那个功率点上，FP16 累加大概本来就到不了 4080S 的真正峰值（规格稠密约 418 TFLOPS）。

所以严格说法是：

$$
\text{出厂}: \quad \text{FP32 累加} = \tfrac{1}{2}\times(\text{同一瓶颈下 FP16 累加的速率})
$$

这个 $\tfrac{1}{2}$ 是确定的；被减半之后的“上限”本身受功耗约束，所以要测“真实天花板”，必须**锁频 + 放开功耗**测，不能用默认开机状态下的数。这也是为什么我建议你先在 3090 上补测 ② 那一格。

---
## 🧑‍💻 User

目前已知的情报，逐条核验：
1. 50系没戏，firmware启动架构什么的都不同，连具体限制如何达成的都还没解析好，更别提解锁了
2. 20/30/40系的解锁原理类似，但是个别需要的操作不同
3. 解锁原理都是在driver接手之前用DMA exploit往卡的firmware/register做操作，来把对应的register改成full issue rate
4. 达成3的方式有两种: EFI application与initramfs kernel module。后者仅限linux，前者可以windows
5. 20/30的路径是speed select有permissions gate，但40路径是WPR_CFG/WPR/FEAT的gate，所以虽然目标操作相同但具体实现不同。
6. 带锁5090凭借高带宽与自带的无锁mxfp8/nvfp4以及快的不可理喻的cuda core依然辗压4090

有阉割的只有tf32, fp16,bf16,fp8的"fp32累加"
其他包括fp16,fp8的fp16累加，int8, int4，以及mxfp8, nvfp4都没有阉割
并且只有tensor core有阉割，对游戏性能基本无影响。

参考：
https://github.com/Zagitta/gpu-unlock
https://github.com/JustAResearcher/gpu-unlock-ada
https://github.com/KohakuBlueleaf/KohakUnlock

---
## 😀 默认助手 (7)

## 核验结论一览

| # | 你的说法 | 判定 |
|---|---|---|
| 1 | 50 系没戏，固件启动架构都不同 | ✅ 成立（措辞上"没戏"→"未解"，官方口径是 *research target*）|
| 2 | 20/30/40 原理类似，个别操作不同 | ✅ 成立 |
| 3 | driver 接管前用 DMA exploit 改 register 到 full issue rate | ⚠️ **方向对，"DMA exploit"措辞不准**，而且漏了最关键的**冷启动前提** |
| 4 | EFI app / initramfs kmod 两条路，后者仅 Linux | ⚠️ **"EFI = Windows" 这个二分是错的**，EFI 是 OS 无关的 |
| 5 | 20/30 是 speed-select 有权限门，40 是 WPR/FEAT 门 | ⚠️ **部分成立**：40 最终写的还是 SS0/SS1，"门"是前置而非替代 |
| 6 | 带锁 5090 辗压 4090 | ⚠️ **结论对，归因要改**：FP16/BF16 那一列反而输给解锁后的 4090 |
| 7 | 只有 TF32/FP16/BF16/FP8 的 **FP32 累加**被阉 | ✅ 成立，但 FP8 那条的证据链有个洞 |
| 8 | FP16-acc / INT8 / INT4 / mxfp8 / nvfp4 没阉 | ⚠️ 前三个 ✅；**mxfp8/nvfp4 目前无法核验**（20/30/40 上没有这硬件）|
| 9 | 只有 tensor core 有阉，游戏基本无影响 | ✅ 成立 |

---

## 逐条核验

<details><summary><b>① 50 系：成立，但要说"未解"而不是"没戏"</b></summary>

KohakUnlock 原文：**"Other board profiles and Blackwell remain research targets."**

所以事实是：**没有任何公开可复现的 Blackwell 解锁**。你的判断在"当前没有可用方案"这个层面是对的。但要区分两件事：

- "**没人做出来**" ≠ "**做不出来**"。三个仓库的口径都是 research target（研究中），不是 "impossible"。
- 而且**动机比以前大得多**：如果 5090 沿用同样的 1:2 分档（规格上 FP8 dense 838 vs 带 FP32 累加 419），解锁收益是 $+419$ TF/s，比 4090 的 $+165$ 还大。所以不是没人想要，是啃不动。

技术上你的归因基本站得住：Blackwell 的启动链（认证层级、GSP 角色、booter 语义）都换了，而且**限制到底落在哪个寄存器/哪层固件上都还没被定位**——而 20/30/40 的所有方案都建立在"已定位 PLM/SS 语义 + 有一条能跑未授权 payload 的 booter 通道"之上。前提没了，方案就无从移植。

</details>

<details><summary><b>② 20/30/40 原理类似：成立，但有个被忽略的实用差异</b></summary>

三家数据放在一起，**1:2 这个比例在四个世代上完全一致**，这是"原理相同"最硬的证据：

| 世代 | 卡 | FP16→FP32 | FP16→FP16（对照） | 比值 |
|---|---|---|---|---|
| Turing | 2080 Ti | $59.9 \to 111.0$ | $119.5 \to 118.5$ | $+85.3\%$ / 不变 |
| Ampere | 3090 | $73.03 \to 122.76$（PyTorch GEMM）| — | $+68.1\%$ |
| Ada | 4090 | $165.6 \to 331.6$ | — | $\approx 2.00$ |
| Ada | 4070 Ti S | $95.0 \to 189.9$ | — | $\approx 2.00$ |

**但"原理相同"会让你低估一个差异：Turing 只有 FP16。** KohakUnlock 明确写 **"BF16/TF32 are not added to Turing"**——不是"不解锁"，是硬件根本没有。所以：

- 2080 Ti 解锁只买到 **FP16 训练**速度，买不到 BF16。
- 3090/4090 解锁才同时拿到 FP16 + BF16 + TF32（+FP8）。

对你们做 BF16 训练的集群来说，**20 系基本没有价值**，别把它和 30/40 混为一谈。

</details>

<details><summary><b>③ "DMA exploit" 措辞不准：这是"借合法通道执行不受信固件"</b></summary>

准确描述应该是：

> 在 `nvidia` 驱动 bind 之前，**通过 GPU 自身的 Booter / FWSEC 固件加载链路**，把一段自制 payload DMA 进显存，让卡上的 Falcon 微控制器**以特权身份执行它**，由它去写 PLM/SS 寄存器。

关键差别：

| 你的说法 | 实际机制 |
|---|---|
| DMA exploit 读写寄存器 | **不是**直接 poke 寄存器，而是**在卡上跑代码**（未授权固件执行）|
| 一次性操作 | Ada 的 payload 是 **fully baked**（"nothing is patched at runtime"），指令地址来自对 **debug booter 的解密分析**，生产 booter 在 GPU 内部认证解密 |

**而"驱动接手之前"这个前提，你只对了一半——真正苛刻的是"冷启动"：**

JustAResearcher 原文：**"Power the machine off and start it again for the first attempt. The module needs a card whose firmware state is fresh."** 原因是要读到 `WPR2 == 0`；暖重启**不保证**回到这个状态，模块会拒绝重跑。

> ⚠️ **这对你的集群是硬约束**：Ada 的失败路径是"必须断电重来"，不能靠 `reboot` 恢复。没有可控 PDU 的机器不能上。

</details>

<details><summary><b>④ 两条交付路径：二分法错了</b></summary>

你的说法 vs 事实：

| 你的说法 | 实际 |
|---|---|
| initramfs kmod → 仅 Linux | ✅ 对 |
| EFI application → 可以 Windows | ⚠️ **EFI 不是 Windows 专属**——它是 OS 无关的。JustAResearcher 明确提到 4090 在 **Linux 上**也用 "persistent EFI default" 完成解锁 |

正确的二分是 **"EFI 方案（OS 无关）" vs "kmod 方案（Linux 专属）"**，不是 "Windows 方案 vs Linux 方案"。

另外还有**第三条历史路径**你没列：20 系的早期路线是**改 vBIOS / nvflash**（Turing-unlock、cmp50hx-unlock 谱系），以及 Windows 下的签名驱动 + 用户态工具。KohakUnlock 的 Turing 实现明确基于 `haohervchb/Turing-unlock` 和 `xrip/cmp50hx-unlock`。

**两条路的实际操作差异（对你的运维有意义）：**

| | EFI app | initramfs kmod |
|---|---|---|
| OS | Windows / Linux 都行 | 仅 Linux |
| Secure Boot | **必须关**（KohakUnlock: "successful trial with Secure Boot off"）| 需 MOK 签名 |
| 内核升级 | 不受影响 | 需重编重打 initramfs |
| 持久化 | 靠 bootloader entry（one-shot entry 不持久）| 随包自动 |

</details>

<details><summary><b>⑤ 30 vs 40 的"门"：你的方向对，但表述需要修</b></summary>

**先说你对的部分：目标操作确实相同。** JustAResearcher 原文：

> "The RTX 4090 path ... writes **SS0 `0x88888888` and SS1 `0x8`**, then performs FLR before the unmodified NVIDIA driver binds."

所以 40 系**最终写的也是 speed select**，不是别的寄存器。你"目标操作相同"这一句是对的。

**要修的是"门"的归属。** 证据来自 Zagitta 主仓的 PR #24：

> "The GA100 Booter the GA10x path uses **halts on Ada without opening FEAT permissions**."

所以准确关系是：

$$
\text{Ampere}:\quad \text{SS 写} + \text{FLR}
$$
$$
\text{Ada}:\quad \underbrace{\text{WPR\_CFG} + \text{WPR} + \text{FEAT}}_{\text{开权限门}} + \underbrace{\text{SS0/SS1}}_{\text{同一个目标操作}} + \text{FLR}
$$

WPR_CFG/FBPA/WPR/FEAT 是**前置门禁**，不是替代品。用 GA102 那套直接去写 Ada 的 SS，会在 Booter 阶段就 halt。

**可观测标志也不同**（这是实操上最有用的一条）：Ada 有一个 `fmla32` 位，解锁后 **$1 \to 0$**，且 NVIDIA RM 会报告 "full FMLA32 rate"——这是比"看日志"更硬的验证手段。Ampere 没有这个读数。

**我不能确认的部分：** 你说"20/30 的 speed select 有 permissions gate"——三个 README 里都没有描述 Ampere/Turing 存在单独的权限门这一步，已知序列就是 SS 写 + FLR。这条标记为**待证据**，别写进文档。

</details>

<details><summary><b>⑥ 5090 辗压 4090：结论对，但有一列是反的</b></summary>

你列的三条依据，逐条核：

| 依据 | 数据 | 判定 |
|---|---|---|
| 高带宽 | 5090 ~1792 GB/s vs 4090 ~1008 GB/s → $1.78\times$ | ✅ |
| 自带无锁 mxfp8/nvfp4 | 40 系**完全没有** FP4/FP6/microscaling 硬件 | ✅ 最大的结构性优势 |
| CUDA core 快 | 5090 ~104.8 vs 4090 ~82.6 TF/s → $1.27\times$ | ✅ |

**但"碾压"在 FP32 累加这一列不成立。** 按公开规格换算（⚠️ 5090 数字请自行核对）：

| 精度（FP32 累加） | 带锁 5090 | 解锁 4090 |
|---|---|---|
| FP16 / BF16 | $\approx 209$ TF/s | **$331.6$ TF/s** |
| FP8 | $\approx 419$ TF/s | $\approx 603$ TF/s（实测 91% 满速）|
| FP4 | $\approx 838$ TF/s | **无这硬件** |

所以精确的说法是：

- **量化和带宽受限的负载**（FP4/FP8 推理、大模型 decode、长上下文）→ 5090 碾压，$\text{4090}$ 无解。
- **BF16/FP16 + FP32 累加的训练** → **解锁后的 4090 反超带锁 5090**，因为 5090 同样被 1:2 分档砍了一半。

这个反直觉的结果恰好解释了"为什么有人拼命要解 5090"：一旦解开，FP8→FP32 从 419 变 838，那才是真正的断层。

</details>

---

## 阉割范围清单核验

| 你列的 | 证据 | 判定 |
|---|---|---|
| FP16→**FP32** | $202.5$ vs $101.5$；$119.5$ vs $59.9$ | ✅ |
| BF16→**FP32** | $157.41$（解锁）；"BF16 16–20% gain" | ✅ |
| TF32 | $42.58 \to 82.61$；$47.49 \to 94.51$（$1.99\times$）| ✅ |
| FP8→**FP32** | 4080S Windows：$199.4 \to 310.4$（$+55.7\%$）；4070TiS：$165.45 \to 323.95$（$1.96\times$）| ✅ 但见下 |
| FP16→**FP16** acc | $202.5 \to 201.8$；$119.5 \to 118.5$ | ✅ 完全未动 |
| FP8→**FP16** acc | 4090 解锁后达 $603$ ≈ $91\%$ 的 $660.6$ 满速 → 间接支持 | ⚠️ 无直接 A/B |
| INT8 | "INT8, dp4a and scalar FP32 stayed near baseline"；$379.5$ TOP/s 不变 | ✅ |
| INT4 | "Unsigned INT4/INT32 stayed at ~$754.8$ TOP/s in both states" | ✅ |
| 标量 FP32（CUDA core） | $37.5 \to 37.7$ TF/s | ✅ |
| mxfp8 / nvfp4 | **这三个仓库无法核验**（20/30/40 无此硬件）| ⚠️ **无证据** |
| 游戏无影响 | 只有 tensor 的 FP32-acc 通路受影响，CUDA core/DLSS 走 FP16-acc | ✅ |

**FP8 那条证据链有个洞，你自己心里要有数：** JustAResearcher 自己承认，4070 Ti SUPER 上做的**匹配 CUTLASS E4M3 GEMM 只差 0.3%**（$59.326 \to 59.495$），理由是 "that kernel reaches only about 17% of this card's FP8 peak, so it cannot resolve an issue-rate change"。所以 FP8 被阉是**靠寄存器级探针（1.96×）和 Windows 上另两个第三方仓库**撑起来的，不是靠自己的应用级 GEMM。作者自己把 FP8 排除在该仓库的 pass/fail 门禁之外。

**mxfp8/nvfp4 没有阉割——这个推断合理但没证据。** 唯一的结构性支持是：分档表是**按操作类别枚举的定长表**（`SS0` 8 个 nibble），新格式不进表就不被限流。但注意出厂值 `0x25121025` / `0x36000134` 的**每个 nibble 都比 8 小**，说明表里不止 4 个类别被动了——**nibble 与"是否被阉"不是 1:1 映射**，这个推断别当结论用。

---

## ⚠️ 最重要的一条：**issue rate ≠ 应用性能**

这条会直接改你的集群容量规划，所以我单独拎出来。

JustAResearcher 自己的原文（把 2× 打回了原型）：

> "Real cuBLAS GEMM median throughput varied by shape: **FP16 gained 12–34%, BF16 gained 16–20%** at two shapes and was **essentially unchanged at 4096³**."

> "These are **register-resident issue rates, not application GEMM or inference speedups**."

| 测量层级 | FP16 增益 | 来源 |
|---|---|---|
| 寄存器级 issue rate（`mmapeak`） | $\approx 2.0\times$ | 4090 / 4070TiS / 4080S |
| 真实 cuBLAS GEMM（450 W 匹配） | **$1.12 \sim 1.34\times$** | 4090 |
| PyTorch 8192³ GEMM | $1.68\times$ | 3090 |
| 量化 MoE 服务负载 | **$0.983\times$（无收益）** | 48 GB 4090 |

**为什么差这么多？** 因为原来被认为"废掉的一半"里，有一部分本来就不是瓶颈：真实 GEMM 里 tensor issue rate 常常**被 memory / L2 / epilogue / 指令混合**挡住，解锁只是把其中一个上限抬走了。而 8192³ 这种巨形状恰好是 issue-rate 受限的，所以数字好看。

**所以：**

- 我前两轮引用的 $1.8\times$ / $2.0\times$ 都是 **issue rate 或特定巨形状 GEMM**，不是你们真实训练跑分的预期。
- 集群容量规划必须用**你们自己的 shape** 实测，别用 README 头条数字。
- 量化/低精度推理（尤其 MoE、GPTQ/AWQ）**基本零收益**——那条路径压根不走 FP32 累加。

---

## 对我前两轮回答的修正

<details><summary><b>两处实质修正（都有新证据）</b></summary>

**1. 功耗不是"看不出来"，实测是 $+10.5\%$。**

JustAResearcher 的 4090 证据：**"an unlocked RTX 4090 peaked at 346 W against 313 W stock"**，且 "spent both MMA and GEMM passes under SW Power Cap"。

$346/313 = 1.105$，对 $2\times$ 的吞吐。这比我预测的 $+15\sim40\%$ **还要低**——进一步支持"占空比里的空拍照样费电，摘掉它接近免费"这个模型。同时也说明：**功耗墙确实会被撞到，撞到后就是用降频换吞吐**。

**2. 支持范围要更正：**

- **Zagitta/gpu-unlock 主仓**：只到 GA10x（3090 系）。**4090 不在里面。**
- **JustAResearcher/gpu-unlock-ada**：这是主仓的 fork（v1.1.1），**4090 已支持**（AD102，2× 实测），另有一张 16 行"实验性 ID 表"，**每个 ID 要单独 `gpu_unlock.xxx_experimental=1`，且必须按精确 PCI 地址点名，`targets=all` 永远选不中它们**。
- 所以"20/30/40 都能解"这个说法，在**可用性**上是有层级的：30 系 = 稳定；40 系 = 4090 可用、其余实验性；20 系 = 只有 FP16，且是另一套 Turing 谱系。

</details>

---

## 一句话总结

你的情报**大概八成像**，四个世代确实是同一个 $1{:}2$ 分档、同一类"跑固件改寄存器"的手法；需要改的是三处措辞（不是纯 DMA poke、EFI 不分 OS、40 系最终也写 SS）、一处待证（Ampere 权限门）、一处反例（FP32 累加列上解锁 4090 反超带锁 5090），以及**一个最重要的认知**——$2\times$ 是 issue rate，真实 GEMM 只有 $1.1\sim1.3\times$，量化推理接近 $0$。

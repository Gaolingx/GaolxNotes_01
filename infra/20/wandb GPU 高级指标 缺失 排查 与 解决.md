# wandb GPU 高级指标 缺失 排查 与 解决

## 🧑‍💻 User

新版wandb部分机器上看不到gpu高级信息如何解决（gpu pcie/nvlink tx/rx等）？

---
## 😀 默认助手 (7)

下面结论是我直接对着 wandb 仓库 `main` 分支的源码核对过的（`symon/` 这个 Rust 系统指标服务 + `wandb/env.py`），不是凭印象。先说结论，再给排查和修复步骤。

---

## 一句话结论

新版的 GPU 高级指标（PCIe/NVLink TX/RX）**本来就不是默认保证有的**，它只有两个来源：

1. **NVML 的 GPM**（GPU Performance Monitoring）——**仅 Hopper 及更新架构（H100/H200…）支持**，`GPM_METRICS` 里包含 `pcieTxBytes/pcieRxBytes/nvlinkTxBytes/nvlinkRxBytes`；
2. **可选的 DCGM profiling**——**默认关闭**，必须显式打开 `WANDB_ENABLE_DCGM_PROFILING=true`，且机器上要有 `libdcgm.so.4` + DCGM host engine 跑在 `localhost:5555`（Linux only）。

所以「部分机器看不到」= 那批机器的 GPU 不是 Hopper+（A100/V100/T4/消费卡）→ 没有 GPM；或者没装/没开 DCGM；或者 MIG / vGPU / 容器里 NVML-DCGM 不可见。这些都属于**采集端拿不到**，不是 UI bug。

<details>
<summary><b>源码依据：新版采集链路（wandb-symon，Rust）</b></summary>

- `symon/hatch.py`：wheel 里会用 cargo 构建并打包一个独立二进制 **`wandb-symon`**；注意如果打包时设了 `WANDB_BUILD_SKIP_WANDB_SYMON=true`，这个包**完全没有系统指标**（自建 wheel/镜像源要留意）。
- `symon/src/main.rs`：symon 是一个 gRPC 服务，由父进程 **wandb-core** 通过 portfile/socket 拉起；日志写到 `{WANDB_CACHE_DIR}/wandb/logs/symon-debug-<ts>.log`（默认 `~/.cache/wandb/logs/`），`WANDB_DEBUG` 非空时输出 debug 级日志。
- 采集器清单（同文件顶部注释 + `monitors.rs`）：
  - `gpu_nvidia.rs`（NVML，Linux/Windows）
  - `gpu_nvidia_dcgm.rs`（DCGM profiling，**仅 Linux，且仅当 `--enable-dcgm-profiling`**）
  - `scrape.rs`（OpenMetrics / DCGM exporter HTTP 抓取）
  - `gpu_amd.rs`、`gpu_apple.rs`、`tpu_libtpu.rs`、`trainium.rs`、`host.rs`
- `wandb/env.py` 里就是那个开关：
  ```python
  ENABLE_DCGM_PROFILING = "WANDB_ENABLE_DCGM_PROFILING"
  # 注释原文：Requires the `nvidia-dcgm` service to be running on the machine.
  ```

**NVML 采集器里哪些指标什么时候有**（`gpu_nvidia.rs` 的 `GpuMetricAvailability::default()`）：

| 指标 | 默认 | 条件 |
|---|---|---|
| `gpu`, `memory`, `memoryAllocated(Bytes)`, `temp`, `powerWatts/powerPercent`, `enforcedPowerLimitWatts`, `energyJoules`, `smClock`, `memoryClock`, `fanSpeed`, `corrected/uncorrectedMemoryErrors`, `name/uuid/brand/cudaCores/architecture/pciBusId/numaNode` | ✅ 开 | 只要 NVML 能初始化 |
| `pcieTxBytes`, `pcieRxBytes` | ✅ 开，但**仅当该卡不支持 GPM** | 走 `nvmlDeviceGetPcieThroughput()`，很多消费卡/vGPU/MIG 上会直接报错 |
| `nvlinkTxBytes`, `nvlinkRxBytes` | ❌ | **只有 GPM（Hopper+）或 DCGM 才可能给** |
| `smActive`, `smOccupancy`, `dramActive`, `pipe*Active` | ❌ | 同上（GPM 或 DCGM） |
| `pcieLinkGen`, `pcieLinkSpeed`, `pcieLinkWidth`, `maxPcieLinkGen/Width` | ❌（代码里写死 `false` + `TODO: enable these metrics`） | **当前版本就是不采**，配不配置都看不到 |
| `graphicsClock`, `encoderUtilization` | ❌（`TODO: questionable utility`） | 同上 |

另外两个容易踩的点：
- **可用性是"一次失败就永久关闭"**：某指标第一次读失败就把 `availability.x = false`，该进程后续整段 run 都不再上报。所以偶发的驱动/权限抖动会导致"同一个 run 后半段没数据"。
- GPM 需要配对采样，两次间隔 <100 ms 就跳过（`GPM_MIN_SAMPLE_INTERVAL`），所以采样频率极高时反而不出值。
</details>

---

## 排查顺序（先定位是"哪一类缺失"）

```bash
# 1) 确认 symon 真的在跑 + 看它自己怎么说（最关键的一步）
ls -lt ~/.cache/wandb/logs/ | head        # 或 $WANDB_CACHE_DIR/wandb/logs
tail -n 200 ~/.cache/wandb/logs/symon-debug-*.log
# 想看详细日志：
WANDB_DEBUG=true python train.py
```

日志里对照这几句就行：

| 日志内容 | 含义 |
|---|---|
| `No NVIDIA driver found; NVIDIA GPU monitoring disabled` | NVML 初始化失败 → 后面全没有 |
| `Failed to initialize NVIDIA GPU monitoring: ...` | 驱动/NVML 版本问题 |
| `Failed to initialize NVIDIA GPU DCGM monitoring: ...` | DCGM 路径失败（未开开关、没有 `libdcgm.so.4`、连不上 `localhost:5555`） |
| `DCGM worker: No supported fields found after filtering` | 卡/驱动/MIG 不支持 PROF 字段 |
| `DCGM worker: Failed to connect to DCGM` | DCGM host engine 没起来 |

```bash
# 2) 机器本身能不能给出这些数据
nvidia-smi -q -d PCIE | head -40        # PCIe 链路 + 吞吐
nvidia-smi nvlink -s                    # 有没有 NVLink / 速率（无 NVLink 的卡会报不支持）
nvidia-smi -q | grep -i -m5 mig         # 是否 MIG 模式
python - <<'PY'
import pynvml; pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print("pcie:", pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES))
try: print("gpm_support:", pynvml.nvmlDeviceGetGpmSample)
except Exception as e: print("no GPM api:", e)
PY

# 3) 容器里（很常见的"部分机器"原因）
ls -l /dev/nvidia*; echo $NVIDIA_VISIBLE_DEVICES; ldconfig -p | grep -i nvidia-ml
```

---

## 修复方案

### 方案 A：要 NVLink/完整 profiling → 打开 DCGM（Linux，推荐）

```bash
# 1) 装 DCGM（Debian/Ubuntu 示例；RHEL 系用 dnf/yum 同名包）
sudo apt-get update && sudo apt-get install -y datacenter-gpu-manager nvtop   # 包名也可能是 nvidia-dcgm
sudo systemctl enable --now nvidia-dcgm        # 或 nv-hostengine
ss -lntp | grep 5555                            # 确认 host engine 在 localhost:5555（代码里写死这个地址）
ldconfig -p | grep libdcgm                      # 需要能被 dlopen 到 libdcgm.so.4

# 2) 打开开关（只影响采集端，无需改代码）
export WANDB_ENABLE_DCGM_PROFILING=true
```

注意点（都是"部分机器"典型差异来源）：
- **只有 Linux + NVIDIA** 走这条路，Windows/macOS 没有。
- **MIG 实例不支持 PROF 类指标**：开了 DCGM 也拿不到 pcie/nvlink。
- **profiling 计数器需要权限**：容器里通常要 `--privileged`（或至少 `--cap-add SYS_ADMIN` + 挂载 `/dev/nvidia*`）；非 root 用户还需要 `NVreg_RestrictProfilingToAdminUsers=0`（数据中心驱动/部分消费卡默认限制）。
- DCGM 的 `nvlinkTxBytes/nvlinkRxBytes/pcieTxBytes/pcieRxBytes` 与 GPM 同名同单位，UI 侧无差别。

### 方案 B：Hopper+ 机器（H100/H200/B200…）什么都不用开

`gpm_support()` 为 true 时会自动用 GPM 提供 `pcieTxBytes/pcieRxBytes/nvlinkTxBytes/nvlinkRxBytes` 和 `smActive/dramActive/pipe*Active`。如果这类机器**也**看不到，基本只剩三种可能：驱动太旧不支持 GPM API、容器里 NVML 不可见、或者该指标首次采样失败被永久关闭（重启进程/run 再看）。

### 方案 C：不装 DCGM，用 DCGM exporter + OpenMetrics 抓取

新 symon 里有独立的 `scrape.rs` 采集器（"OpenMetrics and **DCGM exporter** endpoints over HTTP"），老 SDK 时代对应的内部设置是 `_stats_open_metrics_endpoints` / `_stats_open_metrics_filters`：

```python
import wandb
wandb.init(settings=wandb.Settings(
    _stats_open_metrics_endpoints={"dcgm": "http://127.0.0.1:9400/metrics"},
    _stats_open_metrics_filters={"dcgm": ["DCGM_FI_PROF_PCIE_", "DCGM_FI_PROF_NVLINK_"]},
))
```

这条路径的好处是：**所有架构**（含 A100/V100）都能拿到 `DCGM_FI_PROF_PCIE_TX_BYTES`、`DCGM_FI_PROF_PCIE_RX_BYTES`、`DCGM_FI_PROF_NVLINK_TX_BYTES`、`DCGM_FI_PROF_NVLINK_RX_BYTES`，配一次就行。

<details>
<summary><b>方案 D：兜底——自己在训练脚本里采（最可控，不依赖 wandb 版本差异）</b></summary>

```python
import pynvml, time, wandb
wandb.init(project="x"); pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
wandb.define_metric("gpu/nvlinkTxBytes", step_metric="trainer/global_step")
for step in range(...):
    ...
    wandb.log({
        "gpu/pcieTxBytes": pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES),
        "gpu/nvlinkTxBytes": pynvml.nvmlDeviceGetNvLinkUtilizationCounter(...),  # 注意该 API 在新架构/新驱动上已废弃
    }, step=step)
```

更稳的写法是用 `dcgm` 的 Python binding 或直接读 DCGM exporter 的 HTTP 文本，然后 `wandb.log` —— 这样指标一定出现在你自己的命名空间里，不受 `system.gpu.*` 采集策略影响。
</details>

---

## 还有两条容易忽略的"整机没数据"原因

1. **装的 wandb 根本没有 symon**：从源码/自制 wheel 安装且设了 `WANDB_BUILD_SKIP_WANDB_SYMON=true` → 连基础的 GPU 利用率都没有，不只是高级指标。先 `pip show -f wandb | grep -i symon` 确认二进制在不在，或换成官方 wheel。
2. **版本混用**：Python SDK ≤0.19 那条老路径（`wandb/sdk/internal/system/assets/gpu.py`）**压根只采 util/memory/temp/power**，没有任何 PCIe/NVLink 字段。所以"部分机器"如果是 wandb 版本不同，先统一版本再看。

---

## 建议的下一步

如果你把这几项贴给我，我可以直接判断那批机器卡在哪一环：
- `wandb --version`（以及是否官方 wheel）
- GPU 型号 / 是否 MIG / 是否容器（挂载方式）
- `~/.cache/wandb/logs/symon-debug-*.log` 里带 `DCGM`、`NVIDIA`、`nvml` 的行
- `nvidia-smi -q -d PCIE` 与 `nvidia-smi nvlink -s` 的片段

**优先级建议**：H100/H200 机器先查 GPM/驱动和容器 NVML（方案 B + 日志）；A100/其它架构要做 NVLink/PCIe 分析，直接上 **方案 A（`WANDB_ENABLE_DCGM_PROFILING=true` + `nvidia-dcgm`）** 或 **方案 C（DCGM exporter + OpenMetrics）**；如果只是自己要画图，方案 D 最省事。

---
## 🧑‍💻 User

排查新版 W&B GPU 高级指标缺失。W&B 版本：0.30.x 或更高；GPU：其他 NVIDIA GPU；现象：基础 GPU 指标正常，但 PCIe/NVLink TX/RX 缺失。请给出针对性排查步骤。

---
## 😀 默认助手 (7)

结论先行：**你这台机器基础指标正常，说明 NVML 初始化成功，问题被锁定在三条支路之一**——① 这张卡物理/架构上不可能通过 NVML 给出 NVLink；② 卡的 `nvmlDeviceGetPcieThroughput` 不支持或"首次失败被永久关掉"；③ DCGM 这条（NVLink 的唯一来源）没开或开了但失败。下面按"先分类再修"的顺序来。

---

## 第一步：3 个 1 分钟判断（决定走哪条支路）

```bash
# ① 这张卡物理上有没有 NVLink？（没有的话 nvlinkTx/Rx 永远不会有，属于预期行为）
nvidia-smi nvlink -s || nvidia-smi nvlink -c        # 报 not supported / No Link 就是没有
nvidia-smi topo -m | head                            # 看 NV# 拓扑列

# ② 本机 NVML 是否导出 GPM（Hopper+ 才需要）
nm -D $(ldconfig -p | awk '/libnvidia-ml\.so/{print $NF; exit}') | grep -i gpm
nvidia-smi --query-gpu=name,driver_version --format=csv

# ③ run 里到底有没有这些 key（避免"其实有、只是面板没画/名字不同"）
```

```python
import wandb
df = wandb.Api().run("entity/project/run_id").history(samples=2000)
keys = sorted(k for k in df.columns if any(s in k.lower() for s in ("pcie", "nvlink", "smactive", "dramactive")))
print("\n".join(keys) or "一个都没有")
print(df.filter(like="pcie").describe())   # 有 key 但全 NaN 也是"没采到"
```

> 注意：symon 里 GPM/DCGM 侧的值是走 `Sample.averages`（聚合通道）的，不同版本落库后的 key 形态（后缀/聚合标记）可能不同，**所以先按关键字全量 dump，别只盯 UI 面板**。0.30 的 symon 里 `pcieLinkGen/pcieLinkSpeed/pcieLinkWidth/maxPcieLink*` 是代码里写死 `false` 的（`TODO: enable these metrics`）——如果你指的是这几个，**当前版本就是故意不采**，任何机器都看不到。

判断结果 → 支路：

| 判断结果 | 缺的指标 | 唯一可行来源 |
|---|---|---|
| 卡无 NVLink（T4/GeForce/L40S…） | nvlinkTx/Rx | 无（预期缺失） |
| 有 NVLink，非 Hopper（A100/V100…） | nvlinkTx/Rx + profiling | **只能 DCGM**（NVML 路径完全没有 NVLink 代码） |
| Hopper+（H100/H200）但 GPM 符号缺失/驱动太旧 | nvlink + pcie + pipe*/dramActive | 升驱动，或退回 DCGM |
| pcieTx/Rx 缺但卡无 GPM | pcieTx/Rx | NVML `pcie_throughput`（需查为何失败） |

---

## 第二步：看 symon 自己的日志（信息量最大）

```bash
WANDB_DEBUG=true python train.py &          # 非空即开 debug 级
ls -lt "${WANDB_CACHE_DIR:-$HOME/.cache}/wandb/logs/" | head
grep -Ei "nvml|dcgm|nvidia|gpm|pcie|nvlink" "${WANDB_CACHE_DIR:-$HOME/.cache}/wandb/logs/"symon-debug-*.log | tail -60
```

对应关系：

| 日志 | 结论 |
|---|---|
| `No NVIDIA driver found; NVIDIA GPU monitoring disabled` | 整体不可用（你这里不适用） |
| `Failed to initialize NVIDIA GPU DCGM monitoring: ...` | DCGM 路径没起来（未开开关 / 无 `libdcgm.so.4` / 连不上 `localhost:5555`） |
| `DCGM worker: No supported fields found after filtering` | **卡/驱动/MIG 不支持 PROF 字段 → DCGM 线程直接退出**，这是非 Hopper 卡最常见的死法 |
| `DCGM worker: Failed to set watches / Failed to create field group` | 权限或 host engine 问题 |
| 无任何 DCGM 日志 | 开关没打开（`WANDB_ENABLE_DCGM_PROFILING` 没生效，见第四步） |

---

## 第三步：NVML 层（只影响 `pcieTxBytes/pcieRxBytes`）

两个容易踩的坑，都值得单独验一遍：

```bash
python - <<'PY'
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
try:
    tx = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
    rx = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES)
    print("pcie throughput OK:", tx, rx)          # 单位 KB/s
except Exception as e:
    print("pcie throughput 不支持:", type(e).__name__, e)   # 用 0/1 当参数再试一次
print("mig mode:", pynvml.nvmlDeviceGetMigMode(h) if hasattr(pynvml, "nvmlDeviceGetMigMode") else "n/a")
PY
```

1. **"一次失败即永久关闭"**：symon 的可用性标志是粘性的（`availability.pcie_throughput = false` 之后本进程内不再重试）。如果驱动在 run 启动瞬间抖动、或容器设备热插拔过，就会出现"基础指标正常、PCIe 全程空白"。**验证方式：重启进程重跑一个短 run**，若前 1–2 个采样点有值后又消失，就是这类问题。
2. **`tx` 与 `rx` 必须同时成功**：代码是 `(device.pcie_throughput(Send), device.pcie_throughput(Receive))` 配对判断，任一失败就把整组关掉。容器只挂了部分设备、或 vGPU/MIG 场景经常踩这个。
3. **GPM 与 NVML 吞吐是互斥的**：`if availability.pcie_throughput && !availability.gpm`。如果这台卡 `gpm_support()` 为 true，PCIe 吞吐就改由 GPM 提供；此时若 GPM 采样失败（`availability.gpm = false`），你**连 pcieTx/Rx 一起丢掉**，而基础指标仍然正常——这正好匹配你的现象。
4. GPM 需要配对采样且**两次间隔 ≥ 100 ms**（`GPM_MIN_SAMPLE_INTERVAL`），采样过密会整轮跳过。

---

## 第四步：要 NVLink → 上 DCGM，四个必查项

```bash
export WANDB_ENABLE_DCGM_PROFILING=true        # 官方开关（wandb/env.py 里的 ENABLE_DCGM_PROFILING）
sudo apt-get install -y datacenter-gpu-manager # 或 nvidia-dcgm
sudo systemctl enable --now nvidia-dcgm        # 或 nv-hostengine
```

这四件事**缺一不可**，按顺序查：

1. **库能被加载**：代码硬编码 `libdcgm.so.4`，用 `ldconfig -p | grep libdcgm` 确认；不在默认路径就 `export LD_LIBRARY_PATH=/usr/local/dcgm/lib:$LD_LIBRARY_PATH`。
2. **host engine 在 `localhost:5555`**（地址写死，改不了）：`ss -lntp | grep 5555`。
3. **profiling 权限**：容器内通常要 `--privileged`（或 `--cap-add SYS_ADMIN` + 完整挂载 `/dev/nvidia*`）；非 root 用户还需要 `NVreg_RestrictProfilingToAdminUsers=0`。
4. **字段真的被支持**：`dcgmi discovery -l` / 看日志里 `Found N unique supported profiling field IDs` 是否为 0。**MIG 实例不支持 PROF 字段**；消费级 GeForce 也基本不支持。另外代码只用 **GPU 0** 去查支持集（`dcgmiProfGetSupportedMetricGroups(gpu_id=0)`）再全局过滤，**混插异构卡时可能把某些卡的指标整体过滤掉**。

行为细节（避免误判"没生效"）：DCGM watch 周期 5 s、`max_keep_samples=2`，且首次可能返回 `DCGM_ST_NO_DATA`——**run 开始后几十秒内空白是正常的**；百分比类字段会被 ×100，字节类字段原样（bytes/s），与 GPM 同名同单位，UI 侧无差别。

---

## 第五步：环境限制确认（"部分机器"的高频真凶）

```bash
nvidia-smi -q | grep -i -m3 mig            # MIG → PROF/NVLink 计数基本没戏
ls -l /dev/nvidia*; echo "$NVIDIA_VISIBLE_DEVICES"   # 容器只映射部分卡/设备节点
cat /proc/driver/nvidia/version            # 驱动版本（GPM 需要较新的数据中心驱动）
```

Windows 上 NVML 路径可用，但 **DCGM 分支只编译在 Linux**；macOS 同理无关。

---

## 快速对照：现象 → 处置

| 现象 | 最可能原因 | 处置 |
|---|---|---|
| 无 NVLink 的卡缺 `nvlinkTx/Rx` | 预期行为 | 不用修，改用带宽/占用替代指标 |
| A100/V100 有 NVLink 但缺 `nvlinkTx/Rx` | 非 Hopper 无 GPM，NVML 路径根本不产 NVLink | 开 DCGM（第四步）或 scrape DCGM exporter |
| Hopper+ 全缺（含 `smActive/dramActive`） | GPM 不可用（驱动旧/NVML 无 GPM 符号） | 升数据中心驱动；或退回 DCGM |
| 只有 `pcieTx/Rx` 缺 | NVML 吞吐不支持 / 首次失败粘住 / 被 GPM 分支接管后失败 | 第三步验证 + 重启 run 复测 |
| 什么都缺、连日志都没有 | wheel 里没有 symon 二进制 | `pip show -f wandb | grep -i symon`，换官方 wheel |
| 期望 `pcieLinkGen/Width` | 0.30 里写死不采 | 自己采或用 DCGM exporter 的链路字段 |

**兜底（不依赖上述任何分支）**：跑一个 DCGM exporter 再用 scrape 通道接进来，或直接在脚本里 `wandb.log`——两条路都能稳定拿到 `DCGM_FI_PROF_PCIE_TX_BYTES` / `DCGM_FI_PROF_NVLINK_TX_BYTES`。

```python
import wandb
wandb.init(settings=wandb.Settings(
    _stats_open_metrics_endpoints={"dcgm": "http://127.0.0.1:9400/metrics"},
    _stats_open_metrics_filters={"dcgm": ["DCGM_FI_PROF_PCIE_", "DCGM_FI_PROF_NVLINK_"]},
))
```

---

## 想让我继续精确定位，请贴这几样

1. `wandb --version` + `pip show -f wandb | grep -i symon`
2. `nvidia-smi -q` 的 GPU 型号段 + `nvidia-smi nvlink -s` + 是否 MIG + 是否容器及挂载方式
3. 第一步 ③ 的 key dump 结果
4. `symon-debug-*.log` 里含 `nvml|dcgm|gpm` 的行

**按你这个现象（基础 OK、PCIe/NVLink 缺、非 Hopper）我的下注顺序是**：先查日志里 DCGM 是否根本没启用或 `No supported fields` → 再验 `nvmlDeviceGetPcieThroughput` 与"粘性关闭"（重启 run 复测）→ 最后确认卡是否本来就无 NVLink。

---
## 🧑‍💻 User

给我一个命令执行列表，告诉我如何安装dcgm并让wandb使用它读取gpu信息。
系统是ubuntu22.04

---
## 😀 默认助手 (7)

下面是一份可直接照抄的执行清单（Ubuntu 22.04）。核心只有 4 件事：**装 DCGM 4.x → 起 host engine 并确保 `localhost:5555` 可连 → 确认 `libdcgm.so.4` 能被加载 → 给训练进程加 `WANDB_ENABLE_DCGM_PROFILING=true`**。

> 关键前提：wandb 0.30 的 symon 里是**硬编码** `libdcgm.so.4` + `localhost:5555`。所以**必须装 DCGM 4.x**（soname 跟大版本走，装成 3.x 就只有 `libdcgm.so.3`，会加载失败）。

---

## 0. 前置检查（30 秒，决定后面会不会白干）

```bash
# GPU 型号 / 驱动
nvidia-smi --query-gpu=index,name,driver_version --format=csv

# 是否 MIG（MIG 实例不支持 PROF 类指标，必须用整卡）
nvidia-smi -q | grep -i -A3 "MIG Mode"

# 是否容器里跑训练（容器场景见文末「容器注意」）
grep -q docker /proc/1/cgroup && echo "in container" || echo "on host"
```

**DCGM profiling 只支持数据中心卡**（V100/A100/H100…）；GeForce/L40S 这类消费卡装了也拿不到字段。如果 `nvidia-smi nvlink -s` 报不支持，`nvlinkTx/Rx` 就不用指望了。

---

## 1. 安装 DCGM（4.x）

```bash
# 1.1 加 NVIDIA CUDA 源（DCGM 包在同一个 repo 里）
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)     # ubuntu22.04
curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${distribution}/x86_64/cuda-keyring_1.1-1_all.deb" \
  -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update

# 1.2 先看清包名（不同小版本命名会变）
apt-cache search 'datacenter-gpu-manager' | sort
apt-cache policy datacenter-gpu-manager

# 1.3 安装 4.x 变体（三选一，按上一步的输出挑）
sudo apt-get install -y datacenter-gpu-manager            # 元包，通常带 4.x
# 或（DCGM 4.x 的拆分包，按你的 CUDA 版本选一个）
sudo apt-get install -y datacenter-gpu-manager-4-core datacenter-gpu-manager-4-cuda12
# （CUDA 11 环境用 -4-cuda11）
```

---

## 2. 启动 host engine，确认 5555 在监听

```bash
sudo systemctl enable --now nvidia-dcgm      # 老版本服务名可能是 nvidia-dcgm.service
systemctl status nvidia-dcgm --no-pager

# DCGM 自身是否正常（能列出卡就 OK）
dcgmi discovery -l

# ★ 最关键：wandb 连的是 TCP localhost:5555
ss -lntp | grep 5555
```

如果 `ss` 里**没有 5555**（新版 DCGM 可能只监听 unix socket），显式用 TCP 起一份：

```bash
sudo nv-hostengine -b 127.0.0.1        # 绑定 TCP，端口 5555；参数名以 nv-hostengine --help 为准
ss -lntp | grep 5555                    # 必须能看到监听
# 可选：用 TCP 再验一次 DCGM 通路（flag 名各版本可能是 --host / -r，先看 dcgmi --help）
dcgmi --host 127.0.0.1:5555 discovery -l
```

---

## 3. 确认 `libdcgm.so.4` 存在且可被加载

```bash
# 3.1 找到并确认版本号
find / -name "libdcgm.so*" 2>/dev/null
ls -l /usr/lib/x86_64-linux-gnu/libdcgm.so*

# 3.2 必须能看到 libdcgm.so.4 ；如果只有 .so.3，说明装的是 DCGM 3.x → 回到第 1 步装 4.x
ldconfig -p | grep libdcgm

# 3.3 确认关键符号存在（symon 用 FFI 调这些）
LIB=$(ldconfig -p | awk '/libdcgm\.so\.4/{print $NF; exit}')
nm -D "$LIB" | grep -E "dcgmInit|dcgmConnect|dcgmProfGetSupportedMetricGroups|dcgmGetLatestValues_v2"
```

如果 `libdcgm.so.4` 不在 `ldconfig` 里，再补一句（不推荐，优先修正安装）：

```bash
echo "/usr/local/dcgm/lib" | sudo tee /etc/ld.so.conf.d/dcgm.conf && sudo ldconfig
```

---

## 4. （可选但推荐）放开非 root 的 profiling 权限

```bash
# 数据中心驱动默认 RestrictProfilingToAdminUsers=1，非 root 采集 profiling 会被拒
cat /proc/driver/nvidia/version
sudo tee /etc/modprobe.d/nvidia-profiler.conf <<'EOF'
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF

# 需要重载驱动才能生效（最稳是重启；或确认无进程占用后 rmmod/modprobe）
sudo update-initramfs -u
sudo reboot          # 或者：sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia
```

---

## 5. 让 wandb 使用 DCGM

```bash
# 只影响采集端，必须让「训练进程」的环境里有这个变量
export WANDB_ENABLE_DCGM_PROFILING=true
env | grep WANDB_ENABLE_DCGM_PROFILING     # 确认生效

# 重跑一个短 run 验证（先别用长任务）
python train.py
```

Jupyter / 调度器场景，在 `wandb.init()` 之前设也可以：

```python
import os
os.environ["WANDB_ENABLE_DCGM_PROFILING"] = "true"
import wandb
wandb.init(project="dcgm-test")
```

---

## 6. 验证真的生效了（两层）

```bash
# 6.1 采集端日志（成功长这样）
export WANDB_DEBUG=true
grep -Ei "dcgm" "${WANDB_CACHE_DIR:-$HOME/.cache}/wandb/logs/"symon-debug-*.log | tail -30
```

期望看到：

```
Successfully initialized NVIDIA GPU DCGM monitoring client.
DCGM worker: Library initialized.
Found N unique supported profiling field IDs.          # N 必须 > 0
DCGM worker: Initial setup successful.
```

失败信号与含义：

| 日志 | 含义 | 回到哪一步 |
|---|---|---|
| `Failed to load DCGM library` | 找不到/加载不了 `libdcgm.so.4` | 第 3 步 |
| `Failed to connect to DCGM` | `localhost:5555` 连不上 | 第 2 步 |
| `No supported fields found after filtering` | 卡不支持 PROF / MIG 模式 | 第 0 步（换整卡、换数据中心卡） |
| 完全没有 `dcgm` 字样 | 开关没传到进程 | 第 5 步（`env` 确认） |

```python
# 6.2 数据端：确认 key 真的落库了
import wandb
df = wandb.Api().run("entity/project/run_id").history(samples=2000)
print([k for k in df.columns if any(s in k.lower() for s in ("pcie", "nvlink", "smactive", "dramactive"))])
print(df.filter(like="nvlink").tail())
```

预期会多出 `pcieTxBytes/pcieRxBytes/nvlinkTxBytes/nvlinkRxBytes`、`smActive/smOccupancy/dramActive/pipe*Active`（DCGM watch 周期 5 s，**run 刚开始几十秒空白是正常的**；百分比类字段 ×100，字节类为 bytes/s，与 GPM 同名同单位）。

---

## 7. 容器场景（如果你是容器里跑训练）

```bash
# symon 与 DCGM host engine 必须在同一网络命名空间且能访问同一批 GPU
docker run --gpus all \
  --network host \                  # ← 否则容器里的 localhost:5555 不是宿主机
  --cap-add SYS_ADMIN \             # 或 --privileged（profiling 计数的常见要求）
  -e WANDB_ENABLE_DCGM_PROFILING=true \
  -v /dev/nvidia0:/dev/nvidia0 ... \
  your-train-image
```

- 容器内**必须也装 `datacenter-gpu-manager-4-core`**（symon 要 dlopen `libdcgm.so.4`）；
- 或者改用官方 DCGM 容器镜像并在容器内起 host engine；
- 两种做法都可以，重点是「`libdcgm.so.4` 在容器里能加载」+「`localhost:5555` 指向的 DCGM 和卡在同一台机器」。

<details>
<summary><b>兜底方案：DCGM exporter + wandb 的 OpenMetrics 抓取（不依赖 <code>libdcgm.so.4</code>/5555）</b></summary>

适合容器里不方便装 DCGM、或想自己控制字段列表的场景。exporter 会把 `DCGM_FI_PROF_PCIE_TX_BYTES` / `DCGM_FI_PROF_NVLINK_TX_BYTES` 等以 Prometheus 格式暴露在 `:9400/metrics`：

```bash
# 用 NGC 镜像（tag 形如 <dcgm>-<exporter>-ubuntu22.04，从 NGC 选一个可用的）
docker run -d --gpus all --rm -p 9400:9400 \
  --cap-add SYS_ADMIN \
  nvcr.io/nvidia/k8s/dcgm-exporter:<tag>-ubuntu22.04

curl -s localhost:9400/metrics | grep -E "PCIE|NVLINK" | head
```

```python
import wandb
wandb.init(settings=wandb.Settings(
    _stats_open_metrics_endpoints={"dcgm": "http://127.0.0.1:9400/metrics"},
    _stats_open_metrics_filters={"dcgm": ["DCGM_FI_PROF_PCIE_", "DCGM_FI_PROF_NVLINK_"]},
))
```
（这两个内部设置名以你所用版本的 `wandb.Settings` 字段为准。）
</details>

<details>
<summary><b>只有 DCGM 3.x 时的应急做法（官方不支持，仅作实验）</b></summary>

```bash
sudo ln -sf /usr/lib/x86_64-linux-gnu/libdcgm.so.3 /usr/lib/x86_64-linux-gnu/libdcgm.so.4
sudo ldconfig
```
可用性不保证：symon 用的 `dcgmProfGetMetricGroups_v3` / `dcgmGetLatestValues_v2` 在 3.x 里也在，但结构体布局与字段支持度可能不同，随时可能崩或返回空。**优先装 4.x**。
</details>

---

## 8. 快速自检清单（跑完逐条打勾）

1. `nvidia-smi -q | grep -A3 "MIG Mode"` → 未启用 MIG，且是数据中心卡
2. `systemctl is-active nvidia-dcgm` → `active`
3. `ss -lntp | grep 5555` → 有监听
4. `ldconfig -p | grep libdcgm.so.4` → 有
5. `nm -D <libdcgm.so.4> | grep dcgmProfGetSupportedMetricGroups` → 有
6. `env | grep WANDB_ENABLE_DCGM_PROFILING` → `true`
7. symon 日志里 `Found N unique supported profiling field IDs`，N > 0
8. `run.history()` 里能搜到 `pcieTxBytes` / `nvlinkTxBytes`

第 1–5 条是环境问题，第 6 条是配置问题，第 7 条是卡/权限问题，第 8 条是数据落库问题——**任何一条卡住，先修那一条再往下走**，不要跳步。

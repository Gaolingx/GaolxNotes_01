# Docker中运行NsightCompute障碍

### 🧑‍💻 User

docker 中运行nsight compute会遇到哪些障碍？

---

### 🤖 Assistant

在 Docker 中运行 NVIDIA Nsight Compute 时，主要障碍来自 GPU 硬件访问、内核性能计数器权限、容器隔离机制以及软件栈兼容性。以下分类列出关键挑战：

## 1. GPU 设备与驱动可用性
- **缺少 GPU 运行环境**：必须使用 `nvidia-docker` 或 `nvidia-container-toolkit`，否则容器内看不到任何 GPU 设备（`/dev/nvidia*` 不存在）。
- **驱动版本不匹配**：挂载进容器的驱动库（`libcuda.so` 等）来自宿主机，可能与容器内 Nsight Compute 期望的驱动 API 版本不一致，导致 `ncu` 启动失败或分析异常。
- **多 GPU 部分可见**：若 Docker 启动时仅映射部分 GPU（例如通过 `NVIDIA_VISIBLE_DEVICES` 环境变量），Nsight Compute 可能无法感知全部 GPU，或者目标应用被指定在不可见的 GPU 上运行。

## 2. 性能计数器权限（最隐蔽的障碍）
- **PMU（Performance Monitoring Unit）访问受限**：Nsight Compute 依赖 GPU 硬件性能计数器，这需要较高特权。默认容器环境通常没有权限读取 PMU 事件，会导致 `ncu` 报错如“ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters”。
- **Capability 不足**：需要显式添加内核能力，如 `CAP_PERFMON`（Linux 5.8+）或 `CAP_SYS_ADMIN`，否则无法打开性能计数器文件描述符。
- **Seccomp 安全配置拦截**：默认的 seccomp profile 可能阻止 `perf_event_open()` 等系统调用，需要添加 `--security-opt seccomp=unconfined` 或自定义 seccomp 配置文件来放行。
- **内核模块限制**：某些高级分析功能可能触发内核模块（如 `nvidia-uvm`）的特殊操作，特权不足将导致功能不可用。

## 3. `/proc` 与 `/sys` 文件系统隔离
- **驱动状态信息缺失**：Nsight Compute 可能需要读取 `/proc/driver/nvidia` 下的 GPU 信息、版本、参数等，而容器默认挂载的 `/proc` 可能被过滤或未包含这些路径。
- **GPU 拓扑与 NVLink 信息**：`/sys/class/nvidia` 等路径提供 GPU 拓扑、电源状态、频率等信息，如果这些未挂载或权限受限，会影响分析准确性或导致警告。
- **解决思路**：通常需要挂载宿主的相关文件系统，或直接使用 `--pid=host --ipc=host` 并配合特定卷挂载（如 `-v /proc/driver/nvidia:/proc/driver/nvidia:ro`）。

## 4. 库与软件栈兼容性
- **容器内缺少 CUDA Toolkit 部分组件**：虽然 Nsight Compute 只需要驱动库即可捕获内核指标，但其命令行接口 `ncu` 在加载结果、生成报告时可能依赖额外的 CUDA 库或工具（如 `nvtx` 库），若仅使用 `nvidia/cuda:devel` 镜像可能满足条件，但极简镜像可能缺失。
- **Nsight Compute 版本与驱动冲突**：每个版本的 Nsight Compute 都规定了最低驱动版本要求（例如 ≥ R525）。容器内安装的 `nsight-systems` 或 `nsight-compute` 若高于宿主机驱动支持的版本，将无法正常工作。
- **目标 CUDA 应用兼容性**：被测应用可能链接容器内的 CUDA 运行时，而 Nsight Compute 在执行动态注入时需要与这些库正确交互，库版本混乱会导致注入失败。

## 5. 容器命名空间与进程管理
- **PID 命名空间导致进程不可见**：如果 `ncu` 需要在同一个容器外在宿主机上监控其他容器内的进程，则 PID 隔离会使其无法找到目标进程。常见场景是在一个服务容器内运行应用，在另一容器（或宿主机）中用 `ncu` 附加分析，此时需要 `--pid=host` 或共享 PID 命名空间。
- **IPC 与共享内存**：Nsight Compute 与被测应用之间可能通过共享内存交互（例如 CUDA 事件同步），若容器未使用 `--ipc=host`，可能因 `shm` 空间不足或隔离导致通信异常（尤其是与图形或大量数据相关的回放）。
- **用户命名空间与文件权限**：容器内运行的用户 UID 可能与宿主机不同，Nsight Compute 生成的报告文件（默认在容器内）可能无法写回挂载的宿主目录，或宿主上后续无法访问。

## 6. GUI 与图形界面障碍（如果使用 UI 版本）
- **X11 显示转发**：Nsight Compute 的图形界面需要 X Server 支持，容器内需配置 `DISPLAY` 环境变量、挂载 `/tmp/.X11-unix` 且宿主允许连接（`xhost +`）。Wayland 环境下会更复杂。
- **OpenGL/Vulkan 依赖**：GUI 中的某些可视化可能依赖动态图形加速，这又需要额外映射图形设备（如 `/dev/dri`）和权限，容易与纯计算 GPU 容器环境冲突。
- **替代方案**：多数情况下推荐使用命令行界面（`ncu`）或生成报告后在外部查看，避开图形问题。

## 7. 安全策略与 Host 环境干扰
- **SELinux/AppArmor**：宿主机的强制访问控制可能阻止容器访问 GPU 设备或注入系统调用，需要在启动容器时添加 `--security-opt label=disable` 或调整策略，这在企业环境中常常受限。
- **虚拟化环境中的嵌套问题**：若 Docker 运行在虚拟机（如云实例、WSL2）或没有完整 GPU 直通的环境中，硬件性能计数器可能根本不可用，即使所有配置正确也无法执行 Profile。

## 8. 其他潜在问题
- **GPU 竞争与互斥**：Nsight Compute 在分析时需要独占某些 GPU 资源（如 profiling 多进程时），如果容器内还有其他 GPU 常驻任务或 MPS 服务，可能导致分析失败或数据不准。
- **容器销毁后清理**：Profiling 生成的临时文件、未正常关闭的会话可能留在宿主机 `/tmp` 下，需要额外处理。
- **自动化与 CI/CD 集成**：在非交互式流水线中，接受 EULA、处理交互式提示、缓存权限提升等都需要脚本额外处理，增加了实现复杂度。

---

<details>
<summary>🔍 如何系统性克服这些障碍？</summary>

通常可以通过以下 “标准做法” 解决大多数问题：

1. **运行时**：使用 `nvidia-docker2` 或 `--runtime=nvidia` 并确保安装 `nvidia-container-toolkit`。
2. **必需参数**：在 `docker run` 中加入：
   ```bash
   --cap-add CAP_PERFMON          # 或 CAP_SYS_ADMIN
   --security-opt seccomp=unconfined
   --pid=host --ipc=host          # 如果需要跨容器监控或 IPC 共享
   -v /proc/driver/nvidia:/proc/driver/nvidia:ro
   ```
3. **权限简便方案**：在受控环境中可临时使用 `--privileged`，但不推荐生产。
4. **版本对齐**：容器镜像使用与宿主机驱动兼容的官方 CUDA 镜像，并在其中安装匹配的 `nsight-compute`。
5. **命令行优先**：用 `ncu` CLI 代替 GUI，减少图形依赖。
6. **验证环境**：先在容器内运行 `ncu --list-devices` 检查能否正确枚禁 GPU 和权限。
7. **查阅日志**：Nsight Compute 的详细日志（`--log-file`）是定位障碍的最直接手段。

</details>

---

### 🧑‍💻 User

为什么性能计数器的access需要提权？但是看gpu利用率/频率/功耗等不需要？

---

### 🤖 Assistant

核心原因在于**数据的粒度与安全风险不同**。`利用率`、`频率`、`功耗`都是驱动层或硬件内部已经汇总过的高级统计值，而性能计数器能直接读取微架构的内部事件（每条指令、缓存、线程束调度），这些细节能被用于侧信道攻击、窃取其他进程的计算模式或逆向 GPU 内核代码，因此操作系统与 GPU 驱动必须对其进行提权保护。

可以从四个层面理解：

---

### 1. 数据的来源与暴露程度
- **利用率/频率/功耗**：
  这些是 GPU 驱动或固件**定期计算或直接读取的硬件状态寄存器**（如时钟分频器、能量采样器）。驱动将它们抽象成简单的数值，通过 `sysfs` 或 NVML 接口提供给用户空间，**不区分进程、不暴露微架构活动**。例如，`nvidia-smi` 看到的利用率只是过去一段时间内 SM 忙碌时间的百分比，根本看不出是哪条指令、哪个线程束造成。
- **性能计数器（PMU）**：
  它们直接读出**硬件事件计数器**的值，例如“已发射的 FP32 指令数”“L1 缓存命中次数”“全局内存读写事务数量”等。这些事件可以精确关联到具体 kernel、具体 warp，甚至组合后能反向推导出计算逻辑和数据布局。如果任意程序都能读取这些计数器，就相当于给攻击者提供了**显微镜级别的硬件观测能力**。

### 2. 安全风险：侧信道与信息泄漏
- **旁路攻击**：通过监控另一个进程执行时的性能计数器事件（如缓存失效、分支预测命中率），可以窃取密钥、模型参数等敏感信息。经典的 CPU 边信道攻击（Spectre, Meltdown）就是利用了相似能力，GPU 上也有研究展示类似威胁。
- **知识产权泄漏**：竞争对手可以通过性能计数器逆向你的 GPU kernel 计算图、矩阵大小甚至卷积核数值。
- **内核态活动泄露**：GPU 驱动内部操作（如显存分配/释放、队列提交）如果在计数器中有迹可循，非特权访问就可能暴露操作系统或驱动的行为模式。

因此，**所有现代硬件 (CPU/GPU) 都严格限制性能计数器的读取权限**，这是安全共识。

### 3. 操作系统与驱动的权限控制
- **Linux 内核**：
  GPU 通过 `perf_event_open()` 系统调用访问硬件计数器。该调用的访问权限由 `/proc/sys/kernel/perf_event_paranoid` 控制，默认值意味着非 root 用户只能测量自身进程，且需要 `CAP_PERFMON` 或 `CAP_SYS_ADMIN` 能力。容器内若没有这些 capabilities，该系统调用就会被拒绝。
- **NVIDIA 驱动**：
  即使 Linux 层面放行，NVIDIA 驱动也会额外检查访问权限。传统上，只有 root 或具备 `CAP_SYS_ADMIN` 的用户才能读取 GPU 性能计数器（部分原因是为了保护多用户 GPU 集群间的隐私）。新版本驱动可以通过内核模块选项 `NVreg_RestrictProfilingToAdminUsers=0` 来关闭此限制，但默认是开启的。

### 4. “非特权”监控接口的刻意设计
NVIDIA 提供了不要求特权的监控 API：
- **NVML**（`nvidia-smi` 底层库）：只能查询利用率、温度、功耗、时钟频率等聚合信息。
- **PerfKit (PAPI/CUPTI)**：需要用户自行提升权限才能访问 PMU 事件的接口，而非基本查询。
- **GPU System Processor (GSP) 路径**：某些新架构将部分监控功能卸载到安全固件，汇总后将结果存入特权隔离的缓冲区，普通用户只能读汇总结果。

这些接口**刻意隐去了微架构细节**，既满足日常运维需求，又不引入安全弱点。

---

<details>
<summary>💡 技术小结：一键理解</summary>

- 利用率/频率/功耗 = **宏观天气报告**（对外公开，无隐私风险）。
- 性能计数器 = **每个家庭的用电明细和电器使用记录**（高度隐私，必须授权才能查看）。
- 容器中若要使用 Nsight Compute 分析，必须拥有读取明细的权限，因此需要 `--cap-add CAP_PERFMON` 等。
- 而运行 `nvidia-smi` 或查看监控看板完全不需要这些权限，因为只消费公开的“天气报告”。

</details>

---

### 🧑‍💻 User

如何在云服务器的docker环境下运行ncu调试算子性能？

---

### 🤖 Assistant

在云服务器的 Docker 容器中使用 Nsight Compute 分析 GPU kernel 性能，核心挑战是**突破虚拟机 + 容器的双重隔离，获得硬件性能计数器访问权限**。下面的步骤将覆盖从环境准备到命令行分析的完整流程。

---

## 1. 云实例选择与前提条件

并非所有云 GPU 实例都支持 `ncu` 所需的 PMU 访问，必须满足：

- **GPU 直通（Passthrough）**：至少是 GPU 全虚拟化（如 NVIDIA vGPU 的“裸金属”模式），而非分片虚拟化（如部分 vGPU 时间片、MIG 实例通常不支持 profiling）。
- **支持的驱动版本**：宿主机驱动 ≥ 与 Nsight Compute 最低要求匹配（例如 Nsight Compute 2023.x 需要 R525 以上）。
- **允许特权操作**：必须能够以 root 或具备 `CAP_PERFMON` 的用户运行容器；某些云商（如 GCP）可能默认禁用 `perf_event` 相关内核功能，需在控制台或镜像中开启。

✅ 推荐实例类型：
- **AWS**：`p3`、`p4d`、`g5`、`g6`（均为 GPU 直通）
- **GCP**：N1 系列 + GPU 附加（直通），需确认 `perf_event_paranoid` 可调
- **Azure**：`NCv3`、`NDv2`、`NVadsA10` 等 vGPU 类型（需是**全 GPU 分配**，非分片）
- **阿里云**：`ecs.gn6v`、`ecs.gn7i`（GPU 直通）

🚫 不适用场景：
- 任何细粒度 vGPU（如 MIG 配置、部分 vGPU 分片）、无法获得 root 或特权容器的环境、共享主机且限制 `CAP_SYS_ADMIN` 的 Kubernetes 平台。

---

## 2. 宿主机准备（如果你有控制权）

如果你能直接操作云服务器宿主机（例如 EC2 实例），需进行以下检查：

### 2.1 确认驱动与 profiler 权限
```bash
nvidia-smi             # 查看驱动版本
cat /proc/driver/nvidia/version
```

### 2.2 调节内核参数（允许非特权 perf 访问）
```bash
# 临时降低（重启失效），推荐设为 1（允许用户模式访问，不需要 root）
sudo sysctl kernel.perf_event_paranoid=1
# 永久写入
echo "kernel.perf_event_paranoid=1" | sudo tee -a /etc/sysctl.conf
```

### 2.3 允许所有用户访问性能计数器（可选，NVIDIA 驱动限制）
在 `/etc/modprobe.d/nvidia.conf` 中添加：
```
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```
然后重新加载驱动或重启。**生产环境谨慎使用，仅限私有分析环境。**

---

## 3. 安装 Docker 与 NVIDIA Container Toolkit

```bash
# 安装 Docker（以 Ubuntu 为例）
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 安装 nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

验证 GPU 在容器中可见：
```bash
docker run --rm --runtime=nvidia nvidia/cuda:12.3.0-devel-ubuntu20.04 nvidia-smi
```

---

## 4. 构建/拉取包含 Nsight Compute 的镜像

官方镜像不预装 `nsight-compute`，需要自己构建一个带分析工具的镜像。

### Dockerfile 示例（基于 CUDA devel）
```dockerfile
FROM nvidia/cuda:12.3.0-devel-ubuntu20.04

# 设置环境变量避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 安装必要工具和下载 Nsight Compute
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 下载并安装 Nsight Compute CLI 版（截止 2023 年，以下为示例 URL）
ARG NC_VERSION=2023.2.1.0
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/nsight-compute-${NC_VERSION}_${NC_VERSION}-1_amd64.deb \
    && dpkg -i nsight-compute-*.deb \
    && apt-get install -f -y \
    && rm nsight-compute-*.deb

# 可选：安装自己依赖的算子库
# RUN pip install torch ...
```

构建并标记：
```bash
docker build -t my-ncu-image .
```

---

## 5. 启动容器并挂载必要权限

关键参数说明：

| 参数 | 作用 |
|------|------|
| `--runtime=nvidia` 或 `--gpus all` | 暴露 GPU |
| `--cap-add CAP_PERFMON` | 允许 `perf_event_open()` 系统调用 |
| `--security-opt seccomp=unconfined` | 避免 seccomp 拦截 perf 相关调用 |
| `--pid=host` | 可选，如果需要在容器内监测宿主机上其他进程 |
| `-v /proc/driver/nvidia:/proc/driver/nvidia:ro` | 提供 GPU 驱动信息 |
| `-v /path/on/host:/workspace` | 存放源码、输出的共享目录 |

**最精简的启动命令**（仅分析容器内的应用）：
```bash
docker run -it --rm \
  --gpus all \
  --cap-add CAP_PERFMON \
  --security-opt seccomp=unconfined \
  -v /proc/driver/nvidia:/proc/driver/nvidia:ro \
  -v $(pwd):/workspace \
  my-ncu-image /bin/bash
```

<details>
<summary>⚠️ 如果上述参数仍报权限错误</summary>

可尝试：
- 添加 `--cap-add CAP_SYS_ADMIN`（更宽松）
- 或直接使用 `--privileged`（仅限信任的测试环境）
- 确认宿主机 `/proc/sys/kernel/perf_event_paranoid` ≤ 1
- 确认驱动模块参数 `NVreg_RestrictProfilingToAdminUsers=0` 已设置（需 root 查看 `/proc/driver/nvidia/params`）

</details>

---

## 6. 在容器内使用 `ncu` 分析算子

进入容器后，编译或准备你的 CUDA 应用，然后运行分析。

### 6.1 基本性能分析（生成 CSV/报告）
```bash
# 假设你的可执行程序是 ./my_cuda_app
ncu --set full --csv -o profile ./my_cuda_app
```
这将生成 `profile.ncu-rep` 文件和 CSV 输出。

### 6.2 只看特定的 kernel 或 metric
```bash
ncu --kernel-name my_kernel --metrics gpu__time_duration,dram__bytes_read.sum ./my_cuda_app
```

### 6.3 交互模式（在没有 GUI 的容器内也可以使用）
```bash
ncu --target-processes all --mode=launch ./my_cuda_app
```
运行后会在命令行中提供交互选项。

---

## 7. 处理云环境的常见问题

### 问题1：`ERR_NVGPUCTRPERM` 或 `The user does not have permission to access NVIDIA GPU Performance Counters`
- **原因**：perf 权限不足或驱动限制。
- **解决**：按第 2 节调整宿主机 `perf_event_paranoid` 和驱动参数，并在容器启动时加上 `--cap-add CAP_PERFMON`。如果云商不允许修改宿主机，可能需要联系支持或换用裸金属实例。

### 问题2：`ncu` 提示 “Unable to profile all requested metrics”
- **原因**：某些硬件计数器被独占或驱动版本不匹配。
- **解决**：确保没有其他 profiling 工具或 GPU 任务同时运行；尝试只收集基础指标。

### 问题3：虚拟机重启后参数失效
- **解决**：将 `perf_event_paranoid` 和 NVIDIA 驱动选项写入持久化配置（`/etc/sysctl.conf` 和 `/etc/modprobe.d/nvidia.conf`）。

### 问题4：容器内无法看到完整的 GPU 拓扑（NVLink、内存带宽指标不全）
- **解决**：映射更多文件系统：
  ```bash
  -v /sys/class/nvidia:/sys/class/nvidia:ro
  --ipc=host
  ```

---

## 8. 自动化分析流程（示例脚本）

用于 CI/CD 的模板脚本 `profile_kernel.sh`：
```bash
#!/bin/bash
set -e

IMAGE=my-ncu-image
APP=./kernel_bench

docker run --rm \
  --gpus all \
  --cap-add CAP_PERFMON \
  --security-opt seccomp=unconfined \
  -v /proc/driver/nvidia:/proc/driver/nvidia:ro \
  -v $(pwd):/workspace \
  -w /workspace \
  $IMAGE \
  ncu --set full --csv -o profile_output $APP

echo "Profile saved to profile_output.ncu-rep"
```

---

<details>
<summary>🔧 附录：快速自检清单</summary>

1. 宿主机 GPU 是否通过 `nvidia-smi` 正常列出？驱动版本 ≥ 525？
2. `cat /proc/sys/kernel/perf_event_paranoid` 的值是否 ≤ 1？
3. `cat /proc/driver/nvidia/params | grep RestrictProfiling` 是否返回 `0` 或未设置？
4. Docker 是否使用 `nvidia-container-toolkit` 且测试通过？
5. 容器启动参数是否包含 `--cap-add CAP_PERFMON` 和 `--security-opt seccomp=unconfined`？
6. 目标算子程序是否编译为 `-g` 包含符号信息（用于 kernel name 匹配）？

满足以上条件后，`ncu` 就能在云服务器 Docker 环境中顺利运行。

</details>
# Windows11识别A100成NPU问题

### 🧑‍💻 User

windows11 taskmgr a100被识别成npu是怎么回事

---

### 🤖 Assistant

# 结论先说

这是 **Windows 11 的"识别归类"问题**，不是你的 A100 坏了，也不是它真的变成了 NPU。A100 仍然是那块 GPU，CUDA/PyTorch 计算一切照常。

---

## 为什么会这样

**1. A100 是"无头"数据中心卡，只跑 TCC 模式**

- A100（以及 H100 这类）没有显示输出，**默认运行在 TCC 模式（Tesla Compute Cluster）**，并且**不支持 WDDM 显示驱动模型**。
- 任务管理器的"GPU"性能页依赖 WDDM 显示适配器才能显示。A100 没有这个"显示身份"，所以 Windows 不会把它当普通 GPU 列出来。
- 这也是为什么你的"GPU"页里看不到它（或看不到它的利用率）。

**2. Windows 11 24H2 起，任务管理器新增了 NPU 检测**

- 微软在任务管理器的"性能"页加入了 **NPU** 面板（基于 Windows ML / DirectML 的设备枚举，以及 WMI 的 `Win32_NPU`、MCDM 计算驱动模型）。
- 判定逻辑是："找出能执行 AI / 机器学习矩阵运算的硬件加速器"。

**3. A100 有 Tensor Core，恰好被 NPU 检测"捡"走了**

- A100 有 Tensor Core，NVIDIA 驱动会把它注册为一个**支持 DirectML/ML 计算的加速设备**。
- 于是 Windows 的 NPU 枚举逻辑扫到了它；又因为它没有 WDDM 显示身份、无法归入"GPU"分类，最后就**被塞进了"NPU"这一栏显示**。

> 一句话概括：Windows 把"能做 AI 运算的加速器"笼统归为 NPU；A100 这种纯计算卡在 TCC 模式下没有"显卡"身份，于是被 NPU 分类捡走，显示成了 NPU。

---

## 有没有影响 / 怎么验证

- **基本没有实际影响**：训练、推理、CUDA 程序都不受影响。
- 验证它是不是 NPU、以及真实状态，用这些命令：

```powershell
# 1) 看 GPU 真实状态（应显示 A100 及其显存/温度/模式）
nvidia-smi

# 2) 看 Windows 是否把它枚举成 NPU（24H2 及以上）
Get-CimInstance -ClassName Win32_NPU | Format-List Name, DeviceID, FriendlyName

# 3) 看当前显卡驱动模式（TCC 还是 WDDM）
nvidia-smi -q | findstr /i "Driver Model"
```

如果 `Win32_NPU` 里出现 NVIDIA A100 的条目，就证实了上面的解释。

---

## 怎么处理

| 情况 | 做法 |
|------|------|
| 只想确认硬件没坏 | 不用处理，`nvidia-smi` 正常即可 |
| 想正确监控 A100 利用率 | 用 `nvidia-smi`、**DCGM**、**GPU-Z / HWiNFO64** 等，别依赖任务管理器 |
| 想让任务管理器当"GPU"显示 | **做不到**——A100 不支持 WDDM，无法像 `nvidia-smi -dm 0` 那样切到显示模式（那是 A40/A6000/RTX 等支持的卡才能用的操作） |

---

### 补充：别和消费级显卡混淆

- **消费级 RTX 卡**（有显示输出、WDDM）会正常显示在"GPU"下；新版 Windows 还可能在 GPU 里额外多一个 **"神经引擎 / Neural engine"** 条目，那是它把 Tensor Core 单独列出来，属于**正常新功能**，不是误识别。
- 只有 **A100/H100 这类 TCC 无头卡**，因为缺少"GPU 身份"，才会整个被顶到"NPU"里显示。

所以你这台机器上的现象属于 Windows 的一个宽松归类（可以说是识别缺陷），**无害、无需修复**，知道原因即可。

---

### 🧑‍💻 User

windows server2022如何将a100切换成wddm mode运行directml/vulkan api？

---

### 🤖 Assistant

我来先核实一下关键事实（A100 是否支持 WDDM，以及 DirectML/Vulkan 对驱动模式的要求），再给你准确答案。

找到了关键信息。让我获取具体的操作步骤（GitHub 的 "Gaming on A100" 项目和 Tesla WDDM 注册表 hack 方法）。

查证完毕。先说结论，再给步骤。

# 结论

**A100 官方只有 TCC 模式，标准驱动下无法切 WDDM** —— `nvidia-smi -dm 0` 会直接报 `Not Supported`（A100 是无显示输出的 GA100 计算卡，NVIDIA 没给它做 WDDM 图形驱动）。

而 **DirectML 和 Vulkan 在 Windows 上都依赖 WDDM**（DirectML 基于 D3D12，Vulkan 在 Windows 上走 WDDM 图形栈），所以 TCC 模式下的 A100 用不了这两个 API。

**但是**，社区有一个针对 P40/P100/V100/A100 的"民间 hack"，通过注册表 + GRID 驱动把 Tesla 卡强开成 WDDM，可以让 DirectML/Vulkan 跑起来。下面按"官方做法 → 民间 hack"的顺序讲。

---

# 一、官方/正规路线（先说清楚）

| 方案 | 说明 |
|------|------|
| **用 CUDA 系 API**（推荐） | A100 在 TCC 下跑 PyTorch / TensorRT / ONNX Runtime CUDA EP 完全没问题，性能最好。要跑 AI 根本不必折腾 WDDM。 |
| **换支持 WDDM 的卡** | A40、A6000、L4、L40/L40S、RTX 6000 Ada、RTX 4090 等，这些卡 `nvidia-smi -dm 0` 就能切 WDDM，直接跑 DirectML/Vulkan。 |
| **走正规 GRID/vGPU** | Tesla 卡要开图形功能，官方要求装 **GRID/vGPU 驱动 + GRID License**，否则切模式会提示 `Not Supported`。 |

所以如果你**必须用这块 A100 跑 DirectML/Vulkan**，只能走下面的非官方方法。

---

# 二、民间 hack：强开 WDDM（A100/GA100 可用）

原理：GRID 驱动其实**内含** WDDM 图形栈，只是默认对 A100 这类纯计算卡关闭了，并需要 License。通过改注册表"伪造"适配器类型和授权特性，把它骗成普通显卡，再切到 WDDM。

## 步骤 1：安装 GRID 驱动

- 下载并安装 **NVIDIA GRID/vGPU driver**（Windows Server 2022 版本），不要用普通的 Data Center 驱动。
- 装完重启。

## 步骤 2：改注册表

以管理员运行 PowerShell，先定位 A100 对应的子项：

```powershell
Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' |
  ForEach-Object {
    $d = (Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue).DeviceDesc
    Write-Host $_.PSChildName '=>' $d
  }
```

找到 `NVIDIA A100 ...` 那项（例如 `0001`），对它做以下修改：

```powershell
$reg = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0001'  # 换成你的子项号

# 把卡"伪装"成支持图形/授权的显卡
New-ItemProperty -Path $reg -Name AdapterType          -PropertyType DWORD -Value 1     -Force
New-ItemProperty -Path $reg -Name FeatureScore         -PropertyType DWORD -Value 0xd1  -Force
New-ItemProperty -Path $reg -Name GridLicensedFeatures -PropertyType DWORD -Value 7     -Force   # 伪造 GRID 授权特性
New-ItemProperty -Path $reg -Name EnableMsHybrid       -PropertyType DWORD -Value 1     -Force   # 关闭虚拟显示驱动
```

> 说明：
> - 另一份 A100 教程（GitHub `Gaming-on-A100`）把 `AdapterType` 设成 **0**，并只加 `EnableMsHybrid=1`。不同驱动版本取值可能不同，**1 和 0 都试一下**，哪个能让卡出现在"显示适配器"里就用哪个。
> - 如果有核显/别的显示卡，James Brown 的方法是给 iGPU 那项加 `EnableMsHybrid=2`。

## 步骤 3：重启并切模式

```powershell
# 重启后，以管理员运行
nvidia-smi -g 0 -dm 0        # 0 = WDDM
nvidia-smi -q | findstr /i "Driver Model"   # 看是否已变 WDDM
```

如果 `nvidia-smi -dm 0` 仍报错，先在**设备管理器**里找到该卡 → 右键"禁用设备"→ 再"启用设备"，再执行上面的命令。

## 步骤 4：验证

- 任务管理器"性能"里应出现 GPU 条目（不再是 NPU）。
- `GPU-Z` 里 `Driver Model` 显示 `WDDM`。
- 跑 DirectML / Vulkan 时，确认选中的是 A100 这张卡。

---

# 三、注意事项（重要）

1. **这是非官方、绕过 GRID License 的做法**，`GridLicensedFeatures=7` 属于伪造授权特性，可能违反 NVIDIA 许可条款，生产/商用环境请谨慎。
2. **A100 没有物理显示输出**，切到 WDDM 后也是"无头显示"模式——不能接显示器，但 DirectML、Vulkan 的 **offscreen/计算渲染** 是可以用的。
3. **不保证成功**，因卡型/驱动版本而异；改之前**务必先导出备份注册表**。
4. **出问题如何恢复**：把上面加/改的注册表值删掉或改回原值，或用 **DDU** 完全卸载驱动重装即可。
5. Windows Server 2022 建议用**带桌面体验（Desktop Experience）**的版本，Core 版没有完整图形栈，会难搞很多。

---

# 四、给你的实际建议

如果最终目的是**用 A100 跑 AI**，强烈建议直接走 CUDA/TensorRT/ONNX Runtime(CUDA EP)，性能和稳定性都更好，完全没必要去碰 WDDM。

只有当你确实被某个**只能用 DirectML 或 Vulkan** 的软件卡住时，才值得按上面的注册表 hack 折腾一次，并做好"可能失败、需要回滚"的心理准备。
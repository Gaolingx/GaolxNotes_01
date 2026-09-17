# H100无WDDM

### 🧑‍💻 User

windows server 2022 nvidia h100使用nvidia-smi -dm 0会报错吗？

---

### 🤖 Assistant

# 结论：会报错 ❌

在 Windows Server 2022 上对 NVIDIA H100 执行 `nvidia-smi -dm 0` **会报错**，因为 H100 不支持 WDDM 模式。

## 先厘清 `-dm` 参数的含义

| 参数值 | 对应驱动模型 | 说明 |
|--------|-------------|------|
| `-dm 0` | **WDDM**（Windows Display Driver Model） | 标准显示驱动模型，带显示/图形栈 |
| `-dm 1` | **TCC**（Tesla Compute Cluster） | 纯计算模式，无显示引擎 |

很多人容易记反，**`0` 是 WDDM，`1` 才是 TCC**。

## 为什么 H100 会报错

H100（Hopper 架构）是一块**纯计算型数据中心 GPU**：

- 没有显示输出接口（PCIe 版也无视频口）
- GPU 内部**没有显示引擎（Display Engine）**
- 因此在 Windows 上**只支持 TCC 模式**，无法切换到 WDDM

典型报错如下：

```
Unable to set driver model for GPU 00000000:01:00.0: Not Supported
Treating as warning and moving on.
All done.
```

（具体 PCI 地址 `XX:XX.X` 会因机器而异，核心信息是 **`Not Supported`**。）

## 补充说明

1. **H100 默认就是 TCC 模式**，无需手动设置。执行 `nvidia-smi` 看输出列表里的 `Mode` 列即可确认，会显示为 `TCC`。
2. 对 H100 这类计算卡，**TCC 反而是性能最优的选择**——它绕过了 WDDM 的图形调度开销，纯 CUDA 计算更高效。
3. 如果你的场景确实需要 WDDM（例如要把它当显示适配器/远程桌面虚拟 GPU），H100 **本身不具备这个能力**，需要换用支持 WDDM 的卡（如 L4 / L40 / L40S 等带显示引擎的数据中心 GPU）。

如果你是想把卡切换/确认到计算模式，正确命令是：

```powershell
nvidia-smi -dm 1      # 设为 TCC 模式（H100 上默认就是 TCC）
nvidia-smi             # 查看当前 Mode 列
```

需要的话，我也可以帮你确认驱动版本或进一步排查 `nvidia-smi` 的其他报错信息。
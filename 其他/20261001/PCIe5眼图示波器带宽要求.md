# PCIe5眼图示波器带宽要求

## 🧑‍💻 User

pcie5眼图最低要多少带宽示波器

---
## 😀 默认助手 (7)

Search engines were mostly blocked, but DuckDuckGo (lite) worked — pulling the primary sources now.

## 结论（先给数字）

| 测试场景（PCIe 5.0 / 32 GT/s NRZ） | 示波器**最低带宽** | 采样率 | 说明 |
|---|---|---|---|
| **BASE（芯片/硅级）Tx**：在发送引脚附近测 | **≥ 50 GHz** | ≥ 128 GSa/s | 最严格场景，眼图/抖动测试的"标配" |
| **CEM（系统/插卡）Tx**：在 worst-case 通道末端测 | **≥ 33 GHz** | ≥ 128 GSa/s（CEM 允许配合 2× sinx/x 插值放宽） | 通道损耗已把高频内容衰减，规范允许降低带宽 |
| **32 GT/s RX 校准**（用示波器采集 Tx 波形解校准，注入 DUT 接收机） | **≥ 50 GHz** | **≥ 128 GSa/s** | Allion 实测要求：带宽 50 GHz 起、采样率 ≥128 GSa/s |

一句话：**做 PCIe 5.0 眼图/BASE Tx 合规测试，50 GHz 是底线；只有在 CEM 插卡/系统级、测量点在通道末端时，33 GHz 才被允许。**

## 数字是怎么来的

- 32 GT/s NRZ，每周期传 2 bit → 基频 $f_0 = 32/2 = 16\ \text{GHz}$，UI $= 1/32\ \text{GT/s} = 31.25\ \text{ps}$
- 3 次谐波 $3f_0 = 48\ \text{GHz}$；再往上的能量已很少 → **50 GHz 实时示波器**刚好覆盖到 3 次谐波（Tektronix 明确按此推导）
- CEM 测量点在 worst-case 通道末端，高频被衰减 → 带宽要求降到 **33 GHz**
- 采样率：SigTest 后处理需要**每 UI 至少 4 个点**，即 $4 \times 32\ \text{GS/s} = 128\ \text{GSa/s}$

<details>
<summary>实操注意点（容易被忽略，但决定你测出来准不准）</summary>

1. **要看"系统带宽"而不是示波器标称带宽**：探头 + 夹具(interposer) + 电缆 + 巴伦组成的链路带宽往往低于示波器本身。50 GHz 示波器配 33 GHz 探头，实际就不是 50 GHz 系统。
2. **去嵌 / 嵌入是必须的**：BASE 测量要求尽可能在器件引脚处，需要 S 参数去嵌 breakout 通道；CEM 则需要把夹具/通道损耗用 embedding 补回来，再交给 SigTest 做时钟恢复 + 参考接收机均衡（CTLE/DFE）后看眼图。
3. **带宽不足的后果**：高频成分被低通滤掉，测出的眼高、抖动会失真（且方向不固定），合规余量没有意义——所以不要用 $0.7\times$ 数据率（≈22 GHz）这类"经验法则"来做合规测试。
4. **常见达标机型**（举例，非唯一）：Tektronix DPO70000SX（50/59/70 GHz）、Keysight UXR / Z 系列 + D9050PCIC 软件、Teledyne LeCroy WaveMaster 8000HD。33 GHz 级示波器只建议用于 CEM 场景。
5. **顺带一提 Gen6**：64 GT/s 是 PAM4，符号率仍是 32 GBd，但电平台阶更多、对噪声/线性度更敏感，通常仍从 50 GHz 起步（很多实验室直接上 59/70 GHz），不能沿用 Gen5 的 33 GHz 结论。
</details>

<details>
<summary>依据来源</summary>

- Tektronix《PCIe Gen 5 Tx Tech Brief》——原文："*putting the 3rd harmonic at 48 GHz … a 50 GHz bandwidth real-time oscilloscope is needed for PCIe 5.0 Base Tx testing. For CEM Tx testing … allowing a bandwidth requirement of 33 GHz*"，并给出 4 点/UI → 128 GS/s：https://www.tek.com/en/documents/technical-brief/pcie-gen-5-tx-tech-brief
- Keysight PCIe 5.0 应用笔记 5992-3141EN——"*to test a PCIe 5.0 transmitter, you should consider a real-time oscilloscope with a minimum bandwidth of 50 GHz*"
- Allion Labs《PCIe 5.0 – Physical Layer Test Challenge》——32 GT/s RX 校准要求带宽 ≥50 GHz、采样率 ≥128 GSa/s：https://www.allion.com/server-series_physical-test-challenge_04/
- Teledyne LeCroy QPHY2-PCIE5-TX-RX 数据手册（WaveMaster 8000HD 覆盖 PCIe 5.0 Tx/Rx 校准需求）
</details>

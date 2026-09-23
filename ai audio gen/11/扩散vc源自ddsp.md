# 扩散vc源自ddsp

## 🧑‍💻 User

为什么diffusion-svc是ddsp-svc衍生出来的？

---
## 🤖 Assistant

## 核心答案

因为 **Diffusion-SVC 本来就不是一个"独立重新发明"的项目，而是把 DDSP-SVC 里那个扩散模块单独拆出来做成的仓库。**

Diffusion-SVC 的官方 README 说得非常直白（原文）：

> 此仓库是 [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) 仓库的扩散部分的单独存放。可单独训练和推理。

所以严格来讲它不是"衍生"关系那么简单，而是**同源代码的拆分/独立化**——DDSP-SVC 里既要训 DDSP 声学模型、又要训扩散模型，而扩散那部分代码本身是自包含的，于是就被独立成了 Diffusion-SVC。

---

## 代码谱系（谁从谁演化）

这条链其实是有明确先后顺序的：

| 项目 | 作者 | 角色 |
|------|------|------|
| **Diff-SVC** (`prophesier/diff-svc`) | prophesier | 早期"纯扩散"SVC，扩散思想来源 |
| **DDSP-SVC** (`yxlllc/DDSP-SVC`) | yxlllc | 轻量 DDSP 声学模型；**3.0 版加入"浅扩散"模型**，README 里自称 "DDSP + Diff-SVC refactor version" |
| **Diffusion-SVC** (`CNChTu/Diffusion-SVC`) | CNChTu | 把 DDSP-SVC 的**扩散部分单独拆出来**，可独立训练/推理 |

关键的一环在 DDSP-SVC 3.0：它把 Diff-SVC 的扩散模型**重构（refactor）**进自己的框架里，用来给 DDSP 的粗糙输出做"浅扩散"精修。等这套扩散代码在 DDSP-SVC 里成熟、且能独立跑之后，CNChTu 就把它抽成了一个独立仓库，也就是 Diffusion-SVC。

DDSP-SVC 的 README 致谢列表里同时列了 `Diff-SVC` 和 `Diffusion-SVC`，也印证了这个"互相引用"的血缘关系。

---

## 为什么在工程上"拆得出来"？

这是理解"为什么是衍生"的技术关键——**扩散模块和 DDSP 模块是解耦的**：

1. **共享同一套前端管线**
两者都用相同的预训练组件：特征编码器（ContentVec / HubertSoft）、基频提取器（RMVPE）、声码器（NSF-HiFiGAN）。数据预处理、采样率、hop size 这些完全一致，所以扩散部分挪出去后能直接复用整套脚手架（`preprocess.py` / `train*.py`）。

2. **浅扩散（shallow diffusion）本身就是级联结构**
DDSP-SVC 的思路是：先用一个轻量声学模型（DDSP，或后来的 "naive 模型"）生成一个粗糙结果，再让扩散模型**只从中间时间步开始做少量去噪**去精修。既然扩散器只需要"一段输入 mel + 少量步数"，它天然就能挂到**任意前端模型**后面，不绑定 DDSP。

3. **面向实时/低显存优化**
Diffusion-SVC 相比 Diff-SVC 显存占用小得多、训练和推理更快，并针对浅扩散和实时用途做了专门优化，还能在较强的 GPU 上实时推理。

---

## 拆出来之后继承了什么、改了什么

<details>
<summary>点击展开：Diffusion-SVC 相对于 DDSP-SVC 的取舍</summary>

**继承自 DDSP-SVC：**
- 数据处理 / 预处理脚本、训练循环、推理脚手架
- 特征编码器、RMVPE 基频、NSF-HiFiGAN 声码器配置

**相对 DDSP-SVC 的变化：**
- 去掉 DDSP 声学模型，专注纯扩散解码器
- 引入 **naive 模型**（小网络声学模型）作为浅扩散起点，让弱 GPU 也能实时出好音质
- 强调"可级联"：既能独立训练完整扩散，也能只训练浅扩散所需的降噪步数，提高训练速度与质量
</details>

---

## 一句话总结

**Diffusion-SVC 是 DDSP-SVC 的"扩散部分"被单独抽出来独立化**——扩散代码在 DDSP-SVC 3.0 就已经是重构自 Diff-SVC、并作为浅扩散精修器存在；因为它是自包含、可独立训练的，CNChTu 就把它拆成了独立仓库，继承了 DDSP-SVC 整套前端管线（ContentVec + RMVPE + NSF-HiFiGAN），只保留并强化扩散器本身。

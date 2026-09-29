# DCO信息作用

## 🧑‍💻 User

github DCO信息有什么用？

---
## 🤖 Assistant

# GitHub DCO 信息的作用

**DCO（Developer Certificate of Origin，开发者原创声明）** 是开源项目用来确认代码来源合法性的一种**轻量级机制**。它和传统的 CLA（Contributor License Agreement，贡献者许可协议）目标类似，但形式简单得多。

## 一、DCO 信息具体指什么

DCO 信息通常体现在提交（commit）的 **`Signed-off-by` 签名行**上，例如：

```
feat: add new feature

Signed-off-by: Zhang San <zhangsan@example.com>
```

开发者通过 `git commit -s` 命令自动生成这一行，声明：

> 我确认这份贡献是我原创的，或我有权按项目许可证提交它。

## 二、DCO 信息的主要用途

| 用途 | 说明 |
|------|------|
| **法律溯源（Provenance）** | 证明每段代码的来源，明确谁对贡献负责，降低版权纠纷风险 |
| **合规审查** | 企业/项目可追溯贡献者身份，满足法务与开源合规要求 |
| **替代 CLA 的轻量方案** | 开发者无需签署正式合同，只需在 commit 中签名，门槛更低 |
| **CI 自动化校验** | GitHub 的 **DCO App** 或机器人会自动检查每个 commit 是否含 `Signed-off-by`，缺失则 PR 无法合并 |
| **许可证一致性** | 确保提交的代码能合法地以项目的开源许可证发布 |

## 三、DCO 与 CLA 的对比

| 维度 | DCO | CLA |
|------|-----|-----|
| 形式 | commit 里一行签名 | 正式签署的法律协议 |
| 门槛 | 低，一次配置即可 | 较高，需签约 |
| 是否转让版权 | 否，保留版权 | 常涉及版权许可/转让 |
| 代表项目 | Linux Kernel、GitLab | Apache、Google 部分项目 |

## 四、实际工作流中的体现

1. 贡献者提交 PR，每个 commit 需带 `Signed-off-by`。
2. **DCO 检查机器人**（如 GitHub DCO App、`probot/dco`）自动扫描。
3. 若缺失，机器人会在 PR 中标记失败，并提示用 `git rebase --signoff` 修复。
4. 全部通过后才允许合并。

## 小结

DCO 信息的核心价值是：**用最小的流程成本，为开源贡献提供法律层面的"原创性与授权"保证**，并通过自动化工具在 PR 环节强制落实，从而保护项目、贡献者和使用者的合法权益。

---

如果你是想在自己的项目里**启用 DCO 检查**，或想**批量给已有 commit 补签**，我可以进一步给出具体配置和命令。需要的话告诉我。

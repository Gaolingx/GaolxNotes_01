# ICLR投稿流程

## 🧑‍💻 User

paper如何投稿iclr

---
## 🤖 Assistant

I'll look up the latest official ICLR submission guidelines to give you accurate information.

Let me also fetch the ICLR 2026 Author Guide for the detailed submission process.

# ICLR 投稿完整指南

以下基于 **ICLR 2026** 官方 *Call for Papers* 与 *Author Guide* 整理（以距今最近的会议为准，各年流程基本一致）。ICLR 全程通过 **OpenReview** 平台完成投稿与公开评审。

---

## 一、投稿核心流程（5 步）

1. **注册 OpenReview 账号** — 所有作者都必须有账号（用真实且最新的邮箱/机构信息）。
2. **提交 Abstract（摘要）** — 在摘要截止日期前录入标题与摘要，用于评审员"投标/认领"论文。
3. **上传完整论文 PDF + 补充材料** — 在论文截止日期前完成。
4. **等待评审 & 参与公开讨论（Rebuttal）** — 与审稿人在 OpenReview 上公开互动、修订论文。
5. **等待最终决定** — 被接收则提交 camera-ready 版本。

> ⚠️ 两个截止时间**均为硬性**，逾期无法补救，官方明确表示不会为任何理由单独调整。

---

## 二、关键日期（ICLR 2026）

所有时间均为 **UTC-12（AoE，"地球上任意地点"）**。

| 环节 | 日期 |
|------|------|
| 摘要提交截止 | 2025-09-19 |
| 论文提交截止 | 2025-09-24 |
| 投稿系统开放 | 2025-09-13 |
| 评审放出 | 2025-11-11 |
| 作者-审稿人公开讨论 | 2025-11-11 → 12-03 |
| 作者最后回复日 | 2025-12-03 |
| 最终决定通知 | 2026-01-22 |
| Camera-ready 上传 | 2026 年 2 月中 |

---

## 三、论文格式与篇幅要求

<details>
<summary><b>点此展开细节</b></summary>

- **篇幅**：投稿时正文 **≤ 9 页**；讨论/rebuttal 阶段与 camera-ready 放宽至 **10 页**（严格执法，超页直接桌面拒稿 desk reject）。
- **参考文献**：不计入页数，页数不限。
- **附录**：参考文献之后可不限页数，但审稿人**没有义务阅读**附录。
- **LaTeX 模板**：官方模板 👉
`https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip`
- **补充材料（Supplementary）**：正文 + 附录建议合并成单个 PDF；代码可打包为 `.zip` 作为补充材料上传，鼓励开源以提升可复现性。
- **伦理声明（Ethics Statement，推荐）**：正文末尾、参考文献之前，≤1 页，**不计入页数**。
- **可复现性声明（Reproducibility Statement，推荐）**：正文末尾，**不计入页数**。
- **致谢（Acknowledgement）**：不计入页数。

</details>

---

## 四、必须遵守的政策（违反即可能拒稿）

<details>
<summary><b>点此展开各项政策</b></summary>

### 1. 双盲评审（Double-blind）
- 论文正文和补充材料中**不得暴露作者身份**，否则直接 desk reject。
- 作者自己的 arXiv 论文：可以引用，但必须用**第三人称**叙述。
- arXiv 上有几乎相同的预印本**不违反匿名政策**，只要不明确指向自己即可。

### 2. 一稿多投政策（Dual Submission）
- 禁止与已发表、已接收、或同时投给其他会议/期刊的论文**相同或高度相似**。
- 出现在 **arXiv** 等非同行评审平台，或曾投给**不出版 proceedings 的 workshop**，不违反政策。
- 评审期间投递 arXiv **是允许的**。

### 3. 互惠审稿要求（Reciprocal Reviewing）
- 提交 **≥3 篇**论文的作者，需作为审稿人**审阅至少 6 篇**论文；未按时完成可能被 desk reject。
- **每个投稿至少有 1 位作者**注册担任审稿人（审至少 3 篇）。
- 资质标准：曾在 ICLR/NeurIPS/ICML 或同等期刊有接收论文；若团队无人符合，则**豁免**（鼓励新研究者投稿）。
- 若无人注册为审稿人，论文将被 desk reject（PC 可个案豁免）。

### 4. 大语言模型（LLM）使用
- 允许把 LLM 当作通用辅助工具。
- **新规**：若 LLM 在研究构思/写作中作用显著到可视为"贡献者"，须在**附录单列 LLM 使用说明**（不计页数）。
- 未披露重要 LLM 使用 → 可能 desk reject。**LLM 不能作为作者**。

### 5. 撤稿政策
- 论文截止前撤稿：从 OpenReview 删除。
- 论文截止后撤稿：仍保留在公开的 "withdrawn papers" 区，**立即去匿名化且无法删除**（与 arXiv 类似，视为存档）。

### 6. 行为准则与伦理准则
- 所有参与者须遵守 ICLR **Code of Conduct** 与 **Code of Ethics**，投稿时需明确确认。

</details>

---

## 五、常见问题（FAQ 精选）

<details>
<summary><b>点此展开常见疑问解答</b></summary>

**Q：没有 OpenReview 账号能投稿吗？**
不能。所有作者都必须有 OpenReview 账号，请在截止前确认。

**Q：摘要可以先用占位符，之后再补全吗？**
不可以。**占位符/重复摘要会被删除**。但标题和摘要可在论文截止前修改，只要内容与原摘要保持接近。

**Q：摘要截止后还能增删作者吗？**
**不能增删作者**（无任何例外）。作者**顺序**可在摘要截止后、论文截止前调整。

**Q：忘记上传、传错版本，能通融吗？**
不能。截止后**任何编辑都不允许**，官方明确不处理此类邮件。

**Q：补充材料什么时候截止？**
与正文**同一截止时间**。

**Q：附录要单独 PDF 吗？**
两者皆可：可接在正文参考文献后，也可作为单独的补充材料文件。

**Q：如何匿名提供代码？**
① 匿名打包为 `.zip` 作补充材料；② 建匿名仓库并把链接放入论文；③ 开放讨论后以评论形式私下发给审稿人和 AC。

**Q：必须在会议注册吗？**
被接收的论文，**至少一位作者需注册**（in-person）。多篇论文只需一次注册。被接收论文需录制 5 分钟视频。

</details>

---

## 六、Camera-ready（接收后）

- 篇幅与投稿版一致（**10 页**）。
- 伦理声明、致谢等不计页数。
- 可更改**作者顺序**，但**不能增删作者**。
- 官网会发邮件（2026 年 2 月中）说明上传流程。

---

## 七、给首次投稿者的建议

1. **尽早提交摘要**，别拖到最后一刻（系统当天会很卡）。
2. **严格按模板排版**，超页 = 直接拒稿，无悬念。
3. **匿名性检查**：正文、附录、代码注释、仓库链接都要脱敏。
4. **主动参与公开讨论**：ICLR 是公开评审，认真 rebuttal 对结果影响很大。
5. **提前找好合格审稿人**，否则可能因互惠审稿要求被拒。

---

### 官方链接汇总

| 资源 | 链接 |
|------|------|
| ICLR 2026 Call for Papers | `https://iclr.cc/Conferences/2026/CallForPapers` |
| ICLR 2026 Author Guide | `https://iclr.cc/Conferences/2026/AuthorGuide` |
| 投稿系统（OpenReview） | `https://openreview.net/group?id=ICLR.cc/2026/Conference` |
| LaTeX 模板 | `https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip` |
| Code of Ethics | `https://iclr.cc/public/CodeOfEthics` |
| Code of Conduct | `https://iclr.cc/public/CodeOfConduct` |

---

> 💡 **提示**：以上为 ICLR 2026 的规格。若你准备投下一年（如 ICLR 2027），请以官网最新 *Call for Papers* 与 *Author Guide* 为准，日期与政策可能微调（例如页数限制近几年有变化）。如需我帮你核对某一年最新要求，或排版/匿名性自检清单，可以告诉我。

需要我进一步展开某一部分吗？例如 **LaTeX 模板使用步骤**、**rebuttal 写作策略**，或 **匿名代码仓库的搭建方法**？

# OpenReview注册无需edu邮箱解决方案

## 🧑‍💻 User

openreview无法获取edu账号没法注册怎么办

---
## 😀 默认助手 (7)

先说结论：**OpenReview 现在早就不要求 edu 邮箱了**；你遇到的"注册不了"，绝大多数情况并非因为缺少 edu 账号，原因通常是注册后卡在人工审核（moderation），或漏了"确认邮箱"这一步。

以下依据是我刚查的官方文档（`docs.openreview.net`，`Signing up for OpenReview` / `Expediting Profile Activation`，近 1 个月更新过）。

---

## 一、先纠正前提

官方原文要点：

- 注册时填**任何邮箱**都能提交。若域名是公共邮箱（gmail / qq / 163 / outlook）或不在机构列表里，只会弹出一个 **Email domain warning 警告**。
- **这个警告不是错误，可以直接继续注册**。用公共域名注册只是会走人工审核，**最多约 2 周**。
- 用机构邮箱（学校、公司、实验室）可以**自动或快速激活**，这才是推荐做法的原因。它并非硬性门槛。

所以"没有 edu 邮箱"并不等于"不能注册"，只是"不能秒过"。

## 二、按你的情况选路径

| 你的情况 | 该怎么做 | 预期耗时 |
|---|---|---|
| 有学校或单位邮箱（`.edu.cn`、学校自定义域名、公司域名都算） | 直接用它注册，警告可忽略 | 立即或几小时 |
| 有机构邮箱但域名不被识别 | 照常注册，OpenReview 会定期把新域名加进机构列表 | 几天 |
| 只有个人邮箱（QQ、163、Gmail） | 照样注册，选 `Independent Researcher`，等审核 | ≤ 2 周 |
| 在职但没公司邮箱 | 用公司名填机构信息（推荐优先于选独立研究者） | 数天 |
| 13–18 岁 | 需家长签 `OpenReview_Parental_Consent_Form.pdf`，用拒信里的专属链接上传，**不要**邮件发给官方 | — |
| 未满 13 岁 | 不能注册 | — |

## 三、完整流程（最容易漏的是第 3 步）

1. `openreview.net/signup` → 填全名 → 勾确认框 → 填邮箱 → Sign Up → 设密码。
2. 收 `OpenReview Signup Confirmation` 邮件，点链接进入 **Profile Registration** 填资料（`Career & Education History` + 机构信息 + homepage）。
3. ⚠️ **要真正"确认"邮箱**：在 profile 里点邮箱旁的 `Confirm`，然后把收到的验证码**在同一个窗口**粘贴并点 `Verify`。成功后邮箱旁会显示 `(confirmed)`。
   - 官方明确说：**很多人卡住就是因为漏了这步**，只填了邮箱没确认。
4. 点 `Register for OpenReview` → 要么立即激活，要么收到 `OpenReview profile activation pending` 进入审核。

## 四、加速审核的 5 个动作

这些动作就是官方给出的"加急"清单：

1. **给 pending 账号加一个机构邮箱并确认**。这是官方原话里的"fastest way"，用自己学校、实验室或公司的邮箱。
2. **不要重复注册第二个账号**。官方特别警告：用机构邮箱再注册一次，会造成 duplicate profile，后续处理更麻烦。正确做法是**在已有 pending 账号上加邮箱**。
3. **完善 homepage**：必须是能同时看到你的**姓名、单位、你注册用的那个邮箱**的页面。可用 Google Scholar / GitHub / 个人主页 / 机构个人页；公司或院系首页、网名社媒账号**不合格**。
4. **`Career & Education History` 不能只写一条 `Independent Researcher`**，否则会被拒（rejection），要求你补机构记录。
5. 万一被拒或卡太久，走官网底部的 **Contact / Feedback 表单**说明情况，附学生证、录取通知、在职证明、论文首页链接等。

## 五、几个实操坑

<details>
<summary>点开看（收不到邮件、审核后被拒、投稿赶不上等）</summary>

- **收不到激活邮件**：用页面上的 `Didn't receive email confirmation?` 重发。国内 qq/163 邮箱经常被拦，建议注册时先用 Gmail/Outlook 类地址。
- **注册用个人邮箱，但想用学校邮箱收通知**：完全可以。账号创建后把 `preferred email` 设成个人邮箱即可，投稿和通知都走这个地址。所以"必须用机构邮箱"只是**激活**阶段的需求，不是**使用**阶段。
- **赶投稿 deadline 怎么办**：留 **1–2 周**缓冲。审核没有"加急通道"，deadline 前一天才注册基本没救。另注意很多会议（ICLR/NeurIPS 等）要求**所有作者**都有 profile，不只是通讯作者。
- **重新触发注册页**：如果激活链接过期，用 `Resending an activation link` 重发，不要在别处新建账号。
- **被人占用了同名或邮箱**：若提示某个链接或邮箱已被其他账号使用，走 Feedback 表单让官方合并（`Merging Profiles`）。
</details>

---

需要我帮你查某个具体会议（比如 ICLR 2026、ACL）的 profile 注册截止时间，或者帮你把加急用的 profile 文案（homepage / 机构信息怎么填才不被拒）写出来吗？

# GitHub仓库可转移所有权

### 🧑‍💻 User

github可以完全转交仓库所有者吗，例如双盲评审期间根据规定（"using anonymous accounts on data hosting platforms and anonymous code where relevant."），将repo link放到一个anonymous github account上，评审后移交给作者

---

### 🤖 Assistant

**是的，GitHub 完全支持彻底转交仓库的所有权（Transfer Ownership）。**

在你提到的双盲评审（Double-Blind Review）场景中，建立一个匿名的 GitHub 账号存放代码，并在论文被录用后将其转移给你的真实账号，这是一个非常标准且合规的操作流程。

不过，**在双盲评审中使用这种方法有一个极其致命的陷阱**，稍有不慎就会导致你失去匿名性并可能被拒稿。以下是详细的操作建议和警告。

### 🚨 双盲评审中最致命的陷阱：Git 提交历史

即使你把代码传到了一个叫做 `anonymous-author-123` 的账号下，**Git 的提交历史（Commit History）中仍然会保留你的真实姓名和邮箱。** 审稿人只需要点击仓库的 "Commits"，就能把你的真实身份看得清清楚楚。

为了确保绝对的匿名，你必须**抹除所有的历史提交记录**，只上传一份纯净的代码。

### 🛡️ 推荐的匿名开源标准工作流

如果你决定使用匿名 GitHub 账号，请严格按照以下步骤操作：

1. **清理代码内容**：
   * 删除代码文件、README 中的作者名字、所属机构。
   * 检查代码中是否有硬编码的绝对路径（例如：`C:/Users/ZhangSan/Dataset/...` 或 `/home/zhangsan/workspace/...`）。
   * 移除 License 文件（因为里面通常有版权所有者的名字）。
2. **彻底抹除 Git 历史并推送**：
   在本地将清理好的代码复制到一个全新的文件夹中，然后重新初始化 Git：
   ```bash
   # 进入全新的纯净代码文件夹
   cd pure_code_folder

   # 重新初始化 git（不要把原来的 .git 文件夹拷过来！）
   git init

   # 临时修改这个仓库的用户名和邮箱为匿名
   git config user.name "Anonymous Author"
   git config user.email "anonymous@example.com"

   # 提交代码（只有一个 "Initial commit"）
   git add .
   git commit -m "Initial commit for double-blind review"

   # 关联匿名 GitHub 账号的仓库并推送
   git branch -M main
   git remote add origin https://github.com/your-anonymous-account/your-repo.git
   git push -u origin main
   ```
3. **评审结束后转移仓库**：
   论文录用后（或者评审结束后），你可以将这个匿名仓库转移到你的主账号。转移后，你可以把原来带有完整提交历史的本地代码强制推送到这个仓库中，恢复所有的开发记录。

---

### 💡 更好的替代方案：Anonymous GitHub

由于手动清理非常容易出错，目前计算机顶会（如 CVPR, NeurIPS, ACL 等）非常推荐使用专门的匿名化代理工具，最著名的是 **Anonymous GitHub** (`https://anonymous.4open.science/`)。

**它的优势在于：**
1. 你只需要把代码正常传到你的**真实 GitHub 账号**里的一个**私有仓库（Private Repo）**。
2. 把私有仓库的链接粘贴到 Anonymous GitHub 网站上。
3. 它会生成一个临时的、匿名的链接（例如 `https://anonymous.4open.science/r/my-project-xyz`）。
4. **这个链接会自动隐藏你的 GitHub ID、隐藏所有的提交历史（Commits）、隐藏 Issues 和 PR，只展示当前的代码。**
5. 你把这个生成的链接写到论文里即可。评审结束后，你只需把你的私有仓库公开（Public），非常省事。

---

<details>
<summary><b>附：如何在 GitHub 上执行“转移仓库（Transfer Ownership）”的具体操作</b></summary>

如果你仍需要手动转移仓库，具体步骤如下：

1. 登录你的**匿名账号**，进入需要转移的仓库。
2. 点击顶部的 **Settings** 选项卡。
3. 将页面拉到最底部的 **Danger Zone**（危险区域）。
4. 找到 **Transfer ownership**，点击 `Transfer` 按钮。
5. 在弹出的对话框中，输入**目标账号**（你的真实 GitHub 用户名）以及当前的仓库名称以确认。
6. 点击确认后，GitHub 会向你的**真实账号**的邮箱发送一封确认邮件（或者在真实账号的 GitHub 页面顶部也会有横幅提示）。
7. 登录你的**真实账号**，点击接受转移（Accept transfer）。

**转移后的效果：**
* 仓库的 URL 会自动重定向（如果你通过旧的匿名链接访问，会自动跳转到你的真实账号下）。
* 所有的代码、Stars、Issues、PRs 都会原封不动地带过去。

</details>
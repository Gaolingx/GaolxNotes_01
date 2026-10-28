# git squash commit 操作方法

## 🧑‍💻 User

git 命令如何git squash commit？

---
## 😀 默认助手 (7)

# Git 压缩提交（Squash Commits）的几种方式

## 一、交互式 rebase（最通用、最灵活）

```bash
git rebase -i HEAD~4        # 整理最近 4 个提交
git rebase -i <commit-id>   # 整理该 commit 之后的所有提交（不含该 commit）
git rebase -i --root        # 从第一个 commit 开始整理
```

编辑器会打开一个待办列表（todo list）：

```
pick a1b2c3d feat: 第一个提交
pick e4f5g6h fix: 修个 bug
pick i7j8k9l feat: 第二个提交
pick m0n1o2p wip
```

把需要合并的提交行的 `pick` 改成：

| 关键字 | 简写 | 作用 |
|---|---|---|
| `pick` | `p` | 保留该提交 |
| `squash` | `s` | 合并到上一个提交，**保留并合并提交信息**（会再弹一次编辑器） |
| `fixup` | `f` | 合并到上一个提交，**丢弃该提交的提交信息** |
| `reword` | `r` | 只改提交信息 |
| `drop` | `d` | 删除该提交 |

例如把后 3 个都合并到第一个：

```
pick a1b2c3d feat: 第一个提交
squash e4f5g6h fix: 修个 bug
squash i7j8k9l feat: 第二个提交
squash m0n1o2p wip
```

保存退出 → 弹出合并后的提交信息编辑器 → 编辑好再保存退出。

> **vim 小技巧**：在 todo 文件里执行 `:2,$s/^pick/squash/` 可把从第 2 行起的所有 `pick` 一次性改成 `squash`。

**非交互式写法**（想脚本化 / 不想打开编辑器）：

```bash
# Linux
GIT_SEQUENCE_EDITOR="sed -i '2,\$s/^pick/squash/'" git rebase -i HEAD~4

# macOS 的 sed 需要额外参数
GIT_SEQUENCE_EDITOR="sed -i '' '2,\$s/^pick/squash/'" git rebase -i HEAD~4
```

<details>
<summary>二、<code>reset --soft</code>（最快，适合"把最近 N 个提交合成 1 个"）</summary>

```bash
git reset --soft HEAD~4     # 回退指针，改动全部保留在暂存区
git commit -m "feat: 合并后的完整提交"
```

- `--soft`：改动进入 **暂存区**（推荐做 squash）
- `--mixed`（默认）：改动进入工作区，需重新 `git add`
- `--hard`：**丢弃改动，慎用**

优点：不触发冲突、不打开交互编辑器。
缺点：会丢失原有的多个提交信息，需要自己重写一条。

</details>

<details>
<summary>三、<code>merge --squash</code>（把一个分支压成一个提交合入当前分支）</summary>

```bash
git checkout main
git merge --squash feature-x     # 把 feature-x 的所有改动放进暂存区，不产生 merge commit
git commit -m "feat: 整个 feature-x 的功能"
```

特点：保留原分支历史不动，只在 `main` 上留一条干净的提交。适合"功能分支内部提交很乱"的场景。

</details>

<details>
<summary>四、<code>--fixup</code> + <code>--autosquash</code>（修复型提交的推荐姿势）</summary>

```bash
# 发现某个老提交有问题，先做一个 fixup 提交
git commit --fixup=<目标commit-id>

# 之后自动归位（无需手动改 todo）
git rebase -i --autosquash HEAD~10
```

可设置一次永久生效：

```bash
git config --global rebase.autoSquash true
```

这样 `git rebase -i` 会自动把 `fixup!`/`squash!` 开头的提交排到目标提交后面并标记好。

</details>

<details>
<summary>五、合并最后一个提交：<code>commit --amend</code></summary>

```bash
git add .
git commit --amend --no-edit   # 把暂存区内容并入上一个提交，提交信息不变
```

这其实是"只 squash 两个提交"最轻量的做法。

</details>

## 六、如何确定"起点"提交

不知道 `HEAD~N` 该写多少时：

```bash
# 与主分支的分叉点
git rebase -i $(git merge-base main HEAD)

# 查看提交列表和数量
git log --oneline -n 20
```

## 七、重要注意事项

1. **rebase / reset 会重写历史**。只对**尚未推送**或**自己独占的分支**做 squash。
2. 如果已经推送过，需要强推，并且**务必用 `--force-with-lease`** 而不是 `--force`：

   ```bash
   git push --force-with-lease
   ```
   它会在远端有你不知道的新提交时拒绝推送，防止覆盖别人的工作。
3. 别人已经基于这些提交开发时，**不要**改写——改用 `merge --squash` 或让 PR 合入时 Squash（GitHub/GitLab 的 "Squash and merge" 按钮）。
4. rebase 过程中冲突：解决后 `git add <files>` 然后 `git rebase --continue`；想放弃用 `git rebase --abort`。
5. 操作前留个后路：

   ```bash
   git branch backup-before-squash   # 或者记下当前 SHA
   ```

   搞砸了可以用 `git reflog` 找回原来的 SHA。

## 快速选择

| 场景 | 推荐命令 |
|---|---|
| 精细控制哪些提交合并、怎么合并 | `git rebase -i HEAD~N` |
| 最近 N 个提交直接合成一个 | `git reset --soft HEAD~N && git commit` |
| 只合并最后两个 | `git commit --amend` |
| 整个功能分支压成一条合入主干 | `git merge --squash feature` |
| 修 bug 的零散提交想归位 | `git commit --fixup=<sha>` + `git rebase -i --autosquash` |
| GitHub 上的 PR | 页面上的 **Squash and merge** |

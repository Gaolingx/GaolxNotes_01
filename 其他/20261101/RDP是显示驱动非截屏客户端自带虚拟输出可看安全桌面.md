# RDP是显示驱动非截屏客户端自带虚拟输出可看安全桌面

## 🧑‍💻 User

众所周知，Windows UAC那个东西是安全桌面，一般的软件根本拿不到UAC的界面（黑屏），截图截不到，也无法注入任何操作，RDP开了口子的，但是别的远程桌面软件处理起来就很麻烦了
像sunshine那种都是靠dxgi抓framebuffer这种邪修办法实现的
但是你得有逻辑显示器去抓
无头不行的（没有物理显示器也得有vdd）
但是RDP可以无头
这是为什么？

---
## 😀 默认助手 (7)

## 先给结论

你把两件事混在一起了：

1. **“无头也能用”** —— 这不是 RDP 的“抓屏能力”，而是 **RDP 自己会作为显示器出现**；
2. **“能看 UAC”** —— 这不是“抓屏”，而是 **RDP 本身就是这条 session 的显示通路**，不在“客户端截屏”的范畴里。

DXGI Desktop Duplication 是**客户端**，RDP 是**显示驱动本身**。两者的权限层级、所依赖的桌面、是否必须存在 output，全都不同。

---

## 一、DXGI 要的是“逻辑显示器”，不是“显卡”

`IDXGIAdapter::EnumOutputs` 枚举的是**输出端（output/连接器）**，不是 GPU 算力。一个 output 要被 Desktop Duplication 用，需要满足：

- `AttachedToDesktop == TRUE`（`DXGI_OUTPUT_DESC` 里的字段）；
- DWM 正在向这个 output 进行合成；
- 该 output 属于**当前进程所在的那个桌面**。

无头机器上没有 EDID 输出时，Windows 会把该连接器标成“未连接、未激活”。于是：

| 现象 | 结果 |
|---|---|
| `EnumOutputs` | 返回 0 个，或返回 `AttachedToDesktop=FALSE` 的输出 |
| `DuplicateOutput` | `DXGI_ERROR_UNSUPPORTED`，或 `DXGI_ERROR_NOT_CURRENTLY_AVAILABLE`，或 `DXGI_ERROR_NOT_FOUND` |

Win10 1703+ 的 DWM 确实支持“headless 合成”（没有显示器也能合成），但 **Desktop Duplication 不支持 headless output**；它要的是“有个真实存在的输出正在被合成”。所以实际工程中就必须用 VDD（Virtual Display Driver / IddSampleDriver / usbmmidd / Parsec VDD）或 HDMI 假负载来“造一个显示器”。

**关键点：VDD 造的是一个 WDDM 显示设备，Windows 认为“有头了”。** 你需要的不是物理显示器，而是一个被 DWM 认作 output 的设备。

---

## 二、UAC 安全桌面为什么截不到

这是另一层问题，跟有没有显示器无关。UAC 提权时，session 会 `SwitchDesktop` 到 `WinSta0\Winlogon`（和登录界面是同一个 secure desktop），`consent.exe` 以 SYSTEM 跑在那上面。于是：

1. **桌面隔离**：`Default` 桌面上的进程无权读 `Winlogon` 桌面的像素。Winlogon 桌面的 DACL 基本只给 SYSTEM，`OpenDesktop` 都要 `DESKTOP_SWITCHDESKTOP` 之类的权限，普通 token 直接 `E_ACCESSDENIED`。
2. **捕获 API 绑定桌面**：Desktop Duplication 在 `DuplicateOutput` 时就把自己绑在“当时的那个桌面”上。一切 `SwitchDesktop`（UAC、锁屏、Ctrl+Alt+Del、屏保）都会让 `AcquireNextFrame` 返回 `DXGI_ERROR_ACCESS_LOST`；你重建 `DuplicateOutput` 时人在 `Default` 桌面，绑不到 `Winlogon`，于是**黑屏/冻帧**。
3. **DWM 不会把 secure desktop 的合成结果提供给你**：合成输出是交给“该 session 显示通路的消费者”的，而用户态 app 不是那个消费者。
4. **注入同样不通**：`SendInput` / `PostMessage` / `SetThreadDesktop` 全被 UIPI + 桌面隔离挡住。唯一“官方”的手段是 `SendSAS`（模拟 Ctrl+Alt+Del），它是走 RPC 到 winlogon 的，不是注入。

> 所以“DXGI 抓不到 UAC”根本不是分辨率问题，是**你不在那个桌面上，而且你拿不到那个桌面的合成输出**。

<details>
<summary>顺带：`Windows.Graphics.Capture`、GDI `BitBlt(GetDC(NULL))`、Magnification API 全都一样失败</summary>

它们底层要么是 DWM/DXGI，要么绑定当前输入桌面，secure desktop 一律拿不到。GDI 在无头 session（比如 session 0）有时能拿到一个默认大小的桌面 DC，但那是“桌面背景”，不是 UAC。

</details>

---

## 三、RDP 为什么两样都行

核心一句话：**在 RDP session 里，那个 session 的“显卡”就是 RDP 虚拟显示适配器。**

打开设备管理器，RDP 会话里 Display adapters 下会多出 **Microsoft Remote Display Adapter**（驱动 `rdpudd.sys`，Win8+ 的 Remote Desktop Virtual Display Driver；老的 XDDM 时代是 `RDPDD`）。它是一个正经的 WDDM/IDD 显示设备，**带 output、被 DWM 认作已连接显示器**。

于是：

- **无头问题不存在**：RDP 运行时自己就作为显示器出现。物理机没插显示器、没 VDD 都没关系，因为 session 的显示适配器是 RDP 的，不是物理 GPU 的。所谓“RDP 可以无头”，准确说是 **RDP 自带了一个虚拟显示器**。
- **UAC 问题不存在**：RDP **并不是**作为用户桌面里的一个“截屏客户端”去读屏，它**就是这条 session 的显示输出通道**。`dwm.exe` 每个 session 一个，`SwitchDesktop` 到 `Winlogon` 时是同一个 DWM 在往同一个 output 上合成；RDP 的显示栈直接消费这个合成结果并编码发走。所以它看到的是“这个 session 当前桌面的一切”，**包括 secure desktop**。
- 旁证：**你能 RDP 到一台停在登录界面的机器**——登录界面就是 `WinSta0\Winlogon`，同一个 secure desktop。微软明确支持在 RDP 下显示登录界面和 UAC 弹窗，这是 RDP 的基本功能，不是漏洞。

---

## 四、一张表看清本质区别

| | DXGI Desktop Duplication | RDP |
|---|---|---|
| 身份 | 用户态**客户端** | session 的**显示驱动/通路** |
| 需要 output | 需要，且需 `AttachedToDesktop` | 不需要，它**就是** output |
| 能无头 | 不行（必须 VDD/假负载） | 可以（自带虚拟显示器） |
| 看 secure desktop | 不行（桌面隔离 + `ACCESS_LOST`） | 可以（显示通路的消费者，非截屏） |
| 能否注入 UAC | 不行（UIPI/桌面隔离） | 可以（RDP 输入通道直接作用于 secure desktop） |
| 失败码 | `8887A0026` / `887A0004` / `887A0001` | — |

---

## 五、一个重要推论（很多人会搞错）

**“RDP 能看到 UAC” ≠ “在 RDP 会话里跑 DXGI 就能看到 UAC”。**

你 RDP 进一台机器，然后在那个 session 里跑一个 DXGI 抓屏程序：UAC 一弹，你的程序照样 `ACCESS_LOST`、照样黑屏。因为此时你**又变回了客户端**，而且你不在 Winlogon 桌面上。RDP 的“看到”来自 `termsrv`/显示栈那条通路，不是来自 DXGI 客户端 API。

所以 “Sunshine 那类工具看不到 UAC” 和 “RDP 能看到 UAC” 完全是两套机制，不能互相替代。

---

## 六、实践上的几条路

- **不想看 secure desktop**：组策略/注册表关掉 `PromptOnSecureDesktop`（`ConsentPromptBehaviorAdmin` 相关），UAC 就弹在普通桌面，DXGI 能抓到——但这是**降低安全等级**，企业环境不该这么干。
- **无头 + 普通桌面抓屏**：装 VDD / 假负载，让 DWM 有个 output 可复制。这是 Sunshine 那套能用的前提。
- **真要抓 UAC**：用户态没有受支持的办法。可行的是
  1. **RDP 回环/MSTSC shadow**（驱动级，天然覆盖 secure desktop），
  2. **内核驱动/镜像驱动/IDD 自己当显示消费者**，
  3. **SYSTEM 身份 + 挂到 Winlogon 桌面**：理论上能过 DACL，但 DXGI 是否肯给你那个桌面的合成输出并不保证，属于未支持、不稳定、随时可能被补丁改掉的行为。

<details>
<summary>为什么“抓到 secure desktop 的像素”在架构上就是反的</summary>

Secure desktop 存在的目的就是**让不可信的用户态代码既看不到也摸不到**。它能被跨会话、跨桌面观察到，唯一合理的理由是“观察者是这条 session 的显示传输层本身”（RDP、Citrix ICA、Hyper-V 的控制台），而不是某个 app。DXGI 被设计成“给本桌面 app 做共享/录屏”的 API，从设计上就不承担这个职责。所以在用户态绕不过去不是 bug，是设计。

</details>

---

**一句话收束**：DXGI 是“在某块已存在的屏幕上读像素”，RDP 是“这块屏幕本身就是我”——无头靠的是后者自带 output，UAC 靠的是后者处在显示通路里而非客户端位置。两个“为什么”答案不同，但根因是同一个：**你在链条里的位置不同**。

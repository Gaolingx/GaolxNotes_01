# Monterey可装Moonlight6.1.0

## 🧑‍💻 User

macos12Monterey还能安装哪个版本的moonlight

---
## 😀 默认助手 (7)

## 结论：能装的最新版本是 **Moonlight-Qt 6.1.0**

在 macOS 12 Monterey 上，官方 Moonlight（`moonlight-stream/moonlight-qt`，也就是官网下载页那个 Mac 客户端）**最高只能装到 v6.1.0**。最新的 v6.2.0 已经把最低系统要求提到了 **macOS 13 Ventura**，在 Monterey 上装好后会提示"此 App 需要 macOS 13 或更高版本"，无法启动。

| 版本 | 发布日期 | 最低 macOS | Monterey 12 可用 |
|---|---|---|---|
| **v6.2.0**（最新） | 2026-10-04 | macOS 13 Ventura | ❌ |
| **v6.1.0** | 2024-09-17 | macOS 11 Big Sur | ✅ **最新可用版本** |
| v6.0.1 | 2024-06-30 | 支持早于 Big Sur 的系统 | ✅ |

### 下载地址（Universal，Intel + Apple Silicon 通用）

```
https://github.com/moonlight-stream/moonlight-qt/releases/download/v6.1.0/Moonlight-6.1.0.dmg
```

安装：打开 dmg，把 `Moonlight` 拖进「应用程序」即可（签名+公证过的官方包）。

---

<details>
<summary><strong>为什么 6.2.0 不行？（技术依据，可展开）</strong></summary>

判断依据来自官方构建流程，而不是猜测：

1. v6.2.0 的 GitHub Actions 构建矩阵里 macOS 用的是 **Qt 6.11.2**（`build-win-mac.yml`，runner 为 `macos-26`）。
2. Qt 官方文档《Supported Platforms》中 Qt 6.11 写的是：**"Target Platform: macOS 13 or higher (including macOS 26)"** —— 即 Qt 6.11 的官方二进制最低部署目标就是 13.0，所以打包出的 app 的 `LSMinimumSystemVersion` 也是 13.0。
3. 对照验证：v6.1.0 用的是 **Qt 6.7.2**，Qt 6.7 文档写的目标平台是 **macOS 11, 12, 13, 14**，而 v6.1.0 的发布说明里也明确写了 "macOS Big Sur or later is now required"。两者完全吻合 —— 所以 6.1.0 在 Monterey 上是正常支持的（官方支持列表里就包含 macOS 12）。

</details>

<details>
<summary><strong>装 6.1.0 之后要注意的事</strong></summary>

- **不要点应用内的更新提示**。Moonlight 会检查更新并提示 "6.2.0 可用"（日志里是 `Latest version of Moonlight for this platform is: ...`），按了之后下载的是不兼容的版本。同理，用 Homebrew cask 之类的"自动装最新版"方式会拿到 6.2.0。
- **安全提醒**：6.2.0 修复了 3 个漏洞（CVE-2026-33546 / CVE-2026-33547 / CVE-2026-41210，恶意 host 可让客户端崩溃），6.1.0 不带这些修复。如果这台 Mac 有条件升级到 Ventura 或更高，建议升级系统后用 6.2.0。
- **Monterey 本身**已经停止安全更新；6.1.0 也不会再有新功能（YUV 4:4:4、AV1 硬解、libplacebo/MoltenVK 渲染器、500 Mbps 码率等都是 6.2.0 才有的）。

</details>

<details>
<summary><strong>其他可选方案（第三方原生 macOS 客户端等）</strong></summary>

如果升级系统这条路走不通（比如 MacBook Pro 2016、MacBook Air 2017、iMac 2015、Mac mini 2014、Mac Pro 2013 这些机器 Monterey 就是终点），还可以考虑：

- **第三方原生 macOS 客户端**：`MichaelMKenny/moonlight-macos`（`mroi/moonlight` 是其活跃 fork），纯 AppKit 原生应用，官网说明**支持到 macOS 10.14 Mojave**，因此 Monterey 完全没问题。代价是功能较少：不支持高于 60Hz、手柄支持受限（仅蓝牙 Xbox 等，单手柄）、没有 6.x 的 4:4:4 / AV1 / 新渲染器等。它基于 `moonlight-ios` 的较旧代码分支。
- **不要指望 iOS 版**：App Store 的 Moonlight（iOS/iPadOS 版）在 Mac 上只有 Apple Silicon 机型能以"iPhone/iPad App"形式安装，Intel Mac 装不了。
- 理论上你也可以拿 6.2.0 的源码自行编译并配一个更老的 Qt（把部署目标降到 12），但 6.2.0 依赖新 Qt/FFmpeg API，实际很难编译通过，不推荐。

</details>

**一句话**：Monterey 就停在 **Moonlight-Qt 6.1.0**，用上面那个 dmg 链接直接下，别让它自动更新到 6.2.0。

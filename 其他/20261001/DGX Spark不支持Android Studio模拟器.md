# DGX Spark不支持Android Studio模拟器

## 🧑‍💻 User

dgx spark support android studio emulator?

---
## 😀 默认助手 (7)

I'll check the current facts on this — DGX Spark's architecture matters a lot here.

## Short answer

**No.** DGX Spark is an **ARM64 (aarch64) Linux** machine, and Google's Android Emulator does not support ARM64 Linux as a host — only **Apple Silicon macOS**. Android Studio itself also has no official Linux-ARM64 build. So the standard `Android Studio + AVD emulator` setup does not work on DGX Spark.

## Why it doesn't work

DGX Spark runs DGX OS (Ubuntu 24.04-based) on the GB10 Grace Blackwell Superchip — a 20-core **Arm** CPU. That single fact breaks the normal emulator path:

| Component | Reality on DGX Spark |
|---|---|
| Android Studio IDE | Google ships **Linux x86_64 only** (the download page lists Windows 64-bit, Mac, Mac ARM, Linux 64-bit, ChromeOS — no Linux ARM) |
| Android Emulator binary | Google ships emulator builds for x86_64 Linux/Windows and aarch64 macOS. No Linux-aarch64 build exists |
| Hardware acceleration | Google's virtualization docs list only **Intel VT-x, AMD-V, and Apple silicon** as supported host CPUs |
| x86_64 system images as a workaround | Impossible to accelerate — KVM on an Arm host cannot virtualize x86 guests, and there is no nested-x86 path. Pure QEMU TCG emulation is unusably slow |

This isn't a "maybe it's just untested" situation — it's an open Google feature request: [issuetracker #386749845 "Emulator is not available on Windows/Linux arm64"](https://issuetracker.google.com/issues/386749845). A Google engineer states plainly:

> "We support the arm64 architecture on arm64 hosts. The only supported arm64 host currently is MacOS on Apple Silicon."

As of the latest comments (Oct 2026) the issue is still **Assigned / P3 / no update**, so treat this as "not supported, no ETA."

<details>
<summary><b>What actually does work (ranked by practicality)</b></summary>

1. **Physical Android device (best option)** — USB or wireless debugging. `adb` is available for arm64 Linux (Ubuntu/Debian package), platform-tools' official binary is x86_64 but the distro build works fine. Deployment, `logcat`, `dumpsys`, and profiling all behave normally.
2. **Remote emulator** — run the emulator on any x86_64 PC/Mac, then `adb connect <host>:5555`. Android Studio's "pair devices over Wi-Fi" works, so the Spark can drive a remote AVD over the network.
3. **Cloud device farms** — Firebase Test Lab, BrowserStack, Sauce Labs, AWS Device Farm, Genymotion SaaS. Good for CI and coverage across API levels/form factors.
4. **Prebuilt APK testing** — build on the Spark headlessly, install/test on a phone or in the cloud.
5. **Community ARM64 Android Studio repack** — e.g. `DesktopECHO/android-studio-aarch64-install` (ideally plus arm64 SDK/NDK build tools). The IDE will launch, but **you still get no emulator**, and some build tools (`aapt2`, etc.) need arm64 substitutes. Treat as unsupported and fragile.
6. **Waydroid / Cuttlefish / aarch64 QEMU emulator forks** — container- or fork-based Android runtimes that *can* work on arm64 Linux, but they need kernel `binder` + `/dev/kvm` support on DGX OS, are not integrated into Android Studio, and have Play-services/GPU caveats. Experimental at best.

</details>

## Bottom line

Use the DGX Spark for what it's built for — local inference, fine-tuning, CUDA/Blackwell work — and do Android UI testing on a **real phone over `adb`** or an **emulator on a small x86_64 box / cloud farm**. If Android app development is a primary use case, DGX Spark is the wrong tool; any x86_64 Linux, Windows, or Apple Silicon Mac is a far better fit.

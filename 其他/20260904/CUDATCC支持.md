# CUDATCC支持

## 🧑‍💻 User

cycles support tcc mode cuda(a100 h200)?

---
## 🤖 Assistant

I'll research this for you. Let me search for relevant information.

Let me dig into the most relevant sources about server GPUs and TCC mode.

Let me look for specific Blender developer discussions on TCC mode.

Let me search for the specific Blender issue/commit about TCC mode detection.

Let me check the Blender developer docs and source, plus the devtalk thread.

I found the actual Cycles CUDA source code. Now let me verify a few specifics (the Stack Exchange report and TCC handling history).

Let me confirm the CUDA architecture support versions (sm_80 for A100, sm_90 for Hopper).

## Short answer

**Yes.** Blender Cycles' CUDA backend enumerates and renders on GPUs running in **TCC (Tesla Compute Cluster) mode** — including the A100 (compute capability $8.0$) and H200/H100 (compute capability $9.0$). TCC devices are treated as normal CUDA compute devices, not filtered out.

---

## Why it works (from the Cycles source)

Cycles' CUDA device enumeration (`src/device/cuda/device.cpp`) deliberately separates *display* GPUs from *compute* GPUs — and TCC GPUs fall into the compute group:

```cpp
/* If device has a kernel timeout and no compute preemption, we assume
 * it is connected to a display and will freeze the display while doing computations. */
int timeout_attr = 0, preempt_attr = 0;
cuDeviceGetAttribute(&timeout_attr, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, num);
cuDeviceGetAttribute(&preempt_attr, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, num);

if (timeout_attr && !preempt_attr) {
  LOG_INFO << "Device is recognized as display.";
  info.description += " (Display)";
  info.display_device = true;
  display_devices.push_back(info);
} else {
  LOG_INFO << "Device has compute preemption or is not used for display.";
  devices.push_back(info);          // <-- TCC GPUs land here, fully usable
}
```

TCC GPUs have **no display / no kernel-exec timeout**, so they are always added to the renderable `devices` list, not the display-only list. Cycles even queries the TCC flag directly (`GET_ATTR(TCC_DRIVER)`) for diagnostics.

Architecture support in modern Cycles CUDA kernels spans `sm_30`–`sm_120`, which covers:
- **A100** → `sm_80`
- **H100 / H200** → `sm_90` (Hopper kernels shipped since roughly Blender 3.5 / CUDA 11.8+; A100 `sm_80` for longer still)

---

## Important caveats

<details>
<summary><strong>1. TCC GPUs cannot drive the Blender UI / OpenGL viewport</strong></summary>

TCC disables Windows graphics output. Blender's GUI and EEVEE/OpenGL viewport need a **WDDM** (display) GPU. So:
- Keep a cheap WDDM GPU for display, and put the A100/H200 in TCC for **Cycles rendering only**.
- On a headless render node, run Blender in background mode (`blender -b`) — this is exactly what TCC is for.
</details>

<details>
<summary><strong>2. A100 / H200 are compute-only on Windows by default</strong></summary>

NVIDIA's data-center driver reports e.g. *"The following GPUs are in TCC mode (compute only) and cannot be used by graphics APIs for rendering"*. Enabling WDDM for the A100/H100/H200 requires a **vGPU driver + GRID license**. You normally do **not** want this — you want TCC + Cycles CUDA.
</details>

<details>
<summary><strong>3. TCC is a Windows-only concept</strong></summary>

On Linux there is no TCC/WDDM distinction — data-center GPUs are simply headless compute devices. Cycles CUDA works the same.
</details>

<details>
<summary><strong>4. Match the Blender build to the GPU architecture</strong></summary>

Official builds ship *precompiled* CUDA kernels. If your GPU arch isn't among them, Blender tries to JIT-compile — which needs `nvcc` present and usually isn't available. So use a Blender version whose CUDA binaries include `sm_80` **and** `sm_90`. If you build Blender yourself, set `CYCLES_CUDA_BINARIES_ARCH` to include both:
```
sm_80;compute_80;sm_90;compute_90
```
</details>

---

## How to verify on your machine

```python
import _cycles
# or, in the GUI: Preferences ▸ System ▸ Cycles Render Devices ▸ CUDA
print(_cycles.available_devices("CUDA"))
```
Or run Blender with `--debug-cycles` and look for the log lines:
- `Device has compute preemption or is not used for display.` → TCC/compute GPU detected correctly
- `Device is recognized as display.` → it's a WDDM display GPU

Then check your device list: a TCC A100/H200 will appear **without** the `(Display)` suffix and can be ticked for rendering.

**Bottom line:** Cycles CUDA + TCC + A100/H200 is a supported, intended configuration — you just need a separate WDDM display GPU (or headless/background rendering), and a Blender build containing the matching `sm_80`/`sm_90` CUDA kernels.

# ROCm Triton无日志崩溃调试方法

### 🧑‍💻 User

rocm triton崩溃没有任何日志只显示核心转储怎么debug？

---

### 🤖 Assistant

**调试 ROCm + Triton 崩溃（无日志、只有 core dump）的实用步骤**

核心思路：先拿到**完整调用栈**，再开启 verbose 日志/序列化执行，最后缩小到最小复现。Triton（尤其 ROCm 后端）崩溃常发生在 JIT 编译、LLVM/MLIR 处理或 HIP runtime 调用时，host 侧 core dump 很常见。

### 1. 确保生成完整可用的 core dump
```bash
ulimit -c unlimited
# 临时设置 core 文件名（推荐）
echo 'core.%e.%p.%t' | sudo tee /proc/sys/kernel/core_pattern
# 或使用 systemd-coredump（更方便）
# 查看：coredumpctl list
# 提取：coredumpctl dump -o core.xxx PID
```

确认磁盘空间足够，core 文件大小合理（不是几 KB 的截断）。

### 2. 用 GDB / rocgdb 分析 core dump（最关键）
找到可执行文件（通常是 `python` 或你的二进制）和 core 文件：

```bash
# 普通 gdb
gdb /usr/bin/python3 ./core.python.xxxxx

# 如果有 ROCm 调试器优先用
rocgdb /usr/bin/python3 ./core.xxx
```

在 gdb 里执行：
```
set pagination off
bt
bt full
info registers
thread apply all bt full
info sharedlibrary
quit
```

**重点看**：
- 崩溃帧在 `libtriton*.so`、`libamdhip64.so`、`libhsa-runtime64.so`、LLVM/MLIR 相关库，还是 Python C 扩展。
- 是否有明显的空指针、断言失败、内存越界。
- 如果有符号缺失，安装对应 debug 包（`rocm-dbgapi`、`hip-dev`、系统 python-debug 等），或从源码带 `-g` 编译 Triton。

把完整 `bt full` 保存下来，这是最有价值的信息。

### 3. 强制 Python 在崩溃时打印 traceback（强烈推荐）
在代码最开头加：
```python
import faulthandler
faulthandler.enable()
# 或者更激进
faulthandler.enable(all_threads=True)
```

或直接运行：
```bash
python -X faulthandler your_script.py
```

很多 Triton 扩展崩溃时，faulthandler 能给出 Python + C 混合栈。

### 4. 开启详细日志 + 序列化执行（减少竞态）
在运行前 export 这些环境变量（按需组合）：

```bash
# Triton 相关
export TRITON_DEBUG=1
export TRITON_PRINT_AUTOTUNING=1
export TRITON_INTERPRET=1          # 解释器模式，慢但更容易定位逻辑问题（先试这个！）
# 如需 dump IR（视 Triton 版本）：
# export MLIR_ENABLE_DUMP=1
# export LLVM_IR_DUMP=1 或类似

# ROCm / HIP
export HIP_LAUNCH_BLOCKING=1
export AMD_SERIALIZE_KERNEL=3      # 或 1/2，强制序列化
export AMD_LOG_LEVEL=4             # 更高更详细，可试 3~7
export HSA_ENABLE_SDMA=0
export HSAKMT_DEBUG_LEVEL=7        # 部分版本有效
export ROCM_LOG_LEVEL=debug        # 视安装情况

# PyTorch 如果在用
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_USE_HIP_DSA=1         # 部分版本
```

同时检查：
```bash
dmesg | tail -100
# 或
journalctl -k -b | grep -iE 'amdgpu|gpu|page fault|oom|segfault'
rocm-smi
```

GPU page fault / 硬件错误有时只出现在 dmesg，不会有用户态日志。

### 5. 从启动就 attach 调试器（比事后分析 core 更直观）
```bash
gdb --args python your_script.py
# 或 rocgdb --args python ...
(gdb) set environment HIP_LAUNCH_BLOCKING=1
(gdb) set environment TRITON_INTERPRET=1
(gdb) run
# 崩溃后直接 bt full
```

如果崩溃在多线程，`thread apply all bt` 很重要。

### 6. 其他快速排查手段
- **strace 看最后系统调用**：
  ```bash
  strace -f -o strace.log -s 200 python your_script.py
  ```
  看崩溃前是否在 `mmap`、`open` 某个 `.so`、`ioctl` 等。

- **最小化复现**：
  - 用最小的 `@triton.jit` kernel + 最简单的 launch。
  - 关掉 autotune，固定 config。
  - 对比 CPU 解释模式 (`TRITON_INTERPRET=1`) 是否还能崩。
  - 换不同 block size / 数据类型 / 是否使用 atomic / 指针 arithmetic。

- **版本兼容性**（非常常见问题）：
  - 确认 Triton 的 ROCm 后端版本与你的 ROCm（`rocminfo` / `hipcc --version`）匹配。
  - PyTorch + Triton + ROCm 三者版本组合是否官方支持。
  - 尝试从源码编译最新 Triton（带 debug）：
    ```bash
    pip install -e . --no-build-isolation   # 在 Triton 源码目录
    ```
    编译时加 debug 符号。

- **内存相关**：
  - 用 `ASAN` 重新编译相关库（复杂但有效）。
  - 检查是否 OOM（`dmesg`、`free -h`、`rocm-smi` 显存）。
  - 尝试减小问题规模、关闭并行。

- **缓存问题**：
  ```bash
  rm -rf ~/.triton
  # 或指定
  export TRITON_CACHE_DIR=/tmp/triton_cache_clean
  ```

### 7. 拿到信息后如何继续
把以下信息收集起来（可打码路径）：
1. 完整 gdb `bt full` + `thread apply all bt`
2. 使用的 ROCm 版本、Triton 版本（`python -c "import triton; print(triton.__version__)"`）、PyTorch 版本（如果有）
3. 最小可复现代码片段
4. 相关环境变量和 dmesg 片段
5. 是否只在特定 GPU（MI200/MI300 等）或特定 kernel 出现

有了 backtrace 后，通常能直接定位是 Triton 编译器 bug、HIP runtime 问题、用户 kernel 写越界，还是 ABI/版本不匹配。

### 快速优先级推荐顺序
1. `python -X faulthandler` + `TRITON_INTERPRET=1` 重跑
2. 生成并分析 core dump（`bt full`）
3. 加 `HIP_LAUNCH_BLOCKING=1` + `AMD_SERIALIZE_KERNEL=3` + 高 log level
4. 最小化 + 清缓存 + 检查 dmesg
5. 源码 debug 编译或提 issue（附上 stack）

按上面做一般能拿到关键栈信息。如果你把 gdb 的 `bt full`（或 faulthandler 输出）贴出来，我可以帮你进一步解读可能的原因和下一步。

先从 faulthandler + 解释模式 + core 分析开始吧。
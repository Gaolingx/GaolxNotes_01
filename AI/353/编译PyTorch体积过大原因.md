# 编译PyTorch体积过大原因

### 🧑‍💻 User

为什么我自己编译的pytorch cuda二进制高达10g而pip安装的仅2g不到？

---

### 🤖 Assistant

**自编译的 PyTorch CUDA 二进制（尤其是 `libtorch_cuda.so` 等）经常达到数 GB 甚至 10G+，而官方 pip wheel 通常只有 ~1.5–3GB（安装后），主要是因为构建配置、包含的 GPU 架构、是否剥离符号、启用的功能与链接方式差异巨大。**

### 主要原因对比

| 因素 | 自编译常见情况 | 官方 pip wheel | 对体积影响 |
|------|----------------|----------------|------------|
| **CUDA 架构（compute capability）** | 默认或手动设成很宽的 `TORCH_CUDA_ARCH_LIST`（多代 GPU + PTX），每个架构都生成 SASS + PTX | 精选常用架构（如 7.0/7.5/8.0/8.6/9.0 等），控制 fat binary 大小 | **最大**（可差数 GB） |
| **构建类型 / 符号** | Debug、RelWithDebInfo，或未 strip | Release + 彻底 strip | 很大（符号能占数 GB） |
| **功能与依赖** | 默认开启几乎所有东西（NCCL、distributed、quant、test、各种后端、Magma 等） | 按需精简 + 优化链接 | 中到大 |
| **中间产物 vs 最终安装** | 常把整个 build 目录（海量 `.o`、CUDA 中间文件）也算进去 | 只有最终 wheel/安装包 | 极大（build 目录轻松 10G+） |
| **链接与优化** | 动态/静态混用、未做 LTO/大小优化、可能带调试信息 | 专门为发布优化、strip、控制依赖 | 中等 |
| **CUDA 运行时等** | 有时把更多 CUDA 组件或本地 toolkit 痕迹带进来 | 依赖系统/conda CUDA 或精简打包 | 视情况 |

**最常见的“体积爆炸”元凶**：
1. **过多 CUDA 架构**：源码构建时如果不限制 `TORCH_CUDA_ARCH_LIST`，会为很多 compute capability 编译内核，体积线性暴涨。官方 wheel 会仔细挑选并控制。
2. **未剥离调试符号**：Debug 信息、未 `strip` 的 `.so` 可以轻松翻倍甚至更多。
3. **把 build 目录当成“二进制”**：编译过程中生成的目标文件、临时 CUDA 编译产物非常大，最终 `python setup.py install` 或 `cmake --install` 后的包才是真正可对比的。
4. **默认开启过多选项**：`USE_DISTRIBUTED`、`USE_NCCL`、`BUILD_TEST`、各种 experimental 功能、完整 third_party 等都会增加体积。
5. **缺少发布级优化**：官方有专门的构建脚本/标志来控制大小；本地随便 `python setup.py bdist_wheel` 或 CMake 默认配置往往更“全”。

pip 安装的包是**已经过优化、strip、架构筛选、功能裁剪**的预编译 wheel，所以体积可控。

### 如何验证与缩小自编译体积

**先确认到底是哪里大**：
```bash
# 安装后的 torch 包
du -sh $(python -c "import torch; print(torch.__path__[0])")

# 关键大的 .so
find $(python -c "import torch; print(torch.__path__[0])") -name "*.so*" -exec ls -lh {} \; | sort -k5 -h

# 尤其看
ls -lh .../lib/libtorch_cuda*.so*
```

**减小体积的有效做法**（构建时设置）：
- **严格限制架构**（最重要）：
  ```bash
  export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"   # 只写你实际用的 GPU
  # 或更精简 "7.5;8.0;8.6"
  ```
  不要用默认的宽列表，也不要无脑加所有 +PTX（除非真需要前向兼容）。

- **Release + 剥离符号**：
  ```bash
  export CMAKE_BUILD_TYPE=Release
  # 安装后手动 strip（效果显著）
  find $TORCH_INSTALL_PATH -name "*.so*" -exec strip --strip-all {} \;
  # 或 strip --strip-debug 保留一点调试能力
  ```

- **关闭不需要的功能**（CMake/环境变量示例）：
  ```bash
  export USE_DISTRIBUTED=0          # 不需要分布式就关
  export USE_NCCL=0
  export USE_MPI=0
  export USE_GLOO=0
  export BUILD_TEST=0
  export USE_QUANTIZED=0            # 视需求
  export USE_FBGEMM=0
  # 还有 USE_MKLDNN、USE_OPENMP 等按需调整
  ```

- **其他实用设置**：
  - 使用官方推荐的构建方式/脚本（参考 PyTorch 的 `cmake` 配置或 docker 构建流程）。
  - `MAX_JOBS` 控制并行，但不直接决定最终体积。
  - 确保用动态链接 CUDA runtime（默认通常如此），避免不必要的静态塞入。
  - 清理 build 目录后再打包/安装，只比较最终安装树或 wheel。
  - 考虑 `python setup.py bdist_wheel` 后检查 wheel 本身大小，再安装验证。

**示例精简构建思路**（仅示意，需结合你的 CUDA/PyTorch 版本和需求）：
```bash
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export CMAKE_BUILD_TYPE=Release
export USE_DISTRIBUTED=1          # 按需
export BUILD_TEST=0
export MAX_JOBS=8
# 然后按官方文档 python setup.py install 或用 CMake 流程
# 安装后 strip
```

### 额外建议
- **优先用官方 pip/conda**：除非你有必须改源码、支持特殊架构、特定 CUDA 版本或自定义算子的需求，否则官方包体积更合理、经过充分测试。
- 如果你必须自编译，对照 [PyTorch 官方构建文档](https://github.com/pytorch/pytorch#from-source) 和他们的 CI/发布构建配置，尽量复现接近 release 的标志。
- 10G 级别通常意味着「架构太多 + 带符号 + 可能统计了 build 中间文件」。把架构砍到你的 GPU 实际需要 + Release + strip 后，体积一般能降到接近甚至优于官方的水平（取决于你开的功能）。
- 不同 PyTorch + CUDA 版本体积差异也很大（新版本算子/架构支持更多，本身就会更大），对比时尽量用相同版本。

按上面限制架构和 strip 后，绝大多数人都能把体积从「夸张的 10G」压到合理范围。如果方便补充你的 `TORCH_CUDA_ARCH_LIST`、CMake 选项、`libtorch_cuda.so` 实际大小、是否 strip、以及你是看 build 目录还是安装目录，可以进一步精确定位。
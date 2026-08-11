# ComfyUI秒加载靠懒加载和按需加载

### 🧑‍💻 User

很好奇comfy那种秒加载到底是怎么做到的，跑flow要几个模型切来切去走正常的torch.load都要很久才能进显存

---

### 🤖 Assistant

**ComfyUI 的“秒加载”主要靠懒加载（lazy loading）+ 内存映射（mmap）+ 虚拟地址空间 + 按需 fault 进物理显存，而不是像普通 `torch.load` + `.to("cuda")` 那样一次性全量加载并拷贝。**

普通流程（比如直接 `torch.load` 或标准 safetensors 再 `.to(device)`）会：
- 把整个模型（几 GB 到几十 GB）反序列化/读进已提交（committed）的系统 RAM。
- 再完整拷贝到 VRAM，PCIe 带宽和内存带宽都是瓶颈，多个模型来回切就更慢。

ComfyUI 的核心优化（尤其是近期的 **Dynamic VRAM / AIMDO** 系统）彻底改变了这一点。

### 1. 自定义 safetensors 加载器 + 未提交内存映射（关键“秒加载”来源）
- ComfyUI 优先用 `.safetensors`（比 pickle 的 `.ckpt/.pt` 快且安全）。
- 它有自己的 loader：用更高效的文件打开模式，把文件 **mmap 到未提交（uncommitted）的 file-backed memory**。
- 权重通过 **指针赋值** 给模型结构，**不做 deep copy** 到进程内存。
- 结果：Load 节点几乎瞬间执行完（只解析 header + 建立映射和指针），几乎不占用实际 committed RAM。OS 可以随时回收这些页面。[[1]](https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local)

对比标准 `safetensors.torch.load_file`，Comfy 的版本更极致地避免了提前物化数据。

### 2. Dynamic VRAM（AIMDO）——虚拟地址 + Just-in-Time 物理分配
这是近年（大约 2026 左右默认开启）的重大优化，专门为模型权重设计：

- 加载模型时创建 **VBAR（Virtual Base Address Register）**：只消耗 GPU **虚拟地址空间**（几乎免费且“无限”），**零物理 VRAM**。
- 张量初始是未分配状态。正常访问会 segfault，但 Comfy 用自定义 `fault()` API。
- 计算真正需要某个权重时（例如 sampler 某层），才调用 `fault()`：
  - 有空闲 VRAM → 分配物理 VRAM 并填充数据（从 mmap 源）。后续复用，保持高速。
  - 压力大 → 用临时张量拷贝当前需要的层，用完可释放；或驱逐低优先级权重。
- 有 **优先级 + watermark 系统**，避免暴力 thrashing（最近加载的模型优先，被驱逐的设置 watermark，上层自动失败 fault 以高效检查）。
- 不再把整个模型从 VRAM 卸回完整 RAM 拷贝。卸载只是释放物理 VRAM，模型回到“未提交”的 mmap 状态，跨 workflow 复用很快。

好处：
- 初始加载极快（几乎秒级）。
- 支持比物理 RAM 还大的模型（不会疯狂用 pagefile）。
- 多模型切换时 VRAM 利用率更高、OOM 更少。
- LoRA 应用也更快。
- 系统 RAM 占用更低更智能（有缓存但不 lock 到 pagefile，其他程序需要时立刻让出）。

Windows 任务管理器可能仍显示较高 RAM（因为有缓存加速），但 Linux 上常显示很低（算作 cache）。OS 需要时自动从磁盘 re-read。

### 3. 整体模型管理系统（`model_management.py` + ModelPatcher）
即使没有 Dynamic VRAM，ComfyUI 一直比 A1111 等更省 VRAM：
- 模型通常先到 CPU/RAM，按需 `load_models_gpu()`。
- 跟踪当前已加载模型列表，智能决定加载/卸载/offload。
- 多模型 flow（base + refiner、多个 ControlNet、不同 UNET/CLIP/VAE、切换 checkpoint 等）只把当前需要的部分放到 VRAM，其他 offload。
- 选项如 `--lowvram` / `--normalvram` / `--highvram` 控制激进程度（Dynamic VRAM 让这更自动化）。
- 文件在 OS 磁盘缓存里时，后续“从磁盘加载”实际很快。
- GGUF 量化模型更极致：加载器几乎瞬间，真正权重按需加载，RAM/VRAM 在 KSampler 阶段才明显涨。

### 为什么感觉“秒切换多个模型”
- 加载节点本身不干活（只建虚拟结构）。
- 数据移动延迟到真正计算时，且粒度细（按层/按权重）。
- 模型保持“已加载”的虚拟状态跨运行。
- 有缓存 + 智能优先级，切换时 PCIe 流量最小化。
- 复杂 flow 中只活跃当前需要的模型。

### 补充技巧与注意
- **用 safetensors**（尤其 fp8/bf16 等），避免老 pickle。
- 模型放 SSD（最好 NVMe），OS 缓存效果好。
- 足够系统 RAM 有助于缓存（但 Dynamic VRAM 降低了对大 RAM 的硬性依赖）。
- GGUF 插件对超大模型加载感更“秒”。
- 如果觉得慢，检查是否禁用了 Dynamic VRAM、是否在 HDD、驱动/CUDA 版本、是否有自定义节点干扰加载路径。
- 旧版本或禁用 Dynamic VRAM 时，主要靠 mmap + 智能 offload，依然比裸 `torch.load` 快很多，但不如现在极致。

简单说：**Comfy 不“加载整个模型进显存”，而是建个虚拟壳 + 文件映射，真正用到哪一层才 fault 进物理显存**。这就是秒加载 + 多模型高效切换的秘密。普通 PyTorch 脚本很难直接复现同样的体验，因为缺少这套自定义分配器和 patcher 管理。

如果你有具体版本、硬件或某类模型（Flux、SDXL、视频模型等）的日志，可以进一步分析。
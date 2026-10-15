# Godot AAA 开放世界升级方案

## 🧑‍💻 User

godot针对aaa开放世界streaming、hlod+gpu driven和World Partitioning要如何修改引擎，有没有previous work

---
## 😀 默认助手 (7)

I'll research this properly — Godot's current rendering/scene internals plus existing proposals, PRs, plugins, and demos. Let me pull up the browser tools first.

下面是基于**实际引擎源码（master = `4.8-dev`）与官方 proposal/PR 仓库核实**后的答案。我把"引擎要改哪里"和"前人做到哪一步"分开写，最后给一条可落地的路线图。

---

# 一、结论速览

| 能力 | 引擎现状（已核实） | 必须动的地方 | 前人工作 |
|---|---|---|---|
| **World Partitioning / Streaming** | 只有 `ResourceLoader.load_threaded_*`；没有 cell 概念、没有节点"休眠/藏匿"（stash）状态、没有异步实例化 | `SceneTree`/`Node` 状态机、物理/导航/音频的批次注册、编辑器分区工具 | **只有提案，无核心实现**：#11728、#7144、#1197、#2889、#10212、#13798（已关） |
| **HLOD** | 只有手工的 `GeometryInstance3D.visibility_range_*` + `visibility_parent` 依赖淡入淡出 + 自动 mesh LOD | `RendererSceneCull` 的实例分组 + 离线烘焙（合并/简化/imposter）+ 编辑器工具 | 部分：visibility range 已合入；**无自动化 HLOD 烘焙**。提案 #14948、#7339、#5174、#6402、#2014 |
| **GPU-driven** | CPU 多线程剔除 + 逐实例写 SSBO；只有 `MultiMesh` 支持 indirect draw；mesh shader 未合入 | `RendererSceneCull` 实例注册表 → GPU scene、compute 剔除、meshlet、bindless、GPU HZB | **基建已就位**：`RD::draw_list_draw_indirect`（4.4）、MultiMesh indirect（4.4）、mesh surface RD buffer 暴露给 GDExtension（已合入）；mesh shader PR #88934 仍 open |

一句话：**Streaming + World Partition 是"必须改引擎"（连官方都承认的架构级缺口）；GPU-driven 目前已经可以在 GDExtension 里做出 70% 的效果，真正要"AAA"还是得加一个新 renderer。**

---

# 二、基线：引擎里已经有什么（这决定了你从哪一层开始做）

<details>
<summary><b>展开：已核实的引擎内部机制与版本</b></summary>

**剔除 / 渲染主链（`servers/rendering/`）**
- `RendererSceneCull`（SceneCull）持有全部 `Instance`，用 `WorkerThreadPool` 并行跑 `_scene_cull_threaded` / `_scene_cull` / `_visibility_cull_threaded`，可在 `rendering/limits/spatial_indexer/threaded_cull_minimum_instances` 以上开启多线程剔除。
- 剔除结果是 `RenderList`（`GeometryInstanceSurfaceDataCache**` + `RenderElementInfo`），交给 `RenderForwardClustered::_render_list_template<>()`，逐 element 直接 `draw_list_add_*`。
- 逐实例数据写进 SSBO：`MultiUmaBuffer<1u> instance_buffer[RENDER_LIST_MAX]`（`InstanceData`，含 `instance_uniforms_ofs`、各种 `INSTANCE_DATA_FLAG_*`，最小 4096 实例 `INSTANCE_DATA_BUFFER_MIN_SIZE`）。
- **唯一的 GPU-driven 入口**：`INSTANCE_DATA_FLAG_MULTIMESH_INDIRECT` → `RD::draw_list_draw_indirect(...)`（`render_forward_clustered.cpp` 第 613-621 行）。
- 遮挡剔除：`RendererSceneOcclusionCull::HZBuffer`（CPU 端 HZB/depth pyramid，4.3 引入）+ 传统 `OccluderInstance3D` 的 BVH 射线系统（`rendering/occlusion_culling/bvh_build_quality`、`occlusion_rays_per_thread`）。
- Mesh LOD 选择在 **CPU 上、element 构建期**完成：`MeshStorage::mesh_surface_get_lod()`（`servers/rendering/renderer_rd/storage_rd/mesh_storage.h`），阈值 `rendering/mesh_lod/lod_change/threshold_pixels`，逐实例 `lod_bias`。

**流式 / 资源（`core/io/`、`servers/rendering/storage/`）**
- `ResourceLoader.load_threaded_request / load_threaded_get_status / load_threaded_get`，`CACHE_MODE_IGNORE[_DEEP]` 可避免常驻缓存。
- 节点级占位：`Node.set_scene_instance_load_placeholder(true)`（编辑器占位，世界分区的编辑器体验可以先用它顶着）。
- `modules/texture_streaming/`（**master 已合入，属 4.8-dev**）是"如何给 Godot 加一个流式子系统"的**现成范本**：单例 + 反馈处理线程 + 专用 I/O 线程 + 自研 SPSC 命令队列 + 内存预算 + 逐纹理 mip 覆盖，并以 `MODULE_TEXTURE_STREAMING_ENABLED` 宏在 `render_forward_clustered.cpp` 里打点接入渲染器。

**工业级工具链已经在树里**
- `thirdparty/meshoptimizer/`（含 `simplifier.cpp`、`clusterizer.cpp`）= **HLOD 合并/简化 + meshlet 构建的现成算法基础**，不用自己写。
- `rendering/renderer/rendering_method` 的取值是 `forward_plus` / `mobile` / `gl_compatibility` —— 这就是**新增一个 GPU-driven renderer 的官方插拔点**。
</details>

---

# 三、三块能力分别怎么改引擎

## 3.1 World Partitioning + Streaming

### 为什么不是 GDExtension 能解决的
1. **每个 `Node` 的成本是非零且巨大的**（ScriptInstance、`_propagate_transform_changed`、groups、physics body、`_notification`）。AAA 里一屏环境物件是 10⁵–10⁶ 量级，用 Node 承载必然崩。
2. `add_child` / `PackedScene.instantiate()` 是**主线程提交**，无法把"构建节点树"放到 worker，所以没有"分帧实例化"就一定有 hitch。提案 #10212 的 Node Stashing 就是想补这个状态机缺口。
3. cell 的语义要横跨**渲染 / 物理 / 导航 / 音频 / 编辑器 / 网络 AOI**，addon 只能各做一套互不兼容的实现（#15350 floating-origin 就是同一个痛点的镜像）。

### 分层改造方案

**L0：不改引擎，今天能做的**（先验证玩法规模）
```gdscript
# cell 调度器：只做"资源加载 → 主线程分帧实例化"
var st = ResourceLoader.load_threaded_request(cell_scene_path, "", true)
# 每帧 load_threaded_get_status() == THREAD_LOAD_LOADED 时
# 取出 PackedScene，然后在 _process 里限时 budget 内 add_child
```
用 `VisibleOnScreenNotifier3D` / `IntersectShape3D` 做激活域，用 `CACHE_MODE_IGNORE` 防止资源池爆炸。缺点：Node 开销与主线程实例化无法回避。

**L1：加一个 module（推荐起点，直接抄 `modules/texture_streaming/` 的骨架）**
```
modules/world_partition/
  world_partition.h/.cpp     # singleton + 调度器（反馈/IO/提交 三条线）
  world_cell.h/.cpp          # Node3D，持有 CellDescriptor
  cell_descriptor.h          # Resource: 场景路径 / bounds / cell 坐标 / 流式等级
  config.py, register_types.*
```
- 流式等级：`AlwaysLoaded` / `Distance` / `Volume` / `Manual`（对应 UE 的 ALWAYS_LOADED / distance-based）。
- 预算驱动：每帧提交额度按 $\text{budget} = k \cdot \max(\text{落帧时间}, \text{带宽})$ 动态调整，而不是固定 cell 数。
- 关键是把"**load → instantiate → commit**"拆成三段，commit 限时。

**L2：改引擎核心（"真 AAA"必须有）**
| 触点 | 改法 |
|---|---|
| `scene/main/node.h/.cpp` | 增加 `set_stashed()/is_stashed()` + `NOTIFICATION_STASHED`，stash 时跳过 process、transform 传播、group 注册、physics RID 创建 |
| `scene/main/scene_tree.cpp` | `_flush_ugc` 支持"批量提交"（一次 deferred 队列提交整个 cell，而不是逐节点） |
| `scene/resources/packed_scene.cpp` | 支持**离线构建节点树**（worker 上构建，主线程只做 attach），或在 `SceneState` 上做"无脚本节点快速实例化"路径 |
| `servers/rendering/renderer_scene_cull.*` | 不只是节点：把 cell 变成**渲染侧聚合实例**（见 3.2 HLOD 同一条通道），否则 1M 实例的 CPU 剔除本身还是瓶颈 |
| `servers/physics_server_3d` + Jolt | cell 级批量 add/remove body，避免逐 body 的 `_update_shapes` 尖峰 |
| `editor/` | 网格化分区可视化 + `WORLD_PARTITION` 导入器；可先用 `set_scene_instance_load_placeholder` 做编辑态 |

> 实务建议：**环境批量内容不要用 Node 承载**。让 `WorldCell` 直接持有 `MultiMesh` / `MeshRD`（`RenderingServer` 层）+ 物理 body，把 SceneTree 完全绕开——这才是 UE World Partition 的 Actor/Component 之外还有 ISM/HISM 的原因。

---

## 3.2 HLOD

### 现有机制（很关键，别重复造）
Godot 把 HLOD 做成了**手工约定**：`GeometryInstance3D.visibility_range_begin/end` + `_margin`（滞后） + `fade_mode`：
- `VISIBILITY_RANGE_FADE_DISABLED`：最快，纯切换（会有 LOD 跳变）
- `VISIBILITY_RANGE_FADE_SELF`：自身淡出
- `VISIBILITY_RANGE_FADE_DEPENDENCIES`：淡入它 `visibility_parent` 指向的"粗模"，这就是官方文档里的 **Visibility ranges (HLOD)** 流程。

导入侧也已打通：`mesh_instance/visibility_range_*` 是 `resource_importer_scene` 的导入选项（`editor/import/3d/resource_importer_scene.cpp`），`ImporterMesh::generate_lods()`（`scene/resources/3d/importer_mesh.h`）负责自动 LOD 生成。

### 缺什么（就是你真正要改的）
1. **没有自动化 HLOD 烘焙**：合并 N 个 Mesh → 简化 → 可选 imposter/billboard，并生成对应 proxy 资源。UE 用 HLOD Builder + Simplygon；Godot 里没有。
2. **没有"多物件共享一个 proxy"**：现在的 visibility range 是**逐 MeshInstance 的**，AAA HLOD 的收益恰恰来自把一整个 cell 的 5000 个物件压成 1 个 draw。
3. **没有 cluster/actor 层的实例分组**：`RendererSceneCull` 没有任何"父实例可见才剔除子实例"的层次结构。
4. **proxy 自身没有流式**：HLOD proxy 也可能是几百 MB，必须和 cell streaming 联合。
5. **不参与遮挡剔除/阴影**：HLOD 状态与 HZB、shadow caster pass 没有联动（提案 #14948 就在抱怨拿不到 HLOD 状态）。
6. **MultiMesh 无逐实例 LOD/剔除**：提案 #10669（这恰好也是 GPU-driven 的入口）。

### 改造方案
**离线烘焙（编辑器/CLI，可先做成 GDExtension 工具）**
- 用已内置的 `thirdparty/meshoptimizer`：
  - `meshopt_simplify()` / `meshopt_simplifyWithAttributes()` 做几何简化
  - `meshopt_generateVertexRemap*` + `meshopt_optimizeVertexCache/Fetch` 做合并后重整
  - `meshopt_buildMeshlets()` 如果走 3.3 的 virtual geometry 路线
- 输出：`HLODProxy` 资源（mesh + material override + bounds + 触发距离），挂到 `ImporterMesh` 同级的导入流程里。

**运行时（改 `RendererSceneCull`）**
```cpp
// 新增：实例组（cluster/actor），一个 group 有合并 AABB 与 HLOD 链
struct InstanceGroup {
    AABB bounds;                 // 合并 AABB，用于一次剔除决定 N 个子实例
    RID  hlod_proxies[MAX_HLOD]; // 距离分级
    float hlod_distances[MAX_HLOD];
    LocalVector<Instance*> children;
};
```
- 在 `_scene_cull()` 里先对 group 做剔除 + 距离分级：命中 → 只把对应 proxy 加进 `RenderList`，`children` 整批跳过；未命中 → 展开 `children`。
- 这样一次剔除决策覆盖 N 个实例，**同时解决 World Partition 的 CPU 剔除开销问题**（3.1 的 L2 表格里那条）。
- `fade_mode` 的 dependency 淡入淡出可以直接复用到 proxy↔children 之间。

---

## 3.3 GPU-driven（这是三块里"基建最完整、但缺最后一段"的）

### 已有的拼图
| 能力 | 状态 |
|---|---|
| 设备端 indirect draw | ✅ `RenderingDevice.draw_list_draw_indirect`（4.4，PR #97247） |
| MultiMesh 走 indirect | ✅ PR #99455（4.4），命令步长 `INDIRECT_MULTIMESH_COMMAND_STRIDE = 5` |
| 写 MultiMesh 实例缓冲的 RD RID | ✅ `multimesh_get_buffer_rd_rid()` / `multimesh_get_command_buffer_rd_rid()` |
| 写 mesh surface 顶点/索引缓冲的 RD RID（compute 直接写） | ✅ `mesh_surface_get_{vertex,index,attribute,skin}_buffer_rd_rid()` + `ARRAY_FLAG_USE_STORAGE_BUFFER`（PR #118973，已合入） |
| Mesh shader | ❌ PR #88934 仍 open |
| `draw_indirect_count`（GPU 决定 draw 数） | ❌ 提案 #13652 open |
| CompositorEffect 拿到 DrawList | ❌ 提案 #13405/#13406 已关闭，官方选择的是"暴露 buffer RID 给 GDExtension"这条路 |
| GPU 剔除 / GPU scene / meshlet / bindless | ❌ 完全没有 |

### 现在就能做的（无 fork，性价比最高）
用 **MultiMesh indirect + compute 剔除** 做出"伪 GPU-driven"：
1. compute shader 读实例列表 → 做视锥 + 距离 LOD 剔除 → 写 `multimesh_get_buffer_rd_rid()` 的实例缓冲；
2. 同时写 `multimesh_get_command_buffer_rd_rid()` 的命令缓冲（`instance_count`、`instance_offset`…）；
3. 把 `MultiMeshInstance3D` 的 **custom AABB 设为极大** 并 `INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING`，避免 CPU 端把这个节点整批剔掉；
4. mesh 侧用 `ARRAY_FLAG_USE_STORAGE_BUFFER` + `mesh_surface_get_vertex_buffer_rd_rid()` 在 compute 里直接生成几何。

这已经能把"逐实例剔除 + LOD 选择"从 CPU 搬走，代价是：仍然走 `RendererSceneCull` 的 element 级 draw（每 MultiMesh 一次），材质系统还是老路。**这是 #10669 想要的，也是目前在 vanilla 引擎里的极限。**

### 要"AAA"必须做的（改引擎 / 新 renderer）
按依赖顺序：

1. **GPU Scene**：在 `RendererSceneCull` 之上加一份持久的 GPU 实例表（primitive / instance data / bounds / material index / LOD set），CPU 侧 `Instance` 降级为"游戏逻辑镜像"。这是后面所有东西的前提。
2. **Compute 剔除 + indirect**：frustum → HZB 遮挡 → LOD 选择 → 写入 per-pass bucket，产出 indirect command + count。抽掉 `RendererSceneCull::_scene_cull` 与 `_render_list_template<>` 之间的 CPU 逐 element 循环。需要先补 `draw_indirect_count`。
3. **把 HZB 搬到 GPU**：现在 `RendererSceneOcclusionCull` 是 CPU 光栅化 depth pyramid，GPU-driven 必须改成两阶段（上一帧 HZB → 本帧 compute 剔除）。配置在 `rendering/occlusion_culling/*`。
4. **Bindless / descriptor indexing**：GPU-driven 的 draw 是"未知材质"的，必须走 bindless texture/material 表（Vulkan `VK_EXT_descriptor_indexing`，D3D12 dynamic resources，Metal argument buffer）。Godot 现在的 `MaterialStorage` 是逐材质 uniform set，需要新增一条 bindless 路径。
5. **Meshlet（Nanite-class）**：import 期用 `meshopt_buildMeshlets()` 生成 cluster，运行期 DAG/group 选择 + cluster 级剔除；geometry 路径要么 mesh shader（依赖 #88934），要么 vertex-shader 里 fetch cluster（Metal/GL 回退）。
6. **集成方式**：**不要在 `forward_clustered` 里打补丁**。Godot 已经有 3 个 renderer，`rendering/renderer/rendering_method` 就是插拔点，加一个 `gpu_driven` 走 `RendererCompositor` / `RendererSceneRender` 接口，干净得多，也便于长期 rebase。

---

# 四、Previous work 清单（全部核实过）

<details>
<summary><b>A. 官方提案（godot-proposals）</b></summary>

**World Partition / Streaming**
| # | 标题 | 状态 |
|---|---|---|
| #11728 | Add a World Partitioning system（onward Open World RPG；**已指派给 `fire`**，47 条讨论，2025-02 起） | open |
| #7144 | Add a dynamic world partitioning system（V-Sekai，2023） | open |
| #1197 | Add level streaming and stitching tools for creating large open worlds | open |
| #2889 | Add a "StreamingScene" node and associated level streaming editor tools | open |
| #13798 | Integrated Scene Streaming and LOD System for High-End 3D Worlds（2025-12） | closed |
| #6109 | Implement Mesh streaming（明确写了"要在 GPU 上完全剔除 + 分页 + 固定显存"） | open |
| #3177 | Add support for texture streaming | closed（**已作为 module 落地**，见 C） |
| #10212 | Implement support for Node Stashing（73 讨论，stash 状态机的官方诉求） | open |
| #1749 | Add Cesium streaming 3D Tiles（地理级流式） | closed |
| #13799 | Next-Gen 3D Features for Godot 5（Procedural Worlds / GPU-Powered …） | closed |

**HLOD / LOD**
| # | 标题 | 状态 |
|---|---|---|
| #14948 | Add methods/signals to get access to visibility range (HLOD) current state | open |
| #7339 | Improved LOD importing（含自动设 visibility range 的诉求） | open |
| #5174 | Allow specifying custom LOD meshes instead of relying on automatic generation | open |
| #6402 | Implement LOD generation for existing meshes at run-time | open |
| #2014 / #3861 | LOD auto system / 简化时保轮廓 | closed |
| #10669 | **per-instance frustum culling and mesh LOD support to MultiMeshInstance3D**（GPU-driven 的实用入口） | open |
| #3920 / #3914 / #4440 | Portals & rooms 移植到 4.x / 代理网格遮挡 / 暴露遮挡查询 API | open |

**GPU-driven / virtual geometry**
| # | 标题 | 状态 |
|---|---|---|
| #2793 | Implement virtualized geometry rendering（Godot 版 Nanite，40 讨论，2022 起） | open |
| #6822 | Implement mesh shading in RenderingDevice | open |
| #7209 | Add new Mesh resource subtype to enable creating surfaces using RenderingDevice buffers | open |
| #14741 / #14329 | Expose mesh surface RD buffer RIDs to GDExtension（GPU-driven 粒子/几何） | closed→**已实现** |
| #13405 / #13406 | Add DrawList access to CompositorEffect for GPU-driven rendering | closed（未采纳该路线） |
| #13652 | Expose `RenderingDevice.draw_list_draw_indirect_count()` + 硬件支持查询 | open |
</details>

<details>
<summary><b>B. 已合入引擎的 PR / 树内模块 / 相关文件</b></summary>

| PR / 位置 | 内容 | 版本 |
|---|---|---|
| PR #97247 | `RenderingDevice.draw_list_draw_indirect` / `compute_list_dispatch_indirect` | 4.4 |
| PR #99455 | MultiMesh indirect draw（`use_indirect`、command buffer） | 4.4 |
| PR #98788 | `multimesh_get_buffer_rd_rid()` | 4.4 |
| PR #118973 | mesh surface RD buffer RID 暴露 + `ARRAY_FLAG_USE_STORAGE_BUFFER` | master（4.8-dev） |
| PR #113429 | **Implement texture mip-level streaming**（114 讨论）→ `modules/texture_streaming/` | master（4.8-dev） |
| PR #122940/#122938/#123656 | texture streaming 后续清理与修复（2026-08/09） | master |
| PR #48050 | 光栅遮挡剔除（`RendererSceneOcclusionCull::HZBuffer`） | 4.3 |
| PR #48847 / #78803 | visibility range 实现 / 场景导入器支持 visibility range | 4.x |
| `servers/rendering/renderer_scene_occlusion_cull.h` | CPU HZB 遮挡剔除（要 GPU 化的对象） | 树内 |
| `thirdparty/meshoptimizer/{simplifier,clusterizer}.cpp` | HLOD 简化 + meshlet 构建，**现成可用** | 树内 |
| `modules/texture_streaming/texture_streaming.h` | 流式子系统的**参考架构**：SPSC 命令队列 + 反馈线程 + IO 线程 + mip 覆盖 + 内存预算 | 树内 |
| PR #88934 | Mesh Shader support（Vulkan + D3D12） | **open** |
| PR #117836 | Implement MeshRD for RenderingDevice buffers | **open** |
</details>

<details>
<summary><b>C. 社区：addon / module / fork / demo（可直接读的"前人怎么做"）</b></summary>

**地形与流式（最接近 World Partition 的实践）**
- `TokisanGames/Terrain3D` —— clipmap LOD 地形，最大规模约 16km×16km，带碰撞与纹理流式，是目前 Godot 侧最"工程化"的实现。
- `mohsenph69/Godot-MTerrain-plugin`（1000+★，GDExtension）—— 四叉树 LOD 地形，宣传 "highly optimized / GPU driven"。
- `Zylann/godot_voxel`（3900+★）—— **C++ engine module**（不是 addon！），内含地形/网格分块与流式，是"改引擎做流式"的最佳样本。
- `Zylann/voxelgame`（770+★）—— 上面的完整世界原型。
- `ozzr/godot_terrain` —— **custom Godot build + terrain module**（fork 路线的现成例子）。
- `Relintai/voxelman` —— C++ engine module 体素引擎。
- `cuberact/godot-cuberact-planet-chunked-lod` —— chunked LOD 星球。

**World Partition 雏形 / 开放世界框架**
- `DigitallyTailored/Godot-Open-World-Database`（400+★）—— 按 chunk 自动加载/保存的开放世界数据库 addon。
- `SlashScreen/chunx` —— Godot 4 开放世界流式插件。
- `SlashScreen/skelerealms`（480+★）—— 开放世界 RPG 框架。
- `04AR/Godot_Chunker` —— 网格分区器，明确目标是"降低总 drawcall"。
- `redhoot-dev/Godot-Open-World-Demo`（230+★）—— 较早的开放世界 demo。
- `KarlJosvah/godot_level_streaming`、`Boyquotes/godot-streaming-addons`。

**"Nanite 类"实验**
- `AtlerGibby/GodotNanite`（82+★）—— Godot 4.2 里用 GDScript 做 mesh streaming + chunk-based LOD；读它的结论比读代码更有价值（GDScript/Node 承载不了这件事）。
- `HammerVala/NaniteGodot`、`Cyan232/godot-nanite`。
- `parth-satija/Vertex-HGT` —— **GPU-driven 地形系统**（GDExtension 路线）。

**LOD 工具史**
- `godot-extended-libraries/godot-lod`（3.x addon，247★）—— 说明"逐节点 visibility 脚本"能走多远，以及为什么不够。
- `R3X-G1L6AME5H/Godot-LOD-Manager`、`OBKF/Godot-LOD-Manager`。
</details>

---

# 五、给你的落地路线图（按投入产出排序）

| 阶段 | 做什么 | 改引擎? | 产出 |
|---|---|---|---|
| **P0（1–2 月）** | cell 调度器 + `load_threaded_*` + 分帧 `add_child`；**环境批量物件直接用 `MultiMesh`** 而不是 Node | 否 | 可跑的开放世界原型，摸清 cell 尺寸/预算/落帧预算 |
| **P1（2–4 月）** | GPU 侧实例剔除：compute 写 MultiMesh instance + command buffer，`draw_list_draw_indirect`，大 AABB + `IGNORE_OCCLUSION_CULLING` | 否（GDExtension 即可，因为 RID 已暴露） | 剔除与 LOD 搬到 GPU，drawcall 大幅下降 |
| **P2（3–6 月）** | `modules/world_partition/`（抄 `texture_streaming` 骨架）+ 离线 HLOD 烘焙器（`meshopt_simplify`） | **是**（module，rebase 成本低） | 真正的分区 + 自动 HLOD proxy，代理合并整个 cell |
| **P3（6–12 月）** | `RendererSceneCull` 加实例组/聚合 HLOD；`HZB` 搬 GPU；补 `draw_indirect_count` | **是**（servers/） | CPU 剔除开销可控到 10⁶ 实例；GPU 两阶段遮挡 |
| **P4** | 新 renderer（`rendering_method = gpu_driven`）：GPU scene + bindless + meshlet（+ 等 #88934 mesh shader） | **是**（大面积） | Nanite-class，长期跟 master rebase |

**三个决策建议：**
1. **别在 `forward_clustered` 里打补丁做 GPU-driven**——那是维护地狱，且会被上游持续冲突。新 renderer 更符合 Godot 现有多渲染器架构。
2. **HLOD 的收益 90% 来自"多物件合并成一个 proxy"，而不是"自动生成 LOD"**。先把 cluster/actor 层（P3 的 `InstanceGroup`）做出来，再谈简化算法。
3. **盯住 #88934（mesh shader）和 #13652（indirect count）**——它们合入时间直接决定 P4 的可行性与工作量。

---

需要我针对其中某一块继续深挖吗？例如：
- 把 `modules/texture_streaming/` 的线程/命令队列架构逐段拆成 World Partition module 的骨架代码；
- 或者写出 P1 那个"compute 剔除 → MultiMesh indirect"的完整 RD 调用序列（含命令缓冲 stride 布局与 `instance_offset` 语义）。

# ONNX静态图展开过程

### 🧑‍💻 User

onnx的静态图展开过程是怎么样的？

---

### 🤖 Assistant

**ONNX 的静态图展开（构建/导出 + 后处理）过程，本质上是把框架中的模型执行路径固定成一张有向无环（或含有限控制流）的计算图（GraphProto），再通过形状推断、常量折叠等步骤完善与简化该图。**

ONNX 模型是**静态计算图**表示：节点（算子）、边（张量/Value）、权重（initializers）、输入输出在导出时基本固定，便于优化、跨框架部署和推理引擎执行。与 PyTorch eager 动态图不同，动态控制流在导出时通常被“展开/固化”（tracing 按一次执行路径记录，或保留 `If`/`Loop` 等控制流节点）。

### 1. 整体流程概览

```
模型定义（PyTorch/TF 等）
    ↓
准备 dummy input + eval 模式
    ↓
Tracing / Scripting（执行路径展开为静态节点序列）
    ↓
算子映射到 ONNX opset → 生成 ModelProto / GraphProto
    ↓
形状/类型推断（Shape & Type Inference）
    ↓
常量折叠 + 图优化（Constant Folding、死代码消除、融合等）
    ↓
（可选）进一步简化 / 函数展开 / 后端 lowering
    ↓
可部署的静态 ONNX 图
```

“展开”主要体现在：
- 把动态 Python/控制流执行路径记录并固化为固定节点列表。
- 可静态确定的循环可 unroll。
- 复合算子/Function 可展开为更细粒度节点。
- 常量子图被求值并替换（折叠，实质是静态计算展开）。

### 2. 核心展开步骤（以 PyTorch → ONNX 为例，最常见）

#### （1）准备阶段
- 模型设为 `model.eval()`，关闭 dropout/batchnorm 训练行为。
- 准备符合真实输入 shape/dtype 的 dummy input（或动态轴示例）。
- 选择 opset 版本（影响可用算子与语义）。

#### （2）Tracing 展开（最核心的“静态化”）
`torch.onnx.export` 内部主要依赖 tracing（`torch.jit.trace` 或类似机制）：

- 用 dummy input **实际跑一遍**前向。
- 记录所有执行到的 ATen 操作、数据依赖、属性。
- 将这次执行路径**展开**成线性的（或带固定控制流的）节点序列。
- 动态 shape、数据依赖的控制流只保留**本次实际走的路径**（其他分支丢失，这是 tracing 的局限）。
- 得到 TorchScript 图，再映射为 ONNX 节点。

**Scripting**（`torch.jit.script`）可更好保留 `if`/`for` 结构，再导出，适合需要控制流的模型。

结果：一张固定拓扑的图。循环若 trip count 静态可知，部分工具/后端可进一步 unroll（真正的循环展开）。

#### （3）映射为 ONNX IR
- 算子名、属性、输入输出映射到对应 ONNX op（受 opset 约束）。
- 权重变成 `initializer`。
- 中间张量用 `value_info` 描述（此时 shape 可能还不全）。
- 生成 `ModelProto`（含 `GraphProto`、opset import、metadata 等）。

图的基本结构：
- **nodes**：`NodeProto` 列表（`op_type`、inputs、outputs、attributes）。
- **initializers**：常量/权重。
- **input / output**：图边界。
- **value_info**：中间张量类型与 shape（推断后填充）。

#### （4）形状与类型推断（完善静态信息）
使用 `onnx.shape_inference.infer_shapes`（或 Runtime 内置）：
- 从已知输入 shape/type 沿数据流传播。
- 推断中间节点输出 shape、elem_type。
- 支持部分动态维（用 symbolic dim 或动态轴）。
- 失败时部分 shape 保持 unknown（影响后续优化）。

这让图从“拓扑静态”进一步变成“shape 尽量静态”。

#### （5）常量折叠与图优化（静态计算展开 + 简化）
- **Constant Folding**：识别输入全为常量的子图，在导出/优化阶段直接计算出结果，用 `Constant` 或 initializer 替换，删除原节点。这是典型的“静态展开求值”。
- 死代码消除、公共子表达式消除。
- 算子融合（如 Conv+BN+ReLU）。
- 冗余节点删除、identity 消除等。
- 工具：ONNX Runtime Graph Transformer、`onnx-simplifier`、`onnxoptimizer` 等在中文社区很常用。

经过这些，图变得更小、shape 更确定、推理更快。

#### （6）控制流与复合算子的处理
- ONNX 原生支持 `If`、`Loop`、`Scan` 等，可保留动态性（不完全静态）。
- 若循环次数/条件可静态确定，可**unroll/展开**成重复节点序列（部分优化器/后端做）。
- ONNX Function（复合函数）可在加载或优化时**inline 展开**为基本算子图。
- 后端（ORT、TensorRT 等）还可能做进一步 lowering/expansion（把高层 op 拆成更底层 kernel 序列）。

### 3. 运行时侧的继续处理（加载后）
ONNX Runtime 等引擎加载后还会：
- 再次图转换/优化（level 分级）。
- 算子融合、布局转换、内存规划。
- 按 Execution Provider（CPU/GPU/TensorRT…）做特定 expansion 或替换。
- 最终变成可执行的 kernel 调度（图本身仍是静态调度的主体）。

### 4. 简单导出示例（PyTorch）

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

model = SimpleModel().eval()
dummy = torch.randn(1, 10)

torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}  # 可选动态轴
)
# 导出后可用 onnx.shape_inference、onnx-simplifier 等进一步处理
```

导出后的图就是“展开”后的静态表示：节点序列固定，权重为 initializer，中间 shape 经推断补全。

### 5. 关键注意点与局限
- **Tracing 的路径敏感性**：只展开实际执行路径，数据依赖的 if/循环可能不正确，需用 scripting 或手动处理控制流。
- **动态 shape / 控制流**：不完全静态时保留 symbolic 或 `Loop`/`If`，推理引擎支持程度不同。
- **Opset 与兼容性**：不同 opset 算子语义/可用性不同，展开结果受影响。
- **验证**：用 `onnx.checker`、Netron 可视化、ORT 跑通 + 数值对比。
- 工具链：`torch.onnx`、`onnx`、`onnxruntime`、`onnx-simplifier`、Netron、onnxoptimizer 等。

### 6. 小结
ONNX 静态图“展开”= **tracing/scripting 固化执行路径 + 算子映射成 GraphProto + shape/type 推断 + 常量折叠/优化（含可能的 loop unroll 与 function inline）**。最终得到一张（尽量）固定的、可高效优化与部署的计算图。动态部分要么被本次执行路径固定，要么显式用控制流节点保留。

若你指的是特定场景（例如 ORT 内部 Graph Transformer 的某个 pass、Loop 的 unroll、Function 展开、或与 TensorRT/TVM 的 lowering 对比），可以补充细节，我可以针对性深入。
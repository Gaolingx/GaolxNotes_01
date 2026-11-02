# MATLAB 与 Simulink 集成四大方向详解

## 🧑‍💻 User

matlab在simulink之类的集成怎么做

---
## 😀 默认助手 (7)

这个问题有几种很不一样的答案，取决于你说的“集成”是指哪一层。我按四个方向梳理，你先看决策表定位，再看对应细节。

## 先定位：你要的是哪种“集成”

| 方向 | 典型诉求 | 主要手段 |
|---|---|---|
| A. MATLAB 代码 → Simulink 模型内 | 模型里有个算法想直接用 `.m` 写 | MATLAB Function 块、MATLAB System 块、S-Function |
| B. MATLAB → 驱动 Simulink | 脚本里跑仿真、扫参、批量跑 | `sim`、`SimulationInput`、`set_param`、`parsim` |
| C. Simulink/MATLAB → 外部系统 | 和 C/Python/其他仿真器联合 | 代码生成、Engine API、FMI/FMU、ROS |
| D. 部署/代码生成 | 模型或算法变成产品代码 | Simulink Coder / Embedded Coder / MATLAB Coder |

---

<details>
<summary><b>A. 把 MATLAB 代码放进 Simulink（最常问的）</b></summary>

### A1. MATLAB Function 块（首选）
路径：`User-Defined Functions → MATLAB Function`。双击写函数，输入输出即端口。

```matlab
function y = fcn(u, k)
%#codegen
    y = k * u / (1 + abs(u));
end
```

- 走 **代码生成子集**：不能用 `eval`、动态类型随意变、`cell` 受限；建议加 `%#codegen` 让编辑器提前报错。
- 要调不支持的函数（画图、读文件）用 `coder.extrinsic('plot')`，但**只在 Normal 仿真模式有效**，加速/代码生成会失败。
- 每次调用有函数开销，采样快的模型里注意性能。

### A2. MATLAB System 块（System object）
写 `matlab.System` 子类，适合**有状态**、需要 `step`/`reset`/`release` 语义的算法，也支持代码生成。

```matlab
classdef MyFilter < matlab.System
    properties (Access = private)
        state = 0;
    end
    methods (Access = protected)
        function y = stepImpl(~, u)
            y = 0.9 * u;   % 这里放你的状态更新
        end
        function resetImpl(obj)
            obj.state = 0;
        end
    end
end
```

### A3. 其它老式/特殊块（了解即可）

| 块 | 说明 |
|---|---|
| `Interpreted MATLAB Function` | 每步调 MATLAB 解释器，**极慢**，不支持代码生成，仅调试用 |
| `Fcn` | 单行表达式，如 `sin(u(1)) + u(2)^2`，快但不能写多行逻辑 |
| `S-Function` / `Level-2 MATLAB S-Function` | 手写 `setup`/`Outputs`/`Derivatives`，实现 $\dot{x}=f(t,x,u),\ y=g(t,x,u)$ 这类自定义动态，灵活性最高、成本最高 |

### A4. 复用已有 `.m` 文件
把 `.m` 文件放到模型同目录或 MATLAB 路径，MATLAB Function 块里直接调用；若要多个块共享，做成**库（Library）**或**引用模型（Model Reference）**。
</details>

<details>
<summary><b>B. 用 MATLAB 控制 Simulink 仿真</b></summary>

最基础的脚本驱动：

```matlab
load_system('myModel');
set_param('myModel/Gain1','Gain','5');
out = sim('myModel','StopTime','10');
y = out.yout;              % 记录信号
```

工程化写法（推荐，便于并行和复用）：

```matlab
in = Simulink.SimulationInput('myModel');
in = in.setVariable('Kp', 2.5) ...
       .setBlockParameter('myModel/Gain1','Gain','2.5') ...
       .setModelParameter('StopTime','10');
out = sim(in);

% 批量/并行扫参
ins = repmat(Simulink.SimulationInput('myModel'),1,5);
for i = 1:5
    ins(i) = ins(i).setVariable('Kp', i);
end
outs = parsim(ins, 'UseParallel', true);
```

要点：
- **变量作用域**：`setVariable` 写入的是模型工作区，比 `assignin('base',...)` 更干净，且支持并行。
- 参数/信号建议用 `Simulink.Parameter`、`Simulink.Signal`、`Simulink.Bus` 显式定义，便于代码生成和团队协作。
- 大型工程用 `Simulink.data.dictionary`（`.sldd`）管理数据，而不是散落在 base workspace。
</details>

<details>
<summary><b>C. 和外部系统集成</b></summary>

| 目标 | 做法 |
|---|---|
| **Python 调 MATLAB/Simulink** | `matlab.engine`（官方 Engine API）：`import matlab.engine; eng = matlab.engine.start_matlab(); eng.sim('myModel')` |
| **MATLAB 调 Python** | 直接 `py.numpy.array(...)`，或用 `pyenv` 指定解释器 |
| **MATLAB ↔ C/C++** | `coder.ceval` 调外部 C；反向用 MATLAB Compiler SDK 打成 `.dll`/`.so`（`libraryCompiler`） |
| **和别的仿真器联合** | **FMI/FMU**：Simulink 导出 `exportToFMU`，导入用 `FMU Import` 块，做联合仿真（co-simulation） |
| **ROS/ROS2** | ROS Toolbox，`ros2subscriber` / ROS 2 块库 |
| **实时/硬件在环** | Simulink Real-Time + Speedgoat；或 SIL/PIL 做处理器在环 |
| **Excel / 数据库 / OPC UA** | Spreadsheet Link、Database Toolbox、OPC Toolbox |

选型经验：**只要对方是“另一个进程里的仿真器”，优先 FMI**；**只要对方是 Python 应用代码，优先 Engine API**；**要嵌进 C 产品，走代码生成而不是接口调用**。
</details>

<details>
<summary><b>D. 代码生成与部署（集成到产品里）</b></summary>

- **MATLAB Coder**：`codegen myfun -args {0}` 把纯 MATLAB 函数生成 C/C++。
- **Simulink Coder**：`slbuild('myModel')` / `rtwbuild` 从模型生成 C。
- **Embedded Coder**：更细的控制（存储类、定点、AUTOSAR、优化），产品级。
- 关键前提：算法必须**代码生成兼容**（无动态内存、无解释执行函数），MATLAB Function 块里加 `%#codegen` 自查。
- 生成后通常配合 **SIL/PIL** 验证生成代码与仿真结果一致，这是“集成”最容易翻车的一步。
</details>

---

## 几个高频坑

1. **`coder.extrinsic` 在 Accelerator/Rapid Accelerator 下失效** —— 仿真模式一换就报错。
2. **MATLAB Function 块里用了不支持代码生成的函数**，Normal 模式能跑，一生成代码就挂。
3. **变量放 base workspace 还是 model workspace 分不清**，导致并行仿真互相污染。
4. **S-Function 里采样时间、过零检测（zero-crossing）没处理好**，出现代数环或步长爆炸。
5. **跨版本/跨工具链的 FMI**：FMU 的 FMI 版本（2.0 vs 3.0）和平台二进制要对齐。

---

## 需要我更具体的话

告诉我其中一点，我可以给可直接跑的完整例子：

- 你是要 **把一段 `.m` 算法塞进模型**（给我代码，我改成 MATLAB Function 块写法）？
- 还是 **脚本批量跑仿真/扫参**？
- 还是 **Simulink 和 Python / C / 另一个仿真器对接**？

另外说明你的 **MATLAB 版本**（R20xxa），因为 FMI、`parsim`、System 块的支持情况随版本差异较大。

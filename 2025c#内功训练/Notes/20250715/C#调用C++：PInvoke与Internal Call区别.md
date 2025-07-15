# C#调用C++：PInvoke与Internal Call区别

### 🧑‍💻 User

c#调用c++，internal call和pinvoke有什么区别？

---

### 🤖 Assistant

在 C# 中调用 C++ 代码时，**Internal Call（内部调用）** 和 **P/Invoke（Platform Invocation Services）** 是两种不同的机制，主要区别如下：

---

### 1. **P/Invoke（平台调用）**
- **定位**：用于调用**标准本地 DLL（如 Windows API 或第三方 C++ 库）**。
- **实现方式**：
  - 通过 `[DllImport]` 属性声明外部函数。
  - 运行时自动加载 DLL 并处理封送（Marshaling）。
- **特点**：
  - **无需修改运行时**：直接依赖 .NET 的 P/Invoke 机制。
  - **有性能开销**：跨越托管/非托管边界时需转换数据类型（封送），影响性能。
  - **适用性广**：标准跨平台方案（Windows/Linux/macOS）。
- **示例**：
  ```csharp
  using System.Runtime.InteropServices;
  
  public class PInvokeExample {
      [DllImport("MyNativeLib.dll", CharSet = CharSet.Auto)]
      public static extern int Add(int a, int b);
  }
  ```

---

### 2. **Internal Call（内部调用）**
- **定位**：用于 **.NET 运行时内部实现**（如 `mscorlib` 中的底层函数）。
- **实现方式**：
  - 通过 `[MethodImpl(MethodImplOptions.InternalCall)]` 标记方法。
  - 需在 CLR（Common Language Runtime）中**显式注册** C++ 函数。
- **特点**：
  - **深度集成运行时**：直接绑定到 CLR 内部，无额外封送开销。
  - **高性能**：跳过了 P/Invoke 的边界检查，速度更快。
  - **需修改运行时源码**：通常仅用于开发 .NET 运行时自身（如 CoreCLR/Mono）。
- **示例**：
  ```csharp
  public class InternalCallExample {
      [MethodImpl(MethodImplOptions.InternalCall)]
      public static extern string GetRuntimeVersion();
  }
  ```
  C++ 端注册（以 Mono 为例）：
  ```cpp
  #include <mono/metadata/object.h>
  
  MonoString* GetRuntimeVersion(MonoObject* obj) {
      return mono_string_new(mono_domain_get(), "1.0.0");
  }
  
  // 在初始化时注册
  mono_add_internal_call("InternalCallExample::GetRuntimeVersion", (void*)GetRuntimeVersion);
  ```

---

### 核心区别总结
| **特性**         | **P/Invoke**                            | **Internal Call**                     |
|------------------|----------------------------------------|---------------------------------------|
| **使用场景**     | 通用本地库调用（如 OpenCV、系统 DLL）   | .NET 运行时底层实现（如 `String` 类） |
| **性能**         | 有封送开销，较慢                       | 无额外开销，极速                      |
| **实现复杂度**   | 简单（声明 `[DllImport]` 即可）        | 复杂（需修改 CLR 源码并注册函数）     |
| **平台依赖性**   | 跨平台支持                             | 依赖特定运行时实现（如 CoreCLR/Mono） |
| **开发者权限**   | 应用层开发者可直接使用                 | 通常仅运行时开发者使用                |

---

### 如何选择？
- **99% 的场景用 P/Invoke**：  
  适用于调用标准 C++ DLL，无需深入运行时。
- **Internal Call 仅适用于**：  
  开发自定义 .NET 运行时（如 Mono/CoreCLR 贡献者），或对性能有极致要求且能控制运行时环境。

> 💡 **简单说**：  
> - 写普通应用 → **P/Invoke**  
> - 写 .NET 虚拟机 → **Internal Call**

---

### 🧑‍💻 User

为什么unity的monobehavior里面大量使用internal call而不是p/invoke？

---

### 🤖 Assistant

Unity 在 `MonoBehaviour` 等核心组件中大量使用 **Internal Call（内部调用）** 而非 P/Invoke，是出于性能、架构设计和对运行时的深度控制需求。以下是具体原因：

---

### 1. **极致的性能优化**
   - **高频调用场景**：  
     Unity 每帧需调用大量脚本方法（如 `Update()`、`Start()`），这些方法可能被**成千上万个 GameObject 同时调用**。
   - **Internal Call 的优势**：  
     - **零封送开销**：直接通过 CLR 内部跳转表调用 C++ 函数，**无参数/返回值封送（Marshaling）**。
     - **免边界检查**：跳过了 P/Invoke 的托管/非托管边界安全检查。
   - **对比 P/Invoke**：  
     即使是最简单的 `int Add(int a, int b)` 调用，P/Invoke 也会有 **~10-30 纳秒级开销**。在 60FPS 下每秒调用 10 万次 `Update()` 时，Internal Call 可节省 **数毫秒/帧** 的 CPU 时间。

---

### 2. **深度集成 Unity 运行时**
   - **需要直接操作引擎核心**：  
     `MonoBehaviour` 的方法（如 `GetComponent()`）需直接访问 Unity 的 **C++ 对象模型**（如 GameObject 的实体组件系统）。
   - **Internal Call 的权限**：  
     - 可直接操作 Unity **内存管理**（如托管对象与 C++ 对象的生命周期绑定）。
     - 可访问 **非公开引擎数据结构**（如场景树、物理引擎状态）。
   - **P/Invoke 的限制**：  
     只能通过公开的 C API 交互，无法直接操作引擎内部状态。

---

### 3. **避免 P/Invoke 的平台适配问题**
   - **跨平台复杂性**：  
     Unity 支持 20+ 平台（iOS、Android、WebGL 等），P/Invoke 在不同平台需处理：
     - DLL 命名差异（`.dll`/`.so`/`.dylib`）
     - 调用约定（`cdecl`/`stdcall`）
     - ABI 兼容性问题
   - **Internal Call 的解决方案**：  
     - Unity 修改了 Mono/IL2CPP 运行时源码，在**所有平台**以统一方式注册 Internal Call。
     - 例如：`Transform.position` 的 getter 在 iOS/Android/Windows 上使用同一套 Internal Call 实现。

---

### 4. **对运行时的定制化控制**
   - **修改 Mono 运行时**：  
     Unity 长期使用自定制的 Mono 版本，允许其：
     1. 添加专属的 Internal Call API（如 `ScriptingInvocationNoArgs`）。
     2. 优化 Internal Call 的分发机制（如虚函数调用）。
   - **IL2CPP 的替代方案**：  
     当 Unity 切换到 IL2CPP（AOT 编译）时：
     - Internal Call 被编译为 **C++ 函数直接调用**（无间接跳转）。
     - P/Invoke 则需通过 **复杂的桥接层**（P/Invoke Wrapper）。

---

### 5. **历史与架构惯性**
   - **早期性能瓶颈**：  
     Unity 1.x 时代（2005年左右）的 Mono P/Invoke 性能极差，Internal Call 是唯一高性能选择。
   - **架构延续性**：  
     即使现代硬件性能提升，但 Unity 的核心层已深度依赖 Internal Call 架构，重构成本过高。

---

### 对比示例：`Transform.position` 的获取
#### **P/Invoke 伪代码（实际不会这样实现）**
```csharp
[DllImport("UnityEngine", CallingConvention = CallingConvention.Cdecl)]
private static extern void GetTransformPosition(IntPtr transformPtr, out Vector3 position);

public Vector3 position {
    get {
        Vector3 pos;
        GetTransformPosition(this.nativePtr, out pos); // 需要封送指针和结构体
        return pos;
    }
}
```
> ⚠️ 每次调用涉及：托管到非托管切换 + `Vector3` 结构体的复制。

#### **Internal Call 真实实现（UnityEngine 源码）**
```csharp
public Vector3 position {
    [MethodImpl(MethodImplOptions.InternalCall)]
    get;
}
```
> ✅ 在 C++ 侧直接访问 `Transform` 对象内存，返回 `Vector3` 的指针引用（零复制）。

---

### 为什么不完全淘汰 Internal Call？
Unity 正在逐步迁移到 **IL2CPP + Burst** 技术栈：
1. **IL2CPP**：将 C# 编译为 C++，使 Internal Call 转为**静态绑定调用**（性能等同 C++ 函数调用）。
2. **Burst Compiler**：对数学计算等高频操作生成 SIMD 机器码，避开任何调用开销。
3. **混合使用策略**：
   - **引擎核心层**：仍用 Internal Call（如 `GameObject` 生命周期管理）。
   - **用户层**：推荐使用 P/Invoke 调用第三方库（如 OpenCV）。

---

### 总结：Unity 的选择本质是**性能与控制的权衡**
| **场景**               | 技术选择          | 原因                                                                 |
|------------------------|------------------|----------------------------------------------------------------------|
| 每帧调用的高频方法     | Internal Call    | 避免纳秒级开销的死亡叠加（60FPS × 10,000 对象 × 多方法 = 性能生死线） |
| 第三方库交互（如 FFmpeg） | P/Invoke         | 通用性优先，调用频率低可接受开销                                      |
| 数学计算（如矩阵运算）   | Burst + SIMD     | 终极优化：跳过任何调用，直接生成汇编                                  |

> 简单说：**Unity 用 Internal Call 是因为它要“直接驾驶引擎，而不是通过遥控器（P/Invoke）”**。

---

### 🧑‍💻 User

我发现Mono原生即支持Internal call(我们简称icall)，那么在Mono中直接执行unity assembly，然后将icall调用直接指向unity engine的native即是最直接的方式？

---

### 🤖 Assistant

您完全抓住了 Unity 运行时架构的核心！是的，**这正是 Unity 在 Mono 模式下实现脚本与引擎通信的核心机制**。让我们拆解这个流程，并解释为什么这种设计是高效且直接的：

---

### ✅ Unity 基于 Mono 的 Internal Call（icall）工作流
```mermaid
graph LR
    A[Unity C# Assembly] -->|声明 icall| B[MonoBehaviour.Update 等]
    B --> C[Mono 运行时]
    C -->|通过 icall 表跳转| D[Unity Engine Native C++]
    D -->|直接操作| E[GameObject/Transform 等]
```

#### 具体步骤：
1. **C# 侧声明 icall**  
   Unity 的 C# 类（如 `MonoBehaviour`）中包含标记为 `[MethodImpl(InternalCall)]` 的方法：
   ```csharp
   // UnityEngine.dll (C#)
   public class MonoBehaviour : Behaviour {
       [MethodImpl(MethodImplOptions.InternalCall)]
       private extern void Internal_Update();
   }
   ```

2. **Mono 运行时注册 icall**  
   Unity 修改 Mono 源码，在引擎初始化时注册 C++ 函数到 icall 表：
   ```cpp
   // Unity 修改的 Mono 源码 (C++)
   void mono_add_internal_call(const char* name, void* method) {
       // 将 "UnityEngine.MonoBehaviour::Internal_Update" 映射到 C++ 函数
       g_icall_table[name] = method; 
   }
   ```

3. **执行时直接跳转**  
   当 C# 调用 `Update()` 时：
   - Mono 虚拟机查找 icall 表 → 获得 C++ 函数指针
   - **直接执行对应的 C++ 函数**（无参数封送/边界检查）
   ```cpp
   // Unity Engine 中的实现 (C++)
   void MonoBehaviour_Internal_Update(MonoObject* thisPtr) {
       GameObject* obj = ((MonoBehaviour*)thisPtr)->nativeObject; // 直接访问关联的 C++ 对象
       obj->UpdateComponents(); // 调用引擎更新逻辑
   }
   ```

---

### ⚡️ 为什么这是 **最高效的方式**？
1. **零封送开销 (Zero Marshaling)**  
   - 直接传递 `thisPtr`（托管对象的内存地址），无需转换。
   - C++ 侧通过指针直接访问 Unity 的底层对象（如 `GameObject` 的 C++ 实例）。

2. **免去 P/Invoke 的边界代价**  
   - 对比 P/Invoke：需切换栈帧 + 复制参数 + 安全性检查（约 10-30 纳秒/调用）。
   - icall 本质是 **函数指针跳转**（约 1-2 纳秒），性能差 **10 倍以上**。

3. **深度访问引擎内部**  
   C++ 函数可直接操作：
   ```cpp
   // 示例：在 icall 中直接修改 Transform 位置
   void Transform_set_position(Transform* self, Vector3 pos) {
       self->position = pos;              // 直接写内存
       self->NotifyPhysicsSystem();        // 调用内部物理系统
       MarkSceneDirty(self->scene);        // 访问场景管理器
   }
   ```

---

### 🌍 Unity 如何保证跨平台一致性？
Unity 通过 **定制 Mono 运行时** 解决：
1. **统一 icall 注册**  
   在 `UnityEngine.dll` 初始化时，调用各平台的统一注册函数：
   ```csharp
   // 跨平台 icall 注册入口 (C#)
   [RuntimeInitializeOnLoadMethod]
   static void RegisterInternalCalls() {
       RegisterMonoBehaviourICalls(); // 向 Mono 注册所有 icall
   }
   ```

2. **平台抽象层**  
   Unity 的 C++ 引擎核心有平台无关的 API，icall 实现只需调用这些 API：
   ```cpp
   // 平台无关的 Transform 实现 (C++)
   #if UNITY_IOS || UNITY_ANDROID || UNITY_WINDOWS
   void Transform_SetPosition(Transform* t, Vector3 pos) {
       // 调用统一的引擎接口
       UnityCore::SetObjectPosition(t->id, pos);
   }
   #endif
   ```

---

### 🔧 关键优化：`ScriptingInvocation` 宏
Unity 在 Mono 源码中添加了 **专属优化宏**，加速 icall 分发：
```cpp
// Unity 修改的 Mono 源码 (calls.c)
#define SCRIPTING_INVOKE(method, ...) \
    if (is_unity_icall(method)) {     \
        unity_icall_func func = get_unity_icall(method); \
        func(__VA_ARGS__); // 直接跳转！无虚拟机开销 \
    } else { \
        mono_runtime_invoke(...); // 普通 C# 方法 \
    }
```
> 此优化使 `Update()` 等高频调用 **接近原生 C++ 速度**。

---

### ⚠️ 注意：IL2CPP 模式的转变
当使用 IL2CPP 时（现代 Unity 默认）：
1. C# 代码被编译为 C++
2. `[MethodImpl(InternalCall)]` 被替换为 **直接调用 C++ 函数**：
```cpp
// 由 IL2CPP 生成的代码 (C++)
void MonoBehaviour_Update(MonoBehaviour* self) {
    // 直接调用 Unity 的 C++ 实现（无跳转表！）
    UnityMonoBehaviour::Internal_Update(self);
}
```
> 此时 icall 进化为 **静态绑定**，性能甚至优于 Mono 模式。

---

### 总结：您的理解完全正确
Unity 在 Mono 模式下正是通过 **深度定制 Mono 的 icall 机制**，将 C# 脚本与引擎 C++ 核心直接焊接，实现了：
1. **纳秒级的高频调用性能**（应对 10 万+ GameObject/帧）✅  
2. **直接内存访问**（托管与非托管对象共享数据）✅  
3. **跨平台统一接口**（iOS/Android/PC 共用一套 icall）✅  

这种设计是 Unity 性能关键的根基，而您准确地指出了其本质：  
> **“通过 Mono icall 将 C# 方法直接映射到 Native 实现”** —— 这就是 Unity 脚本引擎的命脉所在。
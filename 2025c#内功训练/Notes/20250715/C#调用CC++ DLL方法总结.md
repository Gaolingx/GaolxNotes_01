# C#调用C/C++ DLL方法总结

### 🧑‍💻 User

c#调用c/c++dll有两种方式，一种是用c#声明DllImport，调用c++里面声明extern "C"的方法（属于p/invoke?），还有别的办法吗？

---

### 🤖 Assistant

在C#中调用C/C++ DLL，除了使用`DllImport`和`extern "C"`（即P/Invoke方式）外，还有以下几种常用方法：

---

### 1. **C++/CLI 包装器（推荐用于复杂对象）**
通过C++/CLI（托管C++）创建托管包装层，将非托管C++代码封装成.NET可直接调用的类库。

**步骤：**
1. 创建**C++/CLI类库项目**（Visual Studio中选“CLR类库”）。
2. 在C++/CLI项目中：
   - 引用原生C++头文件和LIB/DLL。
   - 编写托管包装类，桥接.NET和原生代码。
3. C#项目直接**引用C++/CLI生成的DLL**。

**示例：**
```cpp
// C++/CLI 包装类 (ManagedWrapper.cpp)
#pragma once
#include "NativeClass.h" // 原生C++头文件

namespace ManagedWrapper {
    public ref class MyDotNetClass {
    public:
        int Add(int a, int b) {
            return NativeClass::Add(a, b); // 调用原生方法
        }
    };
}
```
**C# 调用：**
```csharp
var obj = new ManagedWrapper.MyDotNetClass();
int result = obj.Add(1, 2); // 像普通.NET对象一样调用
```

**优点**：  
- 支持C++类、复杂数据类型、异常传递。
- 避免P/Invoke的类型转换问题。

**缺点**：  
- 需维护额外的C++/CLI项目。

---

### 2. **COM 互操作**
将C++代码封装为COM组件，C#通过COM接口调用。

**步骤：**
1. 用C++编写**COM组件**（实现`IDispatch`或自定义接口）。
2. 注册COM组件（`regsvr32`）。
3. C#项目中**添加COM引用**，IDE自动生成互操作程序集。

**C# 调用：**
```csharp
var comObj = new MyCOMComponentLib.MyClass();
int result = comObj.Add(1, 2);
```

**优点**：  
- 跨语言标准化方案。
- 支持事件、接口继承等高级特性。

**缺点**：  
- COM开发复杂，需处理GUID、注册等。
- 部署需注册（或使用免注册COM）。

---

### 3. **动态加载 + 函数指针（手动P/Invoke）**
使用`LoadLibrary` + `GetProcAddress`动态加载DLL，配合委托调用。

**示例：**
```csharp
using System.Runtime.InteropServices;

delegate int AddDelegate(int a, int b); // 定义委托签名

public class DllDynamicLoader {
    [DllImport("kernel32.dll")]
    private static extern IntPtr LoadLibrary(string path);

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetProcAddress(IntPtr hModule, string name);

    public static T GetFunction<T>(string dllPath, string funcName) {
        IntPtr hModule = LoadLibrary(dllPath);
        IntPtr funcPtr = GetProcAddress(hModule, funcName);
        return Marshal.GetDelegateForFunctionPointer<T>(funcPtr);
    }
}

// 调用示例
var addFunc = DllDynamicLoader.GetFunction<AddDelegate>("NativeLib.dll", "Add");
int result = addFunc(1, 2);
```

**优点**：  
- 运行时动态加载，灵活控制生命周期。
- 避免DLL依赖问题。

**缺点**：  
- 手动管理内存和委托。
- 需处理32/64位兼容性。

---

### 4. **第三方工具生成绑定**
使用工具自动生成P/Invoke代码或包装层：
- **[SWIG](https://www.swig.org/)**：自动生成多语言绑定（支持C#）。
- **[CppSharp](https://github.com/mono/CppSharp)**：基于LLVM的C++到C#绑定生成器。

**优点**：  
- 自动化处理复杂API。
- 减少手动包装错误。

---

### 选择建议：
| 场景                          | 推荐方式              |
|-------------------------------|-----------------------|
| 简单C函数                     | P/Invoke (`DllImport`) |
| 复杂C++类/高性能交互          | **C++/CLI 包装器**    |
| 遗留COM组件                   | COM互操作             |
| 运行时动态加载DLL             | 动态加载 + 委托       |
| 大型C++库的完整绑定           | SWIG/CppSharp         |

> **重要提示**：  
> - 跨平台项目推荐使用P/Invoke（通过`[DllImport]`支持Linux/macOS）。  
> - 确保DLL的平台（x86/x64）与C#项目匹配！
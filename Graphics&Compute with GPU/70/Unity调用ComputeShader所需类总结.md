# Unity调用ComputeShader所需类总结

### 🧑‍💻 User

unity调用computeshader需要哪些类？

---

### 🤖 Assistant

在 Unity 中使用 Compute Shader 主要涉及以下几个核心类：

---

### 1. **ComputeShader**
- **作用**：表示一个 Compute Shader 资源（`.compute` 文件）。
- **关键操作**：
  - `FindKernel(string name)`：获取内核函数（Kernel）的索引
  - `SetBuffer(int kernel, string name, ComputeBuffer buffer)`：绑定缓冲区
  - `SetTexture(int kernel, string name, Texture texture)`：绑定纹理
  - `SetInt/SetFloat/SetVector` 等：设置标量参数
  - `Dispatch(int kernel, int x, int y, int z)`：调度计算任务

```csharp
public ComputeShader shader; // Inspector 中拖入 .compute 文件
int kernel = shader.FindKernel("CSMain");
shader.Dispatch(kernel, 32, 32, 1); // 调度线程组
```

---

### 2. **ComputeBuffer**
- **作用**：在 CPU 和 GPU 间传递结构化数据（如数组、结构体）。
- **构造函数**：`ComputeBuffer(int count, int stride)`
  - `count`：元素数量
  - `stride`：单个元素大小（字节数），用 `System.Runtime.InteropServices.Marshal.SizeOf` 获取
- **关键方法**：
  - `SetData(Array data)`：从 CPU 上传数据
  - `GetData(Array data)`：从 GPU 下载数据
  - `Release()`：手动释放显存（重要！）

```csharp
struct DataStruct { public float value; }
DataStruct[] cpuData = new DataStruct[1024];
int stride = Marshal.SizeOf(typeof(DataStruct));
ComputeBuffer buffer = new ComputeBuffer(1024, stride);
buffer.SetData(cpuData); // 上传数据
shader.SetBuffer(kernel, "dataBuffer", buffer); // 绑定到Shader
buffer.Release(); // 使用后必须释放！
```

---

### 3. **RenderTexture**
- **作用**：存储计算结果（需开启随机写入）。
- **关键设置**：`enableRandomWrite = true`
- **创建方式**：
  ```csharp
  RenderTexture rt = new RenderTexture(512, 512, 0);
  rt.enableRandomWrite = true;
  rt.Create();
  shader.SetTexture(kernel, "Result", rt); // 绑定到Shader
  ```

---

### 4. **Graphics**
- **作用**：辅助类（用于非计算着色器操作）。
- **常用方法**：
  - `Graphics.Blit()`：全屏绘制时复制纹理
  - `Graphics.SetRenderTarget()`：设置渲染目标

---

### 使用流程示例
```csharp
using UnityEngine;

public class ComputeExample : MonoBehaviour {
    public ComputeShader computeShader;
    public RenderTexture outputTexture;

    void Start() {
        // 创建输出纹理
        outputTexture = new RenderTexture(256, 256, 0);
        outputTexture.enableRandomWrite = true;
        outputTexture.Create();

        // 设置Shader参数
        int kernel = computeShader.FindKernel("CSMain");
        computeShader.SetTexture(kernel, "Result", outputTexture);
        computeShader.SetFloat("Time", Time.time);

        // 调度计算 (256/8=32 线程组)
        computeShader.Dispatch(kernel, 32, 32, 1);
    }

    void OnDestroy() {
        outputTexture.Release(); // 销毁纹理
    }
}
```

---

### Compute Shader 文件示例（.compute）
```hlsl
// test.compute
#pragma kernel CSMain

RWTexture2D<float4> Result; // 可读写纹理
float Time;

[numthreads(8,8,1)] // 线程组大小
void CSMain (uint3 id : SV_DispatchThreadID)
{
    // 计算像素位置 (id.xy)
    float2 uv = id.xy / 256.0;
    float3 color = float3(sin(Time + uv.x), cos(Time + uv.y), 1);
    Result[id.xy] = float4(color, 1);
}
```

---

### ⚠️ 注意事项
1. **显存管理**：`ComputeBuffer` 和 `RenderTexture` 必须手动释放（`Release()` 或 `Destroy()`）。
2. **线程组配置**：`Dispatch(x,y,z)` 参数需与 `[numthreads(X,Y,Z)]` 匹配。
3. **数据对齐**：结构体在 GPU 内存中按 4 字节对齐（避免使用 `bool` 等非对齐类型）。
4. **编辑器支持**：在 Editor 中修改 Compute Shader 后需重新绑定参数。

通过组合这些类，可实现 GPU 通用计算（如粒子系统、物理模拟、图像处理等）。
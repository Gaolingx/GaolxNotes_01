# URP 14 RTHandle渲染问题解决

## Question 1

urp14 renderfeature有办法输出到一张RT上吗，我新建了一个RTHandle但是Framedebugger里看不到，如何解决？

## Answer 1

在URP 14中，使用RenderFeature将内容渲染到RTHandle需要确保正确创建、配置和使用RTHandle。以下是分步解决方案：

### **1. 正确创建RTHandle**
在RenderFeature中，使用`RTHandleSystem`分配RTHandle，并确保描述符与相机目标一致：

```csharp
private RTHandle m_CustomRT;

public override void Create()
{
    // 在创建时初始化RTHandle
    m_CustomRT = RTHandles.Alloc(
        name: "_CustomRT",
        descriptor: new RenderTextureDescriptor(
            width: 256, 
            height: 256, 
            GraphicsFormat.R8G8B8A8_UNorm, 
            0
        ),
        useDynamicScale: true // 根据相机缩放动态调整尺寸
    );
}
```

### **2. 在RenderPass中配置目标**
在自定义`ScriptableRenderPass`的`Configure`方法中设置渲染目标：

```csharp
public override void Configure(CommandBuffer cmd, RenderTextureDescriptor cameraTextureDescriptor)
{
    // 动态调整RTHandle尺寸以匹配相机目标
    RenderTextureDescriptor desc = cameraTextureDescriptor;
    desc.depthBufferBits = 0; // 不需要深度
    desc.msaaSamples = 1;
    m_CustomRT.ScaledRTSize = new Vector2Int(desc.width, desc.height);
    RenderingUtils.ReAllocateIfNeeded(ref m_CustomRT, desc, FilterMode.Bilinear, TextureWrapMode.Clamp);

    // 设置当前Pass的渲染目标
    ConfigureTarget(m_CustomRT);
    ConfigureClear(ClearFlag.Color, Color.clear);
}
```

### **3. 执行渲染命令**
在`Execute`方法中使用CommandBuffer进行绘制：

```csharp
public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
{
    CommandBuffer cmd = CommandBufferPool.Get("Custom Pass");

    // 示例：绘制一个红色全屏Quad
    cmd.SetRenderTarget(m_CustomRT);
    cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, m_Material, 0, 0);

    context.ExecuteCommandBuffer(cmd);
    CommandBufferPool.Release(cmd);
}
```

### **4. 将结果传递到后续流程**
若需将结果用于后续渲染，可在Pass结束后将RTHandle传递给材质：

```csharp
// 在Execute方法末尾添加：
cmd.SetGlobalTexture("_CustomRT", m_CustomRT);
```

### **5. 检查Frame Debugger**
- 在Unity编辑器中打开**Window > Analysis > Frame Debugger**。
- 展开渲染事件，找到你的RenderPass名称（如"Custom Pass"）。
- 确认其中有`SetRenderTarget`到`_CustomRT`，以及后续的`Draw Mesh`调用。

### **常见问题排查**
- **RTHandle未正确分配**：检查描述符参数，确保尺寸非零且格式有效。
- **未设置渲染目标**：确认在CommandBuffer中调用了`SetRenderTarget`或`ConfigureTarget`。
- **生命周期问题**：确保RTHandle在渲染期间未被释放，避免在单帧内重复分配。
- **未在材质中采样RT**：在后续Shader中，使用`sampler2D _CustomRT`采样纹理。

### **完整代码示例**
```csharp
public class CustomRenderFeature : ScriptableRendererFeature
{
    private CustomPass m_CustomPass;
    private RTHandle m_CustomRT;

    public override void Create()
    {
        m_CustomPass = new CustomPass();
        m_CustomRT = RTHandles.Alloc(
            name: "_CustomRT",
            useDynamicScale: true
        );
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(m_CustomPass);
    }

    private class CustomPass : ScriptableRenderPass
    {
        private Material m_Material;
        private RTHandle m_CustomRT;

        public CustomPass()
        {
            m_Material = CoreUtils.CreateEngineMaterial("Hidden/Internal-Colored");
        }

        public override void Configure(CommandBuffer cmd, RenderTextureDescriptor cameraTextureDescriptor)
        {
            RenderTextureDescriptor desc = cameraTextureDescriptor;
            desc.depthBufferBits = 0;
            RenderingUtils.ReAllocateIfNeeded(ref m_CustomRT, desc, FilterMode.Bilinear, TextureWrapMode.Clamp);
            ConfigureTarget(m_CustomRT);
            ConfigureClear(ClearFlag.Color, Color.black);
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get("Custom Pass");
            
            // 绘制红色全屏Quad
            m_Material.SetColor("_Color", Color.red);
            Blitter.BlitCameraTexture(cmd, m_CustomRT, m_CustomRT, m_Material, 0);

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
    }
}
```

### **验证结果**
- 在Frame Debugger中找到你的Pass，确认`SetRenderTarget`指向`_CustomRT`。
- 若仍不可见，尝试将RTHandle Blit到屏幕：在Pass结束后添加`Blitter.BlitCameraTexture(cmd, m_CustomRT, renderer.cameraColorTargetHandle);`。

通过上述步骤，RTHandle的内容应能正确渲染并在Frame Debugger中可见。
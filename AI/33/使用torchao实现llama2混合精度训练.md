# 使用torchao实现llama2混合精度训练

### 🧑‍💻 User

既然pytorch lightning原生不支持fp8 dtype训练，那要如何使用nvfp4 fwd+mxfp8 bwd混合精度训练，模型架构是llama2，已知torch ao包含fp4/8mm加速算子

---

### 🤖 Assistant

在PyTorch Lightning (PTL) 原生不支持这种极端混合精度（Asymmetrical Precision）的情况下，要实现 **NVFP4 Forward + MXFP8 Backward** 的 LLaMA-2 训练，核心思路是**绕过 PTL 的全局精度管理机制**。

你需要将 PTL 的全局精度设置为基础高精度（如 `bf16-true`），然后通过自定义 `torch.autograd.Function` 并在底层调用 `torchao` 的算子，来手动接管前向和反向传播的张量转换与矩阵乘法（Matrix Multiplication, MM）。

以下是实现这一前沿架构的具体步骤：

### 1. 核心数学逻辑与 `torch.autograd.Function`

在线性层 $Y = XW^T$ 中，我们需要在前向和反向过程中分别进行不同精度的量化和矩阵乘法。

*   **前向传播 (Forward):** 输入 $X$ 和权重 $W$ 被量化为 NVFP4，进行矩阵乘法后，反量化回 BF16。
*   **反向传播 (Backward):** 梯度 $dY$、输入 $X$ 和权重 $W$ 参与计算。计算 $dX = dY \cdot W$ 和 $dW = dY^T \cdot X$ 时，将参与运算的张量量化为 MXFP8 进行计算。

我们需要编写一个自定义的 Autograd Function 来实现这一点：

```python
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd, custom_bwd
# 假设 torchao 提供了以下伪代码 API，具体视你使用的 torchao 版本而定
# import torchao.ops as ao_ops 

class Fp4FwdFp8BwdLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.bfloat16) # 强制输入为 bf16
    def forward(ctx, input, weight, bias=None):
        # 1. 保存高精度张量用于反向传播 (或者保存量化后的以节省显存，视策略而定)
        ctx.save_for_backward(input, weight, bias)
      
        # 2. 前向 NVFP4 量化与计算
        # 注意: 实际 torchao API 需要处理 scaling factors (scales, zeros)
        # 伪代码: input_fp4 = ao_ops.quantize_nvfp4(input)
        # 伪代码: weight_fp4 = ao_ops.quantize_nvfp4(weight)
        # 伪代码: output = ao_ops.nvfp4_mm(input_fp4, weight_fp4)
      
        # 这里用占位函数表示调用 torchao 的 NVFP4 算子
        output = torchao_nvfp4_linear_fwd(input, weight) 
      
        if bias is not None:
            output += bias
          
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # 1. 将梯度量化为 MXFP8
        # 伪代码: grad_output_fp8 = ao_ops.quantize_mxfp8(grad_output)
      
        # 2. 计算输入梯度: dX = dY * W
        if ctx.needs_input_grad[0]:
            # 伪代码: weight_fp8 = ao_ops.quantize_mxfp8(weight)
            # grad_input = ao_ops.mxfp8_mm(grad_output_fp8, weight_fp8)
            grad_input = torchao_mxfp8_linear_bwd_data(grad_output, weight)

        # 3. 计算权重梯度: dW = dY^T * X
        if ctx.needs_input_grad[1]:
            # 伪代码: input_fp8 = ao_ops.quantize_mxfp8(input)
            # grad_weight = ao_ops.mxfp8_mm(grad_output_fp8.transpose(), input_fp8)
            grad_weight = torchao_mxfp8_linear_bwd_weight(grad_output, input)

        # 4. 计算偏置梯度
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        return grad_input, grad_weight, grad_bias

# 包装成 nn.Module
class MixedPrecisionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return Fp4FwdFp8BwdLinearFunction.apply(input, self.weight, self.bias)
```

### 2. 替换 LLaMA-2 模型中的线性层

在将模型放入 PTL 之前，你需要递归遍历 LLaMA-2 模型，将其原生的 `nn.Linear` 替换为你自定义的 `MixedPrecisionLinear`。

<details>
<summary><b>点击展开：模型层替换代码</b></summary>

```python
def replace_linear_with_mixed_precision(model, target_classes=(nn.Linear,)):
    """
    递归替换模型中的 Linear 层
    """
    for name, module in model.named_children():
        if isinstance(module, target_classes):
            # 获取原层参数
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
          
            # 创建新层
            new_layer = MixedPrecisionLinear(
                in_features, out_features, bias=has_bias, 
                device=module.weight.device, dtype=module.weight.dtype
            )
          
            # 拷贝权重 (确保是在高精度下拷贝)
            new_layer.weight.data.copy_(module.weight.data)
            if has_bias:
                new_layer.bias.data.copy_(module.bias.data)
              
            # 替换原层
            setattr(model, name, new_layer)
        else:
            # 递归子模块
            replace_linear_with_mixed_precision(module, target_classes)
    return model

# 使用示例 (假设使用 HuggingFace Transformers)
from transformers import LlamaForCausalLM, LlamaConfig

config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LlamaForCausalLM(config)

# 执行替换
model = replace_linear_with_mixed_precision(model)
```
</details>

### 3. 配置 PyTorch Lightning Trainer

在 PTL 中，**不要设置 `precision="fp8"`**。因为底层的张量转换已经在你的自定义 Autograd Function 中完成了。你需要将 PTL 设置为 `bf16-true`（推荐）或 `32-true`，让 PTL 以纯 BF16 模式运行梯度缩放和优化器状态更新，而具体的 FP4/FP8 降级发生在算子内部。

<details>
<summary><b>点击展开：PTL Trainer 设置代码</b></summary>

```python
import pytorch_lightning as pl

class LlamaLightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        # 模型内部会自动调用 Fp4FwdFp8BwdLinearFunction
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # 优化器状态保持在高精度 (FP32)
        return torch.optim.AdamW(self.model.parameters(), lr=1e-5)

# 初始化 Trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices="auto",
    strategy="deepspeed_stage_2", # 建议搭配 DeepSpeed 或 FSDP 处理 LLaMA
    precision="bf16-true",        # 关键点：让 PTL 认为是在做纯 BF16 训练
    max_epochs=3
)

lightning_model = LlamaLightningModule(model)
# trainer.fit(lightning_model, train_dataloader)
```
</details>

### 4. 需要特别注意的坑 (Gotchas)

1.  **Scaling Factors (缩放因子):** 
    FP4 和 FP8 的动态范围非常有限。`torchao` 的底层算子（如 `cublasLt` 包装）通常要求提供缩放因子（`amax`, `scale`）。在上述 Autograd 函数中，你必须在调用算子前实时计算张量的 `max` 并得出缩放因子，或者使用 Delayed Scaling（延迟缩放，利用前一个 step 的 amax）。
    *公式参考：* $X_{\text{scaled}} = \text{clip}(\text{round}(X / S), \text{MIN}, \text{MAX})$
2.  **内存布局 (Memory Layout):** 
    NVIDIA 的 Tensor Cores 对 FP8/FP4 有严格的内存布局要求（通常要求连续的内存块，甚至特定的维度对齐，如 16 的倍数）。在使用 `torchao` 进行 `cast` 时，务必确保张量是 `.contiguous()` 的。
3.  **梯度累积与优化器:** 
    既然反向传播输出的是 BF16/FP32 的梯度（在 Autograd 最后一步由 FP8 累加结果转换而来），你的优化器（如 AdamW）依然会接收高精度梯度。这意味着优化器状态（Momentum, Variance）依然占用高显存。如果需要极致省显存，需结合 8-bit Optimizer。
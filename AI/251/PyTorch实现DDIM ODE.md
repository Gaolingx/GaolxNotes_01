# PyTorch实现DDIM ODE

### 🧑‍💻 User

在pytorch中，如何用x0-pred ode实现ddim？

---

### 🤖 Assistant

在PyTorch中使用DDIM(Denoising Diffusion Implicit Models)时，x0-prediction(预测原始数据)是实现的一种方式。我会为你解释DDIM的原理，并提供完整的实现代码。

## DDIM基本原理

DDIM是扩散模型的一种变体，它允许我们进行更少步骤的采样。与标准DDPM不同，DDIM通过一个非马尔可夫过程来生成数据，使得我们可以用更少的步数达到相似的生成效果。

在DDIM中，我们可以选择预测噪声(ε-prediction)或预测原始数据(x0-prediction)。这里我们主要关注x0-prediction。

## DDIM x0-prediction ODE实现

以下是DDIM x0-prediction在PyTorch中的完整实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        betas,
        loss_type="l2",
        rescale_timesteps=False,
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # 注册betas及其相关变量
        if isinstance(betas, str):
            betas = self.get_betas(betas)
      
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all(), "betas must be in (0, 1]"
      
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, axis=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod))
      
        # 计算后验方差
        self.register_buffer("posterior_variance", betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
      
        # 计算对数方差
        self.register_buffer("posterior_log_variance_clipped", torch.log(self.posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))
      
        # DDIM参数
        self.register_buffer("ddim_sqrt_alphas_cumprod", self.sqrt_alphas_cumprod)
        self.register_buffer("ddim_sqrt_one_minus_alphas_cumprod", self.sqrt_one_minus_alphas_cumprod)
        self.register_buffer("ddim_sigmas", torch.sqrt(betas))
      
    def get_betas(self, beta_schedule):
        if beta_schedule == "linear":
            scale = 1000 / self.num_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float64)
        elif beta_schedule == "cosine":
            return self.cosine_beta_schedule(self.num_timesteps)
        else:
            raise NotImplementedError(f"unknown beta schedule: {beta_schedule}")
  
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
  
    def q_mean_variance(self, x_start, t):
        """
        计算q(x_t | x_0)的均值和方差
        """
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
  
    def q_sample(self, x_start, t, noise=None):
        """
        扩散过程：q(x_t | x_0)的计算
        """
        if noise is None:
            noise = torch.randn_like(x_start)
      
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
      
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
  
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None):
        """
        计算p(x_{t-1} | x_t)的均值和方差（使用x0-prediction）
        """
        # 预测x_0
        pred_x0 = model(x, t)
      
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)
      
        if denoised_fn is not None:
            pred_x0 = denoised_fn(pred_x0)
      
        # 计算后验均值和方差
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_x0, x_t=x, t=t)
      
        return model_mean, self.posterior_variance
  
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        计算后验q(x_{t-1} | x_t, x_0)的均值和方差
        """
        at = self._extract(self.alphas_cumprod, t, x_t.shape)
        at_prev = self._extract(self.alphas_cumprod_prev, t, x_t.shape)
        sqrt1 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
      
        # 计算后验均值
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
      
        # 计算后验方差
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
      
        return posterior_mean, posterior_variance, sqrt1
  
    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None):
        """
        从p(x_{t-1} | x_t)采样
        """
        mean, variance = self.p_mean_variance(model, x, t, clip_denoised, denoised_fn)
        noise = torch.randn_like(x)
      
        # 不添加噪声（t=0时）
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
      
        return mean + nonzero_mask * torch.sqrt(variance) * noise
  
    def ddim_step(self, model, x, t, timestep_prev=None, eta=0.0):
        """
        DDIM采样步骤（使用x0-prediction）:
        x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1-alpha_{t-1}-sigma_t^2) * direction + sigma_t * noise
      
        其中eta控制随机性，当eta=0时相当于ODE，即确定性过程
        """
        # 预测x_0
        pred_x0 = model(x, t)
      
        # 获取alpha值
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
      
        if timestep_prev is None:
            timestep_prev = t - 1
      
        alpha_cumprod_t_prev = self._extract(self.alphas_cumprod_prev, t, x.shape)
        sqrt_alpha_cumprod_t_prev = self._extract(self.sqrt_alphas_cumprod_prev, x.shape)
      
        # 计算方向指向x_t
        direction = torch.sqrt(1 - alpha_cumprod_t_prev) * pred_x0
      
        # 计算噪声项
        noise = torch.randn_like(x)
        sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
      
        # 计算DDIM采样结果
        x_prev = sqrt_alpha_cumprod_t_prev * pred_x0 + direction + sigma_t * noise
      
        return x_prev
  
    def ddim_sample(self, model, shape, num_steps=50, eta=0.0, clip_denoised=True):
        """
        DDIM采样过程（ODE版本，eta=0）
      
        参数:
        - model: 训练好的扩散模型
        - shape: 生成样本的形状
        - num_steps: DDIM采样步数（少于原始扩散步数）
        - eta: 随机性参数，0表示确定性过程（ODE）
        - clip_denoised: 是否对预测的x0进行裁剪
      
        返回:
        - 生成的样本
        """
        # 计算采样的时间步
        time_steps = self.get_ddim_timesteps(num_steps)
      
        # 从纯噪声开始
        img = torch.randn(shape)
      
        # 遍历每个时间步
        for i, (t, timestep_prev) in enumerate(time_steps):
            t_tensor = torch.full((shape[0],), t, device=img.device, dtype=torch.long)
          
            # 使用DDIM步骤更新图像
            img = self.ddim_step(model, img, t_tensor, timestep_prev, eta)
          
            # 如果需要，对预测的x0进行裁剪
            if clip_denoised:
                img = img.clamp(-1, 1)
      
        return img
  
    def get_ddim_timesteps(self, num_steps):
        """
        获取DDIM采样的时间步序列
      
        参数:
        - num_steps: 采样步数
      
        返回:
        - 时间步列表，每个元素为(t, timestep_prev)
        """
        # 计算原始扩散过程的时间步间隔
        step_ratio = self.num_timesteps // num_steps
      
        # 创建等间隔的时间步
        timesteps = (torch.arange(num_steps, 0, -1) * step_ratio).long()
      
        # 创建时间步对
        timesteps_with_prev = []
        for i in range(len(timesteps)):
            t = timesteps[i]
            if i < len(timesteps) - 1:
                timestep_prev = timesteps[i+1]
            else:
                timestep_prev = None
            timesteps_with_prev.append((t, timestep_prev))
      
        return timesteps_with_prev
  
    def _extract(self, a, t, x_shape):
        """
        从a中提取特定时间步t的值，并调整形状以匹配x_shape
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
  
    def loss(self, x_start, t, noise=None):
        """
        计算损失函数
        """
        if noise is None:
            noise = torch.randn_like(x_start)
      
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
      
        # 使用模型预测x_0
        pred_x0 = self.model(x_noisy, t)
      
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_x0, x_start)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred_x0, x_start)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(pred_x0, x_start)
        else:
            raise NotImplementedError()
      
        return loss
```

## 如何使用上述代码

以下是使用上述DDIM实现的简单示例：

```python
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

# 定义一个简单的UNet模型
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 这里简化了UNet结构，实际应用中应该使用更完整的架构
        self.conv1 = nn.Conv2d(in_channels + 1, 32, kernel_size=3, padding=1)  # +1 for time embedding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
      
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
      
        self.time_embedding = nn.Linear(1, 1)
      
    def forward(self, x, t):
        # 时间嵌入
        t = t.float()
        t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t = self.time_embedding(t) / 1000.0
      
        # 扩展时间嵌入到与x相同的尺寸
        t = t.expand(-1, -1, x.shape[2], x.shape[3])
      
        # 连接输入和时间嵌入
        x = torch.cat([x, t], dim=1)
      
        # 下采样
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
      
        # 上采样
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
      
        # 输出
        x = torch.sigmoid(self.conv4(x))
      
        return x

# 设置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 10
timesteps = 1000  # 扩散过程的时间步数
ddim_steps = 20   # DDIM采样的时间步数

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建模型和扩散过程
model = SimpleUNet().to(device)
betas = torch.linspace(0.0001, 0.02, timesteps)
diffusion = GaussianDiffusion(model, betas).to(device)

# 优化器
optimizer = Adam(model.parameters(), lr=1e-4)

# 训练模型
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
      
        # 随机时间步
        t = torch.randint(0, timesteps, (images.shape[0],), device=device)
      
        # 计算损失
        loss = diffusion.loss(images, t)
      
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        if i % 100 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss {loss.item():.4f}")

# 使用DDIM生成样本
print("Generating samples using DDIM...")
with torch.no_grad():
    shape = (16, 1, 28, 28)  # 16个28x28的MNIST样本
    samples = diffusion.ddim_sample(model, shape, num_steps=ddim_steps, eta=0.0)
  
    # 可视化生成的样本
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ddim_samples.png')
    plt.show()
```

## DDIM x0-prediction与ε-prediction的区别

DDIM主要有两种预测方式：

1. **x0-prediction**：模型直接预测原始数据x0。这也是我们在代码中实现的方式。

2. **ε-prediction**：模型预测添加的噪声ε。这是原始DDPM论文中使用的方式。

这两种方式可以通过简单的数学关系相互转换：
- 如果模型预测x0，那么预测的噪声为：\(\hat{\epsilon} = \frac{x_t - \sqrt{\bar{\alpha}_t} \hat{x}_0}{\sqrt{1 - \bar{\alpha}_t}}\)
- 如果模型预测ε，那么预测的原始数据为：\(\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}\)

## DDIM的ODE解释

当eta=0时，DDIM采样过程变成了一个确定性的ODE（常微分方程）过程。这使得我们可以用更少的步数从噪声生成数据，同时保持与原始DDPM相似的分布。

数学上，DDIM的ODE形式可以表示为：
\[dx_t = f(x_t, t) dt\]

其中：
- \(x_t\) 是时间步t的数据
- \(f(x_t, t)\) 是由网络决定的向量场

这种确定性采样过程与随机采样（对应于SDE，随机微分方程）相比，有以下优点：
1. 更快的采样速度（可以用更少的步骤）
2. 更一致和可重复的采样结果
3. 更容易进行插值和操纵潜在空间

## 总结

以上代码展示了如何在PyTorch中实现DDIM的x0-prediction ODE版本。关键点包括：

1. 实现了扩散过程的前向和后验分布
2. 实现了DDIM采样步骤，特别是x0-prediction版本
3. 通过将eta参数设为0来实现确定性ODE采样
4. 提供了完整的训练和采样流程

这种方法在保持生成质量的同时，大大减少了采样所需的步骤数，使得扩散模型在实际应用中更加高效。
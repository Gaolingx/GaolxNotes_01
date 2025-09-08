# PyTorch训练状态保存方法总结

### 🧑‍💻 User

pytorch中，如何dump每一步的param gradient optim state？

---

### 🤖 Assistant

# PyTorch 中 Dump 参数、梯度和优化器状态的方法

在 PyTorch 中，您可以通过多种方式在训练过程中 dump 每一步的参数、梯度和优化器状态。以下是几种常用方法：

## 方法一：手动记录每个训练步骤

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os

# 创建保存目录
os.makedirs('./training_dumps', exist_ok=True)

def dump_training_state(step, model, optimizer, loss):
    """保存训练状态"""
    state_dict = {
        'step': step,
        'loss': loss.item(),
        'model_params': {name: param.detach().clone() for name, param in model.named_parameters()},
        'model_grads': {name: param.grad.detach().clone() if param.grad is not None else None 
                       for name, param in model.named_parameters()},
        'optimizer_state': optimizer.state_dict()
    }
    
    # 保存到文件
    with open(f'./training_dumps/step_{step}.pkl', 'wb') as f:
        pickle.dump(state_dict, f)
    
    return state_dict

# 示例训练循环
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for step in range(100):
    # 模拟训练数据
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 在优化器步骤之前保存状态
    if step % 10 == 0:  # 每10步保存一次
        dump_training_state(step, model, optimizer, loss)
    
    # 更新参数
    optimizer.step()
```

## 方法二：使用钩子（Hooks）自动记录梯度

```python
import torch
import torch.nn as nn

# 存储梯度
gradients = {}

def save_grad(name):
    """保存梯度的钩子函数"""
    def hook(grad):
        gradients[name] = grad.detach().clone()
    return hook

# 注册梯度钩子
model = nn.Linear(10, 1)
for name, param in model.named_parameters():
    param.register_hook(save_grad(name))

# 训练循环中
for step in range(100):
    # 训练步骤...
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 此时gradients字典中已保存所有梯度
    print(f"Step {step} gradients:", gradients)
    
    optimizer.step()
```

## 方法三：使用回调函数的高级实现

```python
import torch
import torch.nn as nn
import json
import numpy as np

class TrainingDumper:
    def __init__(self, model, optimizer, save_dir='./training_dumps'):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储历史记录
        self.history = []
    
    def dump_step(self, step, loss):
        """保存当前步骤的状态"""
        # 获取参数和梯度
        params = {name: param.detach().cpu().numpy() 
                 for name, param in self.model.named_parameters()}
        
        grads = {name: param.grad.detach().cpu().numpy() if param.grad is not None else None
                for name, param in self.model.named_parameters()}
        
        # 获取优化器状态
        optim_state = self.optimizer.state_dict()
        
        # 转换为可JSON序列化的格式
        state = {
            'step': step,
            'loss': float(loss),
            'params': {k: v.tolist() for k, v in params.items()},
            'grads': {k: v.tolist() if v is not None else None for k, v in grads.items()},
            'optimizer': {
                'state': {k: {k2: v2.tolist() if isinstance(v2, torch.Tensor) else v2 
                             for k2, v2 in v.items()} 
                         for k, v in optim_state['state'].items()},
                'param_groups': optim_state['param_groups']
            }
        }
        
        # 保存到历史记录
        self.history.append(state)
        
        # 保存到文件
        with open(f'{self.save_dir}/step_{step}.json', 'w') as f:
            json.dump(state, f, indent=2)
            
        return state

# 使用示例
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
dumper = TrainingDumper(model, optimizer)

for step in range(100):
    # 训练步骤...
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 保存状态
    if step % 10 == 0:
        dumper.dump_step(step, loss)
    
    optimizer.step()
```

## 方法四：使用PyTorch Lightning的回调

如果您使用PyTorch Lightning，可以利用其回调系统：

```python
import pytorch_lightning as pl
import torch

class DumpCallback(pl.Callback):
    def __init__(self, save_dir='./lightning_dumps'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在每个训练批次结束时调用"""
        if batch_idx % 10 == 0:  # 每10个批次保存一次
            # 保存参数和梯度
            state = {
                'step': trainer.global_step,
                'params': {name: param.detach().cpu().numpy() 
                          for name, param in pl_module.named_parameters()},
                'grads': {name: param.grad.detach().cpu().numpy() if param.grad is not None else None
                         for name, param in pl_module.named_parameters()}
            }
            
            # 保存到文件
            torch.save(state, f'{self.save_dir}/step_{trainer.global_step}.pt')

# 在Lightning训练中使用
trainer = pl.Trainer(callbacks=[DumpCallback()])
trainer.fit(model, train_loader)
```

## 注意事项

1. **性能考虑**：频繁保存状态会显著减慢训练速度并占用大量存储空间，请根据需要调整保存频率。
2. **内存管理**：使用`.detach().cpu().numpy()`将张量移至CPU并转换为NumPy数组可以减少GPU内存使用。
3. **选择性保存**：考虑只保存您关心的特定参数，而不是所有参数。
4. **序列化格式**：JSON适合小数据且需要可读性的情况，Pickle或PyTorch的保存格式适合大型张量。

选择哪种方法取决于您的具体需求、模型复杂度和存储约束。
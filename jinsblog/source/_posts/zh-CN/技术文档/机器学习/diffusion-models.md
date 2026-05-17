---
title: 扩散模型详解
date: 2026-01-14
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 深度学习, 生成模型, 扩散模型]
---

## 扩散模型原理

### 扩散过程概述

扩散模型通过：
- 前向扩散：逐步添加噪声
- 反向去噪：逐步恢复数据

### 前向扩散过程

逐步向数据添加噪声：
$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t$

其中 $\epsilon_t \sim \mathcal{N}(0, I)$。

### 反向去噪过程

学习逐步去噪：
$p(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$

神经网络学习去噪函数。

### 与其他生成模型对比

| 模型 | 原理 |
|------|------|
| VAE | 编码-解码 |
| GAN | 对抗训练 |
| 扩散模型 | 逐步去噪 |

### 扩散模型优势

| 优势 | 描述 |
|------|------|
| 训练稳定 | 无对抗不稳定 |
| 高质量生成 | 生成质量优秀 |
| 理论清晰 | 概率框架明确 |
| 灵活条件 | 易加条件控制 |

## DDPM算法

### DDPM原理

Denoising Diffusion Probabilistic Models（DDPM）：
- 固定前向过程
- 学习反向过程

### 前向扩散

**噪声调度**：$\beta_1, \beta_2, ..., \beta_T$

$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$

**直接跳转**：
$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$

其中 $\bar{\alpha}_t = \prod_{i=1}^t(1-\beta_i)$。

### 反向去噪

学习去噪网络 $\epsilon_\theta(x_t, t)$：
$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)) + \sigma_t z$

### 训练目标

**简化目标**：
$L = \mathbb{E}_{t, x_0, \epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$

预测添加的噪声。

### 噪声调度

| 类型 | 公式 |
|------|------|
| 线性 | $\beta_t$线性增长 |
| 余弦 | $\beta_t$余弦调度 |
| 自适应 | 根据数据调整 |

```python
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

## 条件扩散模型

### 条件生成

加入条件信息：
$p(x_{t-1}|x_t, c)$

### 条件注入方式

| 方式 | 描述 |
|------|------|
| 输入拼接 | 条件与x_t拼接 |
| 条件编码 | 条件编码器 |
| 交叉注意力 | 注意力注入 |
| Classifier-free | 无分类器引导 |

### Classifier-Free Guidance

同时训练条件和无条件模型：
$\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, t, c) + s(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t))$

**s为引导强度**。

```python
def classifier_free_guidance(model, x_t, t, condition, guidance_scale=7.5):
    # 无条件预测
    uncond_pred = model(x_t, t, None)
    # 条件预测
    cond_pred = model(x_t, t, condition)
    
    # 引导
    return uncond_pred + guidance_scale * (cond_pred - uncond_pred)
```

### 文本条件扩散

如Stable Diffusion：
- 文本编码器（CLIP）
- 交叉注意力注入
- Classifier-free引导

## 高效扩散模型

### 采样加速

扩散模型采样慢（需T步）。

| 方法 | 描述 |
|------|------|
| DDIM | 非马尔可夫采样 |
| DPM-Solver | 高效ODE求解 |
| 跳步采样 | 减少步数 |

### DDIM采样

非马尔可夫采样，可跳步：
$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t, t)$

```python
def ddim_sample(model, x, steps, eta=0):
    for i in reversed(range(steps)):
        t = torch.full((x.size(0),), i)
        
        eps = model(x, t)
        x0_pred = (x - sqrt(1 - alpha_bar[t]) * eps) / sqrt(alpha_bar[t])
        
        if i > 0:
            noise = torch.randn_like(x)
            x = sqrt(alpha_bar[i-1]) * x0_pred + \
                sqrt(1 - alpha_bar[i-1] - eta**2 * sigma[i]**2) * eps + \
                eta * sigma[i] * noise
    return x
```

### 潜在扩散模型（LDM）

在潜在空间扩散：
- 先编码到低维空间
- 在潜在空间扩散
- 再解码回像素空间

**优势**：计算效率高。

Stable Diffusion采用此架构。

## 扩散模型架构

### U-Net去噪网络

典型去噪网络：
- U-Net结构
- 时间步嵌入
- 自注意力/交叉注意力

```python
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # 下采样
        self.conv1 = ConvBlock(in_channels, 64, time_dim)
        self.conv2 = ConvBlock(64, 128, time_dim)
        self.conv3 = ConvBlock(128, 256, time_dim)
        
        # 上采样
        self.up1 = ConvBlock(256, 128, time_dim)
        self.up2 = ConvBlock(128, 64, time_dim)
        self.up3 = ConvBlock(64, out_channels, time_dim)
        
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        d1 = self.conv1(x, t_emb)
        d2 = self.conv2(d1, t_emb)
        d3 = self.conv3(d2, t_emb)
        
        u1 = self.up1(d3, t_emb)
        u2 = self.up2(u1 + d2, t_emb)
        u3 = self.up3(u2 + d1, t_emb)
        
        return u3
```

### 时间步嵌入

将时间步编码为向量：
$PE(t) = [\sin(t/10000^{2i/d}), \cos(t/10000^{2i/d})]$

```python
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

### 注意力机制

在U-Net中加入：
- 自注意力：处理全局信息
- 交叉注意力：注入条件

## 案例实践

### DDPM实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDiffusion:
    def __init__(self, model, timesteps=1000, beta_schedule='linear'):
        self.model = model
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
    
    def q_sample(self, x0, t, noise=None):
        """前向扩散"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t])
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x0, t):
        """训练损失"""
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        
        predicted_noise = self.model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    def p_sample(self, x, t):
        """单步去噪"""
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t])
        sqrt_recip_alphas_t = 1.0 / self.alphas[t]
        
        predicted_noise = self.model(x, t)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t > 0:
            noise = torch.randn_like(x)
            model_std = betas_t.sqrt()
            return model_mean + model_std * noise
        else:
            return model_mean
    
    def sample(self, shape):
        """完整采样"""
        x = torch.randn(shape)
        
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
        
        return x

# 使用
model = UNet(in_channels=1, out_channels=1, time_dim=256)
diffusion = GaussianDiffusion(model)

# 训练
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(epochs):
    for x0 in train_loader:
        t = torch.randint(0, diffusion.timesteps, (x0.size(0),))
        loss = diffusion.p_losses(x0, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 简化扩散模型

```python
class SimpleDiffusion(nn.Module):
    def __init__(self, model, steps=100):
        super().__init__()
        self.model = model
        self.steps = steps
        
    def add_noise(self, x, t):
        alpha = 1 - t / self.steps
        return alpha * x + (1 - alpha) * torch.randn_like(x)
    
    def forward(self, x):
        """训练：预测噪声"""
        t = torch.randint(1, self.steps, (x.size(0),))
        noisy_x = self.add_noise(x, t)
        predicted_noise = self.model(noisy_x, t)
        return predicted_noise
    
    def generate(self, batch_size):
        """生成"""
        x = torch.randn(batch_size, *self.input_shape)
        
        for t in reversed(range(1, self.steps)):
            t_batch = torch.full((batch_size,), t)
            noise_pred = self.model(x, t_batch)
            
            alpha = 1 - t / self.steps
            x = (x - (1 - alpha) * noise_pred) / alpha
            
            if t > 1:
                x = x + torch.randn_like(x) * 0.1
        
        return x
```

### 条件扩散实现

```python
class ConditionalDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
    
    def forward(self, x, condition, t):
        """条件去噪"""
        return self.model(x, t, condition)
    
    def train_step(self, x0, condition):
        """训练"""
        t = torch.randint(0, self.timesteps, (x0.size(0),))
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        
        # 随机丢弃条件（classifier-free）
        drop_mask = torch.rand(x0.size(0)) < 0.1
        condition[drop_mask] = None
        
        noise_pred = self.model(x_noisy, t, condition)
        return F.mse_loss(noise, noise_pred)
    
    def sample(self, condition, guidance_scale=7.5):
        """条件生成"""
        x = torch.randn(condition.size(0), *self.shape)
        
        for t in reversed(range(self.timesteps)):
            # 无条件
            uncond_pred = self.model(x, t, None)
            # 条件
            cond_pred = self.model(x, t, condition)
            
            # 引导
            noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            x = self.p_sample(x, t, noise_pred)
        
        return x
```

### 图像生成示例

```python
from diffusers import DDPMPipeline

# 使用预训练模型
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")

# 生成图像
images = pipeline(batch_size=4).images

# 保存
for i, img in enumerate(images):
    img.save(f"generated_{i}.png")
```

### Stable Diffusion使用

```python
from diffusers import StableDiffusionPipeline

# 加载模型
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
)

# 文本生成图像
image = pipeline("A beautiful sunset over mountains").images[0]
image.save("sunset.png")

# 多张
images = pipeline(["A cat", "A dog"], num_images_per_prompt=2).images
```

## 扩散模型应用

### 图像生成

| 应用 | 模型 |
|------|------|
| 文生图 | Stable Diffusion, DALL-E |
| 图生图 | img2img |
| 图像修复 | Inpainting |
| 超分辨率 | 扩散超分 |

### 其他生成

| 应用 | 描述 |
|------|------|
| 音频生成 | DiffWave, AudioLDM |
| 视频生成 | 视频扩散模型 |
| 3D生成 | 3D扩散模型 |

### 应用案例

```python
# 图像修复
def inpaint(pipeline, image, mask, prompt):
    result = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=50
    )
    return result.images[0]

# 图生图
def img2img(pipeline, init_image, prompt, strength=0.75):
    result = pipeline(
        prompt=prompt,
        image=init_image,
        strength=strength
    )
    return result.images[0]
```

## 扩散模型的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 高质量生成 | 生成质量优秀 |
| 训练稳定 | 无对抗不稳定 |
| 理论清晰 | 概率框架明确 |
| 灵活控制 | 易加条件 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 采样慢 | 需多步迭代 |
| 计算成本 | 推理时间长 |
| 内存占用 | 高分辨率需大内存 |

## 总结

扩散模型是重要的生成模型。核心内容包括：
- 前向扩散：逐步添加噪声
- 反向去噪：逐步恢复数据
- DDPM：基础扩散算法
- 条件扩散：Classifier-free引导
- 高效采样：DDIM、DPM-Solver
- U-Net架构：去噪网络设计

扩散模型生成质量高，是当前主流生成技术。

## 延伸阅读

- [变分自编码器](/2026/05/10/zh-CN/技术文档/机器学习/vae/)
- [生成对抗网络](/2026/05/10/zh-CN/技术文档/机器学习/gan/)
- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
- [概率论基础](/2026/05/10/zh-CN/技术文档/机器学习/probability-theory/)
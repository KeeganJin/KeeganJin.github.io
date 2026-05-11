---
title: 变分自编码器（VAE）
date: 2026-04-23
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 深度学习, 生成模型, VAE]
---

## VAE基本原理

### 生成模型概述

生成模型学习数据分布，生成新样本。

**目标**：
$P(X) = \int P(X|Z)P(Z)dZ$

### VAE的核心思想

变分自编码器（Variational Autoencoder）结合：
- 自编码器结构
- 变分推断
- 概率生成

**架构**：
```
输入 X → 编码器 → 潜在空间 Z → 解码器 → 重建 X'
```

### VAE与传统AE的区别

| 方面 | 传统AE | VAE |
|------|--------|-----|
| 潜在空间 | 确定点 | 概率分布 |
| 学习目标 | 重建误差 | ELBO |
| 生成能力 | 无 | 有 |
| 连续性 | 不连续 | 连续 |

## 变分推断基础

### 问题设定

目标是最大化数据对数似然：
$\log P(X) = \log \int P(X|Z)P(Z)dZ$

**困难**：积分难以直接计算。

### 变分推断

引入近似分布 $q(Z|X)$：
$\log P(X) \geq ELBO$

### ELBO推导

**推导过程**：
$\log P(X) = \log P(X)\int q(Z|X)dZ$
$= \int q(Z|X)\log P(X)dZ$
$= \int q(Z|X)\log\frac{P(X,Z)}{P(Z|X)}dZ$
$= \int q(Z|X)\log\frac{P(X,Z)}{q(Z|X)}dZ + \int q(Z|X)\log\frac{q(Z|X)}{P(Z|X)}dZ$
$= ELBO + KL(q(Z|X)||P(Z|X))$

由于KL散度 ≥ 0：
$\log P(X) \geq ELBO$

### ELBO分解

$ELBO = \mathbb{E}_{q(Z|X)}[\log P(X|Z)] - KL(q(Z|X)||P(Z))$

**两部分**：
- 重建项：$\mathbb{E}_{q}[\log P(X|Z)]$（重建准确）
- 正则项：$KL(q(Z|X)||P(Z))$（接近先验）

## VAE架构详解

### 编码器

编码器输出潜在分布的参数：
$q(Z|X) = \mathcal{N}(Z; \mu(X), \sigma^2(X))$

**网络结构**：
```
输入 X
  ↓
隐藏层 (全连接)
  ↓
μ(X) 和 σ(X) (两个输出)
  ↓
采样 Z = μ + σ·ε, ε~N(0,1)
```

### 解码器

解码器从潜在变量重建数据：
$P(X|Z)$

**输出**：
- 连续数据：$\mathcal{N}(X; \mu(Z), I)$
- 离散数据：Bernoulli分布

### 先验分布

通常选择标准正态：
$P(Z) = \mathcal{N}(Z; 0, I)$

### 重参数化技巧

**问题**：采样不可微分。

**解决**：
$Z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$

梯度可通过 $\mu$ 和 $\sigma$ 传播。

```python
# 重参数化
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
```

## VAE损失函数

### 损失组成

$L = L_{reconstruction} + L_{KL}$

### 重建损失

对于连续数据：
$L_{recon} = MSE(X, X')$

对于离散数据：
$L_{recon} = CrossEntropy(X, X')$

### KL散度计算

对于正态分布：
$KL(\mathcal{N}(\mu, \sigma^2)||\mathcal{N}(0, I)) = \frac{1}{2}\sum(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$

### 总损失

```python
def vae_loss(x, x_recon, mu, log_var):
    # 重建损失
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss
```

## VAE的变体

### β-VAE

增加KL权重：
$L = L_{recon} + \beta \cdot L_{KL}$

**效果**：
- β > 1：更解耦的表示
- 可能影响重建质量

### 条件VAE（CVAE）

加入条件信息：
$q(Z|X, c), P(X|Z, c)$

**应用**：条件生成

### VQ-VAE

向量量化VAE：
- 离散潜在空间
- 使用codebook
- 适合图像生成

### Hierarchical VAE

多层次潜在变量：
$Z = (Z_1, Z_2, ...)$

不同层捕获不同特征。

## 案例实践

### 基本VAE实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def loss_function(self, x, x_recon, mu, log_var):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_loss

# 使用
vae = VAE(input_dim=784, latent_dim=20)
optimizer = torch.optim.Adam(vae.parameters())

for epoch in range(epochs):
    for batch in train_loader:
        x = batch.view(-1, 784)
        x_recon, mu, log_var = vae(x)
        loss = vae.loss_function(x, x_recon, mu, log_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### VAE图像生成

```python
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # 编码器（CNN）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(64*7*7, latent_dim)
        self.fc_var = nn.Linear(64*7*7, latent_dim)
        
        # 解码器（CNN）
        self.fc_decode = nn.Linear(latent_dim, 64*7*7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder(self.fc_decode(z)), mu, log_var

# 生成新图像
def generate_images(vae, num_images=10):
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim)
        images = vae.decode(z)
    return images
```

### β-VAE实现

```python
class BetaVAE(VAE):
    def __init__(self, input_dim, latent_dim, beta=4):
        super().__init__(input_dim, latent_dim)
        self.beta = beta
    
    def loss_function(self, x, x_recon, mu, log_var):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.beta * kl_loss
```

### 潜在空间可视化

```python
import matplotlib.pyplot as plt

def visualize_latent_space(vae, data_loader):
    """可视化潜在空间"""
    mus = []
    labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            mu, _ = vae.encode(x.view(-1, 784))
            mus.append(mu)
            labels.append(y)
    
    mus = torch.cat(mus).numpy()
    labels = torch.cat(labels).numpy()
    
    plt.scatter(mus[:, 0], mus[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.title('VAE Latent Space')
    plt.show()
```

### 潜在空间插值

```python
def interpolate(vae, x1, x2, steps=10):
    """潜在空间插值"""
    with torch.no_grad():
        z1, _ = vae.encode(x1)
        z2, _ = vae.encode(x2)
        
        # 线性插值
        alphas = torch.linspace(0, 1, steps)
        images = []
        
        for alpha in alphas:
            z = z1 + alpha * (z2 - z1)
            img = vae.decode(z)
            images.append(img)
        
        return torch.stack(images)
```

## VAE的应用

### 图像生成

| 应用 | 描述 |
|------|------|
| 生成新图像 | 从随机Z采样 |
| 图像重建 | 编码+解码 |
| 图像插值 | 潜在空间平滑过渡 |
| 异常检测 | 重建误差大 |

### 数据压缩

潜在维度低于输入维度，实现压缩。

### 表示学习

学习数据的潜在表示。

### 异常检测

```python
def detect_anomaly(vae, x, threshold=100):
    """异常检测"""
    with torch.no_grad():
        x_recon, mu, log_var = vae(x)
        recon_error = F.mse_loss(x_recon, x, reduction='sum')
        return recon_error > threshold
```

## VAE的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 可解释 | 概率框架清晰 |
| 潜在空间 | 连续且有结构 |
| 生成能力 | 可生成新样本 |
| 训练稳定 | 无对抗训练 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 生成模糊 | 重建偏模糊 |
| 先验限制 | 标准正态可能不最优 |
| 模式覆盖 | 可能覆盖不全 |

## 总结

VAE是重要的生成模型。核心内容包括：
- 变分推断：近似后验分布
- ELBO：变分下界优化
- 重参数化：使采样可微分
- 编码器-解码器：概率生成框架
- VAE变体：β-VAE、CVAE、VQ-VAE

VAE提供可控的生成框架，适合表示学习和条件生成。

## 延伸阅读

- [生成对抗网络](/2026/05/10/zh-CN/技术文档/机器学习/gan/)
- [扩散模型详解](/2026/05/10/zh-CN/技术文档/机器学习/diffusion-models/)
- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
- [概率论基础](/2026/05/10/zh-CN/技术文档/机器学习/probability-theory/)
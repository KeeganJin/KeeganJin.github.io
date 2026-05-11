---
title: 生成对抗网络（GAN）
date: 2026-04-23
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 深度学习, 生成模型, GAN]
---

## GAN基本原理

### 生成对抗思想

GAN（Generative Adversarial Network）通过对抗训练生成数据。

**两大组件**：
- 生成器（Generator）：生成假数据
- 判别器（Discriminator）：区分真假

### 博弈论框架

**对抗过程**：
- 生成器努力骗过判别器
- 判别器努力识别假数据
- 双方博弈达到平衡

### GAN目标函数

$\min_G \max_D V(D, G) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]$

**解读**：
- 判别器：最大化识别真实概率
- 生成器：最小化判别器识别假数据

### 理论分析

最优判别器：
$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$

最优时：
$p_g = p_{data}$

## 生成器架构

### 生成器作用

从随机噪声生成数据：
$G: Z \rightarrow X_{fake}$

### 生成器结构

**基本结构**：
```
随机噪声 Z
  ↓
全连接层 / 卷积层
  ↓
上采样（Upsampling）
  ↓
输出层（生成图像）
```

### 输入噪声

通常使用：
- 正态分布噪声
- 均匀分布噪声

$Z \sim \mathcal{N}(0, I)$

### 激活函数

| 层 | 激活函数 |
|------|----------|
| 中间层 | ReLU, LeakyReLU |
| 输出层 | Tanh, Sigmoid |

## 判别器架构

### 判别器作用

区分真实和生成数据：
$D: X \rightarrow [0, 1]$

### 判别器结构

**基本结构**：
```
输入数据 X
  ↓
卷积层 / 全连接层
  ↓
下采样
  ↓
输出层（真/假概率）
```

### 输出

输出为概率：
- D(x)接近1：真实数据
- D(x)接近0：生成数据

### 激活函数

| 层 | 激活函数 |
|------|----------|
| 中间层 | LeakyReLU |
| 输出层 | Sigmoid |

## GAN训练策略

### 交替训练

**流程**：
```
1. 固定G，训练D（更新判别器）
2. 固定D，训练G（更新生成器）
3. 重复交替
```

### 判别器训练

```python
# 训练判别器
real_labels = torch.ones(batch_size)
fake_labels = torch.zeros(batch_size)

# 真数据
real_output = D(real_data)
loss_real = F.binary_cross_entropy(real_output, real_labels)

# 假数据
fake_data = G(z)
fake_output = D(fake_data.detach())
loss_fake = F.binary_cross_entropy(fake_output, fake_labels)

d_loss = loss_real + loss_fake
d_loss.backward()
d_optimizer.step()
```

### 生成器训练

```python
# 训练生成器
fake_data = G(z)
fake_output = D(fake_data)
g_loss = F.binary_cross_entropy(fake_output, real_labels)  # 目标是骗过D

g_loss.backward()
g_optimizer.step()
```

### 训练比例

通常：
- 每训练1次G，训练D k次（k=1-5）
- 保持判别器略强

## GAN训练稳定性

### 常见问题

| 问题 | 描述 |
|------|------|
| 模式崩溃 | 只生成少数模式 |
| 不收敛 | 挟摆不收敛 |
| 梯度消失 | D太强，G无梯度 |
| 梯度爆炸 | 训练不稳定 |

### 模式崩溃（Mode Collapse）

生成器只生成少数几种输出。

**原因**：
- 生成器找到"欺骗"判别器的捷径
- 判别器对某些模式判别弱

### 稳定性技巧

| 技巧 | 方法 |
|------|------|
| 标签平滑 | 真标签用0.9代替1 |
| 噪声标签 | 随机翻转部分标签 |
| 特征匹配 | 匹配特征而非输出 |
| 历史样本 | 判别器看历史生成 |

### 梯度问题解决

**判别器太强**：
- 降低D训练频率
- 减小D网络规模
- 使用Wasserstein损失

## GAN变体

### DCGAN

深度卷积GAN：
- 使用卷积网络
- 批归一化
- LeakyReLU
- 移除全连接层

```python
class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z.view(z.size(0), z.size(1), 1, 1))
```

### WGAN

Wasserstein GAN：
- 使用Wasserstein距离
- 移除输出Sigmoid
- 权重裁剪或梯度惩罚

**Wasserstein距离**：
$W(P_r, P_g) = \max_{||f||_L\leq 1} \mathbb{E}_{x\sim P_r}[f(x)] - \mathbb{E}_{x\sim P_g}[f(x)]$

```python
# WGAN判别器损失
d_loss = -torch.mean(D(real_data)) + torch.mean(D(fake_data))

# WGAN生成器损失
g_loss = -torch.mean(D(fake_data))
```

### WGAN-GP

梯度惩罚替代权重裁剪：
$L = L_{original} + \lambda \mathbb{E}[(||\nabla_x D(x)||_2 - 1)^2]$

```python
def gradient_penalty(D, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    
    d_output = D(interpolated)
    gradients = torch.autograd.grad(d_output, interpolated, 
                                    grad_outputs=torch.ones_like(d_output),
                                    create_graph=True)[0]
    
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
```

### StyleGAN

风格生成GAN：
- 风格注入机制
- 噪声注入
- Progressive Growing
- 高质量人脸生成

**特点**：
- 控制生成风格
- 不同层注入不同风格
- 高分辨率生成

### Conditional GAN（cGAN）

条件GAN：
- 生成器和判别器接收条件
- 条件生成

$\min_G \max_D V(D, G) = \mathbb{E}[\log D(x, c)] + \mathbb{E}[\log(1 - D(G(z, c), c))]$

### CycleGAN

循环一致GAN：
- 无配对数据转换
- 域适应

**循环一致损失**：
$L_{cycle} = ||G_{Y\rightarrow X}(G_{X\rightarrow Y}(x)) - x||$

## 案例实践

### 基本GAN实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 训练
latent_dim = 100
output_dim = 784  # MNIST

G = Generator(latent_dim, output_dim)
D = Discriminator(output_dim)

g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(epochs):
    for real_data in train_loader:
        batch_size = real_data.size(0)
        
        # 训练判别器
        real_data = real_data.view(-1, output_dim)
        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)
        
        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z)
        
        real_output = D(real_data)
        fake_output = D(fake_data.detach())
        
        d_loss = F.binary_cross_entropy(real_output, real_labels) + \
                 F.binary_cross_entropy(fake_output, fake_labels)
        
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # 训练生成器
        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z)
        fake_output = D(fake_data)
        
        g_loss = F.binary_cross_entropy(fake_output, real_labels)
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
```

### DCGAN图像生成

```python
class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)

# 生成图像
def generate_images(G, num=10):
    with torch.no_grad():
        z = torch.randn(num, latent_dim)
        images = G(z)
    return images
```

### WGAN-GP实现

```python
class WGANCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # 无Sigmoid
        )
    
    def forward(self, x):
        return self.model(x)

def train_wgan_gp(G, C, real_data, lambda_gp=10):
    batch_size = real_data.size(0)
    
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    
    # Critic损失
    c_loss = -torch.mean(C(real_data)) + torch.mean(C(fake_data))
    
    # 梯度惩罚
    alpha = torch.rand(batch_size, 1)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    c_interp = C(interpolated)
    
    gradients = torch.autograd.grad(c_interp, interpolated,
                                    grad_outputs=torch.ones_like(c_interp),
                                    create_graph=True)[0]
    
    gp = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    total_loss = c_loss + gp
    return total_loss
```

### 条件GAN实现

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.model(x)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = torch.cat([x, c], dim=1)
        return self.model(x)
```

## GAN评估

### 评估指标

| 指标 | 描述 |
|------|------|
| Inception Score | 生成质量和多样性 |
| FID | Fréchet Inception Distance |
| 人工评估 | 视觉质量 |

### Inception Score

$IS = \exp(\mathbb{E}_x[KL(p(y|x)||p(y))])$

**含义**：
- 高质量：p(y|x)尖锐
- 多样性：p(y)均匀

### FID

$FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$

**越低越好**。

```python
from pytorch_fid import fid_score

fid_value = fid_score.calculate_fid_given_paths(
    ['real_images', 'generated_images'],
    batch_size=50
)
```

## GAN应用

### 图像生成

| 应用 | 描述 |
|------|------|
| 人脸生成 | StyleGAN |
| 图像修复 | 填充缺失部分 |
| 超分辨率 | SRGAN |
| 图像转换 | CycleGAN |

### 数据增强

生成数据扩充训练集。

### 其他应用

| 应用 | 描述 |
|------|------|
| 文本生成 | SeqGAN |
| 音频生成 | WaveGAN |
| 视频生成 | VideoGAN |

## GAN的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 高质量生成 | 生成图像清晰 |
| 无需马尔可夫链 | 直接采样 |
| 隐式密度 | 不需显式建模 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 训练不稳定 | 需技巧稳定 |
| 模式崩溃 | 可能只生成少数模式 |
| 超参数敏感 | 需仔细调参 |
| 评估困难 | 难量化评估 |

## 总结

GAN是强大的生成模型。核心内容包括：
- 对抗训练：生成器vs判别器
- 博弈框架：双方博弈达到平衡
- 训练策略：交替训练
- 稳定性技巧：标签平滑、梯度惩罚
- GAN变体：DCGAN、WGAN、StyleGAN、CycleGAN

GAN生成质量高，但训练需技巧和经验。

## 延伸阅读

- [变分自编码器](/2026/05/10/zh-CN/技术文档/机器学习/vae/)
- [扩散模型详解](/2026/05/10/zh-CN/技术文档/机器学习/diffusion-models/)
- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
- [优化算法详解](/2026/05/10/zh-CN/技术文档/机器学习/optimization-algorithms/)
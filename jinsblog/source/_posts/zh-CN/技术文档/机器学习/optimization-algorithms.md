---
title: 优化算法详解
date: 2026-05-08
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 神经网络, 优化算法]
---

## SGD基础

### SGD原理

随机梯度下降（Stochastic Gradient Descent）是最基础的优化算法。

**更新规则**：
$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$

其中：
- $\theta$：参数
- $\alpha$：学习率
- $\nabla L$：损失函数梯度

### SGD的特点

| 特点 | 描述 |
|------|------|
| 简单 | 容易理解和实现 |
| 无偏 | 梯度估计无偏 |
| 噪声 | 单样本梯度噪声大 |
| 收敛慢 | 震荡可能严重 |

### SGD的实现

```python
import numpy as np

def sgd_update(params, gradients, learning_rate):
    """SGD更新"""
    for param, grad in zip(params, gradients):
        param -= learning_rate * grad
    return params

def sgd_train(model, X, y, epochs, learning_rate, batch_size=1):
    """SGD训练"""
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            gradients = compute_gradients(model, X_batch, y_batch)
            model.params = sgd_update(model.params, gradients, learning_rate)
```

### Mini-batch SGD

使用批量数据计算梯度：

$\theta_{t+1} = \theta_t - \alpha \frac{1}{m}\sum_{i=1}^{m}\nabla L_i(\theta_t)$

**优点**：
- 梯度估计更稳定
- 可利用GPU并行
- 收敛更快

**批量大小选择**：
| 批量大小 | 特点 |
|----------|------|
| 小（16-64） | 噪声大，收敛慢 |
| 中（128-512） | 平衡效率和稳定性 |
| 大（1024+） | 稳定，但可能陷入局部最优 |

## Momentum与Nesterov

### Momentum原理

Momentum积累历史梯度，平滑更新方向。

**更新规则**：
$v_t = \gamma v_{t-1} + \alpha \nabla L(\theta_t)$
$\theta_{t+1} = \theta_t - v_t$

其中 $\gamma$ 是动量系数（通常0.9）。

### Momentum的作用

| 作用 | 描述 |
|------|------|
| 加速收敛 | 沿梯度方向加速 |
| 减少震荡 | 平滑梯度方向 |
| 逃离浅坑 | 动量帮助逃离 |

```python
def momentum_update(params, gradients, velocities, learning_rate, gamma=0.9):
    """Momentum更新"""
    for param, grad, v in zip(params, gradients, velocities):
        v = gamma * v + learning_rate * grad
        param -= v
    return params, velocities
```

### Nesterov加速梯度（NAG）

**更新规则**：
$v_t = \gamma v_{t-1} + \alpha \nabla L(\theta_t - \gamma v_{t-1})$
$\theta_{t+1} = \theta_t - v_t$

**特点**：
- 在"预测"位置计算梯度
- 提前修正方向
- 收敛更快

```python
def nesterov_update(params, gradients, velocities, learning_rate, gamma=0.9):
    """Nesterov更新"""
    # 预测位置
    params_lookahead = [p - gamma * v for p, v in zip(params, velocities)]
    
    # 在预测位置计算梯度
    gradients_lookahead = compute_gradients(params_lookahead)
    
    for param, grad, v in zip(params, gradients_lookahead, velocities):
        v = gamma * v + learning_rate * grad
        param -= v
    
    return params, velocities
```

### Momentum vs Nesterov

| 方面 | Momentum | Nesterov |
|------|----------|----------|
| 梯度位置 | 当前位置 | 预测位置 |
| 收敛速度 | 快 | 更快 |
| 计算成本 | 低 | 略高 |

## AdaGrad

### AdaGrad原理

自适应学习率，根据参数历史梯度调整。

**更新规则**：
$G_t = G_{t-1} + (\nabla L_t)^2$
$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \nabla L_t$

**特点**：
- 常更新参数：学习率小
- 少更新参数：学习率大
- 适合稀疏数据

```python
def adagrad_update(params, gradients, G, learning_rate, epsilon=1e-8):
    """AdaGrad更新"""
    for param, grad, g in zip(params, gradients, G):
        g += grad ** 2
        param -= learning_rate * grad / (np.sqrt(g) + epsilon)
    return params, G
```

### AdaGrad的缺点

| 缺点 | 描述 |
|------|------|
| 学习率单调递减 | 最终学习率过小 |
| 不适合深度学习 | 训练后期无法学习 |

## RMSprop

### RMSprop原理

解决AdaGrad学习率递减问题，使用滑动平均。

**更新规则**：
$E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g_t^2$
$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t$

**特点**：
- 学习率不单调递减
- 适合非平稳目标
- 深度学习常用

```python
def rmsprop_update(params, gradients, Eg2, learning_rate, gamma=0.9, epsilon=1e-8):
    """RMSprop更新"""
    for param, grad, eg in zip(params, gradients, Eg2):
        eg = gamma * eg + (1 - gamma) * grad ** 2
        param -= learning_rate * grad / (np.sqrt(eg) + epsilon)
    return params, Eg2
```

### RMSprop vs AdaGrad

| 方面 | AdaGrad | RMSprop |
|------|---------|----------|
| 学习率变化 | 单调递减 | 动态调整 |
| 深度学习 | 不适合 | 适合 |
| 长期训练 | 学习率太小 | 正常 |

## Adam与AdamW

### Adam原理

结合Momentum和RMSprop的优点。

**更新规则**：
$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$

**偏差修正**：
$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$
$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$

**参数更新**：
$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

### Adam参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| $\alpha$ | 0.001 | 学习率 |
| $\beta_1$ | 0.9 | 一阶矩衰减率 |
| $\beta_2$ | 0.999 | 二阶矩衰减率 |
| $\epsilon$ | 1e-8 | 防止除零 |

```python
def adam_update(params, gradients, m, v, t, learning_rate=0.001, 
                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Adam更新"""
    t += 1
    for param, grad, m_i, v_i in zip(params, gradients, m, v):
        # 更新矩估计
        m_i = beta1 * m_i + (1 - beta1) * grad
        v_i = beta2 * v_i + (1 - beta2) * grad ** 2
        
        # 偏差修正
        m_hat = m_i / (1 - beta1 ** t)
        v_hat = v_i / (1 - beta2 ** t)
        
        # 参数更新
        param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return params, m, v, t
```

### Adam的特点

| 特点 | 描述 |
|------|------|
| 自适应学习率 | 每参数独立调整 |
| 偏差修正 | 初期估计准确 |
| 快速收敛 | 综合Momentum和RMSprop |
| 广泛使用 | 深度学习默认选择 |

### AdamW

**问题**：Adam中的权重衰减与自适应学习率冲突。

**AdamW改进**：
$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \alpha \lambda \theta_t$

将权重衰减从梯度中分离，正确实现L2正则化。

```python
def adamw_update(params, gradients, m, v, t, learning_rate=0.001, 
                  beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
    """AdamW更新"""
    t += 1
    for param, grad, m_i, v_i in zip(params, gradients, m, v):
        # 权重衰减（独立于梯度）
        param -= learning_rate * weight_decay * param
        
        # 更新矩估计
        m_i = beta1 * m_i + (1 - beta1) * grad
        v_i = beta2 * v_i + (1 - beta2) * grad ** 2
        
        # 偏差修正
        m_hat = m_i / (1 - beta1 ** t)
        v_hat = v_i / (1 - beta2 ** t)
        
        # 参数更新
        param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return params, m, v, t
```

### Adam vs AdamW

| 方面 | Adam | AdamW |
|------|------|-------|
| 权重衰减 | 与梯度混合 | 独立实现 |
| 正则化效果 | 可能不稳定 | 更好的正则化 |
| 泛化性能 | 较好 | 更好 |

## 学习率调度策略

### 固定学习率

$\alpha_t = \alpha_0$

**特点**：简单但不够灵活。

### 阶梯衰减

$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/T \rfloor}$

每隔T步衰减$\gamma$倍。

```python
def step_decay(learning_rate, epoch, decay_rate=0.1, decay_epochs=30):
    """阶梯衰减"""
    return learning_rate * (decay_rate ** (epoch // decay_epochs))
```

### 指数衰减

$\alpha_t = \alpha_0 \cdot e^{-kt}$

```python
def exponential_decay(learning_rate, epoch, k=0.001):
    """指数衰减"""
    return learning_rate * np.exp(-k * epoch)
```

### 余弦退火

$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t\pi}{T}))$

```python
def cosine_annealing(learning_rate_max, learning_rate_min, epoch, total_epochs):
    """余弦退火"""
    return learning_rate_min + 0.5 * (learning_rate_max - learning_rate_min) * \
           (1 + np.cos(epoch * np.pi / total_epochs))
```

### Warmup

初期学习率从0逐渐增加到目标值：

```python
def warmup(learning_rate_target, epoch, warmup_epochs):
    """Warmup"""
    if epoch < warmup_epochs:
        return learning_rate_target * (epoch + 1) / warmup_epochs
    return learning_rate_target
```

### 学习率调度对比

| 策略 | 特点 |
|------|------|
| 固定 | 简单，可能不最优 |
| 阶梯 | 常用，需调整衰减时机 |
| 指数 | 平滑衰减 |
| 余弦 | 自动调整，效果好 |
| Warmup | 防止初期不稳定 |

## 二阶优化方法简介

### Newton法

使用二阶信息（Hessian矩阵）：

$\theta_{t+1} = \theta_t - H^{-1} \nabla L(\theta_t)$

**优点**：收敛快（二次收敛）
**缺点**：计算Hessian逆矩阵成本高

### 拟Newton法

不直接计算Hessian，用近似矩阵：
- BFGS
- L-BFGS（有限内存）

### 自然梯度

考虑参数空间几何结构：
$\theta_{t+1} = \theta_t - \alpha F^{-1} \nabla L$

其中F是Fisher信息矩阵。

### 二阶方法的应用

| 方法 | 适用场景 |
|------|----------|
| Newton法 | 小规模问题 |
| L-BFGS | 中等规模 |
| 自然梯度 | 概率模型 |

## 优化算法选择指南

### 选择原则

| 场景 | 推荐算法 |
|------|----------|
| 一般深度学习 | Adam/AdamW |
| 稀疏数据 | AdaGrad/RMSprop |
| Transformer | AdamW + Warmup |
| CNN | SGD + Momentum |
| 小数据 | L-BFGS |

### 各算法特点总结

| 算法 | 自适应 | Momentum | 收敛速度 | 稳定性 |
|------|--------|----------|----------|--------|
| SGD | 否 | 否 | 慢 | 低 |
| Momentum | 否 | 是 | 中 | 中 |
| AdaGrad | 是 | 否 | 中 | 中 |
| RMSprop | 是 | 否 | 快 | 高 |
| Adam | 是 | 是 | 快 | 高 |

## 案例实践

### PyTorch优化器

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 不同优化器
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'AdamW': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
}

# 训练循环
for name, optimizer in optimizers.items():
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
```

### TensorFlow优化器

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# 不同优化器
optimizers = {
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
    'Momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
    'AdamW': tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
}

model.compile(optimizer=optimizers['AdamW'],
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
```

### 学习率调度实现

```python
# PyTorch学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(epochs):
    train_one_epoch()
    scheduler.step()  # 更新学习率

# TensorFlow学习率调度
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=100
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### 优化器对比实验

```python
import matplotlib.pyplot as plt

# 对比不同优化器收敛
results = {}
for name, opt_class in optimizer_classes.items():
    model = create_model()
    optimizer = opt_class(model.parameters())
    
    losses = []
    for epoch in range(epochs):
        loss = train_epoch(model, optimizer)
        losses.append(loss)
    
    results[name] = losses

# 可视化
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimizer Comparison')
plt.show()
```

## 总结

优化算法是神经网络训练的关键。核心内容包括：
- SGD基础：最简单的梯度下降
- Momentum与Nesterov：加速收敛，减少震荡
- AdaGrad：自适应学习率，适合稀疏数据
- RMSprop：解决AdaGrad学习率递减问题
- Adam与AdamW：综合优点，广泛使用
- 学习率调度：动态调整学习率
- 二阶优化：利用Hessian信息

Adam/AdamW是现代深度学习的默认选择，配合学习率调度效果更佳。

## 延伸阅读

- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [正则化技术](/2026/05/10/zh-CN/技术文档/机器学习/regularization/)
- [超参数调优](/2026/05/10/zh-CN/技术文档/机器学习/hyperparameter-tuning/)
- [优化理论基础](/2026/05/10/zh-CN/技术文档/机器学习/optimization/)
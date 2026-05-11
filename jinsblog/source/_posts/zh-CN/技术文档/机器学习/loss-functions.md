---
title: 损失函数详解
date: 2026-05-02
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 神经网络, 损失函数]
---

## 损失函数的作用

### 什么是损失函数

损失函数（Loss Function）衡量模型预测与真实值之间的差异。

**作用**：
- 量化模型误差
- 提供优化目标
- 指导参数更新

### 损失函数与目标函数

**损失函数**：单个样本的误差
$L(y, \hat{y})$

**代价函数**：所有样本损失的平均
$J = \frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{y}_i)$

**目标函数**：代价函数加正则化
$Obj = J + \lambda R$

### 损失函数的选择影响

| 影响 | 描述 |
|------|------|
| 训练方向 | 决定优化目标 |
| 收敛速度 | 影响梯度大小 |
| 模型性能 | 影响最终结果 |
| 过拟合风险 | 某些损失更易过拟合 |

## 回归损失：MSE、MAE、Huber

### MSE（均方误差）

**定义**：
$L_{MSE} = (y - \hat{y})^2$

**代价函数**：
$J_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

**导数**：
$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$

```python
import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true)
```

### MSE的特点

| 特点 | 描述 |
|------|------|
| 对异常值敏感 | 大误差被放大 |
| 梯度简单 | 线性梯度 |
| 凸函数 | 有全局最优 |
| 可微分 | 优化友好 |

### MAE（平均绝对误差）

**定义**：
$L_{MAE} = |y - \hat{y}|$

**代价函数**：
$J_{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

**导数**：
$\frac{\partial L}{\partial \hat{y}} = \text{sign}(\hat{y} - y)$

```python
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_gradient(y_true, y_pred):
    return np.sign(y_pred - y_true)
```

### MAE的特点

| 特点 | 描述 |
|------|------|
| 对异常值不敏感 | 所有误差权重相同 |
| 零点不可微 | 需要特殊处理 |
| 非光滑 | 优化可能不稳定 |

### MSE vs MAE

| 方面 | MSE | MAE |
|------|-----|-----|
| 异常值敏感度 | 高 | 低 |
| 梯度 | 连续变化 | 恒定值 |
| 零点可微 | 是 | 否 |
| 适用场景 | 无异常值 | 有异常值 |

### Huber损失

**定义**：
$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y-\hat{y})^2 & |y-\hat{y}| \leq \delta \\ \delta|y-\hat{y}| - \frac{1}{2}\delta^2 & |y-\hat{y}| > \delta \end{cases}$

**特点**：
- 小误差：使用MSE（精确）
- 大误差：使用MAE（抗异常）
- 结合两者优点

```python
def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    return np.where(
        error <= delta,
        0.5 * error ** 2,
        delta * error - 0.5 * delta ** 2
    )
```

### 其他回归损失

| 损失函数 | 特点 |
|----------|------|
| Log-Cosh | 类似Huber，全程可微 |
| Quantile Loss | 分位数回归 |
| Smooth L1 | 目标检测常用 |

## 分类损失：交叉熵、Focal Loss

### 0-1损失

**定义**：
$L_{0-1} = \mathbb{1}[y \neq \hat{y}]$

**特点**：
- 直观但不优化友好
- 不可微分
- 很少直接使用

### 交叉熵损失（二分类）

**定义**：
$L_{CE} = -y\ln\hat{y} - (1-y)\ln(1-\hat{y})$

其中 $\hat{y}$ 是预测概率。

**导数**：
$\frac{\partial L}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$

```python
def binary_crossentropy(y_true, y_pred):
    # 防止log(0)
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

### 交叉熵损失（多分类）

**定义**：
$L_{CE} = -\sum_{k=1}^{K} y_k \ln \hat{y}_k$

其中 $y_k$ 是真实标签的one-hot编码，$\hat{y}_k$ 是预测概率。

```python
def categorical_crossentropy(y_true, y_pred):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))
```

### 交叉熵与Softmax

配合使用时梯度简化：
$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$

其中 $z_i$ 是Softmax输入。

### 交叉熵的特点

| 特点 | 描述 |
|------|------|
| 优化友好 | 可微分，梯度简单 |
| 信息论基础 | 衡量分布差异 |
| 与Softmax配合良好 | 梯度形式简单 |
| 凸函数 | 有全局最优 |

### Hinge损失

**定义**：
$L_{Hinge} = \max(0, 1 - y\hat{y})$

其中 $y \in \{-1, +1\}$。

**特点**：
- SVM使用
- 鼓励正确分类且置信度高
- 不关心置信度超过阈值的样本

```python
def hinge_loss(y_true, y_pred):
    # y_true应为-1或+1
    return np.maximum(0, 1 - y_true * y_pred)
```

### Focal Loss

**定义**：
$L_{Focal} = -\alpha(1-\hat{y})^\gamma y\ln\hat{y} - (1-\alpha)\hat{y}^\gamma(1-y)\ln(1-\hat{y})$

**参数**：
- $\alpha$：类别权重
- $\gamma$：聚焦参数（通常为2）

**特点**：
- 解决类别不平衡
- 关注困难样本
- 减少简单样本权重

```python
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # 正样本损失
    pos_loss = -alpha * (1 - y_pred)**gamma * y_true * np.log(y_pred)
    
    # 负样本损失
    neg_loss = -(1 - alpha) * y_pred**gamma * (1 - y_true) * np.log(1 - y_pred)
    
    return np.mean(pos_loss + neg_loss)
```

### Focal Loss的作用

| 作用 | 描述 |
|------|------|
| 类别不平衡 | α平衡正负样本 |
| 困难样本 | γ聚焦困难样本 |
| 目标检测 | RetinaNet使用 |

## 对比损失

### Contrastive Loss

**定义**：
$L = (1-y)\frac{1}{2}d^2 + y\frac{1}{2}\max(0, m-d)^2$

其中：
- $y=0$：相似样本
- $y=1$：不相似样本
- $d$：样本距离
- $m$：边际阈值

**目的**：
- 相似样本：距离小
- 不相似样本：距离大于边际

### Triplet Loss

**定义**：
$L = \max(0, d(a, p) - d(a, n) + m)$

其中：
- $a$：锚点样本
- $p$：正样本（与锚点相似）
- $n$：负样本（与锚点不相似）
- $m$：边际

**目的**：
- 锚点与正样本距离小于锚点与负样本距离减边际

```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = np.sum((anchor - positive)**2)
    neg_dist = np.sum((anchor - negative)**2)
    return np.maximum(0, pos_dist - neg_dist + margin)
```

### 对比损失的应用

| 应用 | 损失类型 |
|------|----------|
| 人脸识别 | Contrastive、Triplet |
| 图像检索 | Contrastive |
| 自监督学习 | Contrastive |

## 损失函数选择策略

### 按任务类型选择

| 任务类型 | 推荐损失函数 |
|----------|--------------|
| 回归 | MSE、Huber |
| 二分类 | 交叉熵 |
| 多分类 | 交叉熵 + Softmax |
| 类别不平衡 | Focal Loss |
| 目标检测 | Focal Loss、Smooth L1 |
| 嵌入学习 | Triplet、Contrastive |

### 按数据特点选择

| 数据特点 | 推荐损失函数 |
|----------|--------------|
| 有异常值 | MAE、Huber |
| 类别不平衡 | Focal Loss |
| 需要置信度 | Hinge、交叉熵 |

### 损失函数设计原则

| 原则 | 描述 |
|------|------|
| 可微分 | 支持梯度优化 |
| 凸性 | 有全局最优 |
| 合理梯度 | 梯度不应太小或太大 |
| 任务匹配 | 反映任务目标 |

## 案例实践

### 回归损失对比

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 生成数据
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1, 2, 4, 3, 100])  # 包含异常预测

# 计算损失
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
huber = np.mean(huber_loss(y_true, y_pred, delta=1.0))

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Huber: {huber:.4f}")
```

### 分类损失示例

```python
import torch
import torch.nn as nn

# PyTorch损失函数
criterion_ce = nn.CrossEntropyLoss()
criterion_bce = nn.BCELoss()
criterion_focal = ...  # 需要自定义

# 使用示例
y_pred = torch.randn(10, 3)  # 10样本，3类别
y_true = torch.randint(0, 3, (10,))  # 真实标签

loss = criterion_ce(y_pred, y_true)
print(f"交叉熵损失: {loss.item():.4f}")
```

### TensorFlow损失函数

```python
import tensorflow as tf

# TensorFlow损失函数
mse_loss = tf.keras.losses.MSE
mae_loss = tf.keras.losses.MAE
ce_loss = tf.keras.losses.CategoricalCrossentropy()
bce_loss = tf.keras.losses.BinaryCrossentropy()

# 使用示例
y_true = tf.constant([0, 1, 0])
y_pred = tf.constant([0.1, 0.9, 0.2])

loss = bce_loss(y_true, y_pred)
print(f"二分类交叉熵: {loss.numpy():.4f}")
```

### 自定义损失函数

```python
# PyTorch自定义损失
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# TensorFlow自定义损失
def focal_loss_tf(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    
    y_true = tf.cast(y_true, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    
    return tf.reduce_mean(loss)

# 在模型中使用
model.compile(optimizer='adam', loss=focal_loss_tf)
```

### 损失函数可视化

```python
import matplotlib.pyplot as plt

# 绘制不同损失函数
errors = np.linspace(-3, 3, 100)

losses = {
    'MSE': errors ** 2,
    'MAE': np.abs(errors),
    'Huber (δ=1)': [huber_loss(0, e, 1.0) for e in errors]
}

plt.figure(figsize=(10, 6))
for name, loss in losses.items():
    plt.plot(errors, loss, label=name)
plt.xlabel('Error')
plt.ylabel('Loss')
plt.legend()
plt.title('Regression Loss Functions')
plt.grid(True)
plt.show()
```

### 损失函数对训练的影响

```python
# 对比不同损失函数训练效果
import tensorflow as tf

losses = ['mse', 'mae']
for loss_fn in losses:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss=loss_fn)
    
    # 假设有训练数据
    history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
    
    plt.plot(history.history['loss'], label=f'{loss_fn} train')
    plt.plot(history.history['val_loss'], label=f'{loss_fn} val')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Function Comparison')
plt.show()
```

## 总结

损失函数是神经网络优化的核心。核心内容包括：
- 损失函数作用：量化误差，指导优化
- 回归损失：MSE（精确）、MAE（抗异常）、Huber（综合）
- 分类损失：交叉熵（主流）、Focal Loss（类别不平衡）
- 对比损失：Contrastive、Triplet（嵌入学习）
- 选择策略：按任务类型和数据特点选择

交叉熵是分类任务的标准选择，MSE是回归任务的常用选择。

## 延伸阅读

- [激活函数详解](/2026/05/10/zh-CN/技术文档/机器学习/activation-functions/)
- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [优化算法详解](/2026/05/10/zh-CN/技术文档/机器学习/optimization-algorithms/)
- [正则化技术](/2026/05/10/zh-CN/技术文档/机器学习/regularization/)
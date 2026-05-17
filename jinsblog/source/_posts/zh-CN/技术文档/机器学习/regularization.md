---
title: 正则化技术
date: 2025-12-13
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 神经网络, 正则化]
---

## 正则化的作用

### 为什么需要正则化

神经网络容易过拟合：
- 参数数量远超过训练数据
- 训练集表现好，测试集差
- 泛化能力不足

**正则化的作用**：
- 限制模型复杂度
- 提高泛化能力
- 减少过拟合

### 正则化的类型

| 类型 | 描述 |
|------|------|
| 参数正则化 | 限制参数值 |
| Dropout | 随机丢弃神经元 |
| Batch Normalization | 控制激活分布 |
| 数据增强 | 扩充训练数据 |
| 早停 | 适时停止训练 |

## L1/L2正则化原理

### L1正则化（Lasso）

**损失函数**：
$J = L + \lambda\sum_j |w_j|$

**特点**：
- 产生稀疏解（部分权重为零）
- 自动特征选择
- 不适合神经网络（计算不稳定）

### L2正则化（Ridge/权重衰减）

**损失函数**：
$J = L + \frac{\lambda}{2}\sum_j w_j^2$

**梯度更新**：
$\frac{\partial J}{\partial w} = \frac{\partial L}{\partial w} + \lambda w$
$w_{new} = w_{old} - \alpha(\frac{\partial L}{\partial w} + \lambda w) = w_{old}(1 - \alpha\lambda) - \alpha\frac{\partial L}{\partial w}$

**特点**：
- 权重衰减（每步减小权重）
- 不产生稀疏解
- 计算稳定，适合神经网络

### L1 vs L2

| 方面 | L1 | L2 |
|------|----|----|
| 稀疏性 | 产生稀疏解 | 不产生稀疏解 |
| 计算稳定性 | 不稳定 | 稳定 |
| 适用场景 | 特征选择 | 神经网络 |
| 解的性质 | 稀疏解 | 平滑解 |

### Elastic Net

结合L1和L2：

$J = L + \lambda_1\sum_j |w_j| + \lambda_2\sum_j w_j^2$

```python
import numpy as np

def l2_regularization(weights, lambda_reg):
    """L2正则化"""
    return lambda_reg * np.sum(weights ** 2)

def l2_gradient(weights, lambda_reg):
    """L2正则化梯度"""
    return lambda_reg * weights

def elastic_net_regularization(weights, lambda_l1, lambda_l2):
    """Elastic Net"""
    return lambda_l1 * np.sum(np.abs(weights)) + lambda_l2 * np.sum(weights ** 2)
```

### 正则化参数λ的选择

| λ值 | 效果 |
|-----|------|
| 太小 | 正则化效果不明显 |
| 太大 | 模型过于简单，欠拟合 |
| 合适 | 平衡训练误差和泛化 |

**选择方法**：交叉验证

```python
import torch
import torch.nn as nn

# PyTorch权重衰减
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# TensorFlow权重衰减
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
```

## Dropout技术

### Dropout原理

训练时随机"丢弃"部分神经元：

- 每次迭代随机选择神经元
- 丢弃率p决定保留比例
- 测试时不丢弃，但输出乘以p

### Dropout的效果

| 效果 | 描述 |
|------|------|
| 防止过拟合 | 减少神经元间依赖 |
| 模型集成 | 相当于训练多个子网络 |
| 提高泛化 | 强制每个神经元独立学习 |

### Dropout的数学解释

**训练时**：
$y = f(\sum_i w_i x_i \cdot \frac{z_i}{p})$

其中 $z_i \in \{0, 1\}$ 是随机变量，保留概率为 $p$。

**测试时**：
$y = f(\sum_i w_i x_i \cdot p)$

权重乘以 $p$（或训练时使用 inverted dropout，输出除以 $p$）。

### Dropout实现

```python
import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p  # 保留概率
    
    def forward_train(self, x):
        """训练时前向传播"""
        mask = np.random.binomial(1, self.p, size=x.shape) / self.p
        return x * mask, mask
    
    def forward_test(self, x):
        """测试时前向传播"""
        return x * self.p
```

### Dropout参数选择

| 场景 | 建议丢弃率 |
|------|------------|
| 全连接层 | 0.5 |
| 输入层 | 0.2-0.3 |
| 卷积层 | 较少使用或不使用 |

```python
# PyTorch Dropout
import torch.nn as nn

dropout = nn.Dropout(p=0.5)

# TensorFlow Dropout
dropout = tf.keras.layers.Dropout(rate=0.5)
```

### Dropout的变体

| 变体 | 特点 |
|------|------|
| Spatial Dropout | 卷积层丢弃整个通道 |
| DropConnect | 丢弃连接而非神经元 |
| DropBlock | 丢弃连续区域 |

## Batch Normalization

### BN原理

标准化每层输入：

$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
$y = \gamma \hat{x} + \beta$

其中：
- $\mu_B$：批次均值
- $\sigma_B^2$：批次方差
- $\gamma, \beta$：可学习参数

### BN的作用

| 作用 | 描述 |
|------|------|
| 标准化输入 | 每层输入分布稳定 |
| 加速收敛 | 更大学习率可用 |
| 减少对初始化依赖 | 初始化影响变小 |
| 防止过拟合 | 有轻微正则化效果 |

### BN的训练和测试

**训练**：使用当前批次统计量
**测试**：使用训练时的全局统计量（滑动平均）

```python
import numpy as np

class BatchNormalization:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.epsilon = epsilon
        self.momentum = momentum
        
        # 运行统计量（测试时使用）
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward_train(self, x):
        """训练时"""
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        
        # 标准化
        x_hat = (x - mean) / np.sqrt(var + self.epsilon)
        y = self.gamma * x_hat + self.beta
        
        # 更新运行统计量
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        
        return y
    
    def forward_test(self, x):
        """测试时"""
        x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.gamma * x_hat + self.beta
```

### BN的位置

通常在激活函数之前：

```
输入 → BN → 激活 → 输出
```

但某些情况下在激活之后效果更好。

### BN的实现

```python
# PyTorch Batch Normalization
bn = nn.BatchNorm1d(num_features)

# TensorFlow Batch Normalization
bn = tf.keras.layers.BatchNormalization()
```

### BN的注意事项

| 注意点 | 描述 |
|------|------|
| 小批量问题 | 批量小时统计量不准确 |
| 训练测试差异 | 确保正确切换模式 |
| 与正则化冲突 | BN有轻微正则化效果 |

## Layer Normalization

### LN原理

标准化每个样本的所有特征：

$\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$
$y = \gamma \hat{x} + \beta$

其中 $\mu_L$ 和 $\sigma_L^2$ 是单个样本的统计量。

### LN vs BN

| 方面 | Batch Norm | Layer Norm |
|------|------------|------------|
| 统计量 | 跨样本 | 单样本 |
| 适用场景 | CNN | RNN、Transformer |
| 批量依赖 | 依赖批量大小 | 不依赖 |
| 测试时 | 需切换模式 | 不需要 |

```python
# PyTorch Layer Normalization
ln = nn.LayerNorm(normalized_shape)

# TensorFlow Layer Normalization
ln = tf.keras.layers.LayerNormalization()
```

### LN的应用场景

| 场景 | 原因 |
|------|------|
| RNN | 时间步维度不同 |
| Transformer | 批量大小可变 |
| NLP任务 | 样本长度不一致 |

## 数据增强

### 数据增强原理

通过变换扩充训练数据：

**目的**：
- 增加数据多样性
- 提高模型泛化
- 减少过拟合

### 图像数据增强

| 方法 | 描述 |
|------|------|
| 翻转 | 水平/垂直翻转 |
| 旋转 | 随机角度旋转 |
| 缩放 | 放大/缩小 |
| 剪裁 | 随机裁剪 |
| 颜色变换 | 调整亮度、对比度 |
| 添加噪声 | 随机噪声 |

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
```

### 文本数据增强

| 方法 | 描述 |
|------|------|
| 同义词替换 | 替换为同义词 |
| 随机插入 | 插入随机词 |
| 随机删除 | 删除随机词 |
| 回译 | 翻译后翻译回来 |

### 数据增强的注意事项

| 注意点 | 描述 |
|------|------|
| 保持语义 | 变换不应改变标签 |
| 适度使用 | 过度可能引入噪声 |
| 任务相关 | 不同任务使用不同方法 |

## 早停策略

### 早停原理

监控验证集性能，适时停止训练：

**流程**：
```
1. 每轮训练后评估验证集
2. 记录最佳验证性能
3. 若连续若干轮无改善，停止训练
4. 使用最佳模型
```

### 早停的参数

| 参数 | 描述 |
|------|------|
| patience | 无改善容忍轮数 |
| min_delta | 最小改善阈值 |
| restore_best_weights | 是否恢复最佳权重 |

```python
# PyTorch早停
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def should_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# TensorFlow早停
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          callbacks=[early_stopping])
```

### 早停的效果

| 效果 | 描述 |
|------|------|
| 防止过拟合 | 在最佳时机停止 |
| 减少训练时间 | 不浪费额外训练 |
| 提高泛化 | 使用验证最佳模型 |

## 案例实践

### 综合正则化示例

```python
import torch
import torch.nn as nn

class RegularizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.dropout1(nn.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(nn.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

# 使用权重衰减
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### TensorFlow综合正则化

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    loss='sparse_categorical_crossentropy'
)

# 早停
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)
model.fit(X_train, y_train, callbacks=[early_stopping])
```

### 正则化效果对比

```python
import matplotlib.pyplot as plt

# 不同正则化强度的模型
regularizations = {
    '无正则化': {'weight_decay': 0, 'dropout': 0},
    '轻微正则化': {'weight_decay': 0.001, 'dropout': 0.2},
    '中等正则化': {'weight_decay': 0.01, 'dropout': 0.5},
    '强正则化': {'weight_decay': 0.1, 'dropout': 0.7}
}

for name, params in regularizations.items():
    model = create_model(params)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50)
    
    plt.plot(history.history['val_accuracy'], label=name)

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Regularization Effect')
plt.show()
```

## 正则化最佳实践

### 正则化组合建议

| 建议 | 描述 |
|------|------|
| 默认组合 | Dropout + L2 |
| CNN | Batch Norm + L2 |
| Transformer | Layer Norm + Dropout |
| RNN | Layer Norm + Dropout |

### 正则化强度调参

| 参数 | 调参范围 |
|------|----------|
| weight_decay | 1e-5 到 1e-2 |
| dropout_rate | 0.2 到 0.5 |
| patience | 3 到 10 |

### 避免过度正则化

**过度正则化症状**：
- 训练误差远高于测试误差
- 模型无法学习训练数据
- 欠拟合表现

**解决**：降低正则化强度。

## 总结

正则化是防止神经网络过拟合的关键技术。核心内容包括：
- L1/L2正则化：限制参数值，L2更适合神经网络
- Dropout：随机丢弃神经元，相当于模型集成
- Batch Normalization：标准化层输入，加速收敛
- Layer Normalization：单样本标准化，适合序列模型
- 数据增强：扩充训练数据，提高泛化
- 早停策略：适时停止训练，防止过拟合

合理使用正则化组合可以显著提高模型泛化能力。

## 延伸阅读

- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [优化算法详解](/2026/05/10/zh-CN/技术文档/机器学习/optimization-algorithms/)
- [超参数调优](/2026/05/10/zh-CN/技术文档/机器学习/hyperparameter-tuning/)
- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
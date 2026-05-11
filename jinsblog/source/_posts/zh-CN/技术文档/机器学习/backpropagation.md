---
title: 反向传播算法详解
date: 2026-04-12
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 神经网络, 反向传播]
---

## 反向传播的历史

### 反向传播的提出

反向传播（Backpropagation）算法在1986年由Hinton等人重新推广，成为训练神经网络的核心方法。

**历史时间线**：
| 时间 | 事件 |
|------|------|
| 1960s | Chain rule用于计算梯度 |
| 1974 | Werbos提出BP算法 |
| 1986 | Hinton等人推广BP |
| 2010s | 深度学习复兴 |

### 反向传播的重要性

**作用**：
- 计算网络中每个参数的梯度
- 使多层网络能够训练
- 是深度学习的基石

## 梯度下降基础

### 梯度下降原理

梯度下降沿负梯度方向更新参数：

$\theta_{t+1} = \theta_t - \alpha \frac{\partial J}{\partial \theta}$

其中：
- $\theta$：参数
- $\alpha$：学习率
- $\frac{\partial J}{\partial \theta}$：梯度

### 梯度的几何意义

- 梯度指向函数增长最快的方向
- 负梯度指向函数下降最快的方向
- 沿负梯度方向可以找到最小值

### 梯度下降的类型

| 类型 | 描述 |
|------|------|
| Batch GD | 使用全部数据计算梯度 |
| Mini-batch GD | 使用部分数据计算梯度 |
| SGD | 使用单个样本计算梯度 |

```python
import numpy as np

def gradient_descent(theta, gradient, learning_rate):
    """梯度下降更新"""
    return theta - learning_rate * gradient

def stochastic_gradient_descent(theta, gradient_func, X, y, learning_rate, epochs):
    """随机梯度下降"""
    for epoch in range(epochs):
        for i in range(len(X)):
            gradient = gradient_func(theta, X[i], y[i])
            theta = theta - learning_rate * gradient
    return theta
```

## 链式法则推导

### 链式法则

**单变量链式法则**：
$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$

**多变量链式法则**：
$\frac{\partial f}{\partial x} = \sum_i \frac{\partial f}{\partial y_i} \cdot \frac{\partial y_i}{\partial x}$

### 神经网络的链式法则应用

神经网络是多层函数复合：
$y = f_3(f_2(f_1(x)))$

梯度需要逐层传递：
$\frac{\partial J}{\partial w_1} = \frac{\partial J}{\partial y} \cdot \frac{\partial y}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial w_1}$

### 链式法则示例

对于简单网络：
$z = wx + b$
$a = \sigma(z)$
$L = L(a, y)$

**梯度计算**：
$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$

$= \frac{\partial L}{\partial a} \cdot \sigma'(z) \cdot x$

```python
def chain_rule_example(w, x, b, y):
    """链式法则示例"""
    # 前向传播
    z = w * x + b
    a = sigmoid(z)
    L = (a - y) ** 2  # MSE
    
    # 反向传播
    dL_da = 2 * (a - y)
    da_dz = sigmoid_derivative(z)
    dz_dw = x
    
    dL_dw = dL_da * da_dz * dz_dw
    
    return dL_dw
```

## 计算图概念

### 计算图

计算图表示计算过程的图结构：
- **节点**：操作或变量
- **边**：数据流

**作用**：
- 可视化计算过程
- 自动微分的基础
- 支持复杂网络结构

### 计算图示例

```
     x ───→ [×] ───→ z ───→ [σ] ───→ a ───→ [L] ───→ Loss
            ↑              ↑              ↑
            w              b              y
```

**节点说明**：
- [×]：乘法操作（z = wx）
- [σ]：激活函数（a = sigmoid(z)）
- [L]：损失函数

### 前向传播与反向传播

**前向传播**：
- 从输入到输出计算各节点值
- 每个节点计算并存储输出

**反向传播**：
- 从输出到输入计算梯度
- 每个节点计算并传递梯度

## 前向传播与反向传播流程

### 前向传播

**流程**：
```
1. 输入数据x
2. 计算第一层: z1 = W1·x + b1, a1 = f(z1)
3. 计算第二层: z2 = W2·a1 + b2, a2 = f(z2)
4. ...
5. 计算输出层: y = f(zL)
6. 计算损失: L = Loss(y, target)
```

### 反向传播

**流程**：
```
1. 计算损失对输出的梯度: dL/dy
2. 计算输出层梯度: dL/dWL = dL/dy · dz/dWL
3. 计算上一层激活的梯度: dL/daL-1
4. 计算上一层权重梯度: dL/dWL-1
5. ...
6. 计算输入层权重梯度: dL/dW1
```

### 算法伪代码

```
前向传播:
for l = 1 to L:
    z[l] = W[l] · a[l-1] + b[l]
    a[l] = activation(z[l])

Loss = compute_loss(a[L], y)

反向传播:
dL/da[L] = loss_gradient(a[L], y)

for l = L to 1:
    dL/dz[l] = dL/da[l] · activation_derivative(z[l])
    dL/dW[l] = dL/dz[l] · a[l-1]
    dL/da[l-1] = W[l]^T · dL/dz[l]
    dL/db[l] = sum(dL/dz[l])

更新参数:
W[l] = W[l] - α · dL/dW[l]
b[l] = b[l] - α · dL/db[l]
```

## 梯度计算示例

### 单层网络梯度

**网络**：$y = \sigma(wx + b)$

**损失**：$L = -y\ln\hat{y} - (1-y)\ln(1-\hat{y})$（交叉熵）

**梯度计算**：
$\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot x$
$\frac{\partial L}{\partial b} = \hat{y} - y$

```python
def single_layer_gradient(x, y_true, w, b):
    """单层网络梯度计算"""
    # 前向传播
    z = w * x + b
    y_pred = sigmoid(z)
    
    # 反向传播
    # 交叉熵 + sigmoid的简化梯度
    dz = y_pred - y_true
    dw = dz * x
    db = dz
    
    return dw, db
```

### 两层网络梯度

**网络**：
$z_1 = W_1 x + b_1, \quad a_1 = \sigma(z_1)$
$z_2 = W_2 a_1 + b_2, \quad a_2 = \sigma(z_2)$

**梯度计算**：
$\frac{\partial L}{\partial W_2} = (a_2 - y) \cdot a_1^T$
$\frac{\partial L}{\partial a_1} = W_2^T (a_2 - y)$
$\frac{\partial L}{\partial W_1} = (a_1(1-a_1) \cdot \frac{\partial L}{\partial a_1}) \cdot x^T$

```python
def two_layer_gradient(x, y_true, W1, b1, W2, b2):
    """两层网络梯度计算"""
    # 前向传播
    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    
    # 反向传播
    # 输出层梯度
    dL_dz2 = a2 - y_true
    dL_dW2 = np.outer(dL_dz2, a1)
    dL_db2 = dL_dz2
    
    # 传递到隐藏层
    dL_da1 = np.dot(W2.T, dL_dz2)
    
    # 隐藏层梯度
    dL_dz1 = dL_da1 * sigmoid_derivative(z1)
    dL_dW1 = np.outer(dL_dz1, x)
    dL_db1 = dL_dz1
    
    return dL_dW1, dL_db1, dL_dW2, dL_db2
```

### Softmax + 交叉熵的梯度

**简化形式**：
$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$

**原因**：交叉熵损失与Softmax配合时梯度计算简化。

```python
def softmax_cross_entropy_gradient(logits, y_true):
    """Softmax + 交叉熵梯度"""
    # 前向传播
    probs = softmax(logits)
    
    # 反向传播（简化形式）
    gradient = probs - y_true
    
    return gradient
```

## 反向传播的实现细节

### 批量处理

**批量梯度计算**：
- 多个样本的梯度平均
- 提高计算效率
- 稳定梯度估计

```python
def batch_gradient(X_batch, y_batch, W, b):
    """批量梯度计算"""
    batch_size = len(X_batch)
    total_dW = 0
    total_db = 0
    
    for x, y in zip(X_batch, y_batch):
        dW, db = single_layer_gradient(x, y, W, b)
        total_dW += dW
        total_db += db
    
    return total_dW / batch_size, total_db / batch_size
```

### 梯度存储与累积

**实现技巧**：
- 中间结果缓存
- 避免重复计算
- 内存与计算效率平衡

### 矩阵运算优化

**向量化计算**：
- 使用矩阵运算替代循环
- 利用GPU并行计算
- 大幅提升效率

```python
def vectorized_gradient(X, y, W, b):
    """向量化梯度计算"""
    # 前向传播（矩阵）
    Z = np.dot(X, W.T) + b
    A = sigmoid(Z)
    
    # 反向传播（矩阵）
    dZ = A - y.reshape(-1, 1)
    dW = np.dot(dZ.T, X) / len(X)
    db = np.mean(dZ, axis=0)
    
    return dW, db
```

## 梯度消失与爆炸问题

### 梯度消失

**现象**：深层网络中梯度逐层衰减，导致深层参数几乎不更新。

**原因**：
- 激活函数导数小（sigmoid最大导数0.25）
- 多个小导数相乘导致梯度极小

**影响**：
- 深层网络难以训练
- 深层权重不学习

### 梯度爆炸

**现象**：深层网络中梯度逐层放大，导致权重更新过大。

**原因**：
- 权重过大
- 多个大导数相乘

**影响**：
- 数值不稳定
- 无法收敛

### 梯度数值范围

假设每层梯度为 $g$，L层网络：

$最终梯度 = g^L$

| g值 | L=10 | L=20 | L=30 |
|-----|------|------|------|
| 0.5 | 0.001 | 0.000001 | 10^-9 |
| 1.0 | 1 | 1 | 1 |
| 1.5 | 57 | 3300 | 190000 |

### 解决方法

| 方法 | 描述 |
|------|------|
| ReLU | 正区间梯度恒为1 |
| Batch Norm | 控制每层输出范围 |
| 梯度裁剪 | 限制梯度大小 |
| 合理初始化 | 初始权重合适 |
| LSTM | 门控机制控制梯度 |

```python
def gradient_clipping(gradient, max_norm):
    """梯度裁剪"""
    norm = np.linalg.norm(gradient)
    if norm > max_norm:
        gradient = gradient * (max_norm / norm)
    return gradient
```

## Python实现

### 手动实现反向传播

```python
import numpy as np

class NeuralNetwork:
    """简单神经网络"""
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        """前向传播"""
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            a = sigmoid(z)
            activations.append(a)
        return activations
    
    def backward(self, activations, y_true):
        """反向传播"""
        gradients_w = []
        gradients_b = []
        
        # 输出层梯度
        dL = activations[-1] - y_true
        
        for l in range(len(self.weights) - 1, -1, -1):
            dw = np.outer(dL, activations[l])
            db = dL
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            
            if l > 0:
                dL = np.dot(self.weights[l].T, dL) * sigmoid_derivative activations[l])
        
        return gradients_w, gradients_b
    
    def update(self, gradients_w, gradients_b, learning_rate):
        """更新参数"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]
    
    def train(self, X, y, epochs, learning_rate):
        """训练"""
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(X, y):
                activations = self.forward(x)
                loss = np.mean((activations[-1] - y_true)**2)
                total_loss += loss
                
                gradients_w, gradients_b = self.backward(activations, y_true)
                self.update(gradients_w, gradients_b, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X):.4f}")
```

### PyTorch自动微分

```python
import torch
import torch.nn as nn

# PyTorch自动计算梯度
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练循环
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        # 前向传播
        output = model(batch_x)
        loss = criterion(output, batch_y)
        
        # 反向传播（自动计算）
        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
```

### TensorFlow自动微分

```python
import tensorflow as tf

# TensorFlow自动微分
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

optimizer = tf.keras.optimizers.Adam()

# 训练循环
for epoch in range(epochs):
    for batch_x, batch_y in train_dataset:
        with tf.GradientTape() as tape:
            output = model(batch_x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, output)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

## 总结

反向传播是神经网络训练的核心算法。核心内容包括：
- 反向传播历史：1986年Hinton推广
- 梯度下降基础：沿负梯度方向更新
- 链式法则：梯度逐层传递的理论基础
- 计算图：可视化计算过程
- 前向传播与反向传播流程：计算值和梯度
- 梯度消失与爆炸：深层网络的挑战

现代深度学习框架自动实现反向传播，但理解其原理对调试和优化至关重要。

## 延伸阅读

- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
- [激活函数详解](/2026/05/10/zh-CN/技术文档/机器学习/activation-functions/)
- [优化算法详解](/2026/05/10/zh-CN/技术文档/机器学习/optimization-algorithms/)
- [正则化技术](/2026/05/10/zh-CN/技术文档/机器学习/regularization/)
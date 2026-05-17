---
title: 激活函数详解
date: 2025-12-04
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 神经网络, 激活函数]
---

## 激活函数的作用

### 为什么需要激活函数

**引入非线性**：
- 没有激活函数，神经网络只是线性变换的组合
- 多层线性网络等价于单层线性网络
- 非线性激活函数赋予网络强大的表达能力

### 激活函数的作用

| 作用 | 描述 |
|------|------|
| 引入非线性 | 使网络能学习复杂模式 |
| 决定神经元输出 | 将输入转换为输出 |
| 影响梯度 | 影响反向传播 |

### 激活函数的位置

激活函数通常在神经元加权求和之后：

$y = f(\sum w_i x_i + b)$

其中 $f$ 是激活函数。

### 激活函数的选择影响

| 影响 | 描述 |
|------|------|
| 网络性能 | 影响模型收敛和准确率 |
| 训练速度 | 影响梯度传播效率 |
| 梯度问题 | 可能导致梯度消失或爆炸 |

## Sigmoid与Tanh

### Sigmoid函数

**定义**：
$\sigma(x) = \frac{1}{1 + e^{-x}}$

**性质**：
- 输出范围：(0, 1)
- 导数：$\sigma'(x) = \sigma(x)(1-\sigma(x))$
- 中心点：$\sigma(0) = 0.5$

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

### Sigmoid的优缺点

| 优点 | 缺点 |
|------|------|
| 输出有界 | 梯度消失问题 |
| 平滑可微 | 输出不是零中心 |
| 适合概率输出 | 指数计算成本高 |

### Sigmoid的梯度消失

当输入很大或很小时：
- $\sigma(x) \approx 1$ 或 $\sigma(x) \approx 0$
- 导数 $\sigma'(x) \approx 0$
- 梯度传播受阻

### Tanh函数

**定义**：
$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

**性质**：
- 输出范围：(-1, 1)
- 导数：$\tanh'(x) = 1 - \tanh^2(x)$
- 中心点：$\tanh(0) = 0$

```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x) ** 2
```

### Tanh vs Sigmoid

| 方面 | Sigmoid | Tanh |
|------|---------|------|
| 输出范围 | (0, 1) | (-1, 1) |
| 零中心 | 否 | 是 |
| 梯度消失 | 有 | 有（比sigmoid轻） |
| 适用场景 | 输出层（概率） | 隐藏层 |

### Tanh的优缺点

| 优点 | 缺点 |
|------|------|
| 零中心 | 梯度消失问题 |
| 输出有界 | 指数计算成本 |
| 梯度比sigmoid大 | 深层网络仍会消失 |

## ReLU及变体

### ReLU函数

**定义**：
$ReLU(x) = \max(0, x)$

**性质**：
- 输出范围：[0, +∞)
- 导数：$ReLU'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$
- 计算简单

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
```

### ReLU的优点

| 优点 | 描述 |
|------|------|
| 计算简单 | 只需比较和取最大 |
| 无梯度消失 | 正区间梯度恒为1 |
| 收敛快 | 梯度传播效率高 |
| 稀疏激活 | 负输入不激活 |

### ReLU的缺点

| 缺点 | 描述 |
|------|------|
| Dead ReLU | 负输入永远不激活 |
| 输出无界 | 可能数值不稳定 |
| 不是零中心 | 可能影响收敛 |

### Dead ReLU问题

当权重更新使得神经元对所有输入都输出负值：
- 神经元"死亡"
- 梯度永远为0
- 无法恢复

**解决方法**：
- 使用LeakyReLU
- 合理初始化权重

### LeakyReLU

**定义**：
$LeakyReLU(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$

其中 $\alpha$ 是小正数（如0.01）。

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

### LeakyReLU的优点

| 优点 | 描述 |
|------|------|
| 解决Dead ReLU | 负区间有梯度 |
| 保持ReLU优点 | 正区间梯度恒为1 |
| 简单有效 | 只多一个参数 |

### ELU（Exponential Linear Unit）

**定义**：
$ELU(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$

```python
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

**特点**：
- 负区间平滑
- 输出均值接近0
- 需要指数计算

### GELU（Gaussian Error Linear Unit）

**定义**：
$GELU(x) = x \cdot P(X \leq x) = x \cdot \Phi(x)$

近似形式：
$GELU(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

**特点**：
- Transformer模型常用
- 平滑非线性
- 表现通常优于ReLU

### ReLU变体对比

| 激活函数 | 负区间梯度 | 平滑性 | 计算复杂度 |
|----------|------------|--------|------------|
| ReLU | 0 | 不平滑 | 低 |
| LeakyReLU | α | 不平滑 | 低 |
| ELU | α(e^x-1) | 平滑 | 中 |
| GELU | 复杂 | 平滑 | 中 |

## Softmax函数

### Softmax定义

**定义**：
$softmax(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$

**性质**：
- 输出范围：(0, 1)
- 输出和为1
- 可解释为概率分布

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 防止溢出
    return exp_x / np.sum(exp_x)
```

### Softmax的作用

| 作用 | 描述 |
|------|------|
| 多分类输出 | 将输出转换为概率 |
| 概率解释 | 输出可解释为类别概率 |
| 与交叉熵配合 | 常用于分类损失 |

### Softmax的计算技巧

**防止溢出**：
减去最大值不影响结果：
$softmax(\mathbf{x})_i = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}$

```python
# 正确实现
def softmax_safe(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)
```

### Softmax导数

$\frac{\partial softmax_i}{\partial x_j} = softmax_i(\delta_{ij} - softmax_j)$

其中 $\delta_{ij}$ 是Kronecker delta。

## 激活函数选择策略

### 选择原则

| 层类型 | 推荐激活函数 |
|--------|--------------|
| 隐藏层 | ReLU、LeakyReLU、GELU |
| 输出层（二分类） | Sigmoid |
| 输出层（多分类） | Softmax |
| 输出层（回归） | 无激活或线性 |

### 不同场景的推荐

| 场景 | 推荐激活函数 |
|------|--------------|
| 一般深度网络 | ReLU |
| Transformer | GELU |
| LSTM/GRU | Tanh、Sigmoid |
| 输出概率 | Sigmoid/Softmax |
| 需要平滑性 | ELU、GELU |

### 避免的选择

| 避免 | 原因 |
|------|------|
| Sigmoid在隐藏层 | 梯度消失严重 |
| Tanh在深层网络 | 深层梯度消失 |
| ReLU在RNN | 可能导致数值问题 |

## 激活函数的梯度问题

### 梯度消失

当激活函数导数很小：
- 梯度在反向传播中衰减
- 深层权重几乎不更新

**容易导致梯度消失的激活函数**：
- Sigmoid（输出接近0或1时）
- Tanh（输出接近-1或1时）

### 梯度爆炸

当激活函数导数很大：
- 梯度在反向传播中放大
- 权重更新过大，数值不稳定

**解决方法**：
- 使用ReLU系列
- 梯度裁剪
- 合理初始化权重

### 激活函数与梯度传播

| 激活函数 | 最大梯度 | 梯度消失风险 |
|----------|----------|--------------|
| Sigmoid | 0.25 | 高 |
| Tanh | 1 | 中 |
| ReLU | 1 | 低（正区间） |
| LeakyReLU | 1 | 低 |

## 案例实践

### 激活函数可视化

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# 各种激活函数
activations = {
    'Sigmoid': sigmoid(x),
    'Tanh': tanh(x),
    'ReLU': relu(x),
    'LeakyReLU': leaky_relu(x),
    'ELU': elu(x),
    'GELU': gelu(x)
}

# 绘制
plt.figure(figsize=(15, 10))
for i, (name, y) in enumerate(activations.items()):
    plt.subplot(2, 3, i+1)
    plt.plot(x, y)
    plt.title(name)
    plt.grid(True)
plt.tight_layout()
plt.show()
```

### 激活函数导数可视化

```python
# 绘制导数
derivatives = {
    'Sigmoid\'': sigmoid_derivative(x),
    'Tanh\'': tanh_derivative(x),
    'ReLU\'': relu_derivative(x),
}

plt.figure(figsize=(15, 5))
for i, (name, y) in enumerate(derivatives.items()):
    plt.subplot(1, 3, i+1)
    plt.plot(x, y)
    plt.title(name)
    plt.grid(True)
plt.tight_layout()
plt.show()
```

### 在PyTorch中使用激活函数

```python
import torch
import torch.nn as nn

# PyTorch激活函数
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(0.01)
elu = nn.ELU()
gelu = nn.GELU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=1)

# 在网络中使用
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
```

### 在TensorFlow中使用激活函数

```python
import tensorflow as tf

# TensorFlow激活函数
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='leaky_relu'),  # 或 tf.nn.leaky_relu
    tf.keras.layers.Dense(10, activation='softmax')
])

# GELU（需要自定义）
def gelu_tf(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / tf.pi) * (x + 0.044715 * x**3)))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=gelu_tf),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 激活函数对比实验

```python
# 对比不同激活函数的效果
activations = ['relu', 'sigmoid', 'tanh', 'elu']

for act in activations:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=act, input_shape=(784,)),
        tf.keras.layers.Dense(128, activation=act),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=0)
    
    print(f"{act}: 最终验证准确率={history.history['val_accuracy'][-1]:.4f}")
```

## 总结

激活函数是神经网络的关键组件。核心内容包括：
- 激活函数作用：引入非线性，赋予网络表达能力
- Sigmoid与Tanh：经典激活函数，但存在梯度消失
- ReLU及变体：现代神经网络主流，解决梯度问题
- Softmax：多分类输出，转换为概率分布
- 选择策略：根据网络类型和任务选择

ReLU是现代神经网络的默认选择，GELU在Transformer中广泛应用。

## 延伸阅读

- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [损失函数详解](/2026/05/10/zh-CN/技术文档/机器学习/loss-functions/)
- [正则化技术](/2026/05/10/zh-CN/技术文档/机器学习/regularization/)
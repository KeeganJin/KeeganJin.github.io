---
title: 神经网络入门
date: 2026-05-07
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 神经网络, 深度学习]
---

## 神经网络历史与发展

### 神经网络的发展历程

| 时期 | 事件 |
|------|------|
| 1940s | McCulloch-Pitts神经元模型 |
| 1950s | 感知机（Perceptron） |
| 1960s | Minsky指出感知机局限 |
| 1980s | 反向传播算法 |
| 1990s | CNN、RNN提出 |
| 2000s | 深度学习萌芽 |
| 2010s | 深度学习爆发（AlexNet、AlphaGo） |
| 2020s | Transformer、大模型时代 |

### 关键里程碑

**感知机局限**（1969）：
- Minsky指出单层感知机无法学习XOR
- 神经网络研究陷入停滞

**反向传播**（1986）：
- Hinton等人提出多层网络训练方法
- 神经网络研究复兴

**AlexNet**（2012）：
- 深度CNN在ImageNet取得突破
- 深度学习时代开启

## 神经元模型

### 生物神经元

**结构**：
- 细胞体：处理信息
- 树突：接收信号
- 轴突：发送信号
- 突触：连接其他神经元

### 人工神经元模型

**数学模型**：
$y = f(\sum_{i=1}^{n} w_i x_i + b)$

其中：
- $x_i$：输入信号
- $w_i$：权重（连接强度）
- $b$：偏置
- $f$：激活函数

### 神经元的组成部分

| 组成 | 描述 |
|------|------|
| 输入 | 来自其他神经元或外界信号 |
| 权重 | 连接强度，可调节 |
| 偏置 | 神经元的阈值 |
| 激活函数 | 决定神经元是否激活 |

### 神元可视化

```python
import numpy as np

def neuron_output(inputs, weights, bias, activation):
    """神经元计算"""
    weighted_sum = np.dot(inputs, weights) + bias
    output = activation(weighted_sum)
    return output

# 示例
inputs = np.array([1, 0.5, -1])
weights = np.array([0.5, 0.3, 0.2])
bias = 0.1

# 使用sigmoid激活
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

output = neuron_output(inputs, weights, bias, sigmoid)
print(f"神经元输出: {output:.4f}")
```

## 感知机与多层感知机

### 感知机

**定义**：最简单的人工神经网络，单层结构。

**模型**：
$y = \text{sign}(\sum_{i=1}^{n} w_i x_i + b)$

**学习规则**：
$w_{new} = w_{old} + \eta (y - \hat{y}) x$

### 感知机的局限

**无法学习非线性函数**：
- XOR问题无法解决
- 只能解决线性可分问题

### XOR问题

```
输入 期望输出
(0,0) → 0
(0,1) → 1
(1,0) → 1
(1,1) → 0
```

单层感知机无法用直线分开。

### 多层感知机（MLP）

**结构**：
- 输入层：接收输入
- 隐藏层：提取特征（多层）
- 输出层：产生输出

**解决XOR**：
- 第一层：学习两个边界
- 第二层：组合边界结果

```python
from sklearn.neural_network import MLPClassifier

# MLP解决XOR
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', random_state=42)
mlp.fit(X, y)

print("预测:")
for x, expected in zip(X, y):
    print(f"{x}: {mlp.predict([x])[0]} (期望: {expected})")
```

## 神经网络的基本结构

### 层的类型

| 层类型 | 描述 |
|--------|------|
| 输入层 | 接收原始数据 |
| 全连接层 | 每个神经元连接所有输入 |
| 卷积层 | 局部连接，提取空间特征 |
| 池化层 | 降低空间维度 |
| 输出层 | 产生最终预测 |

### 网络架构

**前向传播**：
$\mathbf{h}_1 = f(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$
$\mathbf{h}_2 = f(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$
$\mathbf{y} = g(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3)$

### 网络深度

| 深度 | 名称 | 特点 |
|------|------|------|
| 1层 | 浅层网络 | 表达能力有限 |
| 2-4层 | 中等网络 | 可解决大部分问题 |
| 多层 | 深度网络 | 强大表达能力 |

### 参数数量

对于全连接网络：
$Params = \sum_{l=1}^{L} (n_{l-1} \times n_l + n_l)$

其中 $n_l$ 是第 $l$ 层的神经元数。

```python
def count_parameters(layer_sizes):
    """计算网络参数数量"""
    total = 0
    for i in range(1, len(layer_sizes)):
        weights = layer_sizes[i-1] * layer_sizes[i]
        biases = layer_sizes[i]
        total += weights + biases
    return total

# 示例：3层网络
layer_sizes = [784, 128, 64, 10]  # 输入784，隐藏128和64，输出10
params = count_parameters(layer_sizes)
print(f"参数数量: {params}")
```

## 神经网络的能力与局限

### 神经网络的能力

| 能力 | 描述 |
|------|------|
| 函数逼近 | 可逼近任意连续函数 |
| 模式识别 | 学习复杂模式 |
| 特征学习 | 自动提取特征 |
| 泛化能力 | 可泛化到未见数据 |

### 万能逼近定理

对于任意连续函数 $f$，存在神经网络可在任意精度上逼近 $f$。

**条件**：
- 单隐藏层足够（足够多的神经元）
- 激活函数满足一定条件（如sigmoid）

### 神经网络的局限

| 局限 | 描述 |
|------|------|
| 黑盒性质 | 可解释性差 |
| 训练困难 | 需要大量数据和计算 |
| 调参复杂 | 超参数众多 |
| 过拟合风险 | 容易过拟合 |
| 数据依赖 | 需要高质量数据 |

### 深度学习的优势

| 优势 | 描述 |
|------|------|
| 特征学习 | 不需要手工特征工程 |
| 表达能力 | 深层网络可学习复杂模式 |
| 自动化 | 减少人工干预 |

### 深度学习的挑战

| 挑战 | 描述 |
|------|------|
| 数据需求 | 需要大量标注数据 |
| 计算成本 | 训练需要大量计算资源 |
| 黑盒问题 | 解释性差 |
| 调参困难 | 超参数众多 |

## 案例实践

### 使用PyTorch构建简单网络

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 输入到隐藏层
        self.fc2 = nn.Linear(128, 64)   # 隐藏层到隐藏层
        self.fc3 = nn.Linear(64, 10)    # 输出层
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型
model = SimpleNet()
print(model)

# 计算参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"参数总数: {total_params}")
```

### 使用TensorFlow/Keras构建网络

```python
import tensorflow as tf

# Keras构建网络
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### MNIST分类示例

```python
from tensorflow.keras.datasets import mnist

# 加载MNIST数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 预处理
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# 训练
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 评估
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"测试准确率: {test_acc:.4f}")
```

### 网络可视化

```python
import matplotlib.pyplot as plt

# 绘制训练曲线
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.title('Training History')
plt.show()
```

### 不同架构对比

```python
# 单层网络
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='softmax', input_shape=(784,))
])

# 中等网络
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 深层网络
model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 对比
for i, model in enumerate([model1, model2, model3]):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"模型{i+1}准确率: {acc:.4f}")
```

## 神经网络的未来发展

### 当前趋势

| 趋势 | 描述 |
|------|------|
| 更大模型 | 参数规模持续增长 |
| Transformer架构 | 统一多种任务 |
| 多模态学习 | 处理多种类型数据 |
| 自监督学习 | 减少标注依赖 |
| 可解释AI | 提高模型透明度 |

### 技术挑战

| 挑战 | 解决方向 |
|------|----------|
| 计算成本 | 模型压缩、分布式训练 |
| 数据需求 | 自监督学习、数据增强 |
| 可解释性 | 可解释AI技术 |
| 通用性 | 多模态学习 |

## 总结

神经网络是深度学习的基础。核心内容包括：
- 神经网络历史：从感知机到深度学习
- 神经元模型：模拟生物神经元的信息处理
- 感知机与多层感知机：解决非线性问题的演进
- 神经网络结构：输入层、隐藏层、输出层
- 神经网络能力与局限：万能逼近定理与实际挑战

神经网络是强大的机器学习工具，为深度学习奠定了基础。

## 延伸阅读

- [激活函数详解](/2026/05/10/zh-CN/技术文档/机器学习/activation-functions/)
- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [损失函数详解](/2026/05/10/zh-CN/技术文档/机器学习/loss-functions/)
- [卷积神经网络](/2026/05/10/zh-CN/技术文档/机器学习/cnn/)
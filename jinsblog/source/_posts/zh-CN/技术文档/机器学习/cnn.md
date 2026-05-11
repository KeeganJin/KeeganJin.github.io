---
title: 卷积神经网络（CNN）
date: 2026-04-15
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 深度学习, CNN]
---

## CNN基本原理

### CNN的概念

卷积神经网络（Convolutional Neural Network, CNN）是专门处理图像数据的神经网络。

**核心思想**：
- 使用卷积操作提取局部特征
- 使用池化操作降低空间维度
- 保持空间结构信息

### CNN与传统网络的区别

| 方面 | 传统网络 | CNN |
|------|----------|-----|
| 输入处理 | 平铺 | 保持空间结构 |
| 特征提取 | 全连接 | 卷积操作 |
| 参数数量 | 大 | 小（共享权重） |
| 空间信息 | 丢失 | 保留 |

### CNN的优势

| 优势 | 描述 |
|------|------|
| 参数效率 | 卷积核共享权重 |
| 空间理解 | 捕捉局部特征 |
| 平移不变 | 特征位置不变 |
| 层次特征 | 从低到高抽象 |

## 卷积操作详解

### 卷积的数学定义

**二维卷积**：
$S(i,j) = \sum_m \sum_n I(i+m, j+n) K(m,n)$

其中：
- $I$：输入图像
- $K$：卷积核
- $S$：输出特征图

### 卷积核（滤波器）

**卷积核的作用**：
- 提取特定特征
- 边缘检测、纹理检测等
- 学习自适应特征

**常见卷积核**：
| 类型 | 效果 |
|------|------|
| 边缘检测 | 检测边缘 |
| 模糊 | 平滑图像 |
|锐化 | 增强边缘 |

### 卷积参数

| 参数 | 描述 |
|------|------|
| 卷积核大小 | 通常3×3、5×5 |
| 步长（Stride） | 卷积移动步距 |
| 填充（Padding） | 边界填充 |

### 输出尺寸计算

$H_{out} = \frac{H_{in} + 2P - K}{S} + 1$

其中：
- $H_{in}$：输入高度
- $P$：填充
- $K$：卷积核大小
- $S$：步长

```python
import numpy as np

def conv2d(input, kernel, stride=1, padding=0):
    """二维卷积"""
    # 添加填充
    if padding > 0:
        input = np.pad(input, padding)
    
    h_in, w_in = input.shape
    k_h, k_w = kernel.shape
    
    h_out = (h_in - k_h) // stride + 1
    w_out = (w_in - k_w) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = input[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            output[i, j] = np.sum(region * kernel)
    
    return output
```

### 多通道卷积

对于RGB图像：
- 每个通道独立卷积
- 结果求和
- 产生一个特征图

**多个卷积核**：
- 每个卷积核产生一个特征图
- 多个卷积核产生多个特征图

```python
def conv2d_multi_channel(input, kernels):
    """多通道卷积"""
    n_channels = input.shape[2]
    n_filters = kernels.shape[0]
    
    output = np.zeros((h_out, w_out, n_filters))
    
    for f in range(n_filters):
        for c in range(n_channels):
            output[:,:,f] += conv2d(input[:,:,c], kernels[f,:,c])
    
    return output
```

## 池化操作

### 池化的作用

| 作用 | 描述 |
|------|------|
| 降低维度 | 减少计算量 |
| 增加感受野 | 扩大视野范围 |
| 提供不变性 | 小位移不影响 |

### 池化类型

| 类型 | 公式 | 特点 |
|------|------|------|
| 最大池化 | $\max$ | 提取显著特征 |
| 平均池化 | $\frac{1}{n}\sum$ | 平滑特征 |
| 全局池化 | 整体统计 | 降维到一维 |

### 池化参数

| 参数 | 描述 |
|------|------|
| 池化窗口 | 通常2×2 |
| 步长 | 通常等于窗口大小 |
| 填充 | 通常不使用 |

```python
def max_pool2d(input, pool_size=2, stride=2):
    """最大池化"""
    h_in, w_in = input.shape
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)
    
    return output
```

### 池化的效果

**降维效果**：
- 输入：28×28
- 池化：2×2，步长2
- 输出：14×14

## 经典架构：LeNet、AlexNet、VGG、ResNet

### LeNet（1998）

**结构**：
```
输入 → Conv → Pool → Conv → Pool → FC → FC → 输出
```

**特点**：
- 最早的CNN
- 手写数字识别
- 结构简单

```python
import tensorflow as tf

def lenet5():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, 5, activation='tanh'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(16, 5, activation='tanh'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='tanh'),
        tf.keras.layers.Dense(84, activation='tanh'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
```

### AlexNet（2012）

**创新**：
- ReLU激活
- Dropout
- 数据增强
- GPU训练

**结构**：
- 5卷积层 + 3全连接层
- 参数约60M

### VGG（2014）

**特点**：
- 使用小卷积核（3×3）
- 深层网络（16-19层）
- 结构简单统一

**VGG16结构**：
```
[Conv3×3-64] ×2 → Pool
[Conv3×3-128] ×2 → Pool
[Conv3×3-256] ×3 → Pool
[Conv3×3-512] ×3 → Pool
[Conv3×3-512] ×3 → Pool
FC-4096 ×3 → Softmax
```

```python
def vgg16(input_shape):
    model = tf.keras.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        # Block 2
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        # Block 3-5 类似...
        # FC layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(1000, activation='softmax')
    ])
    return model
```

### ResNet（2015）

**核心创新**：残差连接

**问题**：深层网络难以训练（梯度消失）

**解决**：跳跃连接
$y = F(x) + x$

### 残差连接原理

**残差块**：
```
输入 x → Conv → BN → ReLU → Conv → BN → + x → ReLU → 输出
         ↑_________________________________|
```

**效果**：
- 梯度可以直接传递
- 解决梯度消失
- 网络可以很深（100+层）

```python
def residual_block(x, filters):
    """残差块"""
    shortcut = x
    
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x
```

### ResNet变体

| 变体 | 深度 | 特点 |
|------|------|------|
| ResNet-18 | 18层 | 基础版 |
| ResNet-50 | 50层 | 常用 |
| ResNet-101 | 101层 | 更深 |
| ResNet-152 | 152层 | 最深 |

## 残差连接原理详解

### 为什么需要残差

**深层网络问题**：
- 梯度消失
- 训练困难
- 性能可能下降

**残差思想**：
- 学习残差 $F(x) = y - x$ 更容易
- 如果 $F(x) = 0$，网络至少不退化

### 残差块的梯度

$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot (1 + \frac{\partial F}{\partial x})$

**关键**：梯度有"1"的常数项，不会消失。

### 残差的效果

| 效果 | 描述 |
|------|------|
| 解决梯度消失 | 梯度直达浅层 |
| 简化学习 | 学习残差而非映射 |
| 防止退化 | 深层至少等于浅层 |

## CNN的变体与发展

### DenseNet

**密集连接**：每层与之前所有层连接

**特点**：
- 特征重用
- 参数效率高
- 梯度传播好

### Inception/GoogLeNet

**多分支结构**：
- 不同卷积核同时处理
- 组合多尺度特征

### MobileNet

**轻量化设计**：
- 深度可分离卷积
- 减少计算量
- 移动端适用

```python
def depthwise_separable_conv(x, filters):
    """深度可分离卷积"""
    # 深度卷积
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 逐点卷积
    x = tf.keras.layers.Conv2D(filters, 1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    return x
```

### EfficientNet

**复合缩放**：
- 同时缩放深度、宽度、分辨率
- 效率最优

## 案例实践：图像分类

### CNN图像分类示例

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# 加载CIFAR-10数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 预处理
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# CNN模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1)

# 评估
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"测试准确率: {test_acc:.4f}")
```

### PyTorch CNN示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(nn.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(nn.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练循环
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

### 使用预训练模型

```python
# TensorFlow预训练模型
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 添加自定义层
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 微调
base_model.trainable = False  # 先冻结
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)

# 解冻部分层继续训练
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)
```

### 特征可视化

```python
import matplotlib.pyplot as plt

# 提取中间层特征
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(X_test[0:1])

# 可视化特征图
for i, activation in enumerate(activations):
    plt.figure(figsize=(10, 10))
    for j in range(min(32, activation.shape[-1])):
        plt.subplot(8, 8, j+1)
        plt.imshow(activation[0, :, :, j], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Layer {i}')
    plt.show()
```

## 总结

CNN是处理图像数据的专用神经网络。核心内容包括：
- CNN基本原理：卷积提取特征，池化降维
- 卷积操作：卷积核、步长、填充
- 池化操作：最大池化、平均池化
- 经典架构：LeNet、AlexNet、VGG、ResNet
- 残差连接：解决深层网络训练问题
- CNN变体：DenseNet、MobileNet、EfficientNet

CNN是图像识别的核心技术，残差连接解决了深层网络训练问题。

## 延伸阅读

- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [循环神经网络](/2026/05/10/zh-CN/技术文档/机器学习/rnn/)
- [Transformer架构详解](/2026/05/10/zh-CN/技术文档/机器学习/transformer/)
---
title: 循环神经网络（RNN）
date: 2026-04-18
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 深度学习, RNN]
---

## RNN基本原理

### RNN的概念

循环神经网络（Recurrent Neural Network, RNN）是处理序列数据的神经网络。

**核心思想**：
- 利用序列的历史信息
- 隐藏状态传递时间步之间
- 捕捉时间依赖关系

### RNN与传统网络的区别

| 方面 | 传统网络 | RNN |
|------|----------|-----|
| 输入类型 | 固定大小 | 序列 |
| 信息传递 | 无时间关联 | 时间步传递 |
| 记忆能力 | 无记忆 | 有隐藏状态 |
| 输出数量 | 单一输出 | 每步输出 |

### RNN的应用场景

| 应用 | 描述 |
|------|------|
| 文本生成 | 生成连续文本 |
| 语言模型 | 预测下一个词 |
| 机器翻译 | 序列到序列 |
| 时间序列预测 | 预测未来值 |
| 语音识别 | 音频到文本 |

## 序列数据处理

### 序列数据特点

| 特点 | 描述 |
|------|------|
| 时间依赖 | 当前值依赖历史 |
| 变长输入 | 序列长度可变 |
| 顺序重要 | 顺序影响含义 |

### 序列建模任务

| 任务类型 | 输入 | 输出 |
|----------|------|------|
| 一对多 | 单输入 | 序列输出 |
| 多对一 | 序列输入 | 单输出 |
| 多对多 | 序列输入 | 序列输出 |

## RNN的梯度问题

### 梯度消失

**问题**：时间步过多时梯度衰减

**原因**：
- 激活函数（sigmoid/tanh）导数小于1
- 多次相乘导致梯度极小

**影响**：
- 无法学习长距离依赖
- 远端信息无法传递

### 梯度爆炸

**问题**：梯度在反向传播中放大

**原因**：
- 权重过大
- 梯度多次乘以大数

**影响**：
- 数值不稳定
- 无法收敛

### 梯度计算分析

假设梯度每步乘以 $g$：
- $T$ 步后：$g^T$
- $g < 1$：梯度消失
- $g > 1$：梯度爆炸

### 解决方案

| 方法 | 解决的问题 |
|------|------------|
| LSTM/GRU | 梯度消失 |
| 梯度裁剪 | 梯度爆炸 |
| ReLU激活 | 减轻梯度消失 |

## 时序反向传播（BPTT）

### BPTT原理

Backpropagation Through Time（BPTT）是RNN的反向传播算法。

**流程**：
```
前向传播:
for t = 1 to T:
    h_t = f(W h_{t-1} + U x_t + b)

反向传播:
for t = T to 1:
    计算各时间步梯度
    累积梯度更新参数
```

### BPTT的计算

**前向传播**：
$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
$y_t = \sigma(W_{hy} h_t + b_y)$

**反向传播**：
$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}$

其中 $\frac{\partial h_t}{\partial W_{hh}}$ 需要展开计算历史影响。

### Truncated BPTT

截断BPTT减少计算量：
- 只反向传播有限步
- 平衡效率和准确度

```python
def truncated_bptt(model, X, y, truncation_length=20):
    """截断BPTT"""
    total_loss = 0
    hidden = model.init_hidden()
    
    for t in range(len(X)):
        output, hidden = model.forward(X[t], hidden)
        loss = compute_loss(output, y[t])
        total_loss += loss
        
        if (t + 1) % truncation_length == 0:
            # 反向传播 truncation_length 步
            hidden = model.init_hidden()  # 重置隐藏状态
            model.backward_and_update()
    
    return total_loss
```

## RNN变体

### Simple RNN

**基本RNN**：
$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$

**特点**：
- 结构简单
- 容易梯度消失
- 适合短序列

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_x = np.random.randn(hidden_size, input_size) * 0.01
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)
    
    def forward(self, x, h_prev):
        """前向传播"""
        h = np.tanh(np.dot(self.W_h, h_prev) + np.dot(self.W_x, x) + self.b_h)
        y = np.dot(self.W_y, h) + self.b_y
        return y, h
    
    def forward_sequence(self, X):
        """处理序列"""
        h = np.zeros(self.W_h.shape[0])
        outputs = []
        
        for x in X:
            y, h = self.forward(x, h)
            outputs.append(y)
        
        return outputs, h
```

### Bidirectional RNN

**双向RNN**：
- 前向处理：从过去到未来
- 后向处理：从未来到过去
- 组合两个方向信息

$h_t^{forward} = f(W_f h_{t-1}^{forward} + U_f x_t)$
$h_t^{backward} = f(W_b h_{t+1}^{backward} + U_b x_t)$
$y_t = g(V [h_t^{forward}; h_t^{backward}])$

**应用场景**：
- 需要全局信息的任务
- 语言理解、翻译

```python
# PyTorch双向RNN
import torch.nn as nn

bi_rnn = nn.RNN(input_size, hidden_size, bidirectional=True)
```

### Deep RNN

**深层RNN**：多个RNN层堆叠

```
输入 → RNN Layer 1 → RNN Layer 2 → ... → 输出
```

**特点**：
- 更强的表达能力
- 更深的特征抽象

```python
# 多层RNN
deep_rnn = nn.RNN(input_size, hidden_size, num_layers=3)
```

## 案例实践：文本生成

### RNN文本生成示例

```python
import numpy as np

class TextRNN:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 初始化参数
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_xh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.W_hy = np.random.randn(vocab_size, hidden_size) * 0.01
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(vocab_size)
    
    def forward(self, x, h_prev):
        """前向传播"""
        h = np.tanh(np.dot(self.W_hh, h_prev) + np.dot(self.W_xh, x) + self.b_h)
        y = np.dot(self.W_hy, h) + self.b_y
        return y, h
    
    def sample(self, seed_char, length):
        """生成文本"""
        h = np.zeros(self.hidden_size)
        x = np.zeros(self.vocab_size)
        x[seed_char] = 1
        
        generated = []
        for _ in range(length):
            y, h = self.forward(x, h)
            # 采样下一个字符
            prob = np.exp(y) / np.sum(np.exp(y))  # softmax
            next_char = np.random.choice(self.vocab_size, p=prob)
            generated.append(next_char)
            
            x = np.zeros(self.vocab_size)
            x[next_char] = 1
        
        return generated
```

### PyTorch RNN示例

```python
import torch
import torch.nn as nn

class TextGenerationRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden
    
    def generate(self, start_token, length):
        """生成文本"""
        hidden = None
        generated = [start_token]
        
        x = torch.tensor([[start_token]])
        
        for _ in range(length):
            output, hidden = self.forward(x, hidden)
            prob = torch.softmax(output[0, -1], dim=0)
            next_token = torch.multinomial(prob, 1).item()
            generated.append(next_token)
            x = torch.tensor([[next_token]])
        
        return generated

# 训练
model = TextGenerationRNN(vocab_size, 128, 256)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output, _ = model(batch_x)
        loss = criterion(output.view(-1, vocab_size), batch_y.view(-1))
        loss.backward()
        optimizer.step()
```

### TensorFlow RNN示例

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128),
    tf.keras.layers.SimpleRNN(256, return_sequences=True),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

model.fit(X_train, y_train, epochs=10, batch_size=32)
```

### 序列预测示例

```python
import numpy as np
import torch
import torch.nn as nn

class SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        output, hidden = self.rnn(x)
        output = self.fc(output[:, -1, :])  # 只使用最后一步输出
        return output

# 时间序列预测
model = SequencePredictor(input_size=1, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 生成时间序列数据
def generate_sequence(length):
    t = np.linspace(0, 20, length)
    return np.sin(t)

X = generate_sequence(1000)
y = generate_sequence(1000)[1:]  # 预测下一个值

# 训练
for epoch in range(epochs):
    for i in range(len(X) - sequence_length):
        seq = X[i:i+sequence_length]
        target = y[i+sequence_length]
        
        optimizer.zero_grad()
        output = model(torch.tensor(seq).float().unsqueeze(0))
        loss = criterion(output, torch.tensor(target).float())
        loss.backward()
        optimizer.step()
```

## RNN的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 处理序列 | 天然处理序列数据 |
| 理论记忆 | 可以记住历史信息 |
| 变长输入 | 接受不同长度序列 |
| 时间建模 | 捕捉时间依赖 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 梯度消失 | 长序列问题 |
| 计算慢 | 串行处理 |
| 训练困难 | 调参复杂 |
| 记忆有限 | 实际记忆能力有限 |

### RNN vs LSTM/GRU

| 方面 | Simple RNN | LSTM/GRU |
|------|------------|----------|
| 长距离依赖 | 困难 | 可以学习 |
| 梯度消失 | 严重 | 解决 |
| 参数数量 | 少 | 多 |
| 训练效率 | 高 | 较低 |

## 总结

RNN是处理序列数据的基础模型。核心内容包括：
- RNN基本原理：隐藏状态传递，处理序列
- 序列数据处理：时间依赖，变长输入
- 梯度问题：梯度消失和爆炸
- BPTT：时序反向传播算法
- RNN变体：双向RNN、深层RNN

Simple RNN存在梯度消失问题，LSTM/GRU是更好的选择。

## 延伸阅读

- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
- [LSTM与GRU](/2026/05/10/zh-CN/技术文档/机器学习/lstm-gru/)
- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [Transformer架构详解](/2026/05/10/zh-CN/技术文档/机器学习/transformer/)
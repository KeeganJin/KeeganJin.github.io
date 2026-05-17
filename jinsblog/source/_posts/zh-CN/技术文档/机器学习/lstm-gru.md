---
title: LSTM与GRU
date: 2025-12-26
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 深度学习, LSTM]
---

## LSTM门控机制详解

### LSTM的提出

长短期记忆网络（Long Short-Term Memory, LSTM）由Hochreiter和Schmidhuber于1997年提出，解决RNN的梯度消失问题。

**核心创新**：门控机制控制信息流动。

### LSTM的结构

**三个门**：
- 遗忘门：决定丢弃多少信息
- 输入门：决定写入多少信息
- 输出门：决定输出多少信息

**两个状态**：
- 细胞状态（Cell State）：长期记忆
- 隐藏状态：短期输出

### LSTM的数学公式

**遗忘门**：
$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$

决定从细胞状态丢弃多少信息（0完全遗忘，1完全保留）。

**输入门**：
$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$

决定写入多少新信息。

**更新细胞状态**：
$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$

**输出门**：
$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
$h_t = o_t \cdot \tanh(C_t)$

### LSTM的工作流程

```
输入 x_t 和 h_{t-1}
    ↓
遗忘门 → 决定遗忘多少 C_{t-1}
    ↓
输入门 → 决定写入多少新信息
    ↓
候选细胞状态 → tanh 生成候选值
    ↓
更新细胞状态 C_t = f·C_{t-1} + i·C̃
    ↓
输出门 → 决定输出多少
    ↓
隐藏状态 h_t = o·tanh(C_t)
```

### LSTM门的作用详解

| 门 | 作用 | 值范围 |
|----|------|--------|
| 遗忘门 | 清理旧记忆 | 0-1 |
| 输入门 | 添加新记忆 | 0-1 |
| 输出门 | 控制输出 | 0-1 |

### LSTM解决梯度消失

**关键**：细胞状态更新是加法操作，而非乘法。

$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$

梯度：
$\frac{\partial C_t}{\partial C_{t-1}} = f_t$

只要 $f_t$ 接近1，梯度可以保持传递。

```python
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # 初始化权重
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_C = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        self.b_f = np.zeros(hidden_size)
        self.b_i = np.zeros(hidden_size)
        self.b_C = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
    
    def forward(self, x, h_prev, C_prev):
        """前向传播"""
        concat = np.concatenate([h_prev, x])
        
        # 遗忘门
        f = sigmoid(np.dot(self.W_f, concat) + self.b_f)
        
        # 输入门
        i = sigmoid(np.dot(self.W_i, concat) + self.b_i)
        C_tilde = tanh(np.dot(self.W_C, concat) + self.b_C)
        
        # 更新细胞状态
        C = f * C_prev + i * C_tilde
        
        # 输出门
        o = sigmoid(np.dot(self.W_o, concat) + self.b_o)
        h = o * tanh(C)
        
        return h, C, f, i, C_tilde, o
```

## LSTM解决梯度问题

### 为什么LSTM能解决梯度消失

**原因分析**：

1. **加法更新**：细胞状态更新是加法，梯度直接传递
2. **门控调节**：遗忘门可以保持梯度接近1
3. **线性路径**：细胞状态提供线性梯度路径

### 梯度传播路径

RNN：$h_t = \tanh(W h_{t-1})$ → 梯度乘以 $\tanh'$ → 衰减

LSTM：$C_t = f_t C_{t-1} + ...$ → 梯度乘以 $f_t$ → 可控制

### 训练LSTM的注意事项

| 注意点 | 描述 |
|------|------|
| 初始化 | 遗忘门初始偏向1 |
| 学习率 | 适当调整 |
| 梯度裁剪 | 仍可能梯度爆炸 |

## GRU简化设计

### GRU的提出

门控循环单元（Gated Recurrent Unit, GRU）是LSTM的简化版本，由Cho等人于2014年提出。

**简化**：只有两个门，无细胞状态。

### GRU的结构

**两个门**：
- 重置门：决定忽略多少过去信息
- 更新门：决定保留多少过去信息

### GRU的数学公式

**重置门**：
$r_t = \sigma(W_r [h_{t-1}, x_t])$

**更新门**：
$z_t = \sigma(W_z [h_{t-1}, x_t])$

**候选隐藏状态**：
$\tilde{h}_t = \tanh(W [r_t \cdot h_{t-1}, x_t])$

**更新隐藏状态**：
$h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t$

### GRU的工作流程

```
输入 x_t 和 h_{t-1}
    ↓
重置门 → 决定重置多少 h_{t-1}
    ↓
候选状态 → tanh 生成候选值
    ↓
更新门 → 决定新旧状态比例
    ↓
隐藏状态 h_t = (1-z)·h_{t-1} + z·h̃
```

### GRU门的作用

| 门 | 作用 |
|----|------|
| 重置门 | 控制过去信息的使用 |
| 更新门 | 控制新旧信息的混合 |

```python
class GRU:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # 初始化权重
        self.W_r = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_z = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        self.b_r = np.zeros(hidden_size)
        self.b_z = np.zeros(hidden_size)
        self.b = np.zeros(hidden_size)
    
    def forward(self, x, h_prev):
        """前向传播"""
        concat = np.concatenate([h_prev, x])
        
        # 重置门
        r = sigmoid(np.dot(self.W_r, concat) + self.b_r)
        
        # 更新门
        z = sigmoid(np.dot(self.W_z, concat) + self.b_z)
        
        # 候选隐藏状态
        h_tilde = tanh(np.dot(self.W, np.concatenate([r * h_prev, x])) + self.b)
        
        # 更新隐藏状态
        h = (1 - z) * h_prev + z * h_tilde
        
        return h, r, z, h_tilde
```

## LSTM与GRU对比

### 结构对比

| 方面 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3 | 2 |
| 状态数量 | 2（细胞+隐藏） | 1（隐藏） |
| 参数数量 | 多 | 少 |
| 计算复杂度 | 高 | 低 |

### 性能对比

| 方面 | LSTM | GRU |
|------|------|-----|
| 长序列能力 | 强 | 略弱 |
| 训练速度 | 较慢 | 较快 |
| 参数效率 | 低 | 高 |
| 表达能力 | 高 | 中等 |

### 选择建议

| 场景 | 推荐 |
|------|------|
| 长序列 | LSTM |
| 计算资源有限 | GRU |
| 数据量小 | GRU |
| 复杂任务 | LSTM |

### 实际性能差异

实践中两者性能相近，GRU更简单高效：

```python
import torch
import torch.nn as nn

# LSTM
lstm = nn.LSTM(input_size, hidden_size, num_layers=2)

# GRU
gru = nn.GRU(input_size, hidden_size, num_layers=2)
```

## 双向LSTM

### BiLSTM原理

双向LSTM同时处理正向和反向序列：

**正向LSTM**：从 $t=1$ 到 $t=T$
**反向LSTM**：从 $t=T$ 到 $t=1$

**输出组合**：
$h_t = [h_t^{forward}; h_t^{backward}]$

### BiLSTM的优势

| 优势 | 描述 |
|------|------|
| 全局信息 | 同时看到前后文 |
| 更好理解 | 结合正反信息 |
| 适合NLP | 语言理解任务 |

### BiLSTM应用场景

| 场景 | 描述 |
|------|------|
| 语言理解 | 理解完整句子 |
| 机器翻译 | 编码器使用 |
| 文本分类 | 分类任务 |

```python
import torch.nn as nn

# 双向LSTM
bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

# 输出维度加倍
# hidden_size * 2
```

## 案例实践：序列预测

### LSTM序列预测示例

```python
import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output[:, -1, :])  # 使用最后一步
        return output, hidden

# 时间序列预测
model = LSTMModel(input_size=1, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练
for epoch in range(epochs):
    for seq, target in train_data:
        optimizer.zero_grad()
        output, _ = model(seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### 文本生成LSTM

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=20)

# 生成文本
def generate_text(model, seed_text, length):
    text = seed_text
    for _ in range(length):
        x = encode_text(text[-sequence_length:])
        pred = model.predict(x)
        next_char = decode_prediction(pred)
        text += next_char
    return text
```

### GRU文本分类

```python
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        # 组合双向隐藏状态
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden)

model = TextClassifier(vocab_size, 128, 256, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
```

### 情感分析示例

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# 评估
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"测试准确率: {test_acc:.4f}")
```

### 股票价格预测

```python
import numpy as np
import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

model = StockPredictor(input_size=5, hidden_size=64, num_layers=2)

# 准备数据：使用滑动窗口
def prepare_sequences(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return np.array(sequences), np.array(targets)

X, y = prepare_sequences(stock_prices, window_size=30)
```

## LSTM/GRU训练技巧

### 遗忘门初始化

**建议**：遗忘门偏置初始化为正值（如1），使初始遗忘门接近1。

```python
# PyTorch设置遗忘门偏置
for name, param in lstm.named_parameters():
    if 'bias' in name:
        n = param.size(0)
        param.data[n//4:n//2].fill_(1)  # 遗忘门偏置
```

### 梯度裁剪

LSTM仍可能梯度爆炸，需要梯度裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
```

### 学习率选择

| 建议 | 学习率 |
|------|--------|
| LSTM | 0.001-0.01 |
| GRU | 类似 |

### 批量大小

序列任务通常使用较小批量：
- 批量大小：32-128
- 序列长度固定或使用填充

## 总结

LSTM和GRU是解决RNN梯度问题的门控网络。核心内容包括：
- LSTM门控机制：遗忘门、输入门、输出门
- LSTM解决梯度消失：细胞状态加法更新
- GRU简化设计：重置门、更新门
- LSTM与GRU对比：LSTM更强大，GRU更高效
- 双向LSTM：同时处理正反向序列

LSTM适合长序列复杂任务，GRU适合计算资源有限场景。

## 延伸阅读

- [循环神经网络](/2026/05/10/zh-CN/技术文档/机器学习/rnn/)
- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [Transformer架构详解](/2026/05/10/zh-CN/技术文档/机器学习/transformer/)
- [注意力机制详解](/2026/05/10/zh-CN/技术文档/机器学习/attention-mechanism/)
---
title: Transformer架构详解
date: 2026-04-22
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 深度学习, Transformer]
---

## Transformer的设计理念

### Transformer的提出

Transformer由Google于2017年在论文"Attention Is All You Need"中提出。

**核心创新**：
- 完全基于注意力机制
- 无循环结构，支持并行
- 位置编码替代序列处理

### Transformer vs RNN

| 方面 | RNN | Transformer |
|------|-----|-------------|
| 计算方式 | 串行 | 并行 |
| 长距离依赖 | 困难 | 直接 |
| 训练效率 | 低 | 高 |
| 位置信息 | 自然 | 需编码 |

### Transformer的影响

- 成为NLP主流架构
- 支撑大模型发展
- 扩展到视觉、多模态

## 自注意力机制详解

### 自注意力的概念

自注意力让序列中每个位置与其他位置交互：

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

**含义**：
- Q（Query）：查询向量
- K（Key）：键向量
- V（Value）：值向量
- 计算每个位置对其他位置的注意力

### 自注意力的计算流程

```
输入 X
  ↓
线性变换 → Q, K, V
  ↓
Q·K^T → 注意力分数
  ↓
缩放 → 分数 / √d_k
  ↓
Softmax → 注意力权重
  ↓
权重·V → 输出
```

### 注意力分数计算

**点积注意力**：
$score(q, k) = q \cdot k^T$

**缩放**：
$scaled\_score = \frac{q \cdot k^T}{\sqrt{d_k}}$

**原因**：防止点积过大导致softmax梯度小。

### 自注意力的矩阵形式

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

其中：
- Q：$[n \times d_k]$ 查询矩阵
- K：$[n \times d_k]$ 键矩阵  
- V：$[n \times d_v]$ 值矩阵

```python
import numpy as np

def self_attention(X, W_Q, W_K, W_V):
    """自注意力实现"""
    # 线性变换
    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)
    
    # 注意力分数
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Softmax
    attention_weights = softmax(scores)
    
    # 输出
    output = np.dot(attention_weights, V)
    
    return output, attention_weights
```

### 自注意力示例

**句子**："The animal didn't cross the street because it was too tired"

**注意力**："it" 对其他词的注意力：
- 高注意力："animal", "tired"
- 低注意力："street", "cross"

## 多头注意力

### 多头注意力的原理

并行运行多个注意力头，捕捉不同关系：

$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$

$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$

### 多头的作用

| 作用 | 描述 |
|------|------|
| 多角度关注 | 不同头关注不同关系 |
| 丰富表示 | 组合多种信息 |
| 提高能力 | 学习复杂模式 |

### 多头注意力实现

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 各头的权重矩阵
        self.W_Q = [np.random.randn(d_model, self.d_k) for _ in range(num_heads)]
        self.W_K = [np.random.randn(d_model, self.d_k) for _ in range(num_heads)]
        self.W_V = [np.random.randn(d_model, self.d_k) for _ in range(num_heads)]
        self.W_O = np.random.randn(d_model, d_model)
    
    def forward(self, X):
        heads = []
        for i in range(self.num_heads):
            Q = np.dot(X, self.W_Q[i])
            K = np.dot(X, self.W_K[i])
            V = np.dot(X, self.W_V[i])
            head = scaled_dot_product_attention(Q, K, V)
            heads.append(head)
        
        # 拼接并变换
        concat = np.concatenate(heads, axis=-1)
        output = np.dot(concat, self.W_O)
        
        return output
```

### 多头数量选择

| 模型 | 头数量 |
|------|--------|
| BERT-base | 12 |
| BERT-large | 16 |
| GPT-2 | 12 |
| GPT-3 | 96 |

## 位置编码

### 为什么需要位置编码

Transformer无循环结构，无法自然感知位置顺序。

**解决**：添加位置信息到输入。

### 位置编码公式

**Sinusoidal位置编码**：
$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$
$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$

其中：
- pos：位置索引
- i：维度索引
- d：编码维度

### 位置编码的特点

| 特点 | 描述 |
|------|------|
| 可扩展 | 任意长度序列 |
| 相对位置 | 通过三角函数感知 |
| 无需学习 | 固定编码 |

```python
def positional_encoding(max_len, d_model):
    """位置编码"""
    PE = np.zeros((max_len, d_model))
    
    for pos in range(max_len):
        for i in range(d_model):
            if i % 2 == 0:
                PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                PE[pos, i] = np.cos(pos / (10000 ** ((i-1) / d_model)))
    
    return PE

# 示例
PE = positional_encoding(512, 512)
```

### 其他位置编码

| 类型 | 特点 |
|------|------|
| Sinusoidal | 固定编码 |
| Learnable | 可学习编码 |
| Relative | 相对位置编码 |
| Rotary (RoPE) | 旋转位置编码 |

## 编码器-解码器结构

### Transformer架构

```
编码器（N层）:
输入 → 位置编码 → 自注意力 → Add&Norm → 前馈网络 → Add&Norm → 输出

解码器（N层）:
输入 → 位置编码 → 带掩码自注意力 → Add&Norm → 交叉注意力 → Add&Norm → 前馈网络 → Add&Norm → 输出
```

### 编码器结构

每层编码器包含：
1. 多头自注意力
2. Add & Norm（残差+归一化）
3. 前馈网络
4. Add & Norm

```python
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x):
        # 自注意力 + 残差 + 归一化
        attn_output = self.mha(x)
        x = self.norm1(x + attn_output)
        
        # 前馈网络 + 残差 + 归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
```

### 解码器结构

每层解码器包含：
1. 带掩码自注意力（防止看到未来）
2. Add & Norm
3. 交叉注意力（关注编码器输出）
4. Add & Norm
5. 前馈网络
6. Add & Norm

```python
class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x, encoder_output, mask=None):
        # 带掩码自注意力
        attn1 = self.masked_mha(x, mask=mask)
        x = self.norm1(x + attn1)
        
        # 交叉注意力
        attn2 = self.cross_mha(x, encoder_output)
        x = self.norm2(x + attn2)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x
```

### 掩码的作用

**编码器掩码**：无掩码，全局注意力

**解码器掩码**：
- 防止看到未来词
- 下三角掩码

```python
def create_mask(seq_len):
    """创建下三角掩码"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask == 0  # True表示可关注
```

## Transformer变体

### Encoder-only（BERT）

只用编码器，适合理解任务：
- 文本分类
- 命名实体识别
- 问答

### Decoder-only（GPT）

只用解码器，适合生成任务：
- 文本生成
- 语言模型
- 代码生成

### Encoder-Decoder（T5）

完整结构，适合转换任务：
- 翻译
- 摘要
- 问答生成

| 变体 | 适用任务 |
|------|----------|
| BERT | 理解 |
| GPT | 生成 |
| T5 | 转换 |

## 案例实践

### PyTorch Transformer

```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_ff)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码
        src_embedded = self.embedding(src) + self.pos_encoding[:src.size(1)]
        encoder_output = self.encoder(src_embedded, src_mask)
        
        # 解码
        tgt_embedded = self.embedding(tgt) + self.pos_encoding[:tgt.size(1)]
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask)
        
        return self.fc(decoder_output)

model = TransformerModel(vocab_size=10000, d_model=512, num_heads=8, num_layers=6, d_ff=2048)
```

### TensorFlow Transformer

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, d_model),
    tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model // 8),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dense(d_ff, activation='relu'),
    tf.keras.layers.Dense(d_model),
    tf.keras.layers.LayerNormalization()
])

# 使用内置Transformer
transformer = tf.keras.layers.Transformer(
    num_heads=8,
    d_model=512,
    d_ff=2048,
    num_layers=6
)
```

### 文本翻译示例

```python
import torch
import torch.nn as nn

class TranslationTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.fc_out = nn.Linear(d_model, tgt_vocab)
    
    def forward(self, src, tgt):
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output)

# 训练
model = TranslationTransformer(src_vocab=10000, tgt_vocab=10000)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    for src_batch, tgt_batch in train_loader:
        optimizer.zero_grad()
        output = model(src_batch, tgt_batch[:, :-1])
        loss = criterion(output.view(-1, tgt_vocab), tgt_batch[:, 1:].view(-1))
        loss.backward()
        optimizer.step()
```

### BERT风格模型

```python
import transformers
from transformers import BertModel, BertTokenizer

# 使用预训练BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 文本编码
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')

# 获取表示
outputs = model(**inputs)
hidden_states = outputs.last_hidden_state  # 每个token的表示
pooled_output = outputs.pooler_output  # 整句表示
```

### GPT风格生成

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 文本生成
input_text = "The quick brown fox"
inputs = tokenizer(input_text, return_tensors='pt')

outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    num_return_sequences=1
)

generated_text = tokenizer.decode(outputs[0])
print(generated_text)
```

## Transformer的关键设计

### 残差连接

$Output = LayerNorm(x + Sublayer(x))$

**作用**：
- 梯度直接传递
- 简化学习

### 层归一化

$LayerNorm(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$

**作用**：
- 稳定训练
- 加速收敛

### 前馈网络

$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$

**作用**：
- 非线性变换
- 增强表达能力

## Transformer的优势与挑战

### 优势

| 优势 | 描述 |
|------|------|
| 并行计算 | 可并行处理序列 |
| 长距离依赖 | 直接全局注意力 |
| 灵活架构 | 编码器/解码器组合 |
| 强表达力 | 多头注意力丰富表示 |

### 挑战

| 挑战 | 描述 |
|------|------|
| 计算复杂度 | 自注意力O(n²) |
| 内存占用 | 长序列内存大 |
| 位置编码 | 需额外编码 |

### 长序列优化

| 方法 | 描述 |
|------|------|
| 稀疏注意力 | 只关注部分位置 |
| 分块注意力 | 分块计算 |
| 线性注意力 | 降低复杂度 |

## 总结

Transformer是现代深度学习的核心架构。核心内容包括：
- Transformer设计理念：完全基于注意力，并行计算
- 自注意力机制：序列内部位置交互
- 多头注意力：多角度关注，丰富表示
- 位置编码：添加位置信息
- 编码器-解码器结构：理解+生成

Transformer成为大模型的基础架构，推动了AI的发展。

## 延伸阅读

- [注意力机制详解](/2026/05/10/zh-CN/技术文档/机器学习/attention-mechanism/)
- [LSTM与GRU](/2026/05/10/zh-CN/技术文档/机器学习/lstm-gru/)
- [预训练语言模型](/2026/05/10/zh-CN/技术文档/机器学习/pretraining-lm/)
- [大模型架构演进](/2026/05/10/zh-CN/技术文档/机器学习/llm-architecture/)
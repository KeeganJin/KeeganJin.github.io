---
title: 注意力机制详解
date: 2026-04-11
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 深度学习, 注意力机制]
---

## 注意力机制起源

### 注意力的引入

注意力机制最早在机器翻译中引入（Bahdanau等人，2014）。

**动机**：
- 解决RNN长序列瓶颈
- 让模型关注重要部分
- 类似人类选择性注意

### 人类注意力类比

人类视觉：
- 不是同时关注所有信息
- 选择性关注重要区域
- 忽略无关信息

**机器注意力**：
- 模拟这种选择性关注
- 动态分配"注意力"权重

### 注意力的作用

| 作用 | 描述 |
|------|------|
| 聚焦重点 | 关注关键信息 |
| 忽略噪声 | 降低无关信息影响 |
| 提高效率 | 计算更精准 |
| 可解释性 | 可查看关注点 |

## 软注意力与硬注意力

### 软注意力

**定义**：对所有位置分配概率权重，加权求和。

$attention\_output = \sum_i \alpha_i v_i$

其中 $\alpha_i$ 是注意力权重，$\sum_i \alpha_i = 1$。

**特点**：
- 可微分，可训练
- 所有位置都参与
- 计算稳定

### 硬注意力

**定义**：随机选择部分位置，只关注选中位置。

$attention\_output = v_j, \quad j \sim p(\alpha)$

**特点**：
- 不可微分，需要强化学习或采样
- 计算高效
- 可能丢失信息

### 软注意力 vs 硬注意力

| 方面 | 软注意力 | 硬注意力 |
|------|----------|----------|
| 可微分 | 是 | 否 |
| 计算成本 | 高（全部位置） | 低（部分位置） |
| 信息完整 | 完整 | 可能丢失 |
| 实用性 | 广泛使用 | 较少使用 |

### 确定性与随机性

软注意力：确定性计算，权重可学习
硬注意力：随机采样，训练复杂

**实践中**：软注意力更常用。

## 自注意力

### 自注意力的概念

自注意力（Self-Attention）让序列内部位置相互交互。

**与普通注意力的区别**：
| 方面 | 普通注意力 | 自注意力 |
|------|------------|----------|
| 来源 | 不同序列 | 同一序列 |
| 应用 | 翻译等 | 序列内部关系 |
| 信息流 | 跨序列 | 序列内部 |

### 自注意力的计算

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

**Q, K, V来自同一输入**：
$Q = XW_Q$
$K = XW_K$
$V = XW_V$

### 自注意力示例

**句子**："I love apples because they are delicious"

"they"的自注意力：
- 高关注："apples"
- 低关注："I", "love"

**矩阵表示**：
```
位置   I  love apples because they are delicious
I      高  中   低     低    低  低    低
love   中  高   中     低    低  低    低
apples 低  中   高     低    高  低    高
...
```

### 自注意力的优势

| 优势 | 描述 |
|------|------|
| 全局视野 | 每位置可关注所有位置 |
| 长距离依赖 | 直接连接，无需传递 |
| 并行计算 | 所有位置同时计算 |
| 灵活权重 | 动态学习关注模式 |

## 多头注意力

### 多头注意力的原理

并行执行多个独立的注意力计算：

$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$

每个头使用不同的参数：
$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$

### 多头的意义

**类比**：多个专家从不同角度分析。

**效果**：
- 不同头学习不同关系
- 捕捉多种模式
- 丰富表示

### 多头示例

**句子**："The cat sat on the mat"

**不同头可能关注**：
- Head 1：语法关系（主语-谓语）
- Head 2：语义关系（cat-mat）
- Head 3：位置关系（on的位置）

### 多头数量选择

| 头数 | 影响 |
|------|------|
| 少（4） | 表达能力有限 |
| 中（8-12） | 平衡效率和效果 |
| 多（16+） | 更丰富，计算更多 |

### 多头计算流程

```
输入 X
  ↓
并行计算（h个头）:
  Head 1: Attention(Q_1, K_1, V_1)
  Head 2: Attention(Q_2, K_2, V_2)
  ...
  Head h: Attention(Q_h, K_h, V_h)
  ↓
Concat（拼接）
  ↓
线性变换 W_O
  ↓
输出
```

## 交叉注意力

### 交叉注意力的概念

不同序列之间的注意力：
- Query来自一个序列
- Key和Value来自另一个序列

**典型应用**：解码器关注编码器输出。

### 交叉注意力计算

$CrossAttention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

其中：
- Q = DecoderOutput × W_Q
- K = EncoderOutput × W_K
- V = EncoderOutput × W_V

### 交叉注意力的作用

| 应用 | 描述 |
|------|------|
| 翻译 | 目标语言关注源语言 |
| 摘要 | 摘要关注原文 |
| 问答 | 答案关注问题 |

### 交叉注意力示例

**翻译**：
- 源语言："I love apples"
- 目标语言："我爱苹果"

生成"苹果"时，交叉注意力关注"apples"。

## 注意力可视化

### 注意力热力图

可视化注意力权重矩阵：

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    """可视化注意力"""
    plt.figure(figsize=(10, 10))
    sns.heatmap(attention_weights, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='viridis')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title('Attention Visualization')
    plt.show()

# 示例
tokens = ['I', 'love', 'apples', 'because', 'they', 'are', 'delicious']
attention_weights = model.get_attention_weights(tokens)
visualize_attention(attention_weights, tokens)
```

### 多头注意力可视化

```python
def visualize_multihead_attention(attention_weights, tokens, head_idx):
    """可视化特定头"""
    plt.figure(figsize=(8, 8))
    sns.heatmap(attention_weights[head_idx],
                xticklabels=tokens,
                yticklabels=tokens)
    plt.title(f'Head {head_idx} Attention')
    plt.show()

# 查看各头
for i in range(num_heads):
    visualize_multihead_attention(all_head_weights, tokens, i)
```

### 注意力案例分析

**BERT注意力**：
- 底层头：关注局部相邻词
- 中层头：关注语法关系
- 高层头：关注语义关系

**GPT注意力**：
- 主要关注最近生成的词
- 部分头关注长距离依赖

## 注意力机制的变体

### 缩放点积注意力

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

**缩放原因**：
- 防止点积过大
- 防止softmax梯度消失

### 加性注意力（Bahdanau）

$score(q, k) = v^T \tanh(W_q q + W_k k)$

**特点**：
- 使用神经网络计算分数
- 计算成本较高
- 早期翻译模型使用

### 线性注意力

降低自注意力复杂度：

$LinearAttention(Q, K, V) = Q \cdot (K^T \cdot V)$

**优点**：复杂度从O(n²)降到O(n)

### 稀疏注意力

只关注部分位置：
- 局部窗口注意力
- 全局注意力（部分位置）
- 随机注意力

### Flash Attention

高效内存计算：
- 分块计算注意力
- 减少内存访问
- 加速训练

## 注意力机制的应用

### 自然语言处理

| 应用 | 注意力类型 |
|------|------------|
| 翻译 | 自注意力 + 交叉注意力 |
| 文本分类 | 自注意力 |
| 阅读理解 | 交叉注意力 |
| 摘要 | 自注意力 + 交叉注意力 |

### 视觉领域

**视觉注意力**：
- 图像区域注意力
- 空间注意力
- 通道注意力

**Vision Transformer**：
- 图像切块作为序列
- 自注意力处理

### 多模态

**图文注意力**：
- 图像关注文本
- 文本关注图像

**应用**：
- 图像描述
- 视觉问答
- 图文匹配

### 其他应用

| 领域 | 应用 |
|------|------|
| 时间序列 | 时间点注意力 |
| 推荐系统 | 用户-商品注意力 |
| 图神经网络 | 节点注意力 |

## 注意力机制的数学分析

### 注意力权重分布

理想情况：权重集中在重要位置

**问题**：
- 权重可能分散
- 可能关注错误位置

### 注意力的梯度分析

注意力层梯度：
$\frac{\partial L}{\partial \alpha} = ...$

梯度通过softmax传播，可能较小。

### 注意力复杂度

自注意力：
- 时间：O(n²d)
- 空间：O(n²)

其中n是序列长度，d是维度。

**长序列问题**：
- n=1000 → 100万次计算
- n=10000 → 1亿次计算

## 案例实践

### PyTorch注意力实现

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)
        self.scale = d_k ** -0.5
    
    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        
        # 输出
        output = torch.matmul(attention, V)
        
        return output, attention

# 使用
attention = SelfAttention(d_model=512, d_k=64, d_v=64)
output, weights = attention(x)
```

### TensorFlow注意力

```python
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = tf.keras.layers.Dense(d_model)
        self.W_K = tf.keras.layers.Dense(d_model)
        self.W_V = tf.keras.layers.Dense(d_model)
        self.W_O = tf.keras.layers.Dense(d_model)
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        # 线性变换
        Q = self.W_Q(q)
        K = self.W_K(k)
        V = self.W_V(v)
        
        # 分割多头
        Q = tf.reshape(Q, (batch_size, -1, self.num_heads, self.d_k))
        K = tf.reshape(K, (batch_size, -1, self.num_heads, self.d_k))
        V = tf.reshape(V, (batch_size, -1, self.num_heads, self.d_k))
        
        # 注意力
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(self.d_k)
        attention = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention, V)
        
        # 合并多头
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.W_O(output)

# 使用内置层
mha = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
output = mha(query, value, key)
```

### 注意力权重提取

```python
import transformers
from transformers import BertModel, BertTokenizer

# 加载模型
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 输入
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors='pt')

# 获取注意力权重
outputs = model(**inputs)
attentions = outputs.attentions  # 各层各头的注意力权重

# 可视化第一层第一个头
attention_weights = attentions[0][0][0].detach().numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
visualize_attention(attention_weights, tokens)
```

### 自定义注意力分析

```python
def analyze_attention_patterns(attention_weights):
    """分析注意力模式"""
    patterns = {
        'diagonal': np.diag(attention_weights).mean(),  # 自关注
        'local': np.mean(attention_weights[:5, :5]),    # 局部关注
        'global': np.mean(attention_weights),           # 全局平均
        'max': np.max(attention_weights)                # 最大权重
    }
    return patterns

# 分析
patterns = analyze_attention_patterns(attention_weights)
print(f"自关注强度: {patterns['diagonal']:.3f}")
print(f"局部关注强度: {patterns['local']:.3f}")
```

## 总结

注意力机制是现代深度学习的核心技术。核心内容包括：
- 注意力起源：模拟人类选择性关注
- 软注意力与硬注意力：可微分 vs 随机采样
- 自注意力：序列内部交互
- 多头注意力：多角度关注
- 交叉注意力：跨序列关注
- 注意力可视化：理解模型关注点

注意力机制赋予模型灵活的信息选择能力，是Transformer等现代架构的基础。

## 延伸阅读

- [Transformer架构详解](/2026/05/10/zh-CN/技术文档/机器学习/transformer/)
- [预训练语言模型](/2026/05/10/zh-CN/技术文档/机器学习/pretraining-lm/)
- [大模型架构演进](/2026/05/10/zh-CN/技术文档/机器学习/llm-architecture/)
- [循环神经网络](/2026/05/10/zh-CN/技术文档/机器学习/rnn/)
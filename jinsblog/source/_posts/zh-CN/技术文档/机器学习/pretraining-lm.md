---
title: 预训练语言模型
date: 2026-05-10
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 大模型, 预训练]
---

## 预训练的思想

### 预训练的概念

预训练是在大规模数据上训练模型，使其获得通用能力。

**流程**：
```
大规模数据 → 预训练 → 预训练模型
    ↓                     ↓
微调数据 → 微调 → 任务模型
```

### 预训练的优势

| 优势 | 描述 |
|------|------|
| 降低成本 | 不需为每任务从头训练 |
| 提高性能 | 预训练获得通用能力 |
| 减少数据 | 微调只需少量数据 |
| 加速开发 | 快速构建应用 |

### 预训练vs从零训练

| 方面 | 从零训练 | 预训练+微调 |
|------|----------|------------|
| 数据需求 | 大 | 小 |
| 训练成本 | 高 | 低 |
| 性能上限 | 可能高 | 通常高 |
| 时间 | 长 | 短 |

## 语言模型基础

### 语言模型定义

语言模型预测文本序列的概率：

$P(w_1, w_2, ..., w_n) = P(w_1)P(w_2|w_1)...P(w_n|w_{1...n-1})$

### 语言模型任务

**自回归语言模型**：预测下一个词
$P(w_t|w_{<t})$

**自编码语言模型**：预测被遮蔽的词
$P(w_t|context)$

### 语言模型的训练目标

| 类型 | 目标 | 模型 |
|------|------|------|
| 自回归 | 预测下一个词 | GPT |
| 自编码 | 预测遮蔽词 | BERT |
| 两者结合 | 多种目标 | T5 |

## BERT：双向编码

### BERT架构

BERT（Bidirectional Encoder Representations from Transformers）是双向编码器。

**结构**：
- 多层Transformer编码器
- 双向上下文理解
- MLM预训练任务

### BERT预训练任务

#### Masked Language Model（MLM）

随机遮蔽15%的词，预测遮蔽词：

**遮蔽策略**：
- 80%替换为[MASK]
- 10%替换为随机词
- 10%保持原词

```python
# MLM示例
原文: "The cat sat on the mat"
遮蔽: "The cat [MASK] on the [MASK]"
预测: "sat", "mat"
```

#### Next Sentence Prediction（NSP）

预测两个句子是否连续：

```
输入: [CLS] Sentence A [SEP] Sentence B [SEP]
输出: IsNext / NotNext
```

### BERT输入表示

```
[CLS] Token1 Token2 ... [SEP] Token1 Token2 ... [SEP]
  ↓      ↓      ↓        ↓      ↓      ↓        ↓
Segment Emb: AAAAAA... A BBBB... B
Position Emb: 0 1 2 ... 0 1 2 ...
```

**组成**：
- Token Embedding：词嵌入
- Segment Embedding：句子标识
- Position Embedding：位置信息

### BERT变体

| 变体 | 特点 |
|------|------|
| BERT-base | 12层，110M参数 |
| BERT-large | 24层，340M参数 |
| RoBERTa | 更大数据，更长训练 |
| ALBERT | 参数共享，轻量 |
| DistilBERT | 轻量蒸馏 |

### BERT微调

```python
from transformers import BertForSequenceClassification

# 加载预训练BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 微调
for epoch in range(epochs):
    for batch in train_loader:
        outputs = model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

## GPT：单向生成

### GPT架构

GPT（Generative Pre-trained Transformer）是单向生成器。

**结构**：
- 多层Transformer解码器
- 单向（只看过去）
- 自回归生成

### GPT预训练任务

**语言模型**：预测下一个词

$L = -\sum_t \log P(w_t|w_{<t})$

**训练方式**：
- 自回归预测
- 大规模文本数据
- 无标签学习

### GPT发展历程

| 版本 | 参数 | 特点 |
|------|------|------|
| GPT-1 | 117M | 初版 |
| GPT-2 | 1.5B | 更大，生成能力强 |
| GPT-3 | 175B | 极大，涌现能力 |
| GPT-4 | 未公开 | 多模态，更强 |

### GPT生成示例

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 生成
input_ids = tokenizer.encode("The quick brown fox", return_tensors='pt')
output = model.generate(input_ids, max_length=50)
generated = tokenizer.decode(output[0])
print(generated)
```

### GPT vs BERT

| 方面 | BERT | GPT |
|------|------|-----|
| 方向 | 双向 | 单向 |
| 任务 | 理解 | 生成 |
| 应用 | 分类、NER | 文本生成 |
| 预训练 | MLM+NSP | LM |

## 预训练任务设计

### 自监督学习

预训练使用自监督，无需人工标注：

| 任务 | 描述 |
|------|------|
| MLM | 预测遮蔽词 |
| LM | 预测下一个词 |
| NSP | 预测句子连续性 |
| SOP | 预测句子顺序 |

### 多任务预训练

T5使用多种任务：
- 翻译
- 摘要
- 问答
- 分类

统一为文本到文本格式。

### 预训练数据

| 模型 | 数据 |
|------|------|
| BERT | Wikipedia + BookCorpus |
| GPT-2 | WebText |
| GPT-3 | 多来源混合 |

## 预训练数据与规模

### 数据规模

| 模型 | 数据量 |
|------|--------|
| BERT | ~3B tokens |
| GPT-2 | ~40GB |
| GPT-3 | ~500B tokens |
| LLaMA | ~1.4TB |

### 数据来源

| 来源 | 描述 |
|------|------|
| Wikipedia | 高质量百科 |
| Books | 书籍文本 |
| Web | 网络文本 |
| Code | 代码数据 |

### 数据处理

**流程**：
1. 收集数据
2. 清洗去噪
3. 分词处理
4. 格式统一

### 数据质量影响

数据质量影响模型能力：
- 高质量数据 → 更好的理解能力
- 多样化数据 → 更强的泛化能力
- 大规模数据 → 更丰富的知识

## 案例实践

### BERT文本分类

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 准备数据
texts = ["I love this movie!", "This movie is terrible."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# 预测
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print(predictions)
```

### GPT文本生成

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

# 生成
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.7,
    top_k=50,
    do_sample=True
)

print(tokenizer.decode(output[0]))
```

### 使用预训练模型进行NER

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')

text = "Apple is looking at buying U.K. startup for $1 billion"
inputs = tokenizer(text, return_tensors='pt')

outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)

# 解码NER结果
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, pred in zip(tokens, predictions[0]):
    if pred > 0:  # 不是'O'
        print(f"{token}: {model.config.id2label[pred.item()]}")
```

### 预训练模型的微调

```python
from transformers import Trainer, TrainingArguments

# 微调配置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 训练
trainer.train()
```

## 总结

预训练语言模型是现代NLP的基础。核心内容包括：
- 预训练思想：大规模数据获取通用能力
- 语言模型基础：自回归和自编码
- BERT：双向编码，适合理解任务
- GPT：单向生成，适合生成任务
- 预训练任务设计：自监督学习
- 数据规模影响：影响模型能力

预训练+微调是现代NLP的标准范式。

## 延伸阅读

- [Transformer架构详解](/2026/05/10/zh-CN/技术文档/机器学习/transformer/)
- [微调技术详解](/2026/05/10/zh-CN/技术文档/机器学习/fine-tuning/)
- [大模型架构演进](/2026/05/10/zh-CN/技术文档/机器学习/llm-architecture/)
- [上下文学习与提示工程](/2026/05/10/zh-CN/技术文档/机器学习/prompt-engineering/)
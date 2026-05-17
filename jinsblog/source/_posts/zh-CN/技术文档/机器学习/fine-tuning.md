---
title: 微调技术详解
date: 2026-02-11
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 大模型, 微调]
---

## 全量微调

### 微调的概念

微调是在预训练模型基础上，针对特定任务继续训练。

**流程**：
```
预训练模型 → 加载 → 在特定数据训练 → 任务模型
```

### 全量微调方法

更新模型所有参数：

**训练过程**：
- 加载预训练权重
- 添加任务特定层
- 用较小学习率训练
- 更新所有参数

### 全量微调参数

| 参数 | 建议值 |
|------|--------|
| 学习率 | 1e-5 - 5e-5 |
| 批量大小 | 8-32 |
| 迭代次数 | 3-10 |

```python
from transformers import BertForSequenceClassification, Trainer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 全量微调
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        learning_rate=2e-5,
        num_train_epochs=3
    ),
    train_dataset=train_dataset
)
trainer.train()
```

### 全量微调的优缺点

| 优点 | 缺点 |
|------|------|
| 效果最好 | 计算成本高 |
| 简单直接 | 需要存储完整模型 |
| 通用性强 | 可能过拟合 |

## 任务特定微调

### 分类任务微调

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# 添加分类头
# 微调训练
```

### 生成任务微调

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('gpt2')

# 继续训练语言模型
# 或特定生成任务
```

### 序列标注微调

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_tags)

# NER、POS等任务
```

### 问答任务微调

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

# 提取式问答
```

## Prompt Tuning

### Prompt Tuning原理

冻结模型，只训练可学习的提示向量：

$输入 = [P_1, P_2, ..., P_m] + [x_1, x_2, ..., x_n]$

其中 $P_i$ 是可学习的提示向量。

### Prompt Tuning流程

```
冻结模型 ← 只更新提示
    ↑
[可学习提示] + [实际输入]
```

### Prompt Tuning参数

| 参数 | 建议值 |
|------|--------|
| 提示长度 | 10-100 |
| 初始化 | 随机或词汇 |
| 训练轮数 | 比全量多 |

```python
from peft import PromptTuningConfig, get_peft_model

config = PromptTuningConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=20
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# 输出: trainable params: 20 * d_model || all params: 110M
```

### Prompt Tuning优缺点

| 优点 | 缺点 |
|------|------|
| 参数极少 | 效果略低于全量 |
| 存储小 | 需较长时间收敛 |
| 易切换任务 | 提示长度需调优 |

## Prefix Tuning

### Prefix Tuning原理

在每层添加可学习的前缀向量：

$h_i = f([prefix; h_{i-1}])$

每层都有独立的可学习前缀。

### Prefix Tuning vs Prompt Tuning

| 方面 | Prompt Tuning | Prefix Tuning |
|------|---------------|---------------|
| 插入位置 | 仅输入层 | 每层 |
| 参数数量 | 少 | 较多 |
| 效果 | 中等 | 更好 |

```python
from peft import PrefixTuningConfig

config = PrefixTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20
)

model = get_peft_model(model, config)
```

### Prefix Tuning的实现

每层添加前缀：
- 编码器每层
- 解码器每层
- 前缀参数可学习

## LoRA低秩适配

### LoRA原理

低秩适配（Low-Rank Adaptation）通过低秩矩阵修改权重：

$W' = W + BA$

其中：
- B: $[d \times r]$
- A: $[r \times d]$
- r << d（低秩）

### LoRA的数学基础

原权重W冻结，只训练A和B：

$\Delta W = BA$

**参数量**：$2 \times d \times r$ vs $d \times d$

当$r$小（如4-8），参数量大幅减少。

### LoRA参数

| 参数 | 建议值 |
|------|--------|
| rank (r) | 4-64 |
| alpha | 16-32 |
| dropout | 0.05 |

### LoRA的优势

| 优势 | 描述 |
|------|------|
| 参数少 | 只训练低秩矩阵 |
| 效果好 | 接近全量微调 |
| 合奏 | 可合并回原权重 |
| 无推理开销 | 合并后与原模型同 |

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: ~0.1% of total
```

### LoRA应用模块

通常应用于：
- 注意力层（q_proj, v_proj等）
- 可扩展到全层

### LoRA合并

训练后可将LoRA合并到原权重：

$W_{merged} = W + \frac{\alpha}{r}BA$

合并后无额外推理开销。

## QLoRA量化微调

### QLoRA原理

结合量化（4-bit）和LoRA：

**流程**：
```
4-bit量化模型 → LoRA适配器 → 微调 → 合并
```

### QLoRA的技术

| 技术 | 描述 |
|------|------|
| 4-bit NormalFloat | 量化数据类型 |
| 双量化 | 进一步压缩 |
| Paged Optimizers | 内存优化 |

### QLoRA的优势

| 优势 | 描述 |
|------|------|
| 内存极少 | 可微调65B模型 |
| 速度快 | 单GPU可行 |
| 效果好 | 接近16-bit LoRA |

```python
from peft import LoraConfig
from transformers import BitsAndBytesConfig

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=bnb_config
)

# LoRA配置
lora_config = LoraConfig(r=16, lora_alpha=32)
model = get_peft_model(model, lora_config)
```

### QLoRA vs LoRA

| 方面 | LoRA | QLoRA |
|------|------|-------|
| 内存 | 中等 | 极低 |
| 速度 | 快 | 更快 |
| 精度 | 高 | 略低 |
| 适用场景 | GPU充足 | GPU有限 |

## 微调策略选择

### 各方法对比

| 方法 | 参数量 | 效果 | 内存 |
|------|--------|------|------|
| 全量微调 | 100% | 最好 | 高 |
| Prompt Tuning | <1% | 中等 | 低 |
| Prefix Tuning | ~1% | 较好 | 低 |
| LoRA | ~1% | 接近全量 | 低 |
| QLoRA | ~1% | 接近LoRA | 极低 |

### 选择建议

| 场景 | 推荐方法 |
|------|----------|
| 效果优先 | 全量微调或LoRA |
| 内存有限 | QLoRA |
| 多任务切换 | Prompt Tuning |
| 快速实验 | LoRA |

## 案例实践

### LoRA微调示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 加载模型
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

# LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# 应用LoRA
model = get_peft_model(model, config)

# 训练
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./output",
        learning_rate=1e-4,
        num_train_epochs=3
    ),
    train_dataset=train_dataset
)
trainer.train()

# 保存LoRA权重
model.save_pretrained("./lora_weights")
```

### 合并LoRA权重

```python
from peft import PeftModel

# 加载原模型 + LoRA
base_model = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(base_model, "./lora_weights")

# 合并
merged_model = model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./merged_model")
```

### QLoRA微调

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)

# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)

# 训练（内存极少）
trainer.train()
```

### Prompt Tuning示例

```python
from peft import PromptTuningConfig, get_peft_model, TaskType

config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    token_dim=768
)

model = get_peft_model(model, config)

# 只训练提示向量
model.print_trainable_parameters()
```

### 多LoRA适配器管理

```python
# 训练多个LoRA适配器
for task in tasks:
    config = LoraConfig(r=8, lora_alpha=32)
    model = get_peft_model(base_model, config)
    trainer.train()
    model.save_pretrained(f"./lora_{task}")

# 使用时加载
model = PeftModel.from_pretrained(base_model, f"./lora_{task}")
```

## 微调最佳实践

### 学习率选择

| 模型规模 | 建议学习率 |
|----------|------------|
| 小（<100M） | 5e-5 - 1e-4 |
| 中（100M-1B） | 1e-5 - 5e-5 |
| 大（>1B） | 1e-6 - 1e-5 |
| LoRA | 1e-4 - 5e-4 |

### 数据准备

| 建议 | 描述 |
|------|------|
| 格式统一 | 标准化输入输出 |
| 清洗数据 | 去噪、去重复 |
| 验证集 | 保留验证数据 |

### 避免过拟合

| 方法 | 描述 |
|------|------|
| 早停 | 监控验证损失 |
| 正则化 | 权重衰减 |
| 数据增强 | 扩充数据 |

## 总结

微调是利用预训练模型的关键技术。核心内容包括：
- 全量微调：更新所有参数，效果最好
- Prompt Tuning：只训练提示向量，参数最少
- Prefix Tuning：每层添加可学习前缀
- LoRA：低秩适配，参数少效果好
- QLoRA：量化+LoRA，内存极少

LoRA和QLoRA是现代大模型微调的主流方法。

## 延伸阅读

- [预训练语言模型](/2026/05/10/zh-CN/技术文档/机器学习/pretraining-lm/)
- [大模型架构演进](/2026/05/10/zh-CN/技术文档/机器学习/llm-architecture/)
- [模型压缩与加速](/2026/05/10/zh-CN/技术文档/机器学习/model-compression/)
- [Transformer架构详解](/2026/05/10/zh-CN/技术文档/机器学习/transformer/)
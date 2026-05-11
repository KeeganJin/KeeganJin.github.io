---
title: 模型压缩与加速
date: 2026-05-05
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 模型压缩, 知识蒸馏, 量化]
---

## 模型压缩动机

### 模型规模增长

| 模型 | 参数量 |
|------|--------|
| BERT-base | 110M |
| GPT-2 | 1.5B |
| GPT-3 | 175B |
| LLaMA-2-70B | 70B |

### 压缩需求

| 需求 | 描述 |
|------|------|
| 部署限制 | 设备内存有限 |
| 推理速度 | 响应时间要求 |
| 能效要求 | 减少计算能耗 |
| 成本控制 | 降低推理成本 |

### 压缩目标

| 目标 | 方向 |
|------|------|
| 模型体积 | 减少存储 |
| 计算量 | 加速推理 |
| 能耗 | 降低功耗 |
| 性能 | 保持精度 |

## 知识蒸馏

### 蒸馏原理

大模型（Teacher）知识转移到小模型（Student）：

$L = L_{hard} + \alpha L_{soft}$

**软标签**：Teacher输出的概率分布，包含更多信息。

### 软标签的优势

| 方面 | 硬标签 | 软标签 |
|------|--------|--------|
| 信息量 | 二值 | 概率分布 |
| 暗知识 | 无 | 类间关系 |
| 学习难度 | 难 | 易 |

### 蒸馏温度

调节软标签的"软度"：
$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$

**温度T**：
- T=1：原始softmax
- T>1：分布更平滑
- T→∞：均匀分布

### 蒸馏流程

```
1. 训练Teacher模型（大模型）
2. 用Teacher生成软标签
3. 训练Student模型（用软+硬标签）
4. 部署Student模型
```

### 蒸馏实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 软标签损失
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss *= self.temperature ** 2
        
        # 综合损失
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# 训练
def train_with_distillation(teacher, student, train_loader, epochs):
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters())
    distill_loss = DistillationLoss(temperature=4)
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            student_logits = student(inputs)
            loss = distill_loss(student_logits, teacher_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 中间层蒸馏

不仅蒸馏输出，还蒸馏中间层：

```python
class IntermediateDistillation(nn.Module):
    def __init__(self, hint_layers, feature_layers):
        super().__init__()
        self.hint_layers = hint_layers
        self.feature_layers = feature_layers
    
    def forward(self, student_features, teacher_features):
        loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # 特征匹配
            loss += F.mse_loss(s_feat, t_feat)
        return loss
```

### 蒸馏效果

| Teacher | Student | 精度保持 |
|---------|---------|----------|
| BERT-large | BERT-small | ~95% |
| ResNet-152 | ResNet-50 | ~90% |
| GPT-3 | GPT-2 | 可行 |

## 模型剪枝

### 剪枝原理

移除对模型贡献小的参数。

**依据**：
- 权重绝对值小
- 对输出影响小
- 对损失影响小

### 剪枝类型

| 类型 | 描述 |
|------|------|
| 结构化剪枝 | 移除整层/通道 |
| 非结构化剪枝 | 移除单个权重 |

### 非结构化剪枝

移除小权重：

```python
def prune_weights(model, threshold):
    """非结构化剪枝"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = param.abs() > threshold
            param.data *= mask
```

### 结构化剪枝

移除通道/层：

```python
def prune_channels(model, prune_ratio):
    """通道剪枝"""
    for layer in model.conv_layers:
        weight = layer.weight.data
        
        # 计算通道重要性
        importance = weight.abs().sum(dim=(1, 2, 3))
        
        # 保留重要通道
        num_keep = int(len(importance) * (1 - prune_ratio))
        keep_indices = torch.argsort(importance)[-num_keep:]
        
        # 剪枝
        layer.weight.data = weight[keep_indices]
        layer.out_channels = num_keep
```

### 剪枝策略

| 策略 | 描述 |
|------|------|
| 一次性剪枝 | 训练后一次性剪枝 |
| 渐进剪枝 | 训练中逐步剪枝 |
| 自动剪枝 | 自动搜索剪枝比例 |

### 渐进剪枝

训练中逐步剪枝：

```python
def gradual_pruning(model, initial_sparsity, final_sparsity, steps):
    """渐进剪枝"""
    for step in range(steps):
        current_sparsity = initial_sparsity + \
            (final_sparsity - initial_sparsity) * step / steps
        
        prune_model(model, current_sparsity)
        finetune(model, epochs=1)
```

### 剪枝后微调

剪枝后重新训练恢复精度：

```python
def prune_and_finetune(model, prune_ratio, finetune_epochs):
    # 剪枝
    prune_model(model, prune_ratio)
    
    # 微调
    for epoch in range(finetune_epochs):
        train_one_epoch(model, train_loader)
```

### 剪枝效果

| 模型 | 剪枝率 | 精度下降 |
|------|--------|----------|
| ResNet-50 | 30% | ~1% |
| BERT-base | 40% | ~2% |
| VGG-16 | 90% | ~5% |

## 模型量化

### 量化原理

降低参数精度：
- FP32 → FP16/INT8/INT4
- 减少存储和计算

### 量化类型

| 类型 | 描述 |
|------|------|
| 训练后量化（PTQ） | 训练后量化 |
| 量化感知训练（QAT） | 训练中模拟量化 |

### INT8量化

将FP32映射到INT8：

$Q(x) = \text{round}(x/scale + zero\_point)$

**scale和zero_point计算**：
- scale = (max - min) / 255
- zero_point = -min / scale

### 训练后量化

```python
import torch

def quantize_model(model):
    """训练后量化"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model

# 使用
model = ResNet18()
model.eval()
quantized = quantize_model(model)

# 比较大小
print(f"Original: {get_model_size(model)} MB")
print(f"Quantized: {get_model_size(quantized)} MB")
```

### 量化感知训练

训练中模拟量化：

```python
def prepare_qat(model):
    """准备量化感知训练"""
    model.qconfig = torch.quantization.get_default_qat_qconfig()
    torch.quantization.prepare_qat(model, inplace=True)
    return model

def train_qat(model, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # 转换为量化模型
    model.eval()
    quantized = torch.quantization.convert(model)
    return quantized
```

### GPTQ量化

大模型高精度量化：

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 配置
quantize_config = BaseQuantizeConfig(
    bits=4,           # 4-bit量化
    group_size=128,   # 分组量化
    desc_act=False    # 激活量化
)

# 加载并量化
model = AutoGPTQForCausalLM.from_pretrained(
    "model_name",
    quantize_config
)

# 执行量化
model.quantize(calibration_data)
model.save_quantized("quantized_model")
```

### 量化精度影响

| 量化 | 精度 | 精度下降 |
|------|------|----------|
| FP32 | 高 | 0% |
| FP16 | 较高 | ~0.5% |
| INT8 | 中 | ~1-2% |
| INT4 | 低 | ~3-5% |

### 量化效果

| 方面 | FP32 | INT8 | INT4 |
|------|------|------|------|
| 存储 | 4x | 1x | 0.5x |
| 计算 |慢 | 快 | 更快 |
| 精度 |高 | 中 | 较低 |

## 模型架构优化

### 网络架构搜索（NAS）

自动搜索最优架构：

```python
class NAS:
    def __init__(self, search_space):
        self.search_space = search_space
    
    def search(self, data, epochs):
        best_architecture = None
        best_accuracy = 0
        
        for architecture in self.search_space:
            model = build_model(architecture)
            accuracy = train_and_eval(model, data)
            
            if accuracy > best_accuracy:
                best_architecture = architecture
                best_accuracy = accuracy
        
        return best_architecture
```

### 轻量化架构

| 模型 | 参数量 | 特点 |
|------|--------|------|
| MobileNet | ~4M | 深度可分离卷积 |
| ShuffleNet | ~2M | 通道shuffle |
| EfficientNet | ~5M | 复合缩放 |
| DistilBERT | ~66M | 蒸馏BERT |

### 深度可分离卷积

分解标准卷积：
- Depthwise：每通道单独卷积
- Pointwise：1x1卷积组合

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            groups=in_channels, padding=kernel_size//2
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

## 案例实践

### BERT蒸馏

```python
from transformers import BertForSequenceClassification

# Teacher模型
teacher = BertForSequenceClassification.from_pretrained('bert-large-uncased')

# Student模型
student = BertForSequenceClassification.from_pretrained('bert-small-uncased')

# 蒸馏训练
def distill_bert(teacher, student, dataset):
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters())
    
    for batch in dataset:
        with torch.no_grad():
            teacher_output = teacher(**batch)
        
        student_output = student(**batch)
        
        # 软标签损失
        loss = distillation_loss(
            student_output.logits,
            teacher_output.logits,
            batch['labels'],
            temperature=4
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### LLM量化

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 检查内存占用
print(f"Memory: {model.get_memory_footprint() / 1024**2:.2f} MB")
```

### 模型剪枝实践

```python
import torch.nn.utils.prune as prune

def prune_llm(model, amount=0.2):
    """剪枝LLM线性层"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    # 移除剪枝掩码，永久化
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    
    return model
```

### 压缩效果评估

```python
def evaluate_compression(original, compressed, test_data):
    """评估压缩效果"""
    # 模型大小
    original_size = get_model_size(original)
    compressed_size = get_model_size(compressed)
    
    # 精度
    original_accuracy = evaluate(original, test_data)
    compressed_accuracy = evaluate(compressed, test_data)
    
    # 推理速度
    original_speed = measure_inference_time(original)
    compressed_speed = measure_inference_time(compressed)
    
    print(f"大小减少: {(original_size - compressed_size) / original_size:.2%}")
    print(f"精度变化: {compressed_accuracy - original_accuracy:.2%}")
    print(f"速度提升: {original_speed / compressed_speed:.2f}x")
```

## 压缩方法对比

### 方法对比

| 方法 | 压缩率 | 精度保持 | 复杂度 |
|------|--------|----------|--------|
| 蒸馏 | 中 | 高 | 中 |
| 剪枝 | 高 | 中 | 中 |
| 量化 | 高 | 中 | 低 |
| NAS | 高 | 高 | 高 |

### 选择建议

| 场景 | 推荐 |
|------|------|
| 推理加速 | 量化 |
| 部署边缘 | 量化+剪枝 |
| 保持精度 | 蒸馏 |
| 自动优化 | NAS |

## 总结

模型压缩与加速是部署的关键。核心内容包括：
- 知识蒸馏：Teacher到Student知识转移
- 模型剪枝：移除冗余参数
- 模型量化：降低精度减少存储
- 架构优化：设计轻量架构

压缩技术使大模型可在资源受限设备部署。

## 延伸阅读

- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
- [大模型架构演进](/2026/05/10/zh-CN/技术文档/机器学习/llm-architecture/)
- [分布式训练](/2026/05/10/zh-CN/技术文档/机器学习/distributed-training/)
- [模型部署实践](/2026/05/10/zh-CN/技术文档/机器学习/model-deployment/)
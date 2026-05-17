---
title: 上下文学习与提示工程
date: 2026-02-14
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 大模型, 提示工程]
---

## 上下文学习（ICL）原理

### ICL的概念

上下文学习（In-Context Learning, ICL）是大模型在提示中学习任务的能力。

**特点**：
- 无需参数更新
- 通过示例学习
- 即时适应任务

### ICL的工作方式

```
提示格式:
[示例1输入] → [示例1输出]
[示例2输入] → [示例2输出]
...
[新输入] → [模型预测输出]
```

### ICL vs 微调

| 方面 | ICL | 微调 |
|------|-----|------|
| 参数更新 | 无 | 有 |
| 训练时间 | 无 | 需训练 |
| 示例需求 | 提示中 | 训练数据 |
| 适应性 | 即时 | 需微调 |

### ICL的优势

| 优势 | 描述 |
|------|------|
| 即时使用 | 无需训练 |
| 灵活切换 | 不同任务不同提示 |
| 降低成本 | 无训练成本 |
| 试错方便 | 快速迭代 |

### ICL的局限

| 局限 | 描述 |
|------|------|
| 上下文有限 | 示例数量受限 |
| 不稳定 | 效果依赖示例 |
| 记忆有限 | 不持久 |

## Prompt设计原则

### Prompt的重要性

Prompt直接影响模型输出质量。

**影响因素**：
- Prompt结构
- 示例选择
- 指令清晰度

### 设计原则

| 原则 | 描述 |
|------|------|
| 清晰明确 | 指令简洁易懂 |
| 示例相关 | 示例与任务相关 |
| 格式统一 | 输入输出格式一致 |
| 避免歧义 | 无模糊表述 |

### Prompt结构

**标准结构**：
```
1. 任务描述
2. 格式说明
3. 示例（可选）
4. 待处理输入
```

### 常见Prompt类型

| 类型 | 示例 |
|------|------|
| 指令型 | "请翻译以下文本..." |
| 示例型 | 提供示例+输入 |
| 角色型 | "作为翻译专家..." |

### Prompt设计示例

```python
# 翻译Prompt
prompt = """
请将以下英文翻译为中文：
Hello, how are you?
"""

# 分类Prompt
prompt = """
分类以下文本的情感：
文本：这部电影太精彩了！
情感：正面

文本：服务很差，很不满意。
情感：负面

文本：产品质量一般。
情感：？
"""

# 角色Prompt
prompt = """
作为一个专业的Python程序员，请解释以下代码：
def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)
"""
```

## Few-shot提示

### Few-shot概念

提供少量示例引导模型：

**格式**：
```
示例1输入: ... 
示例1输出: ...
示例2输入: ...
示例2输出: ...
新输入: ...
输出: ?
```

### Few-shot示例数量

| 类型 | 示例数 |
|------|--------|
| Zero-shot | 0 |
| Few-shot | 1-10 |
| Many-shot | 10+ |

### Few-shot效果

| 示例数 | 效果 |
|--------|------|
| 0 | 可能不稳定 |
| 1-3 | 有所改善 |
| 5-10 | 通常较好 |
| 更多 | 可能饱和 |

### Few-shot示例选择

**原则**：
- 代表任务特点
- 覆盖不同情况
- 格式一致
- 避免偏见

```python
# Few-shot情感分析
prompt = """
对以下电影评论进行情感分类：

评论："这部电影太棒了，强烈推荐！"
情感：正面

评论："太无聊了，浪费时间。"
情感：负面

评论："演员表演不错，但剧情拖沓。"
情感：中性

评论："视觉效果震撼，但内容空洞。"
情感：？
"""
```

### Few-shot示例顺序

示例顺序可能影响结果：
- 按相似度排序
- 随机顺序
- 避免误导模式

## Chain-of-Thought提示

### CoT概念

让模型展示推理过程：

**格式**：
```
问题: ...
推理步骤:
1. ...
2. ...
...
答案: ...
```

### CoT的优势

| 优势 | 描述 |
|------|------|
| 提高准确率 | 步骤分解更准确 |
| 可解释性 | 推理过程可见 |
| 复杂任务 | 处理复杂推理 |

### CoT示例

```python
# 数学推理CoT
prompt = """
问题：小明有5个苹果，给了小红2个，又买了3个，现在有多少个？

推理过程：
1. 小明原有5个苹果
2. 给小红2个后：5 - 2 = 3个
3. 又买了3个：3 + 3 = 6个
答案：6个苹果

问题：一个班级有30人，男生占40%，女生有多少人？

推理过程：
答案：
"""
```

### Zero-shot CoT

不加示例，只要求步骤：

```python
prompt = """
请一步步思考并解决以下问题：
...
"""
```

### CoT变体

| 变体 | 描述 |
|------|------|
| Zero-shot CoT | "一步步思考" |
| Manual CoT | 手写示例步骤 |
| Auto CoT | 自动生成步骤 |

## 提示模板设计

### 模板的作用

统一Prompt格式，提高效率。

### 模板结构

```python
template = """
任务：{task_description}

格式要求：{format_requirement}

示例：
{examples}

当前输入：{input}
输出：
"""
```

### 模板参数化

```python
from string import Template

prompt_template = Template("""
请完成以下$task：
$examples
输入：$input
输出：
""")

# 使用
prompt = prompt_template.substitute(
    task="情感分析",
    examples="...",
    input="这部电影很好看"
)
```

### 常见模板类型

| 类型 | 适用 |
|------|------|
| 分类模板 | 分类任务 |
| 生成模板 | 文本生成 |
| QA模板 | 问答任务 |
| 翻译模板 | 翻译任务 |

## 提示调优技巧

### 调优方向

| 方向 | 方法 |
|------|------|
| 指令优化 | 更清晰明确 |
| 示例优化 | 选择更好示例 |
| 格式优化 | 更好输出格式 |

### 迭代调优流程

```
设计初始Prompt → 测试 → 分析问题 → 修改 → 再测试 → 循环
```

### 常见问题与解决

| 问题 | 解决 |
|------|------|
| 输出格式不对 | 明确格式要求 |
| 答案不准确 | 添加更多示例 |
| 偏题 | 修改指令 |
| 输出过长 | 限制长度 |

### Prompt调优示例

**问题**：模型输出太长

**解决**：
```python
prompt = """
请用一句话概括以下文章：
...
"""
```

**问题**：格式不统一

**解决**：
```python
prompt = """
请按以下JSON格式输出：
{"sentiment": "正面/负面", "reason": "..."}
"""
```

## 案例实践

### 分类任务Prompt

```python
def create_classification_prompt(texts, labels, new_text):
    """创建分类Prompt"""
    examples = ""
    for text, label in zip(texts[:3], labels[:3]):
        examples += f"文本：{text}\n分类：{label}\n\n"
    
    prompt = f"""
请对文本进行情感分类（正面/负面/中性）：

{examples}
文本：{new_text}
分类：
"""
    return prompt

# 使用
prompt = create_classification_prompt(
    texts=["这部电影很精彩", "很无聊"],
    labels=["正面", "负面"],
    new_text="还可以，不算特别好"
)
```

### CoT推理示例

```python
def create_cot_prompt(question):
    """创建CoT Prompt"""
    prompt = f"""
请按以下格式一步步解决问题：

问题：如果一本书有300页，每天读30页，需要几天读完？

推理过程：
1. 总页数：300页
2. 每天读：30页
3. 所需天数：300 ÷ 30 = 10天
答案：10天

问题：{question}

推理过程：
"""
    return prompt

# 数学问题
question = "商店有100件商品，卖出了35件，又进货20件，现在有多少件？"
prompt = create_cot_prompt(question)
```

### 使用LangChain的Prompt

```python
from langchain.prompts import PromptTemplate

template = """
作为一个{role}，请回答以下问题：
{question}

请按以下格式回答：
{format_instructions}
"""

prompt = PromptTemplate(
    input_variables=["role", "question", "format_instructions"],
    template=template
)

# 使用
final_prompt = prompt.format(
    role="Python专家",
    question="如何处理列表中的重复元素？",
    format_instructions="先解释，然后给代码示例"
)
```

### Few-shot学习

```python
def few_shot_classification(train_data, test_input, n_examples=5):
    """Few-shot分类"""
    # 选择示例
    examples = train_data[:n_examples]
    
    prompt = "请根据示例对文本进行分类：\n\n"
    
    for text, label in examples:
        prompt += f"文本：{text}\n标签：{label}\n\n"
    
    prompt += f"文本：{test_input}\n标签："
    
    return prompt

# 使用
train_data = [("产品很好", "正面"), ("很差", "负面")]
test_input = "质量不错"
prompt = few_shot_classification(train_data, test_input)
```

### 提示管理

```python
class PromptManager:
    """Prompt管理器"""
    def __init__(self):
        self.templates = {}
    
    def add_template(self, name, template):
        self.templates[name] = template
    
    def get_prompt(self, name, **kwargs):
        return self.templates[name].format(**kwargs)

# 使用
manager = PromptManager()
manager.add_template("classify", """
分类以下文本：{input}
类别：{categories}
""")

prompt = manager.get_prompt("classify", 
                            input="很好", 
                            categories="正面/负面")
```

## 提示工程最佳实践

### 安全性考虑

| 考虑 | 方法 |
|------|------|
| 输入验证 | 过滤恶意输入 |
| 输出控制 | 限制敏感内容 |
| 防注入 | 明确边界 |

### 成本优化

| 方法 | 描述 |
|------|------|
| 精简Prompt | 减少无效内容 |
| 批量处理 | 合并请求 |
| 缓存结果 | 重复问题缓存 |

### 效果评估

| 指标 | 描述 |
|------|------|
| 准确率 | 分类准确 |
| 完成率 | 任务完成比例 |
| 质量评分 | 输出质量评估 |

## 总结

上下文学习与提示工程是大模型应用的关键。核心内容包括：
- ICL原理：无需训练，通过示例学习
- Prompt设计：清晰、相关、统一
- Few-shot提示：少量示例引导
- Chain-of-Thought：展示推理过程
- 提示模板：统一格式，参数化
- 调优技巧：迭代优化

Prompt工程直接影响大模型应用效果。

## 延伸阅读

- [大模型架构演进](/2026/05/10/zh-CN/技术文档/机器学习/llm-architecture/)
- [预训练语言模型](/2026/05/10/zh-CN/技术文档/机器学习/pretraining-lm/)
- [微调技术详解](/2026/05/10/zh-CN/技术文档/机器学习/fine-tuning/)
- [Transformer架构详解](/2026/05/10/zh-CN/技术文档/机器学习/transformer/)
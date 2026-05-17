---
title: Semantic Kernel 实践
date: 2026-03-25
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, Semantic Kernel, Microsoft]
---

## Semantic Kernel 设计理念

Semantic Kernel 是微软开发的企业级 Agent SDK，旨在将 LLM 与传统编程语言无缝集成。

### 核心设计原则

1. **代码优先**：以代码为核心，而非 Prompt
2. **企业级**：支持安全、合规、可扩展
3. **模块化**：Skills 组织能力，易于管理
4. **自动规划**：Planner 自动组合 Skills

### 与其他框架的区别

| 特性 | Semantic Kernel | LangChain | AutoGen |
|------|-----------------|-----------|---------|
| 开发方式 | 代码优先 | Prompt 优先 | 对话优先 |
| 目标用户 | 企业开发者 | 一般开发者 | 研究者 |
| 企业支持 | 强 | 一般 | 研究友好 |
| Azure集成 | 原生 | 需配置 | 需配置 |
| 语言支持 | C#/Python | Python | Python |

## Skills、Functions、Planners

### Skills（技能）

Skills 是 Semantic Kernel 的核心抽象，代表 Agent 的能力集合。

```python
from semantic_kernel import Kernel
from semantic_kernel.skills import SkillCollection

# Skills 是 Function 的集合
skill_collection = SkillCollection()
```

### Functions（函数）

Function 是 Skills 中的具体能力：

```python
from semantic_kernel.skills import kernel_function

# 定义 Semantic Function（LLM驱动）
@kernel_function(
    description="生成产品描述",
    name="generate_description"
)
def generate_product_description(product_name: str, features: str) -> str:
    """生成产品描述"""
    # 由 LLM 执行
    pass

# 定义 Native Function（代码实现）
@kernel_function(
    description="计算总价",
    name="calculate_total"
)
def calculate_total_price(price: float, quantity: int) -> float:
    """计算总价"""
    return price * quantity
```

### Function 类型

| 类型 | 描述 | 执行方式 |
|------|------|----------|
| Semantic Function | AI 驱动函数 | LLM 执行 |
| Native Function | 代码函数 | 直接执行 |

### Planners（规划器）

Planner 自动组合 Functions 完成任务：

```python
from semantic_kernel.planners import SequentialPlanner, ActionPlanner

# Sequential Planner：顺序执行
sequential_planner = SequentialPlanner(kernel)

# Action Planner：选择最佳单一 Action
action_planner = ActionPlanner(kernel)

# 执行规划
plan = await sequential_planner.create_plan("分析销售数据并生成报告")
result = await plan.invoke()
```

### Planner 类型

| Planner | 描述 | 适用场景 |
|---------|------|----------|
| SequentialPlanner | 顺序执行多个 Function | 多步骤任务 |
| ActionPlanner | 选择最佳单一 Function | 单步骤任务 |
| StepwisePlanner | 逐步推理执行 | 复杂推理任务 |

## 与 Microsoft 生态集成

### Azure OpenAI 集成

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

kernel = Kernel()

# 添加 Azure OpenAI 服务
kernel.add_chat_service(
    "azure_chat",
    AzureChatCompletion(
        deployment_name="gpt-4",
        endpoint="https://your-resource.openai.azure.com/",
        api_key="your-api-key"
    )
)
```

### Azure Cognitive Services 集成

```python
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

# 添加 Embedding 服务
kernel.add_text_embedding_generation_service(
    "azure_embedding",
    AzureTextEmbedding(
        deployment_name="text-embedding-ada-002",
        endpoint="https://your-resource.openai.azure.com/",
        api_key="your-api-key"
    )
)
```

### Azure 存储集成

```python
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore

# 使用 Azure Cognitive Search 作为记忆存储
memory_store = AzureCognitiveSearchMemoryStore(
    endpoint="https://your-search.search.windows.net",
    api_key="your-api-key"
)

memory = SemanticTextMemory(storage=memory_store)
kernel.register_memory(memory)
```

## 实战案例：企业自动化 Agent

### 定义目标

创建一个企业自动化 Agent：
- **数据处理**：读取和分析企业数据
- **报告生成**：生成业务报告
- **邮件通知**：发送通知邮件

### 完整实现

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.skills import kernel_function, SkillCollection
from semantic_kernel.planners import SequentialPlanner

# 创建 Kernel
kernel = Kernel()

# 配置 Azure OpenAI
kernel.add_chat_service(
    "azure_chat",
    AzureChatCompletion(
        deployment_name="gpt-4",
        endpoint="https://your-resource.openai.azure.com/",
        api_key="your-api-key"
    )
)

# 定义 Skills

class DataSkill:
    """数据处理技能"""
    
    @kernel_function(
        description="读取CSV数据文件",
        name="read_csv"
    )
    def read_csv_data(self, file_path: str) -> str:
        """读取CSV文件并返回数据摘要"""
        import pandas as pd
        df = pd.read_csv(file_path)
        return f"数据行数: {len(df)}, 列数: {len(df.columns)}, 列名: {list(df.columns)}"
    
    @kernel_function(
        description="分析数据趋势",
        name="analyze_trend"
    )
    def analyze_data_trend(self, data_summary: str) -> str:
        """分析数据趋势"""
        # 这个是 Semantic Function，由 LLM 执行
        return "由 LLM 分析数据趋势"

class ReportSkill:
    """报告生成技能"""
    
    @kernel_function(
        description="生成业务报告",
        name="generate_report"
    )
    def generate_business_report(self, analysis_result: str) -> str:
        """生成业务报告"""
        return f"基于分析结果生成报告: {analysis_result}"
    
    @kernel_function(
        description="格式化报告为HTML",
        name="format_html"
    )
    def format_to_html(self, report_content: str) -> str:
        """将报告格式化为HTML"""
        return f"<html><body>{report_content}</body></html>"

class NotificationSkill:
    """通知技能"""
    
    @kernel_function(
        description="发送邮件通知",
        name="send_email"
    )
    def send_email_notification(self, recipient: str, subject: str, content: str) -> str:
        """发送邮件"""
        # 这里可以集成实际的邮件发送逻辑
        return f"邮件已发送给 {recipient}, 主题: {subject}"

# 注册 Skills
data_skill = kernel.import_skill(DataSkill(), "data")
report_skill = kernel.import_skill(ReportSkill(), "report")
notification_skill = kernel.import_skill(NotificationSkill(), "notification")

# 创建 Planner
planner = SequentialPlanner(kernel)

# 执行任务
async def run_automation():
    # 创建计划
    plan = await planner.create_plan(
        "读取 /data/sales.csv 数据，分析趋势，生成报告，并发送给 manager@company.com"
    )
    
    # 执行计划
    result = await plan.invoke()
    
    print("执行结果:", result)

# 运行
asyncio.run(run_automation())
```

### Semantic Function 定义

```python
# 使用 Prompt 定义 Semantic Function
from semantic_kernel.semantic_functions import PromptTemplate

prompt_template = PromptTemplate(
    template="""分析以下数据的趋势：
    
    {{$data_summary}}
    
    请提供：
    1. 主要趋势分析
    2. 关键发现
    3. 建议""",
    execution_settings={
        "max_tokens": 1000,
        "temperature": 0.7
    }
)

# 注册 Semantic Function
kernel.import_semantic_function_from_prompt(
    "analysis",
    "analyze_trend",
    prompt_template
)
```

### 完整企业 Agent 示例

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.semantic_functions import PromptTemplate
from semantic_kernel.skills import kernel_function
from semantic_kernel.planners import SequentialPlanner

class EnterpriseAgent:
    def __init__(self):
        self.kernel = Kernel()
        self.setup_kernel()
        self.register_skills()
    
    def setup_kernel(self):
        """配置 Kernel"""
        self.kernel.add_chat_service(
            "azure_chat",
            AzureChatCompletion(
                deployment_name="gpt-4",
                endpoint="https://your-resource.openai.azure.com/",
                api_key="your-api-key"
            )
        )
    
    def register_skills(self):
        """注册技能"""
        
        # 数据技能
        class DataSkill:
            @kernel_function
            def read_data(self, file_path: str) -> str:
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.describe().to_string()
        
        # 报告技能（Semantic）
        report_prompt = PromptTemplate(
            template="""基于以下数据生成业务报告：
            
            {{$data}}
            
            报告要求：
            - 包含数据概览
            - 包含趋势分析
            - 包含建议"""
        )
        
        # 邮件技能
        class EmailSkill:
            @kernel_function
            def send_email(self, to: str, subject: str, body: str) -> str:
                # 集成邮件服务
                return f"邮件发送成功: {to}"
        
        # 注册
        self.kernel.import_skill(DataSkill(), "data")
        self.kernel.import_semantic_function_from_prompt(
            "report", "generate", report_prompt
        )
        self.kernel.import_skill(EmailSkill(), "email")
    
    async def run_task(self, task_description: str):
        """执行任务"""
        planner = SequentialPlanner(self.kernel)
        plan = await planner.create_plan(task_description)
        result = await plan.invoke()
        return result

# 使用
async def main():
    agent = EnterpriseAgent()
    result = await agent.run_task(
        "读取 /data/monthly_sales.csv，生成销售报告，发送给 sales@company.com"
    )
    print(result)

asyncio.run(main())
```

## Semantic Kernel 进阶用法

### 记忆系统

```python
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.connectors.memory.volatile_memory_store import VolatileMemoryStore

# 创建记忆
memory = SemanticTextMemory(storage=VolatileMemoryStore())
kernel.register_memory(memory)

# 保存信息
await memory.save_information(
    "company_docs",
    id="doc1",
    text="公司年度报告内容...",
    description="2024年度报告"
)

# 检索信息
results = await memory.search("company_docs", "销售数据")
```

### 上下文变量

```python
from semantic_kernel.orchestration.context_variables import ContextVariables

# 创建上下文
context = ContextVariables()
context["data_file"] = "/data/sales.csv"
context["recipient"] = "manager@company.com"

# 执行 Function 时使用上下文
result = await kernel.run_async(
    data_skill["read_data"],
    input_vars=context
)
```

### 流式输出

```python
# 流式获取 LLM 输出
async def stream_example():
    stream = await kernel.run_stream_async(
        semantic_function,
        input_context
    )
    
    for chunk in stream:
        print(chunk, end="", flush=True)
```

### 并行执行

```python
import asyncio

async def parallel_example():
    # 并行执行多个 Function
    results = await asyncio.gather(
        kernel.run_async(skill1["func1"], input1),
        kernel.run_async(skill2["func2"], input2),
        kernel.run_async(skill3["func3"], input3)
    )
    
    # 整合结果
    combined = "\n".join(results)
    return combined
```

## Semantic Kernel 最佳实践

### Skills 组织

1. **分类组织**：按功能分类 Skills
2. **命名规范**：Function 名称清晰
3. **描述准确**：描述帮助 Planner 正确选择

### Planner 使用

1. **任务清晰**：任务描述具体
2. **限制范围**：设置 Function 约束
3. **验证计划**：检查生成的计划

### 企业部署

1. **安全配置**：配置安全策略
2. **日志记录**：记录执行日志
3. **错误处理**：完善的错误处理
4. **监控告警**：设置监控机制

## 总结

Semantic Kernel 以 Skills-Functions-Planners 架构组织 Agent 能力。Skills 是 Function 集合，Function 包括 Semantic（LLM驱动）和 Native（代码）两种类型，Planner 自动组合 Functions 执行任务。

Semantic Kernel 与 Azure 服务无缝集成，适合企业级应用开发。

## 延伸阅读

- [Agent 框架概览](/2026/05/10/zh-CN/技术文档/Agent/agent-frameworks/)
- [工具调用机制详解](/2026/05/10/zh-CN/技术文档/Agent/tool-use/)
- [Agent 安全考量](/2026/05/10/zh-CN/技术文档/Agent/agent-security/)
- [LangChain Agent 实践](/2026/05/10/zh-CN/技术文档/Agent/langchain-agent/)
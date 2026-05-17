---
title: 数据分析 Agent
date: 2026-04-10
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, 数据分析, 可视化]
---

## 数据分析 Agent 的角色

数据分析 Agent 是专门用于数据处理、分析和可视化的智能助手。它可以：

- 自动化数据处理流程
- 执行复杂的数据分析
- 生成可视化图表
- 撰写分析报告

### 与传统数据分析的区别

| 特性 | 传统数据分析 | Agent 数据分析 |
|------|--------------|----------------|
| 操作方式 | 手动编写代码 | 自然语言描述 |
| 技术门槛 | 需编程知识 | 降低门槛 |
| 分析深度 | 依赖人工经验 | 自动探索 |
| 报告生成 | 手动撰写 | 自动生成 |
| 迭代效率 | 编码调试慢 | 快速迭代 |

## 数据获取与预处理

### 数据获取能力

Agent 可以从多种来源获取数据：

```python
# 定义数据获取工具
@tool
def read_csv(file_path: str) -> str:
    """读取 CSV 文件"""
    import pandas as pd
    df = pd.read_csv(file_path)
    return f"数据形状: {df.shape}, 列: {list(df.columns)}"

@tool
def read_json(file_path: str) -> str:
    """读取 JSON 文件"""
    import json
    with open(file_path) as f:
        data = json.load(f)
    return str(data[:100])  # 返回前100条

@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询"""
    import pandas as pd
    # 实际实现会连接数据库
    return "查询结果..."

@tool
def fetch_api(url: str) -> str:
    """调用 API 获取数据"""
    import requests
    response = requests.get(url)
    return str(response.json())
```

### 数据预处理能力

```python
@tool
def clean_data(df_json: str) -> str:
    """数据清洗"""
    import pandas as pd
    df = pd.read_json(df_json)
    
    # 处理缺失值
    df = df.dropna()
    
    # 处理异常值
    for col in df.select_dtypes(include=['number']).columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
    
    return df.to_json()

@tool
def transform_data(df_json: str, operations: str) -> str:
    """数据转换"""
    import pandas as pd
    df = pd.read_json(df_json)
    
    # 执行转换操作
    # operations 是 JSON 格式的操作列表
    ops = json.loads(operations)
    for op in ops:
        if op["type"] == "normalize":
            df[op["column"]] = (df[op["column"]] - df[op["column"].min()) / (df[op["column"].max() - df[op["column"].min())
        elif op["type"] == "encode":
            df[op["column"]] = pd.get_dummies(df[op["column"]])
    
    return df.to_json()
```

## 分析任务编排

### 分析任务分解

Agent 可以自动分解复杂的分析任务：

```
用户: "分析这份销售数据，找出影响销售额的关键因素"

Agent 分解:
1. 读取数据文件
2. 数据清洗和预处理
3. 探索性分析
4. 相关性分析
5. 关键因素识别
6. 生成分析报告
```

### 分析任务执行

```python
class DataAnalysisAgent:
    def __init__(self, tools):
        self.tools = tools
    
    async def analyze(self, request):
        # 规划分析步骤
        steps = self.plan_analysis(request)
        
        # 执行每一步
        results = []
        for step in steps:
            result = self.execute_step(step)
            results.append(result)
        
        # 整合结果
        return self.synthesize(results)
    
    def plan_analysis(self, request):
        prompt = f"""
        分析请求: {request}
        
        请规划分析步骤，包括：
        1. 数据获取和准备
        2. 需要执行的分析方法
        3. 输出格式
        """
        return parse_plan(llm.generate(prompt))
    
    def execute_step(self, step):
        tool = self.tools[step["tool"]]
        return tool.invoke(step["params"])
```

### 分析方法选择

Agent 根据数据和分析目标选择方法：

| 分析目标 | 方法 |
|----------|------|
| 数据概览 | describe(), info() |
| 分布分析 | histogram, boxplot |
| 相关性 | correlation matrix |
| 趋势分析 | time series, regression |
| 分类分析 | groupby, pivot_table |
| 异常检测 | outlier detection |

## 可视化生成

### 可视化工具集成

```python
@tool
def create_histogram(data_json: str, column: str) -> str:
    """创建直方图"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_json(data_json)
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=30)
    plt.title(f'{column} 分布')
    plt.savefig('histogram.png')
    return "图表已保存: histogram.png"

@tool
def create_scatter_plot(data_json: str, x_col: str, y_col: str) -> str:
    """创建散点图"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_json(data_json)
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs {y_col}')
    plt.savefig('scatter.png')
    return "图表已保存: scatter.png"

@tool
def create_correlation_heatmap(data_json: str) -> str:
    """创建相关性热力图"""
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    df = pd.read_json(data_json)
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.savefig('correlation.png')
    return "图表已保存: correlation.png"
```

### 可视化生成流程

```
Agent 接收请求 → 选择合适图表类型 → 
生成可视化代码 → 执行代码 → 保存图表 → 
返回图表路径
```

## 报告撰写自动化

### 报告结构

数据分析报告通常包含：

1. **概述**：分析背景和目标
2. **数据概览**：数据基本信息
3. **分析方法**：使用的分析方法
4. **关键发现**：重要发现和洞察
5. **可视化**：图表展示
6. **结论建议**：总结和建议

### 报告生成实现

```python
@tool
def generate_report(analysis_results: str) -> str:
    """生成分析报告"""
    prompt = f"""
    根据以下分析结果生成报告：
    
    {analysis_results}
    
    报告结构：
    1. 分析概述
    2. 数据概览
    3. 关键发现
    4. 图表说明
    5. 结论和建议
    
    格式：Markdown
    """
    return llm.generate(prompt)
```

### 报告模板

```python
REPORT_TEMPLATE = """
# {title}

## 分析概述

{overview}

## 数据概览

- 数据量: {data_count}
- 时间范围: {time_range}
- 关键指标: {key_metrics}

## 关键发现

{findings}

## 可视化分析

{visualizations}

## 结论和建议

{conclusions}

---

报告生成时间: {timestamp}
"""

def format_report(results):
    return REPORT_TEMPLATE.format(
        title=results["title"],
        overview=results["overview"],
        data_count=results["data_count"],
        ...
    )
```

## 数据分析 Agent 完整实现

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import matplotlib.pyplot as plt
import json

# 数据工具
@tool
def load_data(file_path: str) -> str:
    """加载 CSV 或 JSON 数据文件"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_json(file_path)
    
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "types": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict()
    }
    return json.dumps(info)

@tool
def analyze_distribution(df_json: str, column: str) -> str:
    """分析列的分布"""
    df = pd.read_json(df_json)
    
    stats = {
        "mean": df[column].mean(),
        "median": df[column].median(),
        "std": df[column].std(),
        "min": df[column].min(),
        "max": df[column].max(),
        "quartiles": df[column].quantile([0.25, 0.5, 0.75]).to_dict()
    }
    return json.dumps(stats)

@tool
def calculate_correlation(df_json: str) -> str:
    """计算相关性矩阵"""
    df = pd.read_json(df_json)
    corr = df.corr()
    return corr.to_json()

@tool
def plot_histogram(df_json: str, column: str) -> str:
    """绘制直方图"""
    df = pd.read_json(df_json)
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=30, edgecolor='black')
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.savefig(f'{column}_histogram.png')
    plt.close()
    return f"Saved: {column}_histogram.png"

@tool
def plot_scatter(df_json: str, x_col: str, y_col: str) -> str:
    """绘制散点图"""
    df = pd.read_json(df_json)
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs {y_col}')
    plt.savefig(f'{x_col}_{y_col}_scatter.png')
    plt.close()
    return f"Saved: {x_col}_{y_col}_scatter.png"

@tool
def write_report(content: str, file_path: str) -> str:
    """写入报告文件"""
    with open(file_path, 'w') as f:
        f.write(content)
    return f"报告已保存: {file_path}"

# 创建 Agent
llm = ChatOpenAI(model="gpt-4")
tools = [
    load_data,
    analyze_distribution,
    calculate_correlation,
    plot_histogram,
    plot_scatter,
    write_report
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个数据分析 Agent。

    你可以：
    - 加载和分析数据
    - 计算统计指标
    - 生成可视化图表
    - 撰写分析报告
    
    分析流程：
    1. 先了解数据结构
    2. 选择合适的分析方法
    3. 执行分析和可视化
    4. 撰写报告总结发现"""),
    ("user", "{input}")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用
result = agent_executor.invoke({
    "input": """分析 sales.csv 数据：
    1. 了解数据结构和分布
    2. 分析销售额趋势
    3. 找出关键影响因素
    4. 生成可视化图表
    5. 撰写分析报告保存为 report.md"""
})

print(result["output"])
```

## 数据分析 Agent 最佳实践

### 数据处理

1. **数据验证**：先检查数据质量
2. **安全处理**：避免数据泄露
3. **增量处理**：大数据增量加载

### 分析方法

1. **方法匹配**：选择合适的分析方法
2. **多角度分析**：从不同角度探索数据
3. **验证假设**：验证分析假设

### 可视化

1. **图表匹配**：选择合适的图表类型
2. **清晰标注**：图表有清晰标题和标签
3. **颜色一致**：保持配色一致性

### 报告撰写

1. **结构清晰**：报告结构层次分明
2. **重点突出**：突出关键发现
3. **可执行建议**：提供可执行的建议

## 总结

数据分析 Agent 可以自动化数据获取、预处理、分析、可视化、报告撰写的整个流程。通过工具集成，Agent 可以执行代码进行数据处理和可视化，生成专业的分析报告。

数据分析 Agent 降低数据分析门槛，提高分析效率，适合需要频繁进行数据分析的场景。

## 延伸阅读

- [Agent 入门指南](/2026/05/10/zh-CN/技术文档/Agent/agent-intro/)
- [工具调用机制详解](/2026/05/10/zh-CN/技术文档/Agent/tool-use/)
- [Agent 规划与推理](/2026/05/10/zh-CN/技术文档/Agent/agent-planning/)
- [代码助手 Agent](/2026/05/10/zh-CN/技术文档/Agent/code-assistant-agent/)
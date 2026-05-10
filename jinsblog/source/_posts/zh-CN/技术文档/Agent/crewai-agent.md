---
title: CrewAI Agent 实践
date: 2026-05-10
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, CrewAI, Multi-Agent]
---

## CrewAI 核心概念

CrewAI 是一个角色扮演式的多 Agent 框架，将 Agent 组织为一个"团队"（Crew）完成任务。

### 三要素模型

CrewAI 的核心是三个概念：

```
Agent: 角色 Agent，代表一个专业角色
Task: 任务定义，描述具体需要完成的工作
Crew: Agent 团队，组织多个 Agent 完成任务
```

### CrewAI 设计理念

- **角色扮演**：每个 Agent 是一个专业角色
- **任务驱动**：明确的任务定义驱动执行
- **团队协作**：Agent 组成团队协同工作
- **流程控制**：Sequential 或 Hierarchical 流程

## Agent、Task、Crew 三要素

### Agent 定义

```python
from crewai import Agent

agent = Agent(
    role="研究员",               # 角色
    goal="收集和分析信息",        # 目标
    backstory="专业研究员...",    # 背景故事
    verbose=True,                # 详细输出
    allow_delegation=False,      # 是否允许委派
    tools=[search_tool],         # 工具列表
    llm="gpt-4"                  # LLM 配置
)
```

### Agent 属性详解

| 属性 | 描述 |
|------|------|
| role | Agent 的角色名称 |
| goal | Agent 的目标 |
| backstory | Agent 的背景故事（影响行为风格） |
| verbose | 是否显示详细执行过程 |
| allow_delegation | 是否允许将任务委派给其他 Agent |
| tools | Agent 可使用的工具 |
| llm | Agent 使用的 LLM |

### Task 定义

```python
from crewai import Task

task = Task(
    description="研究 AI Agent 技术的最新进展",  # 任务描述
    expected_output="一份详细的研究报告",       # 预期输出
    agent=researcher,                           # 负责的 Agent
    tools=[search_tool, web_tool],              # 可用工具
    context=[previous_task],                    # 上下文（依赖的任务）
    async_execution=False                       # 是否异步执行
)
```

### Task 属性详解

| 属性 | 描述 |
|------|------|
| description | 任务详细描述 |
| expected_output | 预期的输出格式和内容 |
| agent | 负责此任务的 Agent |
| tools | 任务可用的工具（可选） |
| context | 此任务依赖的其他任务 |
| async_execution | 是否异步执行（并行任务） |
| output_file | 输出文件路径（可选） |

### Crew 定义

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer, editor],  # Agent 列表
    tasks=[research_task, write_task, edit_task],  # 任务列表
    process=Process.sequential,           # 流程类型
    verbose=True                          # 详细输出
)

# 执行
result = crew.kickoff()
```

### Crew 属性详解

| 属性 | 描述 |
|------|------|
| agents | 团队中的 Agent 列表 |
| tasks | 团队需要完成的任务列表 |
| process | 流程类型：sequential 或 hierarchical |
| verbose | 是否显示详细过程 |
| memory | 是否启用记忆系统 |
| manager_llm | Manager Agent 的 LLM（hierarchical 流程） |

## 流程类型：Sequential、Hierarchical

### Sequential 流程（顺序流程）

任务按顺序依次执行：

```
Task1 → Task2 → Task3 → 最终结果
```

#### 顺序流程示例

```python
from crewai import Agent, Task, Crew, Process

# 定义 Agent
researcher = Agent(
    role="研究员",
    goal="收集研究信息",
    backstory="资深研究员，擅长信息收集和分析",
    verbose=True
)

writer = Agent(
    role="作家",
    goal="撰写高质量内容",
    backstory="专业作家，擅长内容创作",
    verbose=True
)

editor = Agent(
    role="编辑",
    goal="审核和优化内容",
    backstory="资深编辑，注重质量",
    verbose=True
)

# 定义任务
research_task = Task(
    description="研究 AI Agent 技术趋势",
    expected_output="研究摘要，包含关键发现",
    agent=researcher
)

write_task = Task(
    description="基于研究结果撰写文章",
    expected_output="完整的文章草稿",
    agent=writer,
    context=[research_task]  # 依赖研究任务的结果
)

edit_task = Task(
    description="审核和修改文章",
    expected_output="最终完善的文章",
    agent=editor,
    context=[write_task]
)

# 创建 Crew（顺序流程）
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,
    verbose=True
)

# 执行
result = crew.kickoff()
print(result)
```

### Hierarchical 流程（层次流程）

Manager Agent 负责分配任务：

```
Manager Agent
     ↓ 分析任务，分配给 Worker
Worker Agents 执行
     ↓ 返回结果
Manager 整合结果
```

#### 层次流程示例

```python
from crewai import Agent, Task, Crew, Process

# 定义 Manager Agent
manager = Agent(
    role="项目经理",
    goal="协调团队完成项目",
    backstory="经验丰富的项目经理",
    allow_delegation=True  # 允许委派
)

# 定义 Worker Agents
researcher = Agent(
    role="研究员",
    goal="收集信息",
    backstory="专业研究员"
)

writer = Agent(
    role="作家",
    goal="撰写内容",
    backstory="专业作家"
)

# 定义任务（不指定 Agent，由 Manager 分配）
main_task = Task(
    description="撰写一篇关于 AI Agent 的文章",
    expected_output="高质量的文章"
)

# 创建 Crew（层次流程）
crew = Crew(
    agents=[manager, researcher, writer],
    tasks=[main_task],
    process=Process.hierarchical,
    manager_llm="gpt-4",  # Manager 的 LLM
    verbose=True
)

# 执行
result = crew.kickoff()
```

### 流程对比

| 特性 | Sequential | Hierarchical |
|------|------------|--------------|
| 执行顺序 | 固定顺序 | Manager 动态分配 |
| 任务分配 | 每个任务指定 Agent | Manager 决定 |
| 灵活性 | 低，任务顺序固定 | 高，可动态调整 |
| 适用场景 | 流程明确的任务 | 复杂不确定的任务 |

## Tools 与知识集成

### 工具定义

CrewAI 使用 LangChain 工具：

```python
from crewai_tools import tool
from langchain.tools import Tool

# 使用 @tool 装饰器
@tool("搜索互联网")
def search_web(query: str) -> str:
    """搜索互联网获取信息"""
    return search_api(query)

# 或使用 LangChain Tool
search_tool = Tool(
    name="search",
    description="搜索互联网",
    func=search_web
)
```

### Agent 工具绑定

```python
researcher = Agent(
    role="研究员",
    goal="收集信息",
    backstory="专业研究员",
    tools=[search_tool, web_scraper_tool]  # 绑定工具
)
```

### CrewAI 内置工具

```python
from crewai_tools import (
    SerperDevTool,       # Google 搜索
    ScrapeWebsiteTool,   # 网页抓取
    FileReadTool,        # 文件读取
    DirectoryReadTool,   # 目录读取
    CodeInterpreterTool  # 代码执行
)

# 使用内置工具
search = SerperDevTool()
scraper = ScrapeWebsiteTool()

researcher = Agent(
    role="研究员",
    tools=[search, scraper]
)
```

### 知识集成

```python
from crewai import Knowledge

# 定义知识源
knowledge = Knowledge(
    sources=[
        {"type": "file", "path": "knowledge.pdf"},
        {"type": "web", "url": "https://example.com/docs"}
    ]
)

# Agent 使用知识
researcher = Agent(
    role="研究员",
    knowledge=knowledge
)
```

## 实战案例：内容创作 Agent 团队

### 定义目标

创建一个内容创作团队：
- **研究员**：研究主题，收集素材
- **策划师**：制定内容大纲
- **作家**：撰写内容
- **编辑**：审核修改

### 完整实现

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# 工具
search_tool = SerperDevTool()
scraper_tool = ScrapeWebsiteTool()

# 定义 Agent
researcher = Agent(
    role="内容研究员",
    goal="深入研究主题，收集高质量素材",
    backstory="""你是一位专业的内容研究员。
    你擅长：
    - 搜索和收集相关信息
    - 分析和提炼关键观点
    - 整理素材供后续使用""",
    verbose=True,
    tools=[search_tool, scraper_tool]
)

planner = Agent(
    role="内容策划师",
    goal="制定内容大纲和结构",
    backstory="""你是一位资深内容策划师。
    你擅长：
    - 分析素材，提炼要点
    - 设计内容结构和逻辑
    - 制定写作大纲""",
    verbose=True
)

writer = Agent(
    role="内容作家",
    goal="撰写高质量的内容",
    backstory="""你是一位专业作家。
    你擅长：
    - 根据大纲撰写内容
    - 语言表达流畅
    - 内容逻辑清晰""",
    verbose=True
)

editor = Agent(
    role="内容编辑",
    goal="审核和优化内容质量",
    backstory="""你是一位资深编辑。
    你擅长：
    - 内容审核和质量把控
    - 语言优化和润色
    - 确保内容专业准确""",
    verbose=True
)

# 定义任务
research_task = Task(
    description="""研究 AI Agent 技术的最新进展。
    
    要求：
    1. 搜索最新的技术文章和报告
    2. 收集关键技术趋势
    3. 提取重要观点和数据""",
    expected_output="一份详细的研究摘要，包含关键发现和素材",
    agent=researcher
)

plan_task = Task(
    description="""基于研究结果制定文章大纲。
    
    要求：
    1. 分析研究素材
    2. 设计文章结构
    3. 明确每个章节要点""",
    expected_output="完整的文章大纲，包含章节结构",
    agent=planner,
    context=[research_task]
)

write_task = Task(
    description="""根据大纲撰写文章。
    
    要求：
    1. 遵循大纲结构
    2. 内容详实准确
    3. 语言流畅专业""",
    expected_output="完整的文章草稿",
    agent=writer,
    context=[plan_task]
)

edit_task = Task(
    description="""审核和修改文章。
    
    要求：
    1. 检查内容准确性
    2. 优化语言表达
    3. 确保逻辑连贯""",
    expected_output="最终完善的文章",
    agent=editor,
    context=[write_task]
)

# 创建 Crew
crew = Crew(
    agents=[researcher, planner, writer, editor],
    tasks=[research_task, plan_task, write_task, edit_task],
    process=Process.sequential,
    verbose=True
)

# 执行
result = crew.kickoff()

print("=== 最终文章 ===")
print(result)
```

### 执行流程

```
1. Researcher: 搜索资料 → 生成研究摘要
2. Planner: 分析素材 → 制定大纲
3. Writer: 根据大纲 → 撰写文章
4. Editor: 审核 → 输出最终文章
```

### 并行任务示例

```python
# 多个研究任务并行执行
research_task1 = Task(
    description="研究 Agent 架构",
    agent=researcher1,
    async_execution=True  # 异步执行
)

research_task2 = Task(
    description="研究 Agent 框架",
    agent=researcher2,
    async_execution=True
)

# 整合任务依赖并行任务
synthesize_task = Task(
    description="整合研究结果",
    agent=planner,
    context=[research_task1, research_task2]  # 依赖并行任务
)

crew = Crew(
    agents=[researcher1, researcher2, planner],
    tasks=[research_task1, research_task2, synthesize_task],
    process=Process.sequential
)
```

## CrewAI 进阶用法

### 任务输出保存

```python
task = Task(
    description="撰写文章",
    expected_output="文章内容",
    agent=writer,
    output_file="output/article.md"  # 输出到文件
)

crew.kickoff()
# 结果保存到 output/article.md
```

### 记忆系统

```python
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential,
    memory=True  # 启用记忆系统
)

# 记忆系统会记录任务结果，供后续任务使用
```

### 回调函数

```python
def task_callback(task_output):
    """任务完成回调"""
    print(f"任务完成: {task_output.task_description}")
    print(f"结果: {task_output.raw}")

task = Task(
    description="...",
    callback=task_callback  # 设置回调
)
```

### 人机交互

```python
# 在任务中请求人工输入
human_input_task = Task(
    description="请用户确认内容方向",
    agent=researcher,
    human_input=True  # 等待人工输入
)
```

## CrewAI 最佳实践

### Agent 设计

1. **角色明确**：role 和 goal 清晰定义
2. **背景丰富**：backstory 影响行为风格
3. **工具匹配**：工具与角色能力匹配
4. **避免重叠**：Agent 职责不重叠

### Task 设计

1. **描述详细**：任务描述包含具体要求
2. **输出明确**：expected_output 明确格式
3. **依赖正确**：context 设置任务依赖
4. **合理分配**：任务与 Agent 能力匹配

### Crew 设计

1. **流程选择**：根据任务选择 sequential 或 hierarchical
2. **数量控制**：Agent 和任务数量适中
3. **成本控制**：verbose=False 减少输出成本

## 总结

CrewAI 以 Agent-Task-Crew 三要素构建多 Agent 系统。Sequential 流程按固定顺序执行，Hierarchical 流程由 Manager 动态分配。工具集成使用 LangChain 工具，知识源支持文件和网页。

CrewAI API 简洁直观，适合角色分工明确、流程清晰的任务。

## 延伸阅读

- [Agent 框架概览](/2026/05/10/zh-CN/技术文档/Agent/agent-frameworks/)
- [多 Agent 协作模式](/2026/05/10/zh-CN/技术文档/Agent/multi-agent-collaboration/)
- [层次化 Agent 系统](/2026/05/10/zh-CN/技术文档/Agent/hierarchical-agent/)
- [AutoGen 多 Agent 实践](/2026/05/10/zh-CN/技术文档/Agent/autogen-agent/)
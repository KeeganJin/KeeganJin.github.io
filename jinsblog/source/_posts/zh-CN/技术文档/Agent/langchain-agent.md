---
title: LangChain Agent 实践
date: 2026-03-16
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, LangChain, LangGraph]
---

## LangChain Agent 核心概念

LangChain 的 Agent 系统基于以下核心组件：

### Agent 类型

LangChain 提供多种 Agent 类型：

| Agent 类型 | 描述 | 适用场景 |
|------------|------|----------|
| Zero-shot Agent | 无示例，直接执行 | 简单任务 |
| Conversational Agent | 支持对话记忆 | 多轮对话 |
| Structured Tool Agent | 支持多输入工具 | 复杂工具调用 |
| OpenAI Functions Agent | 使用 Function Calling | OpenAI 模型 |

### 核心组件

```
Agent: 决策组件，决定下一步行动
AgentExecutor: 执行组件，管理 Agent 执行循环
Tools: 工具集合，Agent 可调用的能力
Prompt: 提示模板，引导 Agent 行为
Memory: 记忆组件，维护对话上下文
```

## AgentExecutor 详解

### AgentExecutor 基本结构

AgentExecutor 是 Agent 的执行引擎：

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,           # Agent 组件
    tools=tools,           # 工具列表
    memory=memory,         # 记忆（可选）
    verbose=True,          # 显示执行过程
    max_iterations=10,     # 最大迭代次数
    handle_parsing_errors=True  # 处理解析错误
)
```

### AgentExecutor 执行流程

```
1. 接收输入
2. Agent 分析输入，决定下一步行动
3. 如果决定调用工具 → 执行工具 → 观察结果
4. 如果决定输出 → 返回最终结果
5. 重复步骤2-4，直到输出或达到最大迭代
```

### AgentExecutor 配置参数

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    
    # 执行控制
    max_iterations=10,        # 最大迭代次数
    max_execution_time=30,    # 最大执行时间（秒）
    early_stopping_method="generate",  # 早停策略
    
    # 错误处理
    handle_parsing_errors=True,  # 处理解析错误
    handle_tool_errors=True,     # 处理工具错误
    
    # 输出控制
    return_intermediate_steps=True,  # 返回中间步骤
    verbose=True,                     # 显示详细日志
    
    # 记忆
    memory=memory  # 对话记忆
)
```

### 执行示例

```python
# 执行 Agent
result = agent_executor.invoke({
    "input": "北京今天天气怎么样？"
})

# 输出结构
{
    "input": "北京今天天气怎么样？",
    "output": "北京今天晴天，气温25°C",
    "intermediate_steps": [
        (AgentAction(tool="search", tool_input="北京天气"), "北京晴天，25°C")
    ]
}
```

## 工具定义与绑定

### 定义工具

#### 使用 @tool 装饰器

```python
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索互联网获取信息。
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果摘要
    """
    results = search_api(query)
    return format_results(results)

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。
    
    Args:
        city: 女城市名称
    
    Returns:
        天气描述
    """
    return weather_api.get(city)
```

#### 使用 Tool 类

```python
from langchain.tools import Tool

search_tool = Tool(
    name="search",
    description="搜索互联网获取信息。输入搜索关键词，返回相关结果。",
    func=search_web
)

weather_tool = Tool(
    name="get_weather",
    description="获取城市天气。输入城市名称，返回天气信息。",
    func=get_weather
)
```

#### 使用 StructuredTool

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str
    unit: str = "celsius"  # 温度单位

def get_weather_structured(input: WeatherInput) -> str:
    return weather_api.get(input.city, input.unit)

weather_tool = StructuredTool(
    name="get_weather",
    description="获取城市天气，可指定温度单位",
    func=get_weather_structured,
    args_schema=WeatherInput
)
```

### 工具集合

```python
tools = [
    search_tool,
    weather_tool,
    calculator_tool,
    file_reader_tool
]
```

### 绑定工具到 Agent

```python
from langchain.agents import create_openai_functions_agent
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools)
```

## LangGraph 状态机模型

### LangGraph 简介

LangGraph 是 LangChain 的扩展，用于构建复杂 Agent 流程：

- 基于状态图（StateGraph）
- 支持循环、分支
- 支持多 Agent 协作
- 更细粒度的流程控制

### 状态图核心概念

```python
from langgraph.graph import StateGraph

# 定义状态
class AgentState(TypedDict):
    messages: List[Message]
    current_step: str
    tool_results: Dict

# 创建状态图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

# 添加边
graph.add_edge("agent", "tool")
graph.add_edge("tool", "agent")

# 设置入口和结束
graph.set_entry_point("agent")
graph.set_finish_point("output")

# 编译
app = graph.compile()
```

### 状态图示例：简单 Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    input: str
    messages: List[str]
    tool_calls: List[dict]
    output: str

def agent_node(state: AgentState):
    """Agent 决策节点"""
    # LLM 分析并决定行动
    response = llm.invoke(state["messages"])
    
    if response.tool_calls:
        return {"tool_calls": response.tool_calls}
    else:
        return {"output": response.content}

def tool_node(state: AgentState):
    """工具执行节点"""
    results = []
    for call in state["tool_calls"]:
        tool = tools[call["name"]]
        result = tool.invoke(call["args"])
        results.append(result)
    
    return {"tool_results": results}

def should_continue(state: AgentState):
    """条件判断"""
    if state["output"]:
        return END
    else:
        return "tool"

# 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tool", "agent")

app = graph.compile()
```

### 状态图示例：多 Agent

```python
from langgraph.graph import StateGraph

class MultiAgentState(TypedDict):
    input: str
    researcher_result: str
    writer_result: str
    editor_result: str
    final_output: str

def researcher_node(state):
    """研究 Agent"""
    result = researcher_agent.invoke(state["input"])
    return {"researcher_result": result}

def writer_node(state):
    """写作 Agent"""
    content = writer_agent.invoke(
        f"基于研究结果写作：{state['researcher_result']}"
    )
    return {"writer_result": content}

def editor_node(state):
    """编辑 Agent"""
    edited = editor_agent.invoke(state["writer_result"])
    return {"final_output": edited}

# 构建多 Agent 图
graph = StateGraph(MultiAgentState)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("editor", editor_node)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "editor")
graph.add_edge("editor", END)

app = graph.compile()
```

### 循环和分支

```python
# 循环示例
graph.add_conditional_edges(
    "agent",
    lambda state: "continue" if not state["done"] else END,
    {"continue": "tool", END: END}
)

# 分支示例
graph.add_conditional_edges(
    "router",
    lambda state: state["next_agent"],
    {"researcher": "researcher", "writer": "writer", "editor": "editor"}
)
```

## 实战案例：构建问答 Agent

### 定义目标

构建一个能够回答问题的 Agent，具备：
- 搜索能力（获取实时信息）
- 计算（精确计算）
- 天气查询（天气信息）

### 实现步骤

#### 1. 定义工具

```python
from langchain.tools import tool
import requests

@tool
def search_web(query: str) -> str:
    """搜索互联网获取实时信息"""
    response = requests.get(f"https://api.search.com/search?q={query}")
    return response.json()["results"][0]["content"]

@tool
def calculate(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)  # 实际应用需使用安全的计算方式
        return str(result)
    except:
        return "计算错误"

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    response = requests.get(f"https://api.weather.com/{city}")
    data = response.json()
    return f"{city}天气：{data['weather']}, 温度：{data['temp']}°C"

tools = [search_web, calculate, get_weather]
```

#### 2. 创建 Agent

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能问答助手。使用工具获取信息来回答用户问题。"),
    ("user", "{input}")
])

agent = create_openai_functions_agent(llm, tools, prompt)
```

#### 3. 创建执行器

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
```

#### 4. 测试运行

```python
# 测试搜索
result1 = agent_executor.invoke({"input": "最新的AI新闻是什么？"})
print(result1["output"])

# 测试计算
result2 = agent_executor.invoke({"input": "计算 123 * 456"})
print(result2["output"])

# 测试天气
result3 = agent_executor.invoke({"input": "北京今天天气怎么样？"})
print(result3["output"])
```

### 添加对话记忆

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 多轮对话
agent_executor.invoke({"input": "北京天气怎么样？"})
agent_executor.invoke({"input": "那上海呢？"})  # 会理解是在问天气
```

### 使用 LangGraph 构建

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class QAState(TypedDict):
    input: str
    messages: List[Message]
    tool_results: List[str]
    output: str

def agent_node(state):
    messages = state["messages"] + [HumanMessage(content=state["input"])]
    response = llm.bind_tools(tools).invoke(messages)
    
    if response.tool_calls:
        return {"tool_calls": response.tool_calls, "messages": messages + [response]}
    return {"output": response.content, "messages": messages + [response]}

def tool_node(state):
    results = []
    for call in state["tool_calls"]:
        tool_func = {t.name: t for t in tools}[call["name"]]
        result = tool_func.invoke(call["args"])
        results.append(result)
    return {"tool_results": results}

def should_continue(state):
    return END if state.get("output") else "tool"

graph = StateGraph(QAState)
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tool", "agent")

# 添加记忆
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 使用
config = {"configurable": {"thread_id": "user1"}}
result = app.invoke({"input": "北京天气怎么样？"}, config)
```

## LangChain Agent 最佳实践

### 工具设计

1. **描述清晰**：工具描述要准确说明用途
2. **参数简洁**：避免过多参数
3. **返回格式**：返回易于理解的结果
4. **错误处理**：优雅处理失败情况

### Prompt 设计

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手。
    
    规则：
    1. 先思考需要什么信息
    2. 使用合适的工具获取信息
    3. 整合信息回答用户
    4. 如果工具失败，说明原因
    
    可用工具：
    - search: 搜索互联网
    - calculate: 数学计算
    - get_weather: 天气查询
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])
```

### 执行控制

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,           # 限制迭代次数
    max_execution_time=60,      # 限制执行时间
    handle_parsing_errors=True, # 处理解析错误
    handle_tool_errors=True     # 处理工具错误
)
```

### 调试技巧

```python
# 启用详细日志
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)

# 查看中间步骤
result = agent_executor.invoke({"input": "问题"})
for step in result["intermediate_steps"]:
    print(f"Action: {step[0].tool}")
    print(f"Input: {step[0].tool_input}")
    print(f"Output: {step[1]}")
```

## 总结

LangChain Agent 基于 Agent + AgentExecutor + Tools 结构。LangGraph 扩展了 LangChain，支持复杂状态流转、多 Agent 协作。实战中需要精心设计工具、Prompt，控制执行参数，并做好错误处理。

LangChain/LangGraph 适合需要灵活控制流程、丰富工具集成的场景。

## 延伸阅读

- [Agent 框架概览](/2026/05/10/zh-CN/技术文档/Agent/agent-frameworks/)
- [工具调用机制详解](/2026/05/10/zh-CN/技术文档/Agent/tool-use/)
- [Agent 规划与推理](/2026/05/10/zh-CN/技术文档/Agent/agent-planning/)
- [Agent 调试技巧](/2026/05/10/zh-CN/技术文档/Agent/agent-debugging/)
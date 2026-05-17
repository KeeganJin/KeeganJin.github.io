---
title: AutoGen 多 Agent 实践
date: 2026-03-19
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, AutoGen, Multi-Agent]
---

## AutoGen 核心概念

AutoGen 是微软开发的多 Agent 框架，以对话为核心协作机制。

### 核心 Agent 类型

```
ConversableAgent: 可对话的 Agent 基类
AssistantAgent: AI Agent（LLM驱动）
UserProxyAgent: 人类代理（可选人工介入）
GroupChatManager: 群聊管理器
```

### AutoGen 设计理念

- **对话驱动**：Agent 通过对话协作
- **人机协同**：支持人类介入
- **灵活定制**：Agent 可高度定制
- **研究友好**：适合实验和探索

## ConversableAgent 设计

### ConversableAgent 基本属性

```python
from autogen import ConversableAgent

agent = ConversableAgent(
    name="agent_name",             # Agent 名称
    system_message="...",          # 系统提示
    llm_config=llm_config,         # LLM 配置
    human_input_mode="NEVER",      # 人工输入模式
    max_consecutive_auto_reply=10, # 最大自动回复次数
    code_execution_config=False,   # 代码执行配置
    function_map=None              # 函数映射
)
```

### 创建 AssistantAgent

```python
from autogen import AssistantAgent

assistant = AssistantAgent(
    name="assistant",
    system_message="""你是一个智能助手，帮助用户完成任务。
    你可以使用工具和代码来解决问题。""",
    llm_config={
        "config_list": [
            {"model": "gpt-4", "api_key": "your-api-key"}
        ]
    }
)
```

### 创建 UserProxyAgent

```python
from autogen import UserProxyAgent

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS",  # 始终等待人工输入
    max_consecutive_auto_reply=0,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# 或半自动模式
semi_automatic = UserProxyAgent(
    name="user",
    human_input_mode="TERMINATE",  # 仅在结束时请求人工
    max_consecutive_auto_reply=10
)
```

### human_input_mode 参数

| 模式 | 描述 |
|------|------|
| ALWAYS | 每次都等待人工输入 |
| NEVER | 不等待人工输入 |
| TERMINATE | 仅在终止信号时请求人工 |

### llm_config 配置

```python
llm_config = {
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": "your-api-key",
            "base_url": "https://api.openai.com/v1"  # 可选
        },
        {
            "model": "gpt-3.5-turbo",
            "api_key": "your-api-key"
        }
    ],
    "temperature": 0,
    "cache_seed": None,  # 缓存种子
    "timeout": 120       # 超时时间
}
```

## GroupChat 与群聊模式

### GroupChat 基本结构

```python
from autogen import GroupChat, GroupChatManager

# 创建 Agent 组
groupchat = GroupChat(
    agents=[agent1, agent2, agent3, user],
    messages=[],             # 消息历史
    max_round=10,            # 最大对话轮次
    admin_name="admin",      # 管理员名称
    speaker_selection_method="auto"  # 发言者选择方法
)

# 创建管理器
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)
```

### speaker_selection_method

| 方法 | 描述 |
|------|------|
| auto | 自动选择下一个发言者 |
| round_robin | 按顺序轮流发言 |
| random | 随机选择发言者 |
| manual | 手动选择 |

### 简单群聊示例

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 定义 Agent
researcher = AssistantAgent(
    name="Researcher",
    system_message="你是研究 Agent，负责收集和分析信息。",
    llm_config=llm_config
)

writer = AssistantAgent(
    name="Writer",
    system_message="你是写作 Agent，负责撰写内容。",
    llm_config=llm_config
)

editor = AssistantAgent(
    name="Editor",
    system_message="你是编辑 Agent，负责审核和修改内容。",
    llm_config=llm_config
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0
)

# 创建群聊
groupchat = GroupChat(
    agents=[user, researcher, writer, editor],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# 启动对话
user.initiate_chat(
    manager,
    message="写一篇关于 AI Agent 技术的文章。"
)
```

### 自定义发言者选择

```python
def custom_speaker_selection(last_speaker, groupchat):
    """自定义发言者选择逻辑"""
    
    # 根据上一发言者决定下一个
    if last_speaker.name == "User":
        return researcher
    
    if last_speaker.name == "Researcher":
        return writer
    
    if last_speaker.name == "Writer":
        return editor
    
    if last_speaker.name == "Editor":
        return user  # 结束
    
    # 默认
    return researcher

groupchat = GroupChat(
    agents=[user, researcher, writer, editor],
    messages=[],
    max_round=10,
    speaker_selection_method=custom_speaker_selection
)
```

### 群聊流程控制

```python
# 添加终止条件
groupchat = GroupChat(
    agents=[...],
    messages=[],
    max_round=20,
    terminate_condition=lambda msg: "任务完成" in msg["content"]
)

# 或在 Agent 中设置终止信号
editor = AssistantAgent(
    name="Editor",
    system_message="""你是编辑 Agent。
    审核完成后，回复 '任务完成' 来结束对话。""",
    llm_config=llm_config
)
```

## 人机协作机制

### 人机协作模式

AutoGen 支持多种人机协作模式：

#### 完全自动化

```python
user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)
```

#### 半自动化

```python
user = UserProxyAgent(
    name="user",
    human_input_mode="TERMINATE",  # 任务结束时请求人工确认
    max_consecutive_auto_reply=10
)
```

#### 全人工介入

```python
user = UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS"  # 每轮都等待人工输入
)
```

### 人工审核示例

```python
# 定义需要人工审核的 Agent
reviewer = UserProxyAgent(
    name="Reviewer",
    human_input_mode="ALWAYS",
    system_message="请审核以下内容并给出修改建议。"
)

# 在对话中等待审核
researcher.initiate_chat(
    reviewer,
    message="以下是研究结果，请审核。"
)
# 系统会等待人工输入
```

### 人工介入时机

```python
# 在关键节点请求人工介入
def should_request_human(state):
    """判断是否需要人工介入"""
    # 检查是否需要人工决策
    if state["decision_needed"]:
        return True
    
    # 检查是否遇到错误
    if state["error_count"] > 3:
        return True
    
    return False

# 设置 Agent
agent = ConversableAgent(
    name="agent",
    llm_config=llm_config,
    human_input_mode="TERMINATE",
    terminate_condition=should_request_human
)
```

## 实战案例：代码生成 Agent 组

### 定义目标

创建一个代码生成 Agent 组：
- **需求 Agent**：理解需求并转化为技术规格
- **编码 Agent**：根据规格生成代码
- **测试 Agent**：编写测试代码
- **人类代理**：审核和确认

### 实现代码

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# LLM 配置
llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": "your-key"}]
}

# 需求分析 Agent
requirements_agent = AssistantAgent(
    name="RequirementsAgent",
    system_message="""你是需求分析专家。
    
    任务：
    1. 分析用户需求
    2. 转化为技术规格
    3. 列出功能点和接口
    
    输出格式：
    - 功能列表
    - API 接口定义
    - 数据结构
    
    完成后回复 '需求分析完成'。""",
    llm_config=llm_config
)

# 编码 Agent
coder_agent = AssistantAgent(
    name="CoderAgent",
    system_message="""你是资深程序员。
    
    任务：
    1. 根据需求规格编写代码
    2. 使用 Python
    3. 代码应清晰、可测试
    
    完成后回复 '编码完成'。""",
    llm_config=llm_config,
    code_execution_config={
        "work_dir": "generated_code",
        "use_docker": False
    }
)

# 测试 Agent
tester_agent = AssistantAgent(
    name="TesterAgent",
    system_message="""你是测试工程师。
    
    任务：
    1. 根据代码编写单元测试
    2. 测试覆盖主要功能
    3. 使用 pytest
    
    完成后回复 '测试完成'。""",
    llm_config=llm_config
)

# 人类代理
user = UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=0,
    code_execution_config={"work_dir": "generated_code"}
)

# 创建群聊
groupchat = GroupChat(
    agents=[user, requirements_agent, coder_agent, tester_agent],
    messages=[],
    max_round=20
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# 启动任务
user.initiate_chat(
    manager,
    message="""请帮我开发一个简单的计算器程序：
    - 支持加减乘除
    - 支持命令行交互
    - 有错误处理"""
)
```

### 工作流程

```
User → 需求分析 → RequirementsAgent
              ↓
需求规格 → 编码 → CoderAgent
              ↓
代码 → 测试 → TesterAgent
              ↓
测试代码 → 审核 → User
              ↓
最终确认 → 完成
```

### 执行结果

Agent 会按顺序：
1. RequirementsAgent 分析需求，输出技术规格
2. CoderAgent 根据规格生成代码
3. TesterAgent 编写测试
4. User 审核并确认

## AutoGen 工具调用

### 注册函数工具

```python
from autogen import ConversableAgent, register_function

# 定义函数
def search_web(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果：{query}"

def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city}：晴天，25°C"

# 创建 Agent
agent = ConversableAgent(
    name="agent",
    llm_config=llm_config
)

# 注册工具
agent.register_function(
    function_map={
        "search_web": search_web,
        "get_weather": get_weather
    }
)

# 使用
user.initiate_chat(
    agent,
    message="北京天气怎么样？"
)
# Agent 会自动调用 get_weather 函数
```

### 工具定义格式

```python
# 函数描述来自 docstring
def calculate(expression: str) -> float:
    """执行数学计算
    
    Args:
        expression: 数学表达式，如 "1+2*3"
    
    Returns:
        计算结果
    """
    return eval(expression)

# 注册时自动解析描述
agent.register_function(
    function_map={"calculate": calculate}
)
```

### 多 Agent 共享工具

```python
# 定义共享工具
shared_tools = {
    "search_web": search_web,
    "get_weather": get_weather,
    "calculate": calculate
}

# 多个 Agent 使用相同工具
agent1 = ConversableAgent(name="agent1", llm_config=llm_config)
agent2 = ConversableAgent(name="agent2", llm_config=llm_config)

for agent in [agent1, agent2]:
    agent.register_function(function_map=shared_tools)
```

## AutoGen 代码执行

### 配置代码执行

```python
code_execution_config = {
    "work_dir": "coding",      # 工作目录
    "use_docker": False,       # 是否使用 Docker
    "timeout": 60,             # 执行超时
    "last_n_messages": 3       # 检查最近N条消息中的代码
}
```

### Agent 执行代码

```python
coder = AssistantAgent(
    name="Coder",
    system_message="你是程序员，可以编写和执行代码。",
    llm_config=llm_config,
    code_execution_config=code_execution_config
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config=code_execution_config
)

# Agent 会自动执行代码块
user.initiate_chat(
    coder,
    message="帮我计算斐波那契数列前10项"
)
# Coder 会生成代码，UserProxyAgent 会执行
```

### 代码块格式

AutoGen 会识别以下格式的代码块：

```
```python
# 代码内容
print("Hello")
```
```

## AutoGen 进阶用法

### Nested Chat（嵌套对话）

```python
# Agent 内部可以有嵌套对话
def nested_chat_workflow():
    # 主对话中的 Agent 可以发起内部对话
    inner_chat = GroupChat(
        agents=[sub_agent1, sub_agent2],
        messages=[]
    )
    
    return inner_chat

agent = ConversableAgent(
    name="MainAgent",
    llm_config=llm_config,
    nested_chat_config={
        "trigger": lambda msg: "复杂任务" in msg["content"],
        "chat": nested_chat_workflow
    }
)
```

### 多轮对话记录

```python
# 获取对话历史
def get_chat_history(chat_result):
    history = []
    for msg in chat_result.chat_history:
        history.append({
            "sender": msg["role"],
            "content": msg["content"]
        })
    return history

# 使用
result = user.initiate_chat(manager, message="...")
history = get_chat_history(result)
```

### 消息摘要

```python
# 对长对话进行摘要
from autogen import ConversableAgent

summary_agent = ConversableAgent(
    name="Summarizer",
    system_message="你负责总结对话内容。",
    llm_config=llm_config
)

# 获取摘要
def summarize_chat(chat_history):
    prompt = f"请总结以下对话：\n{format_history(chat_history)}"
    return summary_agent.generate_reply([{"content": prompt}])
```

## AutoGen 最佳实践

### Agent 定义

1. **明确角色**：system_message 清晰定义职责
2. **限制范围**：避免 Agent 职责重叠
3. **终止信号**：设置明确的完成信号

### 群聊设计

1. **控制轮次**：设置合理的 max_round
2. **发言选择**：根据需求选择 speaker_selection_method
3. **终止条件**：设置明确的终止条件

### 成本控制

1. **限制轮次**：避免无限对话
2. **缓存配置**：启用缓存减少重复调用
3. **模型选择**：根据任务选择合适模型

## 总结

AutoGen 以对话为核心构建多 Agent 系统。ConversableAgent 是核心 Agent 类，GroupChat 支持群聊协作。人机协同机制支持不同级别的人工介入。

AutoGen 适合需要多 Agent 协作、对话式任务处理、研究实验的场景。

## 延伸阅读

- [Agent 框架概览](/2026/05/10/zh-CN/技术文档/Agent/agent-frameworks/)
- [多 Agent 协作模式](/2026/05/10/zh-CN/技术文档/Agent/multi-agent-collaboration/)
- [层次化 Agent 系统](/2026/05/10/zh-CN/技术文档/Agent/hierarchical-agent/)
- [CrewAI Agent 实践](/2026/05/10/zh-CN/技术文档/Agent/crewai-agent/)
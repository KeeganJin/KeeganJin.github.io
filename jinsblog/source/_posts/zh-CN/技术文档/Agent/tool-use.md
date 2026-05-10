---
title: 工具调用机制（Tool Use）
date: 2026-05-10
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, Tool Use, Function Calling]
---

## 工具调用的基本原理

工具调用（Tool Use）是 Agent 扩展自身能力的关键机制。通过工具调用，Agent 可以突破 LLM 的局限，与外部世界交互、获取实时信息、执行实际操作。

### 为什么需要工具调用？

LLM 有明确的局限：
- 无法获取实时信息（训练数据截止后的事件）
- 无法执行精确计算（数学运算可能出错）
- 无法直接操作外部系统（数据库、API 等）
- 无法持久化存储（无法自身维护记忆）

工具调用解决了这些问题，让 Agent 能够：
- 搜索网络获取最新信息
- 执行代码进行精确计算
- 调用 API 操作外部系统
- 读写文件持久化数据

### 工具调用的基本流程

```
用户请求 → LLM 分析 → 选择工具 → 生成参数 → 执行工具 → 返回结果 → LLM 整合 → 输出响应
```

完整流程：
1. **意图分析**：LLM 分析用户请求，判断是否需要调用工具
2. **工具选择**：从可用工具列表中选择合适的工具
3. **参数生成**：根据工具定义生成调用参数
4. **工具执行**：系统执行工具调用
5. **结果处理**：获取工具返回结果
6. **结果整合**：LLM 将工具结果整合到响应中
7. **用户响应**：向用户返回最终答案

## 工具定义与描述（Function Calling）

### 工具定义格式

工具需要以结构化方式定义，让 LLM 能够理解工具的功能和使用方式。

**JSON Schema 格式示例**：

```json
{
  "name": "search_web",
  "description": "搜索互联网获取实时信息，适用于需要最新数据或外部知识的查询",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "搜索关键词或查询语句"
      },
      "num_results": {
        "type": "integer",
        "description": "返回结果数量，默认为5",
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

### 定义要素解析

#### name（工具名称）
- 唯一标识工具
- 建议使用动词+名词格式（如 `search_web`, `send_email`）
- 应简洁明了，便于理解

#### description（工具描述）
- 描述工具的功能和用途
- 说明何时应该使用此工具
- 说明工具返回什么类型的结果
- 描述应足够详细，帮助 LLM 正确选择

#### parameters（参数定义）
- 使用 JSON Schema 定义参数结构
- 每个参数需要类型和描述
- 标注必填参数（required）
- 提供默认值（可选）

### 工具描述的最佳实践

#### 1. 描述工具何时使用

```json
{
  "description": "当用户询问当前天气、天气预报或与天气相关的问题时使用此工具。不要用于历史天气数据查询。"
}
```

#### 2. 描述参数的预期值

```json
{
  "query": {
    "description": "城市名称或地理位置，如'北京'、'New York'。支持中文和英文。"
  }
}
```

#### 3. 说明工具的限制

```json
{
  "description": "搜索最近的新闻文章。注意：只能搜索最近30天的新闻，不支持历史新闻搜索。"
}
```

### 多工具定义示例

```json
{
  "tools": [
    {
      "name": "search_web",
      "description": "搜索互联网获取信息",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "搜索查询"}
        },
        "required": ["query"]
      }
    },
    {
      "name": "execute_code",
      "description": "执行Python代码进行计算或数据处理",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {"type": "string", "description": "要执行的Python代码"},
          "timeout": {"type": "integer", "description": "执行超时时间（秒）", "default": 30}
        },
        "required": ["code"]
      }
    },
    {
      "name": "read_file",
      "description": "读取本地文件内容",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "文件路径"}
        },
        "required": ["path"]
      }
    }
  ]
}
```

## 工具选择策略

### LLM 自动选择

最常见的方式是让 LLM 根据工具描述自动选择：

- LLM 分析用户请求
- 比对工具描述
- 选择最匹配的工具
- 生成调用参数

**优点**：
- 灵活，适应性强
- 可以处理模糊请求
- 可以组合多个工具

**缺点**：
- 可能选择错误
- 依赖描述质量
- 增加推理成本

### 规则驱动选择

对明确场景使用规则选择：

```python
def select_tool(query):
    if "天气" in query or "weather" in query:
        return "get_weather"
    if "搜索" in query or "查找" in query:
        return "search_web"
    if "计算" in query or any(op in query for op in ["+", "-", "*", "/"]):
        return "execute_code"
    return None
```

**优点**：
- 确定性强
- 响应快
- 无额外成本

**缺点**：
- 灵活性差
- 需要维护规则
- 难覆盖所有情况

### 混合策略

结合两种方式：

1. 先用规则快速匹配明确场景
2. 规则无法匹配时使用 LLM 选择
3. LLM 选择后可验证是否符合规则

## 工具执行与结果处理

### 工具执行架构

```
┌─────────────┐
│  Agent Core │
└─────────────┘
      ↓ 工具调用请求
┌─────────────┐
│ Tool Manager│
└─────────────┘
      ↓ 调度执行
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Web Tool   │ │  Code Tool  │ │  File Tool  │
└─────────────┘ └─────────────┘ └─────────────┘
      ↓ 结果返回
┌─────────────┐
│  Result     │
│  Processor  │
└─────────────┘
```

### 工具执行流程

#### 1. 参数验证
- 检查必填参数是否存在
- 检查参数类型是否正确
- 检查参数值是否合法

#### 2. 权限检查
- 检查是否有执行权限
- 检查是否超出操作范围
- 记录操作日志

#### 3. 执行调用
- 调用外部 API
- 执行代码
- 操作文件系统

#### 4. 结果处理
- 格式化返回结果
- 处理错误情况
- 限制返回数据量

### 结果处理策略

#### 结果截断

工具返回大量数据时需要截断：

```python
def process_result(result, max_length=2000):
    if len(result) > max_length:
        return result[:max_length] + "...[截断]"
    return result
```

#### 结果格式化

将原始结果转换为 LLM 易理解的格式：

```python
def format_search_results(results):
    formatted = "搜索结果：\n"
    for i, item in enumerate(results[:5], 1):
        formatted += f"{i}. {item['title']}\n"
        formatted += f"   {item['snippet']}\n"
        formatted += f"   来源: {item['url']}\n\n"
    return formatted
```

#### 错误处理

工具执行失败时的处理：

```python
def handle_tool_error(error):
    if "timeout" in str(error):
        return "工具执行超时，请稍后重试"
    if "permission" in str(error):
        return "权限不足，无法执行此操作"
    return f"工具执行失败: {str(error)}"
```

## 工具调用的错误处理

### 常见错误类型

| 错误类型 | 原因 | 处理方式 |
|----------|------|----------|
| 参数错误 | 参数缺失或类型错误 | 返回错误信息，请求重新生成 |
| 权限错误 | 无执行权限 | 返回权限错误，建议替代方案 |
| 执行超时 | 工具响应慢 | 超时处理，可选重试 |
| 结果异常 | 返回值非预期 | 错误处理，LLM 重新决策 |
| 工具不存在 | 调用未定义工具 | 返回工具列表，请求重新选择 |

### 错误处理流程

```
工具调用 → 执行 → 成功? → 是 → 返回结果
                  ↓否
              错误分类 → 参数错误 → 重新生成参数
                       → 执行错误 → 返回错误信息
                       → 超时 → 重试或替代方案
                       → 其他 → LLM 重新决策
```

### 重试机制

对于可恢复错误实施重试：

```python
async def call_tool_with_retry(tool_name, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = await execute_tool(tool_name, params)
            return result
        except TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
                continue
            raise
```

## 常见工具类型

### API 调用工具

调用外部服务 API：

```json
{
  "name": "call_api",
  "description": "调用外部API获取数据或执行操作",
  "parameters": {
    "type": "object",
    "properties": {
      "endpoint": {"type": "string", "description": "API端点URL"},
      "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
      "body": {"type": "object", "description": "请求体（POST/PUT时）"}
    },
    "required": ["endpoint", "method"]
  }
}
```

### 文件操作工具

读写文件系统：

```json
{
  "name": "read_file",
  "description": "读取指定路径的文件内容",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "文件路径"}
    },
    "required": ["path"]
  }
}
```

```json
{
  "name": "write_file",
  "description": "写入内容到指定文件",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "文件路径"},
      "content": {"type": "string", "description": "要写入的内容"}
    },
    "required": ["path", "content"]
  }
}
```

### 数据库查询工具

执行数据库操作：

```json
{
  "name": "query_database",
  "description": "执行SQL查询获取数据",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "SQL查询语句"},
      "database": {"type": "string", "description": "数据库名称"}
    },
    "required": ["query"]
  }
}
```

### 代码执行工具

执行代码进行计算：

```json
{
  "name": "execute_python",
  "description": "执行Python代码进行计算、数据处理或可视化",
  "parameters": {
    "type": "object",
    "properties": {
      "code": {"type": "string", "description": "要执行的Python代码"}
    },
    "required": ["code"]
  }
}
```

### 搜索工具

搜索网络或内部知识：

```json
{
  "name": "search_web",
  "description": "使用搜索引擎搜索互联网信息",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "搜索查询"}
    },
    "required": ["query"]
  }
}
```

## 工具安全性考量

### 安全风险

工具调用引入的安全风险：

1. **Prompt Injection**：恶意用户通过输入诱导 Agent 执行危险操作
2. **权限滥用**：Agent 执行超出预期的操作
3. **数据泄露**：工具调用暴露敏感数据
4. **资源滥用**：消耗系统资源或产生费用

### 安全防护措施

#### 1. 工具权限控制

限制每个工具的能力范围：

```python
# 定义工具权限
tool_permissions = {
    "read_file": {
        "allowed_paths": ["/data/", "/tmp/"],
        "max_size": 10MB
    },
    "execute_python": {
        "allowed_modules": ["numpy", "pandas", "matplotlib"],
        "network_access": False,
        "file_access": False
    }
}
```

#### 2. 操作审计

记录所有工具调用：

```python
def log_tool_call(tool_name, params, result, user_id):
    log_entry = {
        "timestamp": datetime.now(),
        "tool": tool_name,
        "params": params,
        "result_summary": summarize(result),
        "user": user_id
    }
    audit_log.append(log_entry)
```

#### 3. 参数过滤

过滤危险参数：

```python
def sanitize_params(tool_name, params):
    if tool_name == "execute_python":
        # 检查危险操作
        dangerous_patterns = ["import os", "subprocess", "eval("]
        for pattern in dangerous_patterns:
            if pattern in params["code"]:
                raise SecurityError(f"禁止执行: {pattern}")
    return params
```

#### 4. 用户确认

敏感操作需用户确认：

```python
def require_confirmation(tool_name, params):
    sensitive_tools = ["delete_file", "send_email", "execute_shell"]
    if tool_name in sensitive_tools:
        return ask_user_confirmation(f"确认执行 {tool_name}?")
    return True
```

### 安全护栏设计

```
用户输入 → 输入检查 → LLM 决策 → 工具选择 → 权限检查 → 参数过滤 → 执行 → 结果检查 → 输出
            ↓           ↓          ↓          ↓          ↓
         危险检测    决策验证    工具白名单   权限验证   参数清理
```

## Function Calling 实现示例

### OpenAI Function Calling

```python
import openai

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 发送请求
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools
)

# 检查是否需要调用工具
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    
    # 执行工具
    result = execute_tool(tool_name, tool_args)
    
    # 将结果返回给模型
    follow_up = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "北京今天天气怎么样？"},
            response.choices[0].message,
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        ]
    )
```

### Anthropic Tool Use

```python
import anthropic

client = anthropic.Anthropic()

# 定义工具
tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的当前天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    }
]

# 发送请求
response = client.messages.create(
    model="claude-3-opus",
    max_tokens=1024,
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools
)

# 处理工具调用
for block in response.content:
    if block.type == "tool_use":
        tool_name = block.name
        tool_input = block.input
        result = execute_tool(tool_name, tool_input)
        
        # 继续对话
        follow_up = client.messages.create(
            model="claude-3-opus",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "北京今天天气怎么样？"},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                ]}
            ],
            tools=tools
        )
```

## 总结

工具调用是 Agent 系统的核心能力，让 LLM 能够突破自身局限，与外部世界交互。工具定义的质量直接影响 LLM 的选择准确性。工具执行需要完善的错误处理和安全防护机制。

设计工具调用系统时，需要在灵活性（LLM 自动选择）和确定性（规则驱动）之间取得平衡，同时确保安全性和可调试性。

## 延伸阅读

- [Agent 入门指南](/2026/05/10/zh-CN/技术文档/Agent/agent-intro/)
- [Agent 记忆系统](/2026/05/10/zh-CN/技术文档/Agent/agent-memory/)
- [Agent 规划与推理](/2026/05/10/zh-CN/技术文档/Agent/agent-planning/)
- [Agent 安全考量](/2026/05/10/zh-CN/技术文档/Agent/agent-security/)
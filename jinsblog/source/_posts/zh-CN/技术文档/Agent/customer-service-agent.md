---
title: 客户服务 Agent
date: 2026-05-10
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, 客服, 对话系统]
---

## 客服 Agent 的优势

客户服务 Agent（客服 Agent）在客户支持领域有显著优势：

### 相比传统客服

| 特性 | 传统客服 | Agent客服 |
|------|----------|-----------|
| 响应时间 | 可能延迟 | 即时响应 |
| 服务时间 | 工作时间 | 24/7全天候 |
| 处理能力 | 受限于人力 | 可扩展 |
| 成本 | 人力成本高 | 成本可控 |
| 一致性 | 可能不一致 | 标准一致 |
| 知识传递 | 需培训 | 知识库驱动 |

### Agent客服的价值

- **降低成本**：减少人力客服需求
- **提升体验**：即时响应、全天候服务
- **提高效率**：批量处理常见问题
- **数据积累**：自动记录对话数据

## 知识库集成

### 知识库的作用

知识库是客服 Agent 的核心支撑：
- 提供产品和服务信息
- 提供常见问题解答
- 提供流程和政策指南

### 知识库类型

| 类型 | 内容 | 示例 |
|------|------|------|
| FAQ库 | 常见问题 | "退货流程是什么？" |
| 产品库 | 产品信息 | "产品规格、价格" |
| 流程库 | 业务流程 | "投诉处理流程" |
| 政策库 | 公司政策 | "退款政策" |

### 知识库集成实现

```python
from langchain.tools import tool
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 创建知识库
embeddings = OpenAIEmbeddings()
knowledge_base = Chroma(
    embedding_function=embeddings,
    persist_directory="knowledge_db"
)

@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库获取相关信息
    
    Args:
        query: 用户问题
    
    Returns:
        相关知识内容
    """
    results = knowledge_base.similarity_search(query, k=3)
    
    if results:
        content = "\n".join([r.page_content for r in results])
        return f"找到相关信息：\n{content}"
    return "未找到相关信息"

@tool
def get_product_info(product_id: str) -> str:
    """获取产品详细信息
    
    Args:
        product_id: 产品ID
    
    Returns:
        产品信息
    """
    # 从产品数据库查询
    product = product_db.get(product_id)
    if product:
        return f"""
        产品名称: {product.name}
        价格: {product.price}
        规格: {product.specs}
        库存: {product.stock}
        """
    return "未找到该产品"
```

### 知识库更新

```python
@tool
def update_knowledge(topic: str, content: str) -> str:
    """更新知识库
    
    Args:
        topic: 知识主题
        content: 知识内容
    
    Returns:
        更新结果
    """
    knowledge_base.add_texts(
        texts=[content],
        metadatas=[{"topic": topic}]
    )
    return f"知识已更新: {topic}"
```

## 多轮对话管理

### 多轮对话的重要性

客服场景通常是多轮对话：
- 用户问题可能需要多步澄清
- 解决方案可能需要多步引导
- 需要保持对话连贯性

### 对话状态管理

```python
class ConversationState:
    def __init__(self):
        self.session_id = None
        self.user_intent = None
        self.current_topic = None
        self.history = []
        self.pending_actions = []
        self.resolved = False

class ConversationManager:
    def __init__(self):
        self.sessions = {}  # session_id -> ConversationState
    
    def get_state(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationState()
        return self.sessions[session_id]
    
    def update_state(self, session_id, key, value):
        state = self.get_state(session_id)
        setattr(state, key, value)
    
    def add_message(self, session_id, role, content):
        state = self.get_state(session_id)
        state.history.append({"role": role, "content": content})
```

### 多轮对话流程

```python
def handle_conversation(session_id, user_message):
    state = conversation_manager.get_state(session_id)
    
    # 分析意图
    intent = analyze_intent(user_message, state.history)
    state.user_intent = intent
    
    # 根据意图和状态决定响应
    if intent == "ask_question":
        # 搜索知识库
        answer = search_knowledge(user_message)
        
        if is_complete_answer(answer):
            return answer
        else:
            # 需要更多信息
            state.pending_actions.append("need_more_info")
            return "请提供更多详细信息..."
    
    elif intent == "complaint":
        # 投诉处理流程
        if state.current_topic == None:
            state.current_topic = "complaint"
            return "请问您投诉的具体内容是什么？"
        elif len(state.history) < 3:
            return "请详细描述您的问题..."
        else:
            # 提交投诉
            submit_complaint(state.history)
            state.resolved = True
            return "您的投诉已提交，我们会尽快处理。"
    
    elif intent == "human_request":
        # 转人工
        return transfer_to_human(session_id, state)
```

### 对话历史管理

```python
def manage_history(history, max_turns=10):
    """管理对话历史长度"""
    if len(history) > max_turns * 2:  # 用户+系统各max_turns条
        # 保留最近的对话
        return history[-max_turns * 2:]
    
    # 或使用摘要压缩
    if len(history) > 5:
        early_history = history[:5]
        summary = summarize_conversation(early_history)
        return [{"role": "system", "content": summary}] + history[5:]
    
    return history
```

## 情绪识别与处理

### 情绪识别

识别用户情绪有助于提供更好的服务：

```python
@tool
def detect_emotion(message: str) -> str:
    """检测用户消息的情绪
    
    Args:
        message: 用户消息
    
    Returns:
        情绪类型和强度
    """
    prompt = f"""
    分析以下消息的情绪：
    
    消息: {message}
    
    请判断：
    1. 情绪类型：愤怒/不满/焦急/中性/满意
    2. 情绪强度：1-5级
    
    格式：情绪类型 | 强度
    """
    return llm.generate(prompt)
```

### 情绪处理策略

```python
def handle_negative_emotion(emotion_type, intensity):
    """处理负面情绪"""
    
    if emotion_type == "愤怒" and intensity >= 4:
        # 高强度愤怒，优先安抚
        return """
        我理解您现在很生气。
        请告诉我具体问题，我会尽力帮您解决。
        如果需要，我可以为您转接人工客服。
        """
    
    elif emotion_type == "不满":
        # 表达不满，主动询问
        return """
        感谢您的反馈。
        请详细说明您不满意的地方，我们会改进。
        """
    
    elif emotion_type == "焦急":
        # 焦急，快速回应
        return """
        我会尽快帮您处理这个问题。
        请稍等，我正在查询相关信息。
        """
    
    return "请问有什么可以帮您的？"

def handle_positive_emotion(emotion_type, intensity):
    """处理正面情绪"""
    
    if emotion_type == "满意":
        return """
        感谢您的认可！
        如有其他需要，随时联系我们。
        """
    
    return "请问还有什么可以帮您的？"
```

### 情绪自适应响应

```python
def adaptive_response(user_message, conversation_state):
    # 先检测情绪
    emotion = detect_emotion(user_message)
    emotion_type, intensity = parse_emotion(emotion)
    
    # 根据情绪调整响应策略
    if emotion_type in ["愤怒", "不满", "焦急"]:
        # 负面情绪
        response_strategy = "empathetic_fast"
        base_response = handle_negative_emotion(emotion_type, intensity)
    else:
        # 正面或中性情绪
        response_strategy = "standard"
        base_response = "请问有什么可以帮您的？"
    
    # 结合实际问题和知识库
    if needs_knowledge_search(user_message):
        knowledge = search_knowledge(user_message)
        return combine_response(base_response, knowledge)
    
    return base_response
```

## 人工转接机制

### 转接触发条件

何时需要转接人工客服：

| 条件 | 描述 |
|------|------|
| 用户请求 | 用户明确要求人工 |
| 情绪极端 | 高强度愤怒情绪 |
| 问题复杂 | Agent 无法解决的问题 |
| 涉及隐私 | 需要验证身份信息 |
| 投诉升级 | 用户不满意 Agent 处理 |
| 技术故障 | Agent 系统故障 |

### 转接实现

```python
@tool
def transfer_to_human(session_id: str, reason: str) -> str:
    """转接人工客服
    
    Args:
        session_id: 会话ID
        reason: 转接原因
    
    Returns:
        转接状态
    """
    # 创建转接请求
    transfer_request = {
        "session_id": session_id,
        "reason": reason,
        "conversation_history": get_history(session_id),
        "priority": calculate_priority(reason),
        "timestamp": datetime.now()
    }
    
    # 发送到人工客服队列
    human_queue.add(transfer_request)
    
    return """
    正在为您转接人工客服...
    请稍候，客服人员很快会接听。
    """

def should_transfer_to_human(conversation_state, user_message):
    """判断是否需要转接"""
    
    # 用户明确请求
    if contains_keyword(user_message, ["人工", "真人客服", "投诉"]):
        return True, "用户请求人工客服"
    
    # 情绪极端
    emotion = detect_emotion(user_message)
    if emotion["type"] == "愤怒" and emotion["intensity"] >= 4:
        return True, "用户情绪激动"
    
    # 问题无法解决
    if conversation_state.failed_attempts >= 3:
        return True, "问题无法自动解决"
    
    # 涉及隐私验证
    if contains_keyword(user_message, ["账户", "密码", "身份证", "银行卡"]):
        return True, "涉及敏感信息验证"
    
    return False, None
```

### 转接信息传递

```python
def prepare_handoff_info(session_id):
    """准备转接信息"""
    state = get_conversation_state(session_id)
    
    return {
        "session_id": session_id,
        "customer_id": state.customer_id,
        "conversation_summary": summarize_conversation(state.history),
        "current_issue": state.current_topic,
        "attempts": len(state.history),
        "pending_actions": state.pending_actions,
        "emotion_state": state.last_emotion
    }
```

## 客服 Agent 完整实现

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# 工具定义
@tool
def search_faq(query: str) -> str:
    """搜索常见问题解答"""
    results = faq_db.search(query)
    return format_results(results)

@tool
def get_product_info(product_id: str) -> str:
    """获取产品信息"""
    return product_db.get(product_id)

@tool
def check_order_status(order_id: str) -> str:
    """查询订单状态"""
    return order_db.get_status(order_id)

@tool
def detect_sentiment(message: str) -> str:
    """检测情绪"""
    prompt = f"分析情绪: {message}"
    return llm.generate(prompt)

@tool
def transfer_to_human_agent(reason: str) -> str:
    """转接人工客服"""
    human_queue.add({"reason": reason, "time": datetime.now()})
    return "正在转接人工客服，请稍候..."

@tool
def log_interaction(issue_type: str, resolution: str) -> str:
    """记录交互"""
    analytics.log(issue_type, resolution)
    return "已记录"

# 创建 Agent
llm = ChatOpenAI(model="gpt-4")
tools = [
    search_faq,
    get_product_info,
    check_order_status,
    detect_sentiment,
    transfer_to_human_agent,
    log_interaction
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个客户服务 Agent。

    服务原则：
    1. 友好礼貌，耐心倾听
    2. 准确理解用户问题
    3. 使用知识库提供准确信息
    4. 注意用户情绪变化
    5. 无法解决时及时转人工

    可用工具：
    - search_faq: 搜索常见问题
    - get_product_info: 产品信息
    - check_order_status: 订单查询
    - detect_sentiment: 情绪检测
    - transfer_to_human_agent: 转人工
    
    如果用户情绪激动或问题复杂，请主动转接人工客服。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 使用
def handle_customer_message(session_id, message):
    # 检查是否需要转接
    need_transfer, reason = should_transfer(message)
    if need_transfer:
        return transfer_to_human_agent(reason)
    
    # Agent 处理
    result = agent_executor.invoke({"input": message})
    
    # 记录交互
    log_interaction(session_id, result["output"])
    
    return result["output"]
```

## 客服 Agent 最佳实践

### 知识库管理

1. **定期更新**：保持知识库时效性
2. **分类清晰**：知识分类便于检索
3. **反馈优化**：根据对话反馈优化知识

### 对话设计

1. **开场友好**：友好开场问候
2. **清晰引导**：清晰引导用户表达
3. **确认理解**：确认理解用户问题
4. **礼貌结束**：礼貌结束对话

### 情绪处理

1. **及时识别**：及时识别负面情绪
2. **主动安抚**：主动安抚不满情绪
3. **灵活转接**：灵活转接人工客服

### 持续优化

1. **对话分析**：分析对话数据找问题
2. **知识补充**：补充缺失的知识
3. **流程优化**：优化处理流程

## 总结

客服 Agent 通过知识库集成、多轮对话管理、情绪识别、人工转接机制，提供高效的客户服务。相比传统客服，Agent客服有即时响应、全天候服务、成本可控的优势。

设计客服 Agent 需要注意：知识库质量、对话连贯性、情绪感知、合理的转接策略。

## 延伸阅读

- [Agent 入门指南](/2026/05/10/zh-CN/技术文档/Agent/agent-intro/)
- [Agent 记忆系统](/2026/05/10/zh-CN/技术文档/Agent/agent-memory/)
- [Agent 安全考量](/2026/05/10/zh-CN/技术文档/Agent/agent-security/)
- [Agent 评测方法](/2026/05/10/zh-CN/技术文档/Agent/agent-evaluation/)
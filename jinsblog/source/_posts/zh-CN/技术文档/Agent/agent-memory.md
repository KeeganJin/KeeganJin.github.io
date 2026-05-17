---
title: Agent 记忆系统
date: 2026-03-07
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, Memory, 向量数据库]
---

## 记忆系统的重要性

记忆系统是 Agent 的关键组件，它让 Agent 能够：
- 维持对话上下文（记住刚才说了什么）
- 保存任务状态（知道任务进展到哪里）
- 存储长期知识（积累经验和知识）
- 跨会话一致性（下次对话还能记得）

没有记忆系统，Agent 每次交互都从零开始，无法处理需要多次交互的复杂任务，也无法提供个性化的服务。

## 记忆系统的分类

Agent 记忆系统通常分为三层：

### 短期记忆（Short-term Memory）

短期记忆存储当前对话的上下文信息。

| 特性 | 描述 |
|------|------|
| 存储内容 | 当前对话历史 |
| 存储时长 | 对话会话期间 |
| 存储容量 | 受 LLM 上下文窗口限制 |
| 访问速度 | 实时访问 |

**典型实现**：
- 直接传递对话历史给 LLM
- 滑动窗口保持最近 N 条消息
- Token 限制截断早期消息

### 工作记忆（Working Memory）

工作记忆存储当前任务的状态和中间结果。

| 特性 | 描述 |
|------|------|
| 存储内容 | 当前任务状态、中间结果、临时变量 |
| 存储时长 | 任务执行期间 |
| 存储容量 | 灵活配置 |
| 访问速度 | 实时访问 |

**典型实现**：
- 任务状态机
- 临时变量存储
- 中间结果缓存

### 长期记忆（Long-term Memory）

长期记忆持久化存储知识、经验、用户偏好等。

| 特性 | 描述 |
|------|------|
| 存储内容 | 知识库、用户偏好、历史经验 |
| 存储时长 | 持久化存储 |
| 存储容量 | 可扩展（GB~TB级） |
| 访问速度 | 需检索，相对较慢 |

**典型实现**：
- 向量数据库（语义检索）
- 知识图谱（结构化知识）
- 关系数据库（用户数据）

## 短期记忆详解

### 对话历史管理

短期记忆的核心是管理对话历史：

```
对话历史:
[
  {"role": "user", "content": "帮我分析这份销售数据"},
  {"role": "assistant", "content": "好的，请提供数据文件路径"},
  {"role": "user", "content": "文件在 /data/sales_2024.csv"},
  {"role": "assistant", "content": "我已读取数据，共有1000条记录..."}
]
```

### 滑动窗口策略

LLM 上下文窗口有限，需要控制历史消息数量：

```python
def manage_conversation_history(history, max_messages=20):
    # 保持系统消息
    system_msg = [m for m in history if m["role"] == "system"]
    
    # 保持最近的消息
    recent_msgs = history[-max_messages:]
    
    # 组合返回
    return system_msg + recent_msgs
```

### Token 限制截断

更精确的方式是按 Token 数限制：

```python
def truncate_by_tokens(history, max_tokens=4000):
    total_tokens = 0
    truncated = []
    
    # 从最新消息开始，倒序添加
    for msg in reversed(history):
        msg_tokens = count_tokens(msg["content"])
        if total_tokens + msg_tokens > max_tokens:
            break
        truncated.insert(0, msg)
        total_tokens += msg_tokens
    
    return truncated
```

### 消息摘要压缩

对于长对话，可以压缩早期消息：

```python
def compress_old_messages(history, compress_threshold=10):
    if len(history) > compress_threshold:
        # 提取早期消息
        old_msgs = history[:compress_threshold]
        
        # 生成摘要
        summary = generate_summary(old_msgs)
        
        # 替换为摘要消息
        compressed = [{"role": "system", "content": f"对话摘要: {summary}"}]
        compressed += history[compress_threshold:]
        
        return compressed
    return history
```

## 工作记忆详解

### 任务状态管理

工作记忆保存当前任务的状态：

```python
class TaskState:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 5
        self.completed_subtasks = []
        self.pending_subtasks = []
        self.intermediate_results = {}
        self.variables = {}
```

### 任务状态示例

```
任务: 分析销售数据并生成报告

工作记忆状态:
{
  "current_step": 2,
  "total_steps": 4,
  "steps": [
    {"id": 1, "name": "读取数据", "status": "completed", "result": "..."},
    {"id": 2, "name": "数据分析", "status": "in_progress"},
    {"id": 3, "name": "生成可视化", "status": "pending"},
    {"id": 4, "name": "撰写报告", "status": "pending"}
  ],
  "variables": {
    "data_path": "/data/sales_2024.csv",
    "data_records": 1000,
    "analysis_type": "trend"
  }
}
```

### 中间结果存储

存储任务执行过程中的中间数据：

```python
class IntermediateResults:
    def __init__(self):
        self.results = {}
    
    def store(self, key, value):
        self.results[key] = value
    
    def get(self, key):
        return self.results.get(key)
    
    def clear(self):
        self.results.clear()
```

### 变量管理

管理任务执行中的临时变量：

```python
# Agent 可以在工作记忆中存储变量
state.variables["current_file"] = "sales_2024.csv"
state.variables["format_preference"] = "markdown"

# 后续步骤可以引用这些变量
file = state.variables["current_file"]
```

## 长期记忆详解

### 向量数据库存储

长期记忆最常用的实现是向量数据库：

#### 工作原理

```
原始文本 → Embedding模型 → 向量 → 存入向量数据库
                     ↓
查询文本 → Embedding模型 → 向量 → 向量检索 → 返回相似文档
```

#### 存储流程

```python
def store_to_memory(content, metadata):
    # 生成向量
    embedding = embedding_model.encode(content)
    
    # 存入向量数据库
    vector_db.insert(
        vector=embedding,
        text=content,
        metadata=metadata
    )
```

#### 检索流程

```python
def retrieve_from_memory(query, top_k=5):
    # 生成查询向量
    query_embedding = embedding_model.encode(query)
    
    # 检索相似内容
    results = vector_db.search(
        vector=query_embedding,
        top_k=top_k
    )
    
    return results
```

### 向量数据库选择

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| Pinecone | 云托管、易用 | 生产环境、快速部署 |
| Weaviate | 开源、功能丰富 | 企业级应用 |
| Milvus | 高性能、可扩展 | 大规模数据 |
| Chroma | 轻量、本地部署 | 开发测试 |
| Qdrant | 高效、Rust实现 | 性能敏感场景 |

### 知识图谱存储

结构化知识使用知识图谱存储：

#### 工作原理

```
知识三元组: (实体, 关系, 实体)
示例: (北京, 是, 中国首都)
     (Python, 用于, 数据分析)
```

#### 存储示例

```python
def add_knowledge(entity1, relation, entity2):
    knowledge_graph.add_triplet(entity1, relation, entity2)
    
# 查询知识
def query_knowledge(entity, relation):
    return knowledge_graph.query(f"{entity} {relation} ?")
```

### 用户数据存储

用户偏好和历史数据存储在关系数据库：

```sql
-- 用户偏好表
CREATE TABLE user_preferences (
    user_id VARCHAR(50),
    preference_key VARCHAR(100),
    preference_value TEXT,
    updated_at TIMESTAMP
);

-- 用户历史交互表
CREATE TABLE user_history (
    id SERIAL,
    user_id VARCHAR(50),
    interaction_type VARCHAR(50),
    content TEXT,
    created_at TIMESTAMP
);
```

## 记忆检索策略

### 检索时机

何时从长期记忆检索信息：

1. **用户查询触发**：根据用户问题检索相关知识
2. **任务开始前**：预加载相关背景知识
3. **执行间隙**：补充任务所需知识
4. **定期更新**：保持记忆时效性

### 检索方法

#### 语义检索

基于向量相似度检索：

```python
def semantic_retrieve(query, top_k=5):
    query_vector = embedding_model.encode(query)
    results = vector_db.search(query_vector, top_k)
    return results
```

#### 关键词检索

基于关键词匹配检索：

```python
def keyword_retrieve(query):
    keywords = extract_keywords(query)
    results = []
    for kw in keywords:
        matches = search_by_keyword(kw)
        results.extend(matches)
    return deduplicate(results)
```

#### 混合检索

结合语义和关键词检索：

```python
def hybrid_retrieve(query):
    # 语义检索
    semantic_results = semantic_retrieve(query, top_k=3)
    
    # 关键词检索
    keyword_results = keyword_retrieve(query)
    
    # 合并排序
    combined = merge_and_rank(semantic_results, keyword_results)
    return combined[:5]
```

### 检索优化

#### 检索过滤

根据元数据过滤检索结果：

```python
def filtered_retrieve(query, filters):
    results = vector_db.search(
        vector=embedding_model.encode(query),
        filter=filters  # 如 {"category": "sales", "date": "2024"}
    )
    return results
```

#### 重排序

检索后根据相关性重排：

```python
def rerank_results(query, results):
    # 使用重排序模型计算精确相关性
    reranker = RerankerModel()
    scores = reranker.score(query, [r["text"] for r in results])
    
    # 按分数排序
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked]
```

## 记忆压缩与摘要

### 压缩必要性

长期记忆数据量增长需要压缩：
- 减少存储空间
- 提高检索效率
- 保持关键信息

### 压缩方法

#### 滚动摘要

定期生成历史摘要：

```python
def rolling_summary(history_entries):
    # 分组
    groups = group_by_time(history_entries, interval="1day")
    
    # 每组生成摘要
    summaries = []
    for group in groups:
        summary = llm.summarize(group)
        summaries.append({
            "date": group["date"],
            "summary": summary,
            "key_events": extract_key_events(group)
        })
    
    return summaries
```

#### 关键信息提取

提取并保留关键信息：

```python
def extract_key_info(content):
    key_info = {
        "entities": extract_entities(content),
        "facts": extract_facts(content),
        "decisions": extract_decisions(content),
        "actions": extract_actions(content)
    }
    return key_info
```

### 摘要存储

```python
def store_summary(summary, metadata):
    # 存储摘要文本
    vector_db.insert(
        vector=embedding_model.encode(summary["text"]),
        text=summary["text"],
        metadata={
            "type": "summary",
            "date_range": summary["date_range"],
            "source_ids": summary["source_ids"]
        }
    )
```

## 记忆更新与遗忘机制

### 记忆更新

根据新信息更新已有记忆：

#### 信息合并

```python
def update_memory(new_info, existing_memory):
    # 检查冲突
    conflicts = find_conflicts(new_info, existing_memory)
    
    if conflicts:
        # 解决冲突（保留更新信息）
        resolved = resolve_conflicts(new_info, existing_memory)
        return resolved
    else:
        # 合合信息
        return merge_info(new_info, existing_memory)
```

#### 版本管理

保留记忆的版本历史：

```python
class MemoryVersion:
    def __init__(self):
        self.versions = []
    
    def add_version(self, content, timestamp):
        self.versions.append({
            "content": content,
            "timestamp": timestamp,
            "version_id": len(self.versions)
        })
    
    def get_latest(self):
        return self.versions[-1]["content"]
```

### 遗忘机制

不重要的记忆需要遗忘：

#### 时间衰减

随时间降低记忆权重：

```python
def time_decay(timestamp, current_time, decay_rate=0.1):
    age_days = (current_time - timestamp).days
    weight = math.exp(-decay_rate * age_days)
    return weight
```

#### 访问频率

根据访问频率决定保留：

```python
def access_frequency_pruning(memory_db, threshold=5):
    # 统计访问频率
    access_counts = count_accesses(memory_db)
    
    # 删除低频记忆
    for item in memory_db:
        if access_counts[item.id] < threshold:
            memory_db.delete(item.id)
```

#### 重要性评分

根据重要性保留记忆：

```python
def importance_pruning(memory_db, min_importance=0.3):
    for item in memory_db:
        importance = calculate_importance(item)
        if importance < min_importance:
            memory_db.delete(item.id)
```

## 记忆系统实现方案

### Redis 实现

Redis 用于快速存储短期和工作记忆：

```python
import redis

class RedisMemory:
    def __init__(self):
        self.client = redis.Redis(host='localhost', port=6379)
    
    # 短期记忆
    def save_conversation(self, session_id, messages):
        key = f"conv:{session_id}"
        self.client.set(key, json.dumps(messages))
        self.client.expire(key, 3600)  # 1小时过期
    
    def get_conversation(self, session_id):
        key = f"conv:{session_id}"
        data = self.client.get(key)
        return json.loads(data) if data else []
    
    # 工作记忆
    def save_task_state(self, task_id, state):
        key = f"task:{task_id}"
        self.client.set(key, json.dumps(state))
    
    def get_task_state(self, task_id):
        key = f"task:{task_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None
```

### Mem0 实现

Mem0 是专门的 Agent 记忆管理工具：

```python
from mem0 import Memory

m = Memory()

# 添加记忆
m.add("我喜欢用Python做数据分析", user_id="user1")

# 搜索记忆
results = m.search("数据分析", user_id="user1")

# 获取所有记忆
all_memories = m.get_all(user_id="user1")

# 更新记忆
m.update(memory_id="mem123", data="新的信息")

# 删除记忆
m.delete(memory_id="mem123")
```

### Letta 实现

Letta 提供完整的 Agent 记忆框架：

```python
from letta import LettaClient

client = LettaClient()

# 创建 Agent（自带记忆系统）
agent = client.create_agent(
    name="data_analyst",
    memory_config={
        "type": "hierarchical",
        "short_term": {"max_tokens": 4000},
        "long_term": {"vector_db": "pinecone"}
    }
)

# Agent 自动管理记忆
response = client.send_message(
    agent_id=agent.id,
    message="分析这份销售报告"
)
```

## 记忆系统最佳实践

### 分层设计

```
Level 1: 当前对话（LLM Context）
Level 2: 任务状态（Redis/内存）
Level 3: 会话记忆（数据库）
Level 4: 长期知识（向量数据库）
Level 5: 用户档案（关系数据库）
```

### 记忆同步

确保记忆一致性：

```python
def sync_memory_levels():
    # 任务完成时，将工作记忆转为长期记忆
    if task_completed:
        important_info = extract_important_info(task_state)
        store_to_long_term(important_info)
    
    # 定期同步用户偏好
    if session_end:
        update_user_preferences(user_id, session_data)
```

### 记忆隔离

不同用户/任务记忆隔离：

```python
# 用户隔离
def get_user_memory(user_id):
    return vector_db.search(filter={"user_id": user_id})

# 任务隔离
def get_task_memory(task_id):
    return redis.get(f"task:{task_id}")
```

## 总结

Agent 记忆系统分为三层：短期记忆（对话上下文）、工作记忆（任务状态）、长期记忆（持久化知识）。每层有不同的存储需求和处理策略。

设计记忆系统时，需要考虑：存储容量、检索效率、更新机制、遗忘策略。合理的记忆架构让 Agent 能够维持连贯对话、完成复杂任务、积累长期知识。

## 延伸阅读

- [Agent 入门指南](/2026/05/10/zh-CN/技术文档/Agent/agent-intro/)
- [工具调用机制详解](/2026/05/10/zh-CN/技术文档/Agent/tool-use/)
- [Agent 规划与推理](/2026/05/10/zh-CN/技术文档/Agent/agent-planning/)
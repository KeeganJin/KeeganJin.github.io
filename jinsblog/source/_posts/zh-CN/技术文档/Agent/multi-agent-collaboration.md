---
title: 多 Agent 协作模式
date: 2026-03-30
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, Multi-Agent, 协作模式]
---

## 协作模式分类

多 Agent 协作模式决定了 Agent 如何分工和协作。不同的协作模式适用于不同的任务类型。

### 主要协作模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| 顺序 | Agent 按顺序处理任务 | 流程式任务 |
| 并行 | Agent 同时处理不同子任务 | 独立子任务 |
| 层次 | Manager 分配任务给 Worker | 大型复杂任务 |
| 对话 | Agent 通过对话协作 | 需要讨论的任务 |

## 顺序协作模式

顺序协作中，Agent 按预定顺序处理任务，每个 Agent 处理完成后传递给下一个。

### 基本流程

```
输入 → Agent A → 结果 A → Agent B → 结果 B → Agent C → 最终输出
```

### 工作机制

1. **任务定义**：确定处理步骤顺序
2. **Agent 分配**：每步分配对应 Agent
3. **顺序执行**：Agent 按顺序依次执行
4. **结果传递**：前一个 Agent 的输出作为下一个的输入
5. **最终整合**：最后一个 Agent 输出最终结果

### 适用场景

顺序协作适用于：
- 有明确步骤顺序的任务
- 每步依赖前一步结果的任务
- 流程化处理任务

**示例**：
- 内容生产：选题 Agent → 撰写 Agent → 编辑 Agent → 发布 Agent
- 数据处理：采集 Agent → 清洗 Agent → 分析 Agent → 报告 Agent
- 软件开发：需求 Agent → 设计 Agent → 开发 Agent → 测试 Agent

### 顺序协作示例

```python
class SequentialPipeline:
    def __init__(self, agents):
        self.agents = agents  # [agent_a, agent_b, agent_c]
    
    def run(self, input):
        result = input
        for agent in self.agents:
            result = agent.process(result)
        return result

# 使用示例
pipeline = SequentialPipeline([
    ResearchAgent(),      # 步骤1：研究
    WriterAgent(),        # 步骤2：撰写
    EditorAgent()         # 步骤3：编辑
])

final_report = pipeline.run("写一篇关于AI Agent的文章")
```

### 优势与局限

| 优势 | 局限 |
|------|------|
| 流程清晰 | 无法并行，效率受限 |
| 容易调试 | 每步依赖前一步 |
| 可预测结果 | 一步失败影响全部 |

## 并行协作模式

并行协作中，多个 Agent 同时处理不同的子任务，最后汇总结果。

### 基本流程

```
                    ┌─── Agent A ──── 结果 A ───┐
                    │                           │
输入 → 任务分解 → ──┼─── Agent B ──── 结果 B ───┼── 结果汇总 → 最终输出
                    │                           │
                    └─── Agent C ──── 结果 C ───┘
```

### 工作机制

1. **任务分解**：将大任务分解为独立子任务
2. **Agent 分配**：每个子任务分配一个 Agent
3. **并行执行**：所有 Agent 同时开始执行
4. **等待完成**：等待所有 Agent 完成
5. **结果汇总**：整合所有 Agent 的结果

### 适用场景

并行协作适用于：
- 可分解为独立子任务的任务
- 需要快速完成的大任务
- 需要多视角处理的任务

**示例**：
- 多源搜索：不同 Agent 搜索不同数据源
- 多文件处理：不同 Agent 处理不同文件
- 多角度分析：不同 Agent 从不同角度分析

### 并行协作示例

```python
import asyncio

class ParallelCoordinator:
    def __init__(self, agents):
        self.agents = agents
    
    async def run(self, subtasks):
        # 分配子任务给不同 Agent
        tasks = []
        for i, subtask in enumerate(subtasks):
            agent = self.agents[i % len(self.agents)]
            tasks.append(agent.process_async(subtask))
        
        # 并行执行
        results = await asyncio.gather(*tasks)
        
        # 汇总结果
        return self.aggregate(results)
    
    def aggregate(self, results):
        return "\n".join(results)

# 使用示例
coordinator = ParallelCoordinator([
    SearchAgent(source="web"),
    SearchAgent(source="news"),
    SearchAgent(source="academic")
])

results = coordinator.run([
    "搜索技术博客",
    "搜索新闻报道",
    "搜索学术论文"
])
```

### 并行执行策略

#### 全并行

所有 Agent 同时开始：

```python
async def run_all_parallel(agents, inputs):
    return await asyncio.gather(*[
        agent.run(input) for agent, input in zip(agents, inputs)
    ])
```

#### 分组并行

Agent 分组并行执行：

```python
async def run_grouped_parallel(agent_groups, inputs):
    results = []
    for group in agent_groups:
        group_results = await asyncio.gather(*[
            agent.run(input) for agent, input in zip(group, inputs)
        ])
        results.extend(group_results)
    return results
```

### 优势与局限

| 优势 | 局限 |
|------|------|
| 处理速度快 | 需要独立子任务 |
| 利用多资源 | 结果汇总复杂 |
| 故障隔离 | 协调开销 |

## 层次协作模式

层次协作中，Manager Agent 负责规划和分配任务，Worker Agent 负责执行。

### 基本流程

```
用户请求 → Manager Agent → 任务分解 → 分配给 Workers
                                    ↓
                            Worker 1 → 执行 → 结果
                            Worker 2 → 执行 → 结果
                            Worker 3 → 执行 → 结果
                                    ↓
                            结果收集 → Manager 整合 → 最终输出
```

### 工作机制

#### Manager Agent 职责

1. **接收任务**：接收用户的原始任务
2. **理解任务**：分析任务需求和目标
3. **分解任务**：将任务分解为子任务
4. **分配任务**：将子任务分配给合适的 Worker
5. **监控进度**：跟踪 Worker 执行状态
6. **收集结果**：汇总 Worker 的执行结果
7. **整合输出**：整合结果并返回用户

#### Worker Agent 职责

1. **接收分配**：接收 Manager 分配的子任务
2. **执行任务**：执行具体子任务
3. **返回结果**：向 Manager 返回执行结果

### 适用场景

层次协作适用于：
- 大型复杂任务
- 需要集中管理的任务
- 任务结构不确定的任务

**示例**：
- 软件项目：Manager 分配需求分析、设计、开发、测试给不同 Worker
- 企业运营：Manager 分配销售、市场、运营任务给专业 Worker
- 研究项目：Manager 分配文献检索、数据分析、报告撰写给不同 Worker

### 层次协作示例

```python
class HierarchicalManager:
    def __init__(self, workers):
        self.workers = workers
    
    def run(self, task):
        # 分解任务
        subtasks = self.decompose(task)
        
        # 分配任务
        assignments = self.assign(subtasks)
        
        # 执行并收集结果
        results = {}
        for worker_id, subtask in assignments.items():
            worker = self.workers[worker_id]
            results[worker_id] = worker.execute(subtask)
        
        # 整合结果
        return self.integrate(results)
    
    def decompose(self, task):
        prompt = f"将以下任务分解为子任务：{task}"
        return llm.generate(prompt)
    
    def assign(self, subtasks):
        # 根据能力分配
        assignments = {}
        for subtask in subtasks:
            best_worker = self.find_best_worker(subtask)
            assignments[best_worker.id] = subtask
        return assignments
    
    def integrate(self, results):
        prompt = f"整合以下结果：{results}"
        return llm.generate(prompt)

class Worker:
    def __init__(self, id, capabilities):
        self.id = id
        self.capabilities = capabilities
    
    def execute(self, subtask):
        return process_subtask(subtask)
```

### 多层层次结构

可以有多层 Manager：

```
        Top Manager
             ↓
    ┌────────┼────────┐
    ↓        ↓        ↓
 Manager1 Manager2 Manager3
    ↓        ↓        ↓
 Workers   Workers   Workers
```

### 优势与局限

| 优势 | 局限 |
|------|------|
| 处理大型任务 | Manager 瓶颈风险 |
| 集中管理 | 层级延迟 |
| 专业化分工 | 信息传递损失 |

## 对话协作模式

对话协作中，Agent 通过对话交流信息、讨论方案、达成共识。

### 基本流程

```
Agent A: 我建议方案 X
    ↓
Agent B: 我同意，但建议补充 Y
    ↓
Agent A: 好建议，修改为 X+Y
    ↓
Agent C: 我验证了 X+Y，可行
    ↓
共识达成 → 执行
```

### 工作机制

1. **发起讨论**：某个 Agent 提出初始方案
2. **轮流发言**：Agent 按顺序发言讨论
3. **观点交换**：Agent 提出观点、质疑、补充
4. **达成共识**：通过讨论达成一致意见
5. **执行方案**：按共识方案执行

### 适用场景

对话协作适用于：
- 需要讨论的任务（如方案设计）
- 需要多方视角的任务（如评审）
- 不确定性高的任务（如复杂问题求解）

### 对话协作示例（AutoGen GroupChat）

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 定义 Agent
researcher = AssistantAgent(
    name="Researcher",
    system_message="你是一个研究Agent，负责收集和分析信息。"
)

writer = AssistantAgent(
    name="Writer",
    system_message="你是一个写作Agent，负责撰写内容。"
)

editor = AssistantAgent(
    name="Editor",
    system_message="你是一个编辑Agent，负责审核和修改内容。"
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER"
)

# 创建群聊
groupchat = GroupChat(
    agents=[researcher, writer, editor, user],
    messages=[]
)

manager = GroupChatManager(
    groupchat=groupchat
)

# 启动对话
user.initiate_chat(
    manager,
    message="写一篇关于AI Agent的文章，请协作完成。"
)
```

### 对话控制机制

#### 自由对话

Agent 自由发言：

```python
def free_conversation(agents, topic):
    while not reached_consensus():
        speaker = select_next_speaker(agents)
        response = speaker.generate_response(context)
        context.append(response)
    return consensus
```

#### 主持对话

有一个主持 Agent 控制：

```python
def moderated_conversation(moderator, agents, topic):
    moderator.open_discussion(topic)
    while not reached_consensus():
        next_agent = moderator.select_speaker(agents)
        response = next_agent.speak(context)
        moderator.evaluate(response)
    return moderator.conclude()
```

#### 结构化对话

按预定结构对话：

```python
def structured_conversation(agents, structure):
    for stage in structure:
        for agent in agents:
            response = agent.respond(stage, context)
            context.append(response)
    return finalize(context)
```

### 优势与局限

| 优势 | 局限 |
|------|------|
| 处理不确定性 | 对话轮次多 |
| 多视角综合 | 成本高 |
| 质量高 | 时间长 |

## Agent 间通信机制

### 通信方式

#### 直接消息

Agent 直接发送消息：

```python
class Agent:
    def send_message(self, target_agent, message):
        target_agent.receive(message)
    
    def receive(self, message):
        self.process_message(message)
```

#### 共享黑板

Agent 共享数据区域：

```python
class Blackboard:
    def __init__(self):
        self.data = {}
    
    def write(self, key, value, agent_id):
        self.data[key] = {"value": value, "writer": agent_id}
    
    def read(self, key):
        return self.data.get(key)
```

#### 消息队列

通过队列传递消息：

```python
import queue

class MessageQueue:
    def __init__(self):
        self.queues = {}  # {agent_id: Queue}
    
    def send(self, target_id, message):
        if target_id not in self.queues:
            self.queues[target_id] = queue.Queue()
        self.queues[target_id].put(message)
    
    def receive(self, agent_id):
        return self.queues[agent_id].get()
```

### 消息格式

标准化消息格式：

```python
class AgentMessage:
    def __init__(self, sender, receiver, content, msg_type):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.type = msg_type  # request/response/info/command
        self.timestamp = datetime.now()
    
    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.type,
            "timestamp": self.timestamp.isoformat()
        }
```

## 任务分配策略

### 能力匹配

根据 Agent 能力分配：

```python
def assign_by_capability(subtasks, agents):
    assignments = {}
    for subtask in subtasks:
        best_agent = max(agents, key=lambda a: capability_match(a, subtask))
        assignments[best_agent.id] = subtask
    return assignments
```

### 负载均衡

平衡 Agent 负载：

```python
def assign_by_load(subtasks, agents):
    assignments = {}
    agent_loads = {a.id: 0 for a in agents}
    
    for subtask in sorted(subtasks, key=lambda t: t.complexity, reverse=True):
        # 选择负载最低的 Agent
        best_agent = min(agents, key=lambda a: agent_loads[a.id])
        assignments[best_agent.id] = subtask
        agent_loads[best_agent.id] += subtask.complexity
    
    return assignments
```

### 动态分配

根据执行情况动态分配：

```python
def dynamic_assign(subtasks, agents, current_status):
    # 考虑当前执行状态
    pending = [t for t in subtasks if t.status == "pending"]
    available = [a for a in agents if a.status == "idle"]
    
    assignments = {}
    for subtask in pending:
        if available:
            agent = select_best_agent(subtask, available)
            assignments[agent.id] = subtask
            available.remove(agent)
    
    return assignments
```

## 结果聚合方法

### 直接拼接

简单拼接各 Agent 结果：

```python
def concatenate_results(results):
    return "\n\n".join([r.content for r in results])
```

### 结构化整合

按结构整合：

```python
def structured_integrate(results, template):
    integrated = {}
    for key in template.keys():
        integrated[key] = [r.get(key) for r in results if r.has(key)]
    return integrated
```

### LLM 整合

用 LLM 整合多 Agent 结果：

```python
def llm_integrate(results, task):
    prompt = f"""
    任务：{task}
    以下是多Agent的结果：
    {format_results(results)}
    
    请整合以上结果，生成最终输出。
    """
    return llm.generate(prompt)
```

## 冲突检测与解决

### 冲突类型

| 册突类型 | 描述 | 解决方式 |
|----------|------|----------|
| 资源冲突 | 多 Agent 竞争同一资源 | 分配/排队 |
| 结果冲突 | Agent 结果不一致 | 仲裁/投票 |
| 依赖冲突 | 执行顺序冲突 | 重排序 |
| 目标冲突 | Agent 目标不一致 | 协调/优先级 |

### 册突检测

```python
def detect_conflicts(results):
    conflicts = []
    for i, r1 in enumerate(results):
        for r2 in results[i+1:]:
            if is_conflicting(r1, r2):
                conflicts.append((r1, r2))
    return conflicts

def is_conflicting(r1, r2):
    # 检查内容是否冲突
    if r1.conclusion != r2.conclusion:
        return True
    return False
```

### 册突解决

#### 仲裁机制

```python
def arbitration(conflicts, arbitrator_agent):
    resolutions = []
    for r1, r2 in conflicts:
        decision = arbitrator_agent.arbitrate(r1, r2)
        resolutions.append(decision)
    return resolutions
```

#### 投票机制

```python
def voting(results):
    # 多数投票
    counts = {}
    for r in results:
        key = r.conclusion
        counts[key] = counts.get(key, 0) + 1
    
    winner = max(counts, key=counts.get)
    return winner
```

#### 协商机制

```python
def negotiate(agents, conflict):
    # Agent 协商解决
    context = [conflict.r1, conflict.r2]
    while not consensus:
        proposal = random.choice(agents).propose(context)
        others = [a for a in agents if a != proposer]
        votes = [a.vote(proposal) for a in others]
        if majority_approve(votes):
            consensus = proposal
    return consensus
```

## 协作协议设计

### 协议要素

好的协作协议应包含：
- **角色定义**：每个 Agent 的职责
- **消息格式**：标准化的消息结构
- **交互顺序**：Agent 的发言顺序
- **决策规则**：如何做出决策
- **异常处理**：失败时的处理方式

### 协议示例

```python
class CollaborationProtocol:
    def __init__(self):
        self.roles = {
            "leader": "负责协调和决策",
            "executor": "负责执行具体任务",
            "reviewer": "负责审核结果"
        }
        self.message_format = {
            "type": ["proposal", "response", "review", "decision"],
            "content": "str",
            "metadata": "dict"
        }
        self.interaction_order = ["leader", "executor", "reviewer", "leader"]
        self.decision_rule = "majority_vote"
```

## 总结

多 Agent 协作模式包括顺序、并行、层次、对话四种主要模式。每种模式适用于不同类型的任务。设计协作系统需要考虑：任务分配策略、结果聚合方法、冲突检测与解决、协作协议设计。

选择协作模式时，应根据任务特点（是否可并行、是否有依赖、复杂度）和系统需求（效率、成本、质量）综合考量。

## 延伸阅读

- [多 Agent 系统概述](/2026/05/10/zh-CN/技术文档/Agent/multi-agent-intro/)
- [层次化 Agent 系统](/2026/05/10/zh-CN/技术文档/Agent/hierarchical-agent/)
- [多 Agent 竞争与博弈](/2026/05/10/zh-CN/技术文档/Agent/multi-agent-competition/)
- [Agent 框架概览](/2026/05/10/zh-CN/技术文档/Agent/agent-frameworks/)
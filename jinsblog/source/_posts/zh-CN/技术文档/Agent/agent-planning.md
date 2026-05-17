---
title: Agent 规划与推理
date: 2026-03-10
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, Planning, Reasoning, CoT, ReAct]
---

## 规划的定义与必要性

规划（Planning）是 Agent 将复杂任务分解为可执行步骤的过程。没有规划能力，Agent 面对复杂任务时会不知所措或盲目行动。

### 为什么需要规划？

1. **任务复杂性**：复杂任务无法一步完成，需要分解
2. **行动有序性**：某些步骤有依赖关系，需确定顺序
3. **资源有限性**：规划可以优化资源使用
4. **结果可控性**：规划让行动有预期，便于验证

### 规划的作用

| 作用 | 描述 |
|------|------|
| 任务分解 | 将大任务拆解为小任务 |
| 步骤排序 | 确定任务执行的顺序 |
| 资源分配 | 决定每个步骤的资源投入 |
| 预期设定 | 设定每步的预期结果 |
| 失败预案 | 规划失败时的处理方式 |

## 思维链（Chain of Thought, CoT）

思维链是最基础的推理增强技术，通过引导 LLM 展示推理过程来提高复杂任务的准确性。

### CoT 原理

传统提示：
```
问题：小明有5个苹果，给了小红2个，又买了3个，现在有多少个？
回答：6个
```

CoT 提示：
```
问题：小明有5个苹果，给了小红2个，又买了3个，现在有多少个？
让我们一步步思考：
1. 小明最初有5个苹果
2. 给了小红2个后，还剩5-2=3个
3. 又买了3个，现在有3+3=6个
答案：6个
```

### CoT 的作用

- **提高准确性**：复杂推理任务准确率显著提升
- **过程透明**：展示推理过程，便于调试
- **错误定位**：容易发现推理错误的位置

### CoT 提示技巧

#### 零样本 CoT

直接要求模型展示思考过程：

```
请一步步思考并回答以下问题：
[问题内容]
```

#### 少样本 CoT

提供示例展示推理过程：

```
示例问题：一箱苹果24个，分给6个人每人几个？
示例推理：
1. 总共有24个苹果
2. 要分给6个人
3. 每人得到24/6=4个苹果
答案：每人4个

现在请用同样的方式推理以下问题：
[新问题]
```

### CoT 在 Agent 中的应用

Agent 使用 CoT 进行任务分析：

```python
def plan_with_cot(task):
    prompt = f"""
    任务：{task}
    
    请一步步分析这个任务：
    1. 理解任务目标
    2. 分析需要什么信息
    3. 确定需要哪些工具
    4. 规划执行步骤
    5. 预期每步的结果
    
    请展示你的思考过程：
    """
    
    reasoning = llm.generate(prompt)
    return parse_plan(reasoning)
```

## ReAct 模式（Reasoning + Acting）

ReAct 是一种结合推理和行动的 Agent 模式，交替进行思考和执行。

### ReAct 原理

ReAct 的核心是 **Thought → Action → Observation** 循环：

```
Thought: 我需要查询北京的天气信息
Action: search("北京天气")
Observation: 北京今天晴，气温25°C

Thought: 用户问的是今天天气，我已获得信息
Action: finish("北京今天晴天，气温25°C")
```

### ReAct 流程图

```
┌─────────────┐
│  用户任务   │
└─────────────┘
      ↓
┌─────────────┐
│ Thought     │ ← 思考下一步该做什么
│ (推理)      │
└─────────────┘
      ↓
┌─────────────┐
│ Action      │ ← 决定执行什么行动
│ (行动决策)  │
└─────────────┘
      ↓
┌─────────────┐
│ Observation │ ← 观察行动结果
│ (结果观察)  │
└─────────────┘
      ↓
   循环继续...
```

### ReAct 示例

**任务**：查找《三体》作者并获得其最新作品

```
Thought 1: 我需要先查找《三体》的作者信息
Action 1: search("三体小说作者")
Observation 1: 《三体》作者是刘慈欣

Thought 2: 现在我知道了作者是刘慈欣，需要查找他的最新作品
Action 2: search("刘慈欣最新作品")
Observation 2: 刘慈欣最新作品是《球状闪电》重写版

Thought 3: 我已获得所需信息，可以回答用户了
Action 3: finish("《三体》作者是刘慈欣，其最新作品是《球状闪电》重写版")
```

### ReAct 实现框架

```python
class ReActAgent:
    def __init__(self, tools):
        self.tools = tools
        self.history = []
    
    def run(self, task):
        while True:
            # 生成 Thought 和 Action
            response = self.generate_thought_action()
            
            # 解析 Action
            action = self.parse_action(response)
            
            if action["name"] == "finish":
                return action["output"]
            
            # 执行 Action
            observation = self.execute_action(action)
            
            # 记录历史
            self.history.append({
                "thought": response["thought"],
                "action": action,
                "observation": observation
            })
    
    def generate_thought_action(self):
        prompt = self.build_prompt()
        return llm.generate(prompt)
    
    def execute_action(self, action):
        tool = self.tools[action["name"]]
        return tool.execute(action["params"])
```

## Plan-and-Execute 模式

Plan-and-Execute 将规划与执行分离，先制定完整计划再执行。

### 基本流程

```
Plan阶段: 分析任务 → 分解步骤 → 制定计划
Execute阶段: 执行步骤 → 监控进度 → 处理异常
```

### Plan-and-Execute vs ReAct

| 特性 | ReAct | Plan-and-Execute |
|------|-------|-------------------|
| 规划时机 | 步步规划 | 先整体规划 |
| 灵活性 | 高，每步可调整 | 低，计划固定 |
| 效率 | 每步需推理 | 执行只需执行 |
| 适用场景 | 不确定任务 | 确定性任务 |

### Plan-and-Execute 流程

```
┌─────────────────────────────────────┐
│           Plan Phase                │
│  任务 → 分解 → 步骤1, 步骤2, ...    │
└─────────────────────────────────────┘
              ↓ 计划
┌─────────────────────────────────────┐
│         Execute Phase               │
│  步骤1 → 步骤2 → 步骤3 → ...        │
│     ↓      ↓      ↓                 │
│   结果1  结果2  结果3                │
└─────────────────────────────────────┘
              ↓ 最终结果
```

### Plan-and-Execute 实现

```python
class PlanExecuteAgent:
    def run(self, task):
        # Plan 阶段
        plan = self.plan(task)
        
        # Execute 阶段
        results = []
        for step in plan.steps:
            result = self.execute_step(step)
            results.append(result)
            
            # 检查是否需要重新规划
            if self.need_replan(result):
                plan = self.replan(task, results)
        
        return self.aggregate_results(results)
    
    def plan(self, task):
        prompt = f"""
        任务：{task}
        
        请制定详细计划：
        1. 分解为具体步骤
        2. 说明每步目标
        3. 指定每步工具
        4. 确定执行顺序
        """
        response = llm.generate(prompt)
        return parse_plan(response)
    
    def execute_step(self, step):
        return execute_action(step.tool, step.params)
```

### 动态重规划

执行中发现问题可以重新规划：

```python
def need_replan(self, result):
    # 检查结果是否偏离预期
    if result.status == "failed":
        return True
    if result.deviation > threshold:
        return True
    return False

def replan(self, task, completed_results):
    prompt = f"""
    原任务：{task}
    已完成：{completed_results}
    需要重新规划剩余步骤。
    """
    return llm.generate(prompt)
```

## 思维树（Tree of Thoughts, ToT）

思维树通过探索多条推理路径来解决复杂问题。

### ToT 原理

CoT 是单条推理路径，ToT 是多条路径的树形结构：

```
                问题
                 ↓
        ┌────────┼────────┐
        思路1    思路2    思路3
         ↓       ↓       ↓
      方案1a  方案2a  方案3a
      方案1b  方案2b  方案3b
         ↓       ↓       ↓
       评估    评估    评估
         ↓       ↓       ↓
       选最优方案
```

### ToT 流程

1. **生成**：生成多个可能的思考路径
2. **评估**：评估每个路径的质量
3. **搜索**：选择最优路径继续探索
4. **回溯**：如果路径失败，回溯尝试其他路径

### ToT 实现

```python
class ToTAgent:
    def __init__(self, max_depth=3, breadth=3):
        self.max_depth = max_depth
        self.breadth = breadth  # 每层探索的分支数
    
    def solve(self, problem):
        root = ThoughtNode(problem)
        return self.search(root)
    
    def search(self, node, depth=0):
        if depth >= self.max_depth:
            return self.evaluate(node)
        
        # 生成多个分支
        branches = self.generate_branches(node, self.breadth)
        
        # 评估每个分支
        evaluations = [self.evaluate(b) for b in branches]
        
        # 选择最优分支继续探索
        best_branch = self.select_best(branches, evaluations)
        
        return self.search(best_branch, depth + 1)
    
    def generate_branches(self, node, count):
        prompt = f"生成{count}种不同的解决思路..."
        thoughts = llm.generate(prompt)
        return [ThoughtNode(t, parent=node) for t in thoughts]
    
    def evaluate(self, node):
        prompt = f"评估这个思路的可行性：{node.thought}"
        return llm.generate(prompt)
```

### ToT 适用场景

- 需要创意解决方案的问题
- 多种可能答案的问题
- 单一路径容易走入歧途的问题
- 可以逐步验证的问题

## 任务分解策略

### 分解原则

1. **独立性**：子任务尽量独立，减少依赖
2. **原子性**：子任务应是单一、明确的操作
3. **可验证性**：每个子任务有明确的完成标准
4. **有序性**：有依赖的任务按正确顺序排列

### 分解方法

#### 按功能分解

```python
def decompose_by_function(task):
    prompt = f"""
    任务：{task}
    
    按功能模块分解：
    1. 数据获取模块
    2. 数据处理模块
    3. 分析模块
    4. 输出模块
    """
    return llm.generate(prompt)
```

#### 按时间分解

```python
def decompose_by_time(task):
    # 按时间顺序分解任务
    prompt = f"""
    任务：{task}
    
    按时间顺序分解：
    1. 第一步做什么
    2. 第二步做什么
    ...
    """
    return llm.generate(prompt)
```

#### 按层级分解

```python
def decompose_hierarchically(task):
    # 先分解大任务
    main_steps = get_main_steps(task)
    
    # 每个大步骤继续分解
    detailed_plan = []
    for step in main_steps:
        sub_steps = decompose_step(step)
        detailed_plan.extend(sub_steps)
    
    return detailed_plan
```

### 任务依赖管理

```python
class TaskDAG:
    def __init__(self):
        self.tasks = {}
        self.dependencies = {}
    
    def add_task(self, task_id, task_info):
        self.tasks[task_id] = task_info
        self.dependencies[task_id] = []
    
    def add_dependency(self, task_id, depends_on):
        self.dependencies[task_id].append(depends_on)
    
    def get_execution_order(self):
        # 拓扑排序
        return topological_sort(self.tasks, self.dependencies)
```

## 规划与执行的反馈循环

### 反馈机制

执行结果需要反馈给规划模块：

```python
def feedback_loop(plan, execution_result):
    if execution_result.success:
        # 成功，继续下一步骤
        return "continue"
    else:
        # 失败，需要调整
        if execution_result.error_type == "tool_error":
            # 工具失败，尝试替代工具
            return "retry_with_alternative"
        elif execution_result.error_type == "unexpected_result":
            # 结果不符预期，重新规划
            return "replan"
        else:
            # 其他错误
            return "handle_error"
```

### 自我修正

Agent 根据反馈自我修正：

```python
def self_correction(current_state, error):
    prompt = f"""
    当前状态：{current_state}
    遇到错误：{error}
    
    请分析错误原因并提出修正方案：
    1. 错误原因是什么？
    2. 如何修正？
    3. 是否需要调整后续计划？
    """
    correction = llm.generate(prompt)
    return apply_correction(correction)
```

### 进度监控

监控任务执行进度：

```python
class ProgressMonitor:
    def __init__(self, plan):
        self.plan = plan
        self.completed = []
        self.current = None
        self.failed = []
    
    def update(self, step_id, status, result):
        if status == "completed":
            self.completed.append(step_id)
        elif status == "failed":
            self.failed.append(step_id)
        
        # 计算进度
        progress = len(self.completed) / len(self.plan.steps)
        return {
            "progress": progress,
            "current_step": self.current,
            "completed": self.completed,
            "failed": self.failed
        }
```

## 规划失败的恢复机制

### 失败类型

| 失败类型 | 原因 | 处理方式 |
|----------|------|----------|
| 工具执行失败 | 工具返回错误 | 重试或替代工具 |
| 参数错误 | 参数生成错误 | 重新生成参数 |
| 结果不符预期 | 执行结果非预期 | 重新规划 |
| 步骤依赖失败 | 前置步骤失败 | 回溯或跳过 |
| 超时 | 执行时间过长 | 中断并重新规划 |

### 恢复策略

#### 重试

```python
def retry_with_backoff(action, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = execute(action)
            if result.success:
                return result
        except Exception as e:
            wait_time = 2 ** attempt
            sleep(wait_time)
    return Failure("重试失败")
```

#### 替代方案

```python
def try_alternative(failed_action):
    alternatives = get_alternative_tools(failed_action.tool)
    for alt_tool in alternatives:
        result = execute_with_alternative(alt_tool, failed_action.params)
        if result.success:
            return result
    return Failure("所有替代方案失败")
```

#### 回溯

```python
def backtrack_to_safe_point(current_state):
    # 找到最后一个成功点
    safe_point = find_last_success(current_state.history)
    # 重置到安全点
    reset_to(safe_point)
    # 重新规划
    return replan_from(safe_point)
```

#### 任务降级

```python
def degrade_task(original_task, failed_part):
    # 如果部分失败，尝试完成剩余部分
    remaining = get_remaining_subtasks(original_task, failed_part)
    if can_complete_partial(remaining):
        return execute_partial(remaining)
    return Failure("无法完成任何部分")
```

## 规划框架对比

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| CoT | 简单、通用 | 单步或简单多步任务 |
| ReAct | 灵活、交互式 | 需要多次工具调用的任务 |
| Plan-Execute | 结构化、高效 | 确定性、可分解的任务 |
| ToT | 探索式、创意 | 需要多种方案的复杂问题 |

## 总结

规划是 Agent 的核心能力之一。不同的规划模式适用于不同场景：CoT 适合简单推理，ReAct 适合需要工具调用的交互式任务，Plan-and-Execute 适合结构化任务，ToT 适合需要探索多种方案的复杂问题。

好的规划系统需要：清晰的任务分解、合理的步骤排序、有效的反馈机制、完善的失败恢复。

## 延伸阅读

- [Agent 入门指南](/2026/05/10/zh-CN/技术文档/Agent/agent-intro/)
- [工具调用机制详解](/2026/05/10/zh-CN/技术文档/Agent/tool-use/)
- [Agent 记忆系统](/2026/05/10/zh-CN/技术文档/Agent/agent-memory/)
- [多 Agent 系统概述](/2026/05/10/zh-CN/技术文档/Agent/multi-agent-intro/)
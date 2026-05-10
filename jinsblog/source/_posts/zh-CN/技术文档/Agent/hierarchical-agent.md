---
title: 层次化 Agent 系统
date: 2026-05-10
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, Multi-Agent, 层次化架构]
---

## 层次化架构的设计理念

层次化 Agent 架构使用层级组织 Agent，上层 Manager Agent 负责决策和分配，下层 Worker Agent 负责执行。这种架构模仿了人类组织的层级管理模式。

### 为什么使用层次化架构？

单 Agent 面对复杂任务时存在局限：
- 无法同时处理大量子任务
- 无法维护复杂的任务状态
- 无法动态调整执行策略

多 Agent 对等协作也存在问题：
- 缺乏统一决策点
- 协调开销大
- 册突解决困难

层次化架构解决了这些问题：
- **集中决策**：Manager 统一决策
- **分工执行**：Worker 专注执行
- **层级管理**：降低协调复杂度

### 层次化架构优势

| 优势 | 描述 |
|------|------|
| 结构清晰 | 职责分工明确 |
| 集中管理 | 统一决策和监控 |
| 专业分工 | Worker 可专业化 |
| 可扩展 | 添加 Worker 扩展能力 |
| 容错性 | Worker 失败可由 Manager 处理 |

## Manager Agent 与 Worker Agent

### Manager Agent 职责

Manager Agent 是层级系统的核心：

#### 任务理解与分解

```python
class ManagerAgent:
    def understand_task(self, user_request):
        # 解析用户意图
        intent = self.parse_intent(user_request)
        
        # 确定任务目标
        goals = self.define_goals(intent)
        
        # 分解为子任务
        subtasks = self.decompose(goals)
        
        return subtasks
```

#### Worker 选择与分配

```python
class ManagerAgent:
    def assign_workers(self, subtasks):
        assignments = []
        for subtask in subtasks:
            # 根据能力选择 Worker
            best_worker = self.select_worker(subtask)
            
            # 创建分配
            assignment = {
                "worker_id": best_worker.id,
                "subtask": subtask,
                "deadline": subtask.deadline
            }
            assignments.append(assignment)
        
        return assignments
```

#### 进度监控

```python
class ManagerAgent:
    def monitor_progress(self):
        # 检查所有 Worker 状态
        statuses = {}
        for worker in self.workers:
            status = worker.get_status()
            statuses[worker.id] = status
        
        # 分析进度
        progress = self.analyze_progress(statuses)
        
        # 处理异常
        if progress.has_issues():
            self.handle_issues(progress.issues)
        
        return progress
```

#### 结果整合

```python
class ManagerAgent:
    def integrate_results(self, worker_results):
        # 收集所有结果
        results = [r for r in worker_results.values()]
        
        # 整合结果
        integrated = self.merge_results(results)
        
        # 生成最终输出
        final_output = self.generate_output(integrated)
        
        return final_output
```

### Worker Agent 职责

Worker Agent 负责具体执行：

#### 任务接收

```python
class WorkerAgent:
    def receive_assignment(self, assignment):
        # 解析分配信息
        self.current_task = assignment["subtask"]
        self.deadline = assignment["deadline"]
        
        # 准备执行
        self.prepare_resources()
        
        # 开始执行
        return self.execute()
```

#### 任务执行

```python
class WorkerAgent:
    def execute(self):
        # 执行具体任务
        result = self.perform_task(self.current_task)
        
        # 验证结果
        if self.validate_result(result):
            return Success(result)
        else:
            return Failure("验证失败")
```

#### 结果上报

```python
class WorkerAgent:
    def report_result(self, result):
        # 上报结果给 Manager
        report = {
            "worker_id": self.id,
            "task_id": self.current_task.id,
            "status": result.status,
            "output": result.output,
            "errors": result.errors
        }
        
        self.manager.receive_report(report)
```

#### 状态报告

```python
class WorkerAgent:
    def get_status(self):
        return {
            "id": self.id,
            "state": self.state,  # idle/working/finished/error
            "current_task": self.current_task,
            "progress": self.progress_percentage,
            "estimated_completion": self.estimate_completion()
        }
```

## 任务下发与结果上报

### 任务下发流程

```
Manager                           Worker
   │                                │
   │  ──── 任务分配请求 ────→       │
   │                                │ 接收并准备
   │                                │
   │  ←─── 确认接收 ──────          │
   │                                │
   │                                │ 执行任务
   │                                │
```

### 任务下发实现

```python
class TaskDispatch:
    def __init__(self, manager, workers):
        self.manager = manager
        self.workers = workers
    
    def dispatch(self, subtasks):
        dispatch_results = []
        
        for subtask in subtasks:
            # 选择 Worker
            worker = self.select_available_worker()
            
            # 下发任务
            assignment = self.create_assignment(subtask, worker)
            
            # 发送给 Worker
            response = worker.receive(assignment)
            
            dispatch_results.append({
                "subtask_id": subtask.id,
                "worker_id": worker.id,
                "status": response.status
            })
        
        return dispatch_results
    
    def create_assignment(self, subtask, worker):
        return {
            "id": generate_assignment_id(),
            "subtask": subtask,
            "worker": worker.id,
            "created_at": datetime.now(),
            "deadline": subtask.deadline,
            "instructions": subtask.instructions
        }
```

### 结果上报流程

```
Worker                           Manager
   │                                │
   │  ──── 开始执行 ────            │
   │                                │
   │                                │ 监控状态
   │                                │
   │  ──── 进度更新 ─────→         │
   │                                │
   │                                │
   │  ──── 完成报告 ─────→         │
   │                                │ 整合结果
   │                                │
```

### 结果上报实现

```python
class ResultReporter:
    def __init__(self, worker, manager):
        self.worker = worker
        self.manager = manager
    
    def report_progress(self, progress):
        update = {
            "worker_id": self.worker.id,
            "task_id": self.worker.current_task.id,
            "progress": progress,
            "timestamp": datetime.now()
        }
        self.manager.receive_progress(update)
    
    def report_completion(self, result):
        report = {
            "worker_id": self.worker.id,
            "task_id": self.worker.current_task.id,
            "status": "completed",
            "result": result,
            "execution_time": self.worker.execution_time,
            "timestamp": datetime.now()
        }
        self.manager.receive_completion(report)
    
    def report_error(self, error):
        report = {
            "worker_id": self.worker.id,
            "task_id": self.worker.current_task.id,
            "status": "error",
            "error_type": error.type,
            "error_message": error.message,
            "timestamp": datetime.now()
        }
        self.manager.receive_error(report)
```

## 层次深度选择

### 层次深度的影响

层次深度（层级数量）影响系统性能：

| 深度 | 优势 | 局限 |
|------|------|------|
| 1层（单Agent） | 简单直接 | 能力受限 |
| 2层（Manager+Worker） | 分工清晰 | Manager 负担大 |
| 3层（Manager+SubManager+Worker） | 更细分工 | 信息传递损失 |
| 多层 | 处理超复杂任务 | 响应慢、复杂度高 |

### 深度选择原则

#### 任务复杂度

根据任务复杂度选择：

```python
def select_depth(task):
    complexity = calculate_complexity(task)
    
    if complexity < 10:
        return 1  # 单 Agent
    elif complexity < 50:
        return 2  # Manager + Worker
    elif complexity < 200:
        return 3  # 多层
    else:
        return calculate_optimal_depth(complexity)
```

#### 任务分解粒度

根据子任务粒度选择：

```python
def depth_by_granularity(task):
    subtasks = decompose(task)
    
    if len(subtasks) <= 5:
        return 2  # 直接分配给 Worker
    
    # 分组处理
    groups = group_subtasks(subtasks)
    return len(groups) + 1  # 每组一个 SubManager
```

#### 响应时间要求

根据响应时间要求选择：

```python
def depth_by_time_requirement(task):
    if task.max_response_time < 30:  # 30秒内
        return 1  # 单 Agent，最快
    
    if task.max_response_time < 120:  # 2分钟内
        return 2
    
    return 3  # 允许多层处理
```

### 深度配置示例

```python
# 2层架构：Manager + Workers
class TwoLayerArchitecture:
    def __init__(self, num_workers=5):
        self.manager = ManagerAgent()
        self.workers = [WorkerAgent(id=i) for i in range(num_workers)]
        self.manager.set_workers(self.workers)

# 3层架构：Manager + SubManagers + Workers
class ThreeLayerArchitecture:
    def __init__(self, num_submanagers=3, workers_per_manager=3):
        self.top_manager = TopManagerAgent()
        
        self.sub_managers = []
        for i in range(num_submanagers):
            sub_manager = SubManagerAgent(id=i)
            workers = [WorkerAgent(id=j) for j in range(workers_per_manager)]
            sub_manager.set_workers(workers)
            self.sub_managers.append(sub_manager)
        
        self.top_manager.set_sub_managers(self.sub_managers)
```

## 案例分析：复杂任务的层次分解

### 案例：撰写研究报告

**任务**：撰写一份关于 AI Agent 技术趋势的研究报告

#### 层次分解方案

```
                    Top Manager
                    (整体规划、质量把控)
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    Sub Manager1    Sub Manager2    Sub Manager3
    (研究管理)      (写作管理)      (审核管理)
         │               │               │
    ┌────┼────┐    ┌────┼────┐    ┌────┼────┐
    │    │    │    │    │    │    │    │    │
   W1   W2   W3   W4   W5   W6   W7   W8   W9
  文献  数据  技术  撰写  撰写  撰写  审核  格式  发布
  检索  收集  分析  概要  正文  结论  内容  整理  准备
```

#### 执行流程

```python
# Top Manager 接收任务
task = "撰写AI Agent技术趋势研究报告"

# 分解为三个阶段
phase1 = {"name": "研究阶段", "manager": sub_manager1}
phase2 = {"name": "写作阶段", "manager": sub_manager2}
phase3 = {"name": "审核阶段", "manager": sub_manager3}

# Sub Manager 1 执行研究阶段
# 分配给 Worker1-3
research_results = sub_manager1.coordinate([
    Worker1.collect_literature("AI Agent"),
    Worker2.collect_data("Agent market"),
    Worker3.analyze_technology("Agent frameworks")
])

# Sub Manager 2 执行写作阶段
# 分配给 Worker4-6
writing_results = sub_manager2.coordinate([
    Worker4.write_outline(research_results),
    Worker5.write_body(research_results),
    Worker6.write_conclusion(research_results)
])

# Sub Manager 3 执行审核阶段
# 分配给 Worker7-9
review_results = sub_manager3.coordinate([
    Worker7.review_content(writing_results),
    Worker8.format_document(writing_results),
    Worker9.prepare_publish(writing_results)
])

# Top Manager 整合最终结果
final_report = top_manager.integrate([
    research_results,
    writing_results,
    review_results
])
```

### 案例：软件开发项目

**任务**：开发一个 Agent 监控系统

#### 层次分解方案

```
                    Project Manager
                    (项目管理、需求把控)
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    Design Manager   Dev Manager    QA Manager
    (设计管理)       (开发管理)      (测试管理)
         │               │               │
    ┌────┼────┐    ┌────┼────┐    ┌────┼────┐
    │    │    │    │    │    │    │    │    │
  UI   API   DB   前端  后端  集成  单测  集测  验收
  设计  设计  设计  开发  开发  开发  编写  编写  测试
```

#### 任务下发示例

```python
# Project Manager 分解项目
project = decompose_project("Agent监控系统")

# 下发给 Design Manager
design_task = project.get_phase("design")
design_manager.receive(design_task)

# Design Manager 分配给 Worker
design_manager.assign([
    {"worker": ui_designer, "task": "设计监控界面"},
    {"worker": api_designer, "task": "设计数据API"},
    {"worker": db_designer, "task": "设计数据模型"}
])

# 收集设计结果
design_results = design_manager.collect_results()

# 下发给 Dev Manager
dev_task = project.get_phase("development")
dev_task.add_input(design_results)  # 设计作为输入
dev_manager.receive(dev_task)

# Dev Manager 分配给 Worker
dev_manager.assign([
    {"worker": frontend_dev, "task": "前端开发", "input": design_results.ui},
    {"worker": backend_dev, "task": "后端开发", "input": design_results.api},
    {"worker": integration_dev, "task": "系统集成", "input": design_results.all}
])

# 类似地处理 QA Manager...
```

## 层次化架构实现框架

### CrewAI Hierarchical 流程

```python
from crewai import Agent, Task, Crew, Process

# 定义 Manager Agent
manager = Agent(
    role="Project Manager",
    goal="协调团队完成项目",
    backstory="经验丰富的项目经理",
    allow_delegation=True  # 允委派任务
)

# 定义 Worker Agents
researcher = Agent(
    role="Researcher",
    goal="研究并收集信息",
    backstory="专业研究员",
    allow_delegation=False
)

writer = Agent(
    role="Writer",
    goal="撰写内容",
    backstory="专业作家",
    allow_delegation=False
)

# 定义任务
task = Task(
    description="撰写AI Agent技术报告",
    agent=manager  # Manager 负责此任务
)

# 创建 Crew，使用 Hierarchical 流程
crew = Crew(
    agents=[manager, researcher, writer],
    tasks=[task],
    process=Process.hierarchical  # 层次化流程
)

# 执行
result = crew.kickoff()
```

### 自定义层次化框架

```python
class HierarchicalAgentSystem:
    def __init__(self, config):
        self.top_manager = self.create_manager(config["top"])
        self.sub_managers = []
        self.workers = []
        
        # 创建层次结构
        for sub_config in config["sub_managers"]:
            sub_manager = self.create_manager(sub_config)
            workers = [
                self.create_worker(w_config) 
                for w_config in sub_config["workers"]
            ]
            sub_manager.add_workers(workers)
            self.sub_managers.append(sub_manager)
        
        self.top_manager.add_sub_managers(self.sub_managers)
    
    def execute(self, task):
        # Top Manager 接收任务
        plan = self.top_manager.plan(task)
        
        # 分配给 Sub Managers
        results = {}
        for sub_manager, subtask in zip(self.sub_managers, plan.subtasks):
            results[sub_manager.id] = sub_manager.execute(subtask)
        
        # 整合结果
        return self.top_manager.integrate(results)
```

## 层次化架构的优化

### Manager 负载均衡

避免 Manager 过载：

```python
class ManagerLoadBalancer:
    def __init__(self, managers):
        self.managers = managers
        self.load_tracking = {m.id: 0 for m in managers}
    
    def assign_task(self, task):
        # 选择负载最低的 Manager
        min_load_manager = min(
            self.managers,
            key=lambda m: self.load_tracking[m.id]
        )
        
        self.load_tracking[min_load_manager.id] += task.complexity
        return min_load_manager
```

### 信息传递优化

减少层级间的信息传递损失：

```python
class OptimizedCommunication:
    def pass_down(self, manager, workers, task):
        # 精简信息传递
        essential_info = {
            "goal": task.goal,
            "constraints": task.constraints,
            "resources": task.available_resources
        }
        
        # 批量传递而非逐个
        manager.broadcast(essential_info, workers)
    
    def pass_up(self, worker, manager, result):
        # 结果摘要而非完整结果
        summary = {
            "status": result.status,
            "key_outputs": result.key_outputs,
            "issues": result.issues
        }
        
        worker.send_summary(summary, manager)
```

### 动态层次调整

根据任务动态调整层次：

```python
class DynamicHierarchy:
    def adjust_hierarchy(self, task, current_state):
        # 检查是否需要调整
        if current_state.manager_overloaded:
            # 增加层级
            self.add_sub_manager()
        
        if current_state.task_simple:
            # 减少层级
            self.flatten_hierarchy()
        
        if current_state.workers_idle:
            # 重新分配
            self.rebalance_workers()
```

## 层次化架构的挑战与解决

### Manager 成为瓶颈

**问题**：Manager 处理太多任务成为瓶颈

**解决方案**：
- 增加 Sub Manager 层级
- 负载均衡分配
- Manager 只做决策，不做执行

### 信息传递延迟

**问题**：多层传递导致延迟

**解决方案**：
- 精简传递信息
- 使用异步通信
- 缓存常用信息

### Worker 间缺乏协调

**问题**：Worker 不知道其他 Worker 的进展

**解决方案**：
- Manager 定期同步状态
- 共享黑板机制
- Worker 间直接通信（需要时）

## 总结

层次化 Agent 架构通过 Manager-Worker 分层组织，解决复杂任务的协调问题。Manager 负责决策、分配、监控、整合；Worker 负责具体执行。层次深度需要根据任务复杂度、粒度、响应时间要求选择。

层次化架构适用于大型复杂任务、企业级应用、项目管理等场景。设计时需注意 Manager 瓶颈、信息传递损失、Worker 协调等问题。

## 延伸阅读

- [Agent 架构模式](/2026/05/10/zh-CN/技术文档/Agent/agent-architecture/)
- [多 Agent 系统概述](/2026/05/10/zh-CN/技术文档/Agent/multi-agent-intro/)
- [多 Agent 协作模式](/2026/05/10/zh-CN/技术文档/Agent/multi-agent-collaboration/)
- [CrewAI Agent 实践](/2026/05/10/zh-CN/技术文档/Agent/crewai-agent/)
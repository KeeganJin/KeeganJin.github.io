---
title: Agent 评测方法
date: 2026-05-10
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, 评测, AgentBench]
---

## Agent 评测的挑战

Agent 评测比传统 AI 评测更复杂：

### 评测复杂性

| 挑战 | 描述 |
|------|------|
| 任务多样性 | Agent 处理的任务类型多样 |
| 执行过程 | Agent 有多步执行过程 |
| 工具依赖 | Agent 结果依赖工具执行 |
| 交互复杂性 | Agent 有复杂的人机交互 |
| 成本考量 | Agent 评测消耗资源多 |
| 标准缺失 | 缺乏统一的评测标准 |

### 评测维度

Agent 评测需要多维度评估：

```
评测维度:
├── 任务完成度
│   ├── 成功率
│   ├── 完成质量
│   └── 部分完成
├── 执行效率
│   ├── 步骤数量
│   ├── 时间消耗
│   ├── 资源消耗
├── 成本效益
│   ├── API调用次数
│   ├── Token消耗
│   ├── 工具调用成本
├── 安全性
│   ├── 指令遵循
│   ├── 工具安全
│   ├── 内容安全
└── 用户体验
    ├── 响应时间
    ├── 交互质量
    ├── 用户满意度
```

## 任务成功率评测

### 任务完成定义

任务成功与否的判断标准：

| 任务类型 | 成功标准 |
|----------|----------|
| 问答任务 | 回答正确、完整 |
| 操作任务 | 操作成功执行 |
| 生成任务 | 生成内容符合要求 |
| 分析任务 | 分析结果正确 |

### 成功率计算

```python
class TaskSuccessEvaluator:
    def __init__(self):
        self.evaluation_criteria = {}
    
    def evaluate_task(self, task, agent_output, ground_truth):
        """评估任务成功与否"""
        
        result = {
            "task_id": task.id,
            "task_type": task.type,
            "success": False,
            "score": 0,
            "details": {}
        }
        
        if task.type == "qa":
            # 问答任务评估
            result["success"] = self.evaluate_qa(agent_output, ground_truth)
            result["score"] = calculate_similarity(agent_output, ground_truth)
        
        elif task.type == "operation":
            # 操作任务评估
            result["success"] = self.evaluate_operation(agent_output)
            result["details"] = check_operation_result(agent_output)
        
        elif task.type == "generation":
            # 生成任务评估
            result["success"] = self.evaluate_generation(agent_output, task.requirements)
            result["score"] = self.quality_score(agent_output)
        
        return result
    
    def calculate_success_rate(self, results):
        """计算成功率"""
        successes = [r for r in results if r["success"]]
        return len(successes) / len(results)
```

### 任务完成质量评分

```python
def evaluate_task_quality(output, requirements):
    """评估任务完成质量"""
    
    scores = {}
    
    # 准确性
    scores["accuracy"] = evaluate_accuracy(output, requirements["expected"])
    
    # 完整性
    scores["completeness"] = evaluate_completeness(output, requirements["scope"])
    
    # 格式符合度
    scores["format"] = evaluate_format(output, requirements["format"])
    
    # 时效性
    scores["timeliness"] = evaluate_timeliness(output, requirements["deadline"])
    
    # 综合得分
    scores["overall"] = weighted_average(scores, weights={
        "accuracy": 0.4,
        "completeness": 0.3,
        "format": 0.2,
        "timeliness": 0.1
    })
    
    return scores
```

## 步骤效率评测

### 步骤数量评测

```python
def evaluate_step_efficiency(agent_trace):
    """评估步骤效率"""
    
    metrics = {
        "total_steps": len(agent_trace["steps"]),
        "tool_calls": count_tool_calls(agent_trace),
        "retries": count_retries(agent_trace),
        "unnecessary_steps": count_unnecessary(agent_trace)
    }
    
    # 计算效率指标
    optimal_steps = estimate_optimal_steps(agent_trace["task"])
    
    metrics["efficiency_ratio"] = optimal_steps / metrics["total_steps"]
    
    # 步骤有效性
    effective_steps = count_effective_steps(agent_trace)
    metrics["effectiveness"] = effective_steps / metrics["total_steps"]
    
    return metrics
```

### 时间效率评测

```python
def evaluate_time_efficiency(agent_trace):
    """评估时间效率"""
    
    metrics = {
        "total_time": agent_trace["total_time"],
        "llm_time": sum_llm_time(agent_trace),
        "tool_time": sum_tool_time(agent_trace),
        "idle_time": calculate_idle_time(agent_trace)
    }
    
    # 时间分布
    metrics["time_distribution"] = {
        "llm_percentage": metrics["llm_time"] / metrics["total_time"],
        "tool_percentage": metrics["tool_time"] / metrics["total_time"],
        "idle_percentage": metrics["idle_time"] / metrics["total_time"]
    }
    
    return metrics
```

### 效率基准

```python
# 定义效率基准
efficiency_benchmarks = {
    "simple_qa": {
        "optimal_steps": 1,
        "max_time": 5,  # 秒
        "max_tokens": 500
    },
    "multi_step_task": {
        "optimal_steps": 5,
        "max_time": 60,
        "max_tokens": 2000
    },
    "complex_analysis": {
        "optimal_steps": 10,
        "max_time": 300,
        "max_tokens": 5000
    }
}

def compare_to_benchmark(metrics, benchmark):
    """与基准比较"""
    comparison = {}
    
    for key, value in metrics.items():
        if key in benchmark:
            comparison[key] = {
                "actual": value,
                "benchmark": benchmark[key],
                "ratio": value / benchmark[key]
            }
    
    return comparison
```

## 成本效益评测

### API 调用成本

```python
def evaluate_api_cost(agent_trace):
    """评估 API 调用成本"""
    
    metrics = {
        "total_calls": count_api_calls(agent_trace),
        "llm_calls": count_llm_calls(agent_trace),
        "tool_calls": count_tool_calls(agent_trace),
        "total_tokens": calculate_total_tokens(agent_trace)
    }
    
    # 计算成本
    metrics["estimated_cost"] = calculate_cost(metrics)
    
    return metrics

def calculate_cost(metrics):
    """计算成本"""
    # LLM 调用成本
    llm_cost = metrics["llm_calls"] * PRICE_PER_LLM_CALL
    
    # Token 成本
    token_cost = metrics["total_tokens"] * PRICE_PER_TOKEN
    
    # 工具成本
    tool_cost = metrics["tool_calls"] * PRICE_PER_TOOL_CALL
    
    return llm_cost + token_cost + tool_cost
```

### 成本效益分析

```python
def cost_benefit_analysis(task_result, cost_metrics):
    """成本效益分析"""
    
    # 任务价值（可以根据任务类型设定）
    task_value = estimate_task_value(task_result["task"])
    
    # 成本
    cost = cost_metrics["estimated_cost"]
    
    # 效益比
    benefit_ratio = task_value / cost
    
    return {
        "task_value": task_value,
        "cost": cost,
        "benefit_ratio": benefit_ratio,
        "is_cost_effective": benefit_ratio > MIN_BENEFIT_RATIO
    }
```

## 主流评测框架

### AgentBench

AgentBench 是一个多维度 Agent 评测框架。

#### AgentBench 评测维度

| 维度 | 描述 |
|------|------|
| Operating System | 操作系统交互任务 |
| Database | 数据库操作任务 |
| Knowledge Graph | 知识图谱任务 |
| Web Browsing | 网页浏览任务 |
| Web Shopping | 网购任务 |
| Tool Use | 工具使用任务 |

#### AgentBench 使用示例

```python
# AgentBench 评测示例
from agentbench import AgentBenchEvaluator

evaluator = AgentBenchEvaluator()

# 运行评测
results = evaluator.evaluate(
    agent=my_agent,
    tasks=["os_task", "db_task", "web_task"],
    metrics=["success_rate", "step_count", "time"]
)

# 输出报告
print(results.summary())
```

### WebShop

WebShop 是一个网购任务评测环境。

#### WebShop 任务类型

- 搜索商品
- 查看商品详情
- 添加购物车
- 完成购买

#### WebShop 评测指标

```python
# WebShop 评测指标
webshop_metrics = {
    "task_success": "是否成功完成购买",
    "price_accuracy": "是否购买正确价格商品",
    "step_efficiency": "完成任务的步骤效率",
    "time_efficiency": "完成任务的时间效率"
}
```

### ToolBench

ToolBench 评测 Agent 的工具使用能力。

```python
# ToolBench 评测
toolbench_tasks = [
    "使用搜索工具查找信息",
    "使用计算工具进行计算",
    "使用文件工具读写文件",
    "组合使用多种工具"
]

def evaluate_tool_usage(agent, tasks):
    """评测工具使用能力"""
    results = []
    
    for task in tasks:
        trace = agent.execute(task)
        
        result = {
            "task": task,
            "tools_used": extract_tools_used(trace),
            "tool_selection_accuracy": evaluate_tool_selection(trace),
            "tool_call_success": evaluate_tool_calls(trace),
            "tool_output_handling": evaluate_output_handling(trace)
        }
        
        results.append(result)
    
    return results
```

### 其他评测框架

| 框架 | 描述 |
|------|------|
| AgentEval | 通用 Agent 评测 |
| AgentInstruct | Agent 教学能力评测 |
| AgentSafety | Agent 安全评测 |
| AgentAttack | Agent 抗攻击评测 |

## 自定义评测设计

### 设计评测任务

```python
def design_evaluation_tasks(domain):
    """设计领域评测任务"""
    
    tasks = []
    
    # 基础能力任务
    tasks.extend(generate_basic_tasks(domain))
    
    # 中等难度任务
    tasks.extend(generate_medium_tasks(domain))
    
    # 高难度任务
    tasks.extend(generate_hard_tasks(domain))
    
    # 边缘案例任务
    tasks.extend(generate_edge_cases(domain))
    
    return tasks

def generate_basic_tasks(domain):
    """生成基础任务"""
    if domain == "coding":
        return [
            "写一个简单的函数",
            "修复一个语法错误",
            "解释一段代码"
        ]
    elif domain == "research":
        return [
            "搜索一个概念的定义",
            "总结一篇文章",
            "查找一个事实"
        ]
```

### 设计评测指标

```python
class CustomEvaluator:
    def __init__(self, domain, custom_metrics):
        self.domain = domain
        self.metrics = custom_metrics
    
    def evaluate(self, agent, task):
        """自定义评测"""
        results = {}
        
        # 执行任务
        trace = agent.execute(task)
        
        # 评估各指标
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(trace)
        
        return results

# 定义自定义指标
custom_metrics = {
    "success": lambda trace: trace["success"],
    "steps": lambda trace: len(trace["steps"]),
    "accuracy": lambda trace: calculate_accuracy(trace),
    "efficiency": lambda trace: calculate_efficiency(trace),
    "safety": lambda trace: check_safety(trace)
}

evaluator = CustomEvaluator("coding", custom_metrics)
```

### 评测结果分析

```python
def analyze_evaluation_results(results):
    """分析评测结果"""
    
    analysis = {
        "overall": {
            "success_rate": calculate_success_rate(results),
            "avg_steps": calculate_avg_steps(results),
            "avg_time": calculate_avg_time(results),
            "avg_cost": calculate_avg_cost(results)
        },
        "by_difficulty": {
            "easy": analyze_easy_tasks(results),
            "medium": analyze_medium_tasks(results),
            "hard": analyze_hard_tasks(results)
        },
        "by_type": {
            "qa": analyze_qa_tasks(results),
            "operation": analyze_operation_tasks(results),
            "generation": analyze_generation_tasks(results)
        },
        "issues": identify_common_issues(results),
        "recommendations": generate_recommendations(results)
    }
    
    return analysis

def identify_common_issues(results):
    """识别常见问题"""
    issues = []
    
    # 失败原因分析
    failures = [r for r in results if not r["success"]]
    failure_patterns = analyze_failure_patterns(failures)
    issues.extend(failure_patterns)
    
    # 效率问题
    inefficient = [r for r in results if r["efficiency"] < THRESHOLD]
    efficiency_issues = analyze_efficiency_issues(inefficient)
    issues.extend(efficiency_issues)
    
    return issues
```

## 评测报告生成

### 报告结构

```python
def generate_evaluation_report(results, analysis):
    """生成评测报告"""
    
    report = f"""
# Agent 评测报告

## 评测概述

- 评测任务数: {len(results)}
- 评测时间: {datetime.now()}
- Agent版本: {agent_version}

## 整体表现

- 成功率: {analysis["overall"]["success_rate"]:.2%}
- 平均步骤: {analysis["overall"]["avg_steps"]}
- 平均时间: {analysis["overall"]["avg_time"]}秒
- 平均成本: ${analysis["overall"]["avg_cost"]}

## 按难度分析

### 简单任务
- 成功率: {analysis["by_difficulty"]["easy"]["success_rate"]}

### 中等任务
- 成功率: {analysis["by_difficulty"]["medium"]["success_rate"]}

### 困难任务
- 成功率: {analysis["by_difficulty"]["hard"]["success_rate"]}

## 常见问题

{format_issues(analysis["issues"])}

## 改进建议

{format_recommendations(analysis["recommendations"])}

## 详细结果

{format_detailed_results(results)}
"""
    
    return report
```

## Agent 评测最佳实践

### 评测设计

1. **覆盖全面**：覆盖多种任务类型和难度
2. **基准对比**：与基准和同类 Agent 对比
3. **真实场景**：使用真实用户场景任务
4. **边缘测试**：包含边缘和异常情况

### 评测执行

1. **多次运行**：多次运行减少随机性
2. **环境隔离**：隔离评测环境避免干扰
3. **成本控制**：控制评测资源消耗
4. **日志记录**：记录完整执行日志

### 结果分析

1. **多维度分析**：从多个维度分析结果
2. **问题识别**：识别常见失败原因
3. **对比分析**：与基准对比找出差距
4. **改进建议**：提出具体改进建议

## 总结

Agent 评测需要多维度评估：任务成功率、步骤效率、成本效益、安全性、用户体验。主流评测框架包括 AgentBench、WebShop、ToolBench 等。

设计自定义评测需要：设计多样化任务、定义评测指标、分析评测结果、生成改进建议。评测最佳实践包括：覆盖全面、基准对比、真实场景、边缘测试。

## 延伸阅读

- [Agent 入门指南](/2026/05/10/zh-CN/技术文档/Agent/agent-intro/)
- [Agent 安全考量](/2026/05/10/zh-CN/技术文档/Agent/agent-security/)
- [Agent 调试技巧](/2026/05/10/zh-CN/技术文档/Agent/agent-debugging/)
- [多 Agent 竞争与博弈](/2026/05/10/zh-CN/技术文档/Agent/multi-agent-competition/)
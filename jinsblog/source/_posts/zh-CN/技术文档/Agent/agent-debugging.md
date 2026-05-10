---
title: Agent 调试技巧
date: 2026-05-10
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, 调试, 日志]
---

## Agent 调试的特殊性

Agent 调试比传统程序调试更复杂：

### 调试复杂性

| 特性 | 传统程序 | Agent |
|------|----------|-------|
| 执行确定性 | 确定性执行 | 部分不确定性 |
| 调用链 | 明确调用链 | 多步骤迭代 |
| 错误来源 | 代码逻辑 | LLM+工具+逻辑 |
| 状态追踪 | 变量追踪 | 多层状态追踪 |
| 日志分析 | 结构化日志 | 自然语言日志 |

### Agent 调试挑战

1. **LLM 输出不确定性**：相同输入可能不同输出
2. **多步骤追踪**：需要追踪多轮交互过程
3. **工具调用错误**：工具执行可能失败
4. **状态复杂**：记忆、规划、执行状态交织
5. **时间依赖**：错误可能在后期显现

## 日志记录策略

### 日志层级设计

```python
class AgentLogger:
    def __init__(self):
        self.logs = []
        self.level = "INFO"  # DEBUG, INFO, WARN, ERROR
    
    def log(self, level, category, message, data=None):
        """记录日志"""
        entry = {
            "timestamp": datetime.now(),
            "level": level,
            "category": category,
            "message": message,
            "data": data
        }
        self.logs.append(entry)
        
        # 输出到控制台/文件
        if level >= self.level:
            self.output(entry)
    
    # 各类日志方法
    def log_input(self, user_input):
        self.log("INFO", "input", "用户输入", {"content": user_input})
    
    def log_llm_call(self, prompt, response):
        self.log("DEBUG", "llm", "LLM调用", {
            "prompt": prompt,
            "response": response
        })
    
    def log_tool_call(self, tool_name, params, result):
        self.log("INFO", "tool", f"工具调用: {tool_name}", {
            "params": params,
            "result": result
        })
    
    def log_decision(self, decision_type, decision):
        self.log("DEBUG", "decision", f"决策: {decision_type}", decision)
    
    def log_error(self, error_type, error_details):
        self.log("ERROR", "error", f"错误: {error_type}", error_details)
```

### 关键日志点

```python
def execute_with_logging(agent, task):
    """带日志的执行"""
    logger = AgentLogger()
    
    # 1. 记录任务开始
    logger.log_input(task)
    
    # 2. 记录任务规划
    plan = agent.plan(task)
    logger.log_decision("plan", plan)
    
    # 3. 记录每步执行
    for step in plan.steps:
        # LLM 调用
        logger.log_llm_call(step.prompt, step.llm_response)
        
        # 决策
        logger.log_decision("step_decision", step.decision)
        
        # 工具调用（如有）
        if step.tool_call:
            logger.log_tool_call(
                step.tool_call.name,
                step.tool_call.params,
                step.tool_result
            )
        
        # 步骤结果
        logger.log("INFO", "step", f"步骤完成: {step.id}", step.result)
    
    # 4. 记录最终结果
    logger.log("INFO", "result", "任务完成", agent.output)
    
    return agent.output, logger.logs
```

### 日志格式优化

```python
def format_log_entry(entry):
    """格式化日志条目"""
    
    # 时间戳格式化
    timestamp = entry["timestamp"].strftime("%H:%M:%S.%f")[:-3]
    
    # 级别颜色
    level_colors = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARN": "\033[33m",   # Yellow
        "ERROR": "\033[31m"   # Red
    }
    
    color = level_colors.get(entry["level"], "")
    
    formatted = f"{timestamp} [{color}{entry['level']}\033[0m] [{entry['category']}] {entry['message']}"
    
    if entry["data"]:
        formatted += f"\n  Data: {json.dumps(entry['data'], indent=2)}"
    
    return formatted
```

## 决策过程可视化

### 决策树可视化

```python
def visualize_decision_tree(trace):
    """可视化决策树"""
    
    tree = "决策树:\n"
    
    for step in trace["steps"]:
        # 步骤决策
        tree += f"├─ 步骤 {step.id}: {step.action}\n"
        
        # 子决策
        if step.sub_decisions:
            for sub in step.sub_decisions:
                tree += f"│  ├─ {sub.type}: {sub.choice}\n"
        
        # 结果
        tree += f"│  └─ 结果: {step.result_summary}\n"
    
    return tree

# 示例输出
"""
决策树:
├─ 步骤 1: analyze_task
│  ├─ intent_recognition: 用户意图=数据分析
│  └─ tool_selection: 选择工具=read_csv
│  └─ 结果: 数据读取成功
├─ 步骤 2: process_data
│  ├─ method_selection: 选择方法=统计分析
│  └─ 结果: 分析完成
└─ 步骤 3: generate_output
   └─ 结果: 报告生成成功
"""
```

### 执行流程可视化

```python
def visualize_execution_flow(trace):
    """可视化执行流程"""
    
    flow = []
    
    for step in trace["steps"]:
        # 添加节点
        flow.append({
            "step": step.id,
            "action": step.action,
            "status": step.status,
            "duration": step.duration
        })
        
        # 添加连接
        if step.tool_call:
            flow.append({
                "type": "tool",
                "tool": step.tool_call.name,
                "success": step.tool_result.success
            })
    
    return render_flow_diagram(flow)
```

### 状态变化追踪

```python
def track_state_changes(trace):
    """追踪状态变化"""
    
    state_history = []
    
    for step in trace["steps"]:
        # 记录状态变化
        state_change = {
            "step": step.id,
            "before": step.state_before,
            "after": step.state_after,
            "changes": diff_states(step.state_before, step.state_after)
        }
        state_history.append(state_change)
    
    return state_history

def diff_states(before, after):
    """对比状态变化"""
    changes = []
    
    for key in after:
        if key not in before:
            changes.append({"key": key, "type": "added", "value": after[key]})
        elif before[key] != after[key]:
            changes.append({"key": key, "type": "modified", 
                           "before": before[key], "after": after[key]})
    
    for key in before:
        if key not in after:
            changes.append({"key": key, "type": "removed"})
    
    return changes
```

## 中间状态检查

### 检查点设置

```python
class AgentCheckpoint:
    def __init__(self):
        self.checkpoints = {}
    
    def save_checkpoint(self, step_id, state):
        """保存检查点"""
        self.checkpoints[step_id] = {
            "state": copy.deepcopy(state),
            "timestamp": datetime.now()
        }
    
    def load_checkpoint(self, step_id):
        """加载检查点"""
        return self.checkpoints.get(step_id)
    
    def rollback(self, step_id):
        """回滚到检查点"""
        checkpoint = self.load_checkpoint(step_id)
        if checkpoint:
            return checkpoint["state"]
        return None

def execute_with_checkpoints(agent, task, checkpoint_steps):
    """带检查点的执行"""
    checkpoint_manager = AgentCheckpoint()
    
    for step in agent.plan(task).steps:
        # 执行步骤
        result = agent.execute_step(step)
        
        # 如果是检查点步骤，保存
        if step.id in checkpoint_steps:
            checkpoint_manager.save_checkpoint(step.id, agent.state)
        
        # 如果失败，可以回滚
        if not result.success:
            # 回滚到上一个检查点
            prev_checkpoint = find_previous_checkpoint(step.id, checkpoint_steps)
            agent.state = checkpoint_manager.rollback(prev_checkpoint)
            break
```

### 中间结果验证

```python
def validate_intermediate_results(trace):
    """验证中间结果"""
    
    validations = []
    
    for step in trace["steps"]:
        validation = {
            "step": step.id,
            "checks": []
        }
        
        # 检查 LLM 输出格式
        if step.llm_response:
            validation["checks"].append({
                "type": "llm_format",
                "passed": validate_llm_format(step.llm_response)
            })
        
        # 检查工具调用参数
        if step.tool_call:
            validation["checks"].append({
                "type": "tool_params",
                "passed": validate_tool_params(step.tool_call)
            })
        
        # 检查工具执行结果
        if step.tool_result:
            validation["checks"].append({
                "type": "tool_result",
                "passed": validate_tool_result(step.tool_result)
            })
        
        validations.append(validation)
    
    return validations
```

### 状态健康检查

```python
def health_check(agent_state):
    """状态健康检查"""
    
    checks = {
        "memory": check_memory_health(agent_state.memory),
        "planning": check_planning_health(agent_state.planning),
        "tools": check_tools_health(agent_state.tools),
        "context": check_context_health(agent_state.context)
    }
    
    return {
        "healthy": all(c["healthy"] for c in checks.values()),
        "checks": checks
    }

def check_memory_health(memory):
    """检查记忆状态"""
    return {
        "healthy": len(memory) < MAX_MEMORY_SIZE,
        "warnings": [
            "记忆过大" if len(memory) > WARNING_MEMORY_SIZE else None
        ]
    }
```

## 常见问题与解决方案

### 问题类型与解决

| 问题类型 | 可能原因 | 解决方案 |
|----------|----------|----------|
| LLM 输出格式错误 | Prompt设计问题 | 优化Prompt格式要求 |
| 工具调用失败 | 参数错误或工具问题 | 参数验证、工具调试 |
| 任务未完成 | 规划不完整或执行中断 | 检查规划、增加恢复机制 |
| 循环执行 | 条件判断错误 | 检查终止条件 |
| 成本过高 | 过多LLM调用 | 优化执行流程 |
| 响应超时 | 执行时间过长 | 设置超时、并行执行 |

### 问题诊断流程

```python
def diagnose_issue(trace, error):
    """诊断问题"""
    
    diagnosis = {
        "error": error,
        "type": classify_error(error),
        "location": locate_error(trace, error),
        "cause": analyze_cause(trace, error),
        "solution": suggest_solution(error)
    }
    
    return diagnosis

def classify_error(error):
    """分类错误"""
    
    if "format" in str(error):
        return "format_error"
    elif "tool" in str(error):
        return "tool_error"
    elif "timeout" in str(error):
        return "timeout_error"
    elif "memory" in str(error):
        return "memory_error"
    else:
        return "unknown_error"

def locate_error(trace, error):
    """定位错误位置"""
    
    # 从最近的步骤开始查找
    for step in reversed(trace["steps"]):
        if step.error:
            return {
                "step": step.id,
                "component": step.error_component,
                "details": step.error_details
            }
    
    return {"location": "unknown"}
```

### LLM 输出问题调试

```python
def debug_llm_output(prompt, response, expected_format):
    """调试 LLM 输出"""
    
    analysis = {
        "prompt": prompt,
        "response": response,
        "format_check": {
            "expected": expected_format,
            "actual": extract_format(response),
            "matches": check_format_match(response, expected_format)
        }
    }
    
    # 如果格式不匹配，分析原因
    if not analysis["format_check"]["matches"]:
        analysis["suggestions"] = [
            "Prompt 中增加格式示例",
            "使用结构化输出约束",
            "增加输出验证步骤"
        ]
    
    return analysis
```

### 工具调用问题调试

```python
def debug_tool_call(tool_name, params, result, expected):
    """调试工具调用"""
    
    debug_info = {
        "tool": tool_name,
        "params": params,
        "result": result,
        "expected": expected,
        "checks": []
    }
    
    # 检查参数
    debug_info["checks"].append({
        "name": "params_valid",
        "passed": validate_params(tool_name, params),
        "issues": find_param_issues(params)
    })
    
    # 检查执行
    debug_info["checks"].append({
        "name": "execution_success",
        "passed": result.success,
        "error": result.error if not result.success else None
    })
    
    # 检查结果
    if result.success:
        debug_info["checks"].append({
            "name": "result_valid",
            "passed": validate_result(result, expected),
            "issues": find_result_issues(result, expected)
        })
    
    return debug_info
```

## Agent 调试工具

### Agent 调试器

```python
class AgentDebugger:
    def __init__(self, agent):
        self.agent = agent
        self.logger = AgentLogger()
        self.checkpoint_manager = AgentCheckpoint()
        self.breakpoints = []
    
    def set_breakpoint(self, step_type):
        """设置断点"""
        self.breakpoints.append(step_type)
    
    def run_with_debug(self, task):
        """调试运行"""
        trace = []
        
        plan = self.agent.plan(task)
        self.logger.log("INFO", "plan", "计划生成", plan)
        
        for step in plan.steps:
            # 检查断点
            if step.type in self.breakpoints:
                self.logger.log("DEBUG", "breakpoint", f"到达断点: {step.type}")
                # 可以在这里暂停，等待用户检查
                user_input = input("继续执行? (y/n): ")
                if user_input != 'y':
                    break
            
            # 执行步骤
            result = self.agent.execute_step(step)
            
            # 记录
            trace.append({
                "step": step,
                "result": result,
                "state": copy.deepcopy(self.agent.state)
            })
            
            self.logger.log("INFO", "step", f"步骤执行: {step.type}", result)
            
            # 如果失败，诊断问题
            if not result.success:
                diagnosis = diagnose_issue(trace, result.error)
                self.logger.log("ERROR", "diagnosis", "问题诊断", diagnosis)
                break
        
        return {
            "output": self.agent.output,
            "trace": trace,
            "logs": self.logger.logs
        }
```

### 追踪可视化工具

```python
def visualize_trace(trace):
    """可视化执行追踪"""
    
    print("\n=== Agent 执行追踪 ===\n")
    
    for i, entry in enumerate(trace):
        step = entry["step"]
        result = entry["result"]
        
        # 步骤信息
        print(f"[Step {i+1}] {step.type}")
        print(f"  Action: {step.action}")
        
        # 工具调用
        if step.tool_call:
            print(f"  Tool: {step.tool_call.name}")
            print(f"  Params: {step.tool_call.params}")
        
        # 结果
        print(f"  Status: {result.status}")
        if result.success:
            print(f"  Result: {truncate(result.output, 100)}")
        else:
            print(f"  Error: {result.error}")
        
        print()
```

## Agent 调试最佳实践

### 日志策略

1. **分层记录**：按级别和类别分层
2. **关键点记录**：记录关键决策和执行点
3. **可追溯性**：确保日志可追溯到具体步骤

### 检查策略

1. **中间检查**：在关键步骤设置检查点
2. **结果验证**：验证每步结果是否符合预期
3. **状态监控**：监控 Agent 状态健康

### 问题处理

1. **快速定位**：使用日志快速定位问题
2. **根因分析**：分析问题根本原因
3. **预防措施**：建立预防类似问题的机制

## 总结

Agent 调试需要处理 LLM 不确定性、多步骤追踪、工具调用错误、复杂状态等挑战。调试策略包括：完善的日志记录、决策过程可视化、中间状态检查、问题诊断分析。

Agent 调试器可以设置断点、追踪执行、诊断问题。最佳实践包括：分层日志记录、关键检查点、快速问题定位。

## 延伸阅读

- [Agent 入门指南](/2026/05/10/zh-CN/技术文档/Agent/agent-intro/)
- [Agent 规划与推理](/2026/05/10/zh-CN/技术文档/Agent/agent-planning/)
- [Agent 评测方法](/2026/05/10/zh-CN/技术文档/Agent/agent-evaluation/)
- [Agent 安全考量](/2026/05/10/zh-CN/技术文档/Agent/agent-security/)
---
title: Agent 框架概览
date: 2026-05-10
categories: [技术文档, Agent]
lang: zh-CN
tags: [Agent, Framework, LangChain, AutoGen]
---

## Agent 框架的演进历史

Agent 框架的发展经历了几个阶段：

### 第一阶段：基础工具（2022-2023）

早期 Agent 开发依赖直接调用 LLM API：
- 手动构建 prompt
- 手动管理对话历史
- 手动实现工具调用

### 第二阶段：应用框架（2023）

LangChain 等框架出现：
- 提供链式调用抽象
- 内置工具管理
- 记忆管理支持

### 第三阶段：Agent 框架（2023-2024）

专门的 Agent 框架出现：
- AutoGen：多 Agent 协作
- CrewAI：角色扮演式 Agent
- LangGraph：状态图式 Agent

### 第四阶段：企业级框架（2024+）

企业级 Agent 框架：
- Semantic Kernel：微软企业级 SDK
- AutoGen Studio：可视化 Agent 构建
- 多框架融合趋势

## 主流框架对比

### LangChain

**特点**：
- 最早的应用框架
- 链式调用抽象
- 丰富的工具集成
- LangGraph 支持复杂 Agent

**优势**：
- 社区活跃，资源丰富
- 生态完善
- 支持多种 LLM

**局限**：
- 抽象层次复杂
- 学习曲线陡峭
- Agent 支持需 LangGraph

### AutoGen

**特点**：
- 微软开发
- 多 Agent 对话协作
- 人机协同支持
- 灵活的 Agent 定义

**优势**：
- 多 Agent 原生支持
- 对话协作直观
- 研究友好

**局限**：
- 相对较新，生态待完善
- 复杂场景配置繁琐
- 成本较高

### CrewAI

**特点**：
- 角色扮演式设计
- Sequential/Hierarchical 流程
- Task-Crew 抽象
- 简洁的 API

**优势**：
- 设计直观易理解
- API 简洁
- 角色分工清晰

**局限**：
- 灵活度受限
- 非对话场景支持有限
- 定制化难度

### Semantic Kernel

**特点**：
- 微软企业级 SDK
- Skills/Functions/Planners
- Azure 服务集成
- 企业级安全

**优势**：
- 企业级支持
- 安全合规
- Azure 生态集成

**局限**：
- 学习曲线陡峭
- Azure 依赖
- 个人开发者门槛高

### 框架对比表

| 特性 | LangChain | AutoGen | CrewAI | Semantic Kernel |
|------|-----------|---------|--------|-----------------|
| 开发者 | LangChain | Microsoft | CrewAI Inc | Microsoft |
| 核心抽象 | Chain/Graph | ConversableAgent | Agent/Task/Crew | Skill/Planner |
| 多 Agent | LangGraph | 原生支持 | 支持 | 支持 |
| 学习难度 | 中高 | 中 | 低 | 高 |
| 企业级 | 一般 | 研究友好 | 一般 | 强 |
| 社区生态 | 丰富 | 发展中 | 发展中 | 发展中 |
| LLM 支持 | 多种 | 多种 | 多种 | 多种/Azure |

## 框架选择考量

### 任务类型

根据任务类型选择框架：

| 任务类型 | 推荐框架 |
|----------|----------|
| 单 Agent 简单任务 | LangChain |
| 多 Agent 协作任务 | AutoGen、CrewAI |
| 企业级应用 | Semantic Kernel |
| 复杂状态流转 | LangGraph |
| 角色分工明确 | CrewAI |
| 研究/实验 | AutoGen |

### 团队背景

根据团队背景选择：

| 团队背景 | 推荐框架 |
|----------|----------|
| Python 开发者 | LangChain、AutoGen、CrewAI |
| .NET/微软生态 | Semantic Kernel |
| 快速原型验证 | CrewAI |
| 需要企业支持 | Semantic Kernel |

### 成本考量

根据成本预算选择：

| 成本预算 | 推荐方案 |
|----------|----------|
| 低成本 | LangChain + 自托管 LLM |
| 中等成本 | CrewAI、AutoGen |
| 企业预算 | Semantic Kernel + Azure |

### 时间要求

根据开发时间选择：

| 时间要求 | 推荐框架 |
|----------|----------|
| 快速原型 | CrewAI（API简洁） |
| 中等时间 | LangChain、AutoGen |
| 企业项目 | Semantic Kernel |

## LangChain/LangGraph 概览

### LangChain 核心概念

```
Chain: 链式调用，组合多个组件
Agent: 自主决策的执行单元
Tool: 工具定义和执行
Memory: 对话和状态记忆
Prompt: 提示模板管理
```

### LangGraph 核心概念

LangGraph 基于 StateGraph 构建 Agent：

```
StateGraph: 状态图，定义 Agent 流程
Node: 状态节点，代表 Agent 或操作
Edge: 边，定义状态转换条件
Graph: 完整的 Agent 执行图
```

### LangChain 适用场景

- 单 Agent 应用
- 工具调用场景
- RAG 应用
- 简单链式处理

### LangGraph 适用场景

- 多 Agent 系统
- 复杂状态流转
- 循环/条件分支
- 需要精确控制流程

## AutoGen 概览

### AutoGen 核心概念

```
ConversableAgent: 可对话的 Agent
AssistantAgent: AI Agent（LLM驱动）
UserProxyAgent: 人类代理
GroupChat: 群聊模式
GroupChatManager: 群聊管理
```

### AutoGen 设计理念

AutoGen 以对话为核心：
- Agent 通过对话协作
- 支持人机协同
- 灵活的 Agent 定制

### AutoGen 适用场景

- 多 Agent 对话协作
- 研究实验
- 人机协同应用
- 需要灵活 Agent 定义

## CrewAI 概览

### CrewAI 核心概念

```
Agent: 角色 Agent
Task: 任务定义
Crew: Agent 组（团队）
Process: 流程类型（Sequential/Hierarchical）
Tool: 工具定义
```

### CrewAI 设计理念

CrewAI 以角色扮演为核心：
- 每个 Agent 是一个角色
- 任务分配给角色
- 团队协作完成任务

### CrewAI 适用场景

- 角色分工明确的任务
- 快速原型开发
- 内容生产团队
- 简单多 Agent 场景

## Semantic Kernel 概览

### Semantic Kernel 核心概念

```
Kernel: 核心引擎
Skill: 技能集合
Function: 具体功能
Planner: 规划器
Connector: 连接器（LLM/存储）
```

### Semantic Kernel 设计理念

Semantic Kernel 以企业级为核心：
- Skills 组织能力
- Planner 自动规划
- 企业安全合规
- Azure 服务集成

### Semantic Kernel 适用场景

- 企业级应用
- Azure 生态项目
- 需要安全合规
- 大型组织部署

## 框架学习路线建议

### 初学者路线

1. **入门**：CrewAI（最简单直观）
2. **进阶**：LangChain（生态最丰富）
3. **深入**：LangGraph（复杂流程控制）
4. **拓展**：AutoGen（多 Agent 协作）

### 开发者路线

1. **基础**：LangChain 核心组件
2. **Agent**：LangChain Agent + Tools
3. **复杂**：LangGraph 状态图
4. **多 Agent**：AutoGen 或 CrewAI

### 企业开发者路线

1. **基础**：Semantic Kernel 核心概念
2. **Skills**：Skills 开发和组合
3. **Planner**：Planner 定制
4. **集成**：Azure 服务集成
5. **部署**：企业级部署

### 研究者路线

1. **基础**：AutoGen 核心概念
2. **Agent 定义**：自定义 Agent
3. **协作模式**：GroupChat 深入
4. **实验**：设计实验方案

## 框架融合趋势

### 跨框架集成

框架间开始出现融合：

```python
# LangChain + AutoGen
from langchain.tools import Tool
from autogen import AssistantAgent

# LangChain 工具在 AutoGen 中使用
langchain_tool = Tool(...)
autogen_agent = AssistantAgent(
    tools=[convert_to_autogen_tool(langchain_tool)]
)
```

### 统一接口趋势

各框架开始统一某些接口：
- 工具定义格式趋同
- LLM 接口标准化
- 消息格式统一

### 混合架构实践

实际项目常使用多个框架：
- LangChain 处理单 Agent
- AutoGen 处理多 Agent 协作
- Semantic Kernel 处理企业集成

## 框架选择决策树

```
需要 Agent？
    │
    ├─ 否 → LangChain Chain
    │
    └─ 是 → 需要多 Agent？
              │
              ├─ 否 → LangChain Agent / LangGraph
              │
              └─ 是 → 企业级？
                        │
                        ├─ 是 → Semantic Kernel
                        │
                        └─ 否 → 角色分工明确？
                                  │
                                  ├─ 是 → CrewAI
                                  │
                                  └─ 否 → AutoGen
```

## 总结

Agent 框架选择需要根据任务类型、团队背景、成本预算、时间要求综合考量。主流框架各有特点：LangChain/LangGraph 生态丰富，AutoGen 多 Agent 原生支持，CrewAI API 简洁直观，Semantic Kernel 企业级支持。

建议根据自身情况选择合适的学习路线：初学者从 CrewAI 入门，开发者深入 LangChain/LangGraph，企业开发者关注 Semantic Kernel，研究者使用 AutoGen。

## 延伸阅读

- [LangChain Agent 实践](/2026/05/10/zh-CN/技术文档/Agent/langchain-agent/)
- [AutoGen 多 Agent 实践](/2026/05/10/zh-CN/技术文档/Agent/autogen-agent/)
- [CrewAI Agent 实践](/2026/05/10/zh-CN/技术文档/Agent/crewai-agent/)
- [Semantic Kernel 实践](/2026/05/10/zh-CN/技术文档/Agent/semantic-kernel/)
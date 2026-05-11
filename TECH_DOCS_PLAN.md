# 技术文档编写计划

## 背景

在 `jinsblog/source/_posts/` 目录下创建 AI Agent 和机器学习两大系列的技术文档。每个文档需要双语版本（中文 + 英文）。

---

## 一、AI Agent 系列文档（27篇）

### 第一章：Agent 基础概念

#### 1.1 Agent 入门指南
- **文件名**: `agent-intro.md`
- **内容要点**: Agent 的定义与核心概念、与传统 AI 系统的区别、三大核心能力（感知、决策、执行）、典型特征（自主性、反应性、主动性、社交性）、应用场景概览
- **前置知识**: 无

#### 1.2 LLM 与 Agent 的关系
- **文件名**: `llm-and-agent.md`
- **内容要点**: LLM 作为 Agent "大脑" 的角色、能力边界与局限、从 Chatbot 到 Agent 的演进、LLM Agent 的典型架构、为何需要 Agent
- **前置知识**: Agent 入门指南

#### 1.3 Agent 架构模式
- **文件名**: `agent-architecture.md`
- **内容要点**: 单 Agent 架构、多 Agent 架构、层次化 Agent 架构、黑板架构模式、消息传递架构、架构选择考量因素
- **前置知识**: Agent 入门指南

### 第二章：Agent 核心组件

#### 2.1 工具调用机制（Tool Use）
- **文件名**: `tool-use.md`
- **内容要点**: 工具调用的基本原理、工具定义与描述（Function Calling）、工具选择策略、工具执行与结果处理、错误处理、常见工具类型、安全性考量
- **前置知识**: Agent 入门指南

#### 2.2 Agent 记忆系统
- **文件名**: `agent-memory.md`
- **内容要点**: 记忆系统的重要性、短期记忆（对话历史、滑动窗口）、工作记忆（当前任务状态）、长期记忆（向量数据库、知识图谱）、检索策略、压缩与摘要、更新与遗忘机制、实现方案（Redis、Mem0、Letta）
- **前置知识**: Agent 入门指南

#### 2.3 Agent 规划与推理
- **文件名**: `agent-planning.md`
- **内容要点**: 规划的定义与必要性、思维链（CoT）、ReAct 模式、Plan-and-Execute 模式、思维树（ToT）、任务分解策略、规划与执行的反馈循环、规划失败的恢复机制
- **前置知识**: Agent 入门指南

### 第三章：多 Agent 系统

#### 3.1 多 Agent 系统概述
- **文件名**: `multi-agent-intro.md`
- **内容要点**: 多 Agent 系统的定义、为何需要多 Agent、系统的分类、优势与挑战
- **前置知识**: Agent 架构模式

#### 3.2 多 Agent 协作模式
- **文件名**: `multi-agent-collaboration.md`
- **内容要点**: 协作模式分类（顺序、并行、层次、对话）、Agent 间通信机制、任务分配策略、结果聚合方法、冲突检测与解决、协作协议设计
- **前置知识**: 多 Agent 系统概述

#### 3.3 多 Agent 竞争与博弈
- **文件名**: `multi-agent-competition.md`
- **内容要点**: Agent 竞争场景、博弈论基础、自我对弈与强化学习、Red Teaming 与安全测试、竞争驱动的性能提升
- **前置知识**: 多 Agent 系统概述

#### 3.4 层次化 Agent 系统
- **文件名**: `hierarchical-agent.md`
- **内容要点**: 层次化架构的设计理念、Manager Agent 与 Worker Agent、任务下发与结果上报、层次深度选择、案例分析
- **前置知识**: 多 Agent 协作模式

### 第四章：Agent 框架与实践

#### 4.1 Agent 框架概览
- **文件名**: `agent-frameworks.md`
- **内容要点**: 框架的演进历史、主流框架对比（LangChain、AutoGen、CrewAI、Semantic Kernel）、框架选择考量、学习路线建议
- **前置知识**: Agent 架构模式

#### 4.2 LangChain Agent 实践
- **文件名**: `langchain-agent.md`
- **内容要点**: LangChain Agent 核心概念、AgentExecutor 详解、工具定义与绑定、LangGraph 状态机模型、实战案例
- **前置知识**: Agent 框架概览

#### 4.3 AutoGen 多 Agent 实践
- **文件名**: `autogen-agent.md`
- **内容要点**: AutoGen 核心概念、ConversableAgent 设计、GroupChat 与群聊模式、人机协作机制、实战案例
- **前置知识**: Agent 框架概览

#### 4.4 CrewAI Agent 实践
- **文件名**: `crewai-agent.md`
- **内容要点**: CrewAI 核心概念、Agent/Task/Crew 三要素、流程类型（Sequential/Hierarchical）、Tools 与知识集成、实战案例
- **前置知识**: Agent 框架概览

#### 4.5 Semantic Kernel 实践
- **文件名**: `semantic-kernel.md`
- **内容要点**: Semantic Kernel 设计理念、Skills/Functions/Planners、与 Microsoft 生态集成、实战案例
- **前置知识**: Agent 框架概览

### 第五章：Agent 应用案例

#### 5.1 代码助手 Agent
- **文件名**: `code-assistant-agent.md`
- **内容要点**: 代码助手的应用场景、核心能力（代码生成、调试、重构）、实现要点、案例分析（Devin、Cursor）
- **前置知识**: Agent 架构模式

#### 5.2 数据分析 Agent
- **文件名**: `data-analysis-agent.md`
- **内容要点**: 数据分析 Agent 的角色、数据获取与预处理、分析任务编排、可视化生成、报告撰写自动化
- **前置知识**: Agent 架构模式

#### 5.3 客户服务 Agent
- **文件名**: `customer-service-agent.md`
- **内容要点**: 客服 Agent 的优势、知识库集成、多轮对话管理、情绪识别与处理、人工转接机制
- **前置知识**: Agent 架构模式

#### 5.4 自动研究 Agent
- **文件名**: `research-agent.md`
- **内容要点**: 研究 Agent 的应用场景、信息检索策略、知识整合与推理、报告生成自动化、案例分析（GPT-Researcher）
- **前置知识**: Agent 架构模式

### 第六章：Agent 安全与评测

#### 6.1 Agent 安全考量
- **文件名**: `agent-security.md`
- **内容要点**: Agent 安全风险分类、Prompt Injection 攻击、工具调用安全、数据隐私保护、权限控制机制、安全护栏设计
- **前置知识**: Agent 入门指南

#### 6.2 Agent 评测方法
- **文件名**: `agent-evaluation.md`
- **内容要点**: Agent 评测的挑战、任务成功率评测、步骤效率评测、成本效益评测、主流评测框架（AgentBench、WebShop）、自定义评测设计
- **前置知识**: Agent 入门指南

#### 6.3 Agent 调试技巧
- **文件名**: `agent-debugging.md`
- **内容要点**: Agent 调试的特殊性、日志记录策略、决策过程可视化、中间状态检查、常见问题与解决方案
- **前置知识**: Agent 入门指南

### 第七章：Agent 进阶主题

#### 7.1 Agent 与传统软件工程
- **文件名**: `agent-software-engineering.md`
- **内容要点**: Agent 在软件工程中的定位、Agent 与微服务架构、Agent 与工作流引擎、Agent 系统的可维护性、测试策略
- **前置知识**: Agent 架构模式

#### 7.2 Agent 伦理考量
- **文件名**: `agent-ethics.md`
- **内容要点**: Agent 的责任归属问题、决策的透明度、Agent 与人类工作关系、失败的影响评估、伦理设计原则
- **前置知识**: Agent 入门指南

#### 7.3 Agent 未来展望
- **文件名**: `agent-future.md`
- **内容要点**: Agent 技术发展趋势、自主 Agent 的愿景、Agent 与具身智能、Agent 与 AGI 的关系、可能的技术突破点
- **前置知识**: 所有 Agent 文档

---

## 二、机器学习系列文档（40篇）

### 第一章：数学基础

#### 1.1 线性代数基础
- **文件名**: `linear-algebra.md`
- **内容要点**: 向量与向量运算、矩阵与矩阵运算、矩阵分解（LU/QR/SVD）、线性变换与空间、特征值与特征向量、张量简介
- **前置知识**: 无

#### 1.2 概率论基础
- **文件名**: `probability-theory.md`
- **内容要点**: 概率基本概念、条件概率与贝叶斯定理、常见概率分布（离散/连续）、期望与方差、协方差与相关系数、大数定律与中心极限定理
- **前置知识**: 无

#### 1.3 统计学基础
- **文件名**: `statistics.md`
- **内容要点**: 统计量与统计推断、点估计与区间估计、假设检验、参数估计方法（MLE/MAP）、贝叶斯统计简介
- **前置知识**: 概率论基础

#### 1.4 优化理论基础
- **文件名**: `optimization.md`
- **内容要点**: 优化问题分类、凸优化基础、梯度与方向导数、无约束优化方法、约束优化方法、拉格朗日乘子法
- **前置知识**: 线性代数基础

### 第二章：机器学习基础

#### 2.1 机器学习概述
- **文件名**: `ml-introduction.md`
- **内容要点**: 机器学习定义与发展历史、学习类型（监督/无监督/半监督/强化）、工作流程、模型评估指标、过拟合与欠拟合、应用领域
- **前置知识**: 无

#### 2.2 数据预处理
- **文件名**: `data-preprocessing.md`
- **内容要点**: 数据清洗、缺失值处理、异常值检测与处理、数据标准化与归一化、特征编码（One-hot/Label Encoding）、数据划分策略
- **前置知识**: 机器学习概述

#### 2.3 特征工程
- **文件名**: `feature-engineering.md`
- **内容要点**: 特征选择方法、特征提取技术、特征构造策略、降维技术（PCA/LDA）、特征重要性评估
- **前置知识**: 数据预处理

### 第三章：经典机器学习模型

#### 3.1 线性回归
- **文件名**: `linear-regression.md`
- **内容要点**: 线性回归模型推导、最小二乘法求解、正则化（L1/Lasso、L2/Ridge）、多元线性回归、模型评估指标、案例实践
- **前置知识**: 线性代数基础、优化理论基础

#### 3.2 逻辑回归
- **文件名**: `logistic-regression.md`
- **内容要点**: 逻辑回归原理、Sigmoid 函数、概率解释与最大似然估计、多分类扩展、正则化方法、案例实践
- **前置知识**: 线性回归、概率论基础

#### 3.3 决策树
- **文件名**: `decision-tree.md`
- **内容要点**: 决策树基本概念、信息增益与 ID3、信息增益率与 C4.5、Gini 指数与 CART、树的剪枝策略、回归树、案例实践
- **前置知识**: 机器学习概述、概率论基础

#### 3.4 支持向量机（SVM）
- **文件名**: `svm.md`
- **内容要点**: SVM 基本原理、最大间隔与支持向量、核函数详解、软间隔与松弛变量、SMO 算法、多分类 SVM、SVR、案例实践
- **前置知识**: 线性代数基础、优化理论基础

#### 3.5 贝叶斯分类器
- **文件名**: `bayesian-classifier.md`
- **内容要点**: 贝叶斯分类原理、朴素贝叶斯、贝叶斯网络、参数估计、半朴素贝叶斯、案例实践
- **前置知识**: 概率论基础、统计学基础

#### 3.6 K 近邻（KNN）
- **文件名**: `knn.md`
- **内容要点**: KNN 基本原理、距离度量方法、K 值选择策略、权重分配方法、KD 树优化、案例实践
- **前置知识**: 机器学习概述

### 第四章：集成学习

#### 4.1 集成学习概述
- **文件名**: `ensemble-learning.md`
- **内容要点**: 集成学习核心思想、偏差与方差分解、集成策略分类、集成学习的优势
- **前置知识**: 决策树、机器学习概述

#### 4.2 Bagging 与随机森林
- **文件名**: `random-forest.md`
- **内容要点**: Bagging 原理、Bootstrap 采样、随机森林算法、特征随机选择、袋外数据（OOB）评估、优缺点、案例实践
- **前置知识**: 决策树、集成学习概述

#### 4.3 Boosting 基础
- **文件名**: `boosting.md`
- **内容要点**: Boosting 原理、加法模型与前向分步算法、AdaBoost 算法详解、Boosting 与 Bagging 对比
- **前置知识**: 决策树、集成学习概述

#### 4.4 GBDT 梯度提升树
- **文件名**: `gbdt.md`
- **内容要点**: GBDT 基本原理、损失函数选择、负梯度拟合、GBDT 回归与分类、优缺点、案例实践
- **前置知识**: Boosting 基础、决策树

#### 4.5 XGBoost 算法详解
- **文件名**: `xgboost.md`
- **内容要点**: XGBoost 目标函数推导、二阶泰勒展开优化、结构分数与增益计算、节点分裂策略、正则化与剪枝、并行化与缓存优化、特征重要性分析、调参技巧、案例实践
- **前置知识**: GBDT、优化理论基础

#### 4.6 LightGBM 算法详解
- **文件名**: `lightgbm.md`
- **内容要点**: LightGBM 设计理念、GOSS（梯度单边采样）、EFB（互斥特征绑定）、Leaf-wise 生长策略、直方图加速算法、与 XGBoost 对比、调参技巧、案例实践
- **前置知识**: XGBoost

#### 4.7 Stacking 与 Blending
- **文件名**: `stacking-blending.md`
- **内容要点**: Stacking 原理、元模型设计、交叉验证 Stacking、Blending 方法、多层 Stacking、案例实践
- **前置知识**: 集成学习概述

### 第五章：神经网络基础

#### 5.1 神经网络入门
- **文件名**: `neural-network-intro.md`
- **内容要点**: 神经网络历史与发展、神经元模型、感知机与多层感知机、神经网络的基本结构、能力与局限
- **前置知识**: 线性代数基础

#### 5.2 激活函数详解
- **文件名**: `activation-functions.md`
- **内容要点**: 激活函数的作用、Sigmoid 与 Tanh、ReLU 及变体（LeakyReLU/ELU/GELU）、Softmax 函数、激活函数选择策略、梯度问题
- **前置知识**: 神经网络入门

#### 5.3 损失函数详解
- **文件名**: `loss-functions.md`
- **内容要点**: 损失函数的作用、回归损失（MSE/MAE/Huber）、分类损失（交叉熵/Focal Loss）、对比损失、损失函数选择策略
- **前置知识**: 神经网络入门、概率论基础

### 第六章：反向传播与训练

#### 6.1 反向传播算法详解
- **文件名**: `backpropagation.md`
- **内容要点**: 反向传播的历史、梯度下降基础、链式法则推导、计算图概念、前向传播与反向传播流程、梯度计算示例、实现细节、梯度消失与爆炸问题
- **前置知识**: 神经网络入门、线性代数基础、优化理论基础

#### 6.2 优化算法详解
- **文件名**: `optimization-algorithms.md`
- **内容要点**: SGD 基础、Momentum 与 Nesterov、AdaGrad、RMSprop、Adam 与 AdamW、学习率调度策略、二阶优化方法简介、优化算法选择指南
- **前置知识**: 反向传播算法详解、优化理论基础

#### 6.3 正则化技术
- **文件名**: `regularization.md`
- **内容要点**: 正则化的作用、L1/L2 正则化原理、Dropout 技术、Batch Normalization、Layer Normalization、数据增强、早停策略
- **前置知识**: 反向传播算法详解

#### 6.4 超参数调优
- **文件名**: `hyperparameter-tuning.md`
- **内容要点**: 超参数分类、网格搜索、随机搜索、贝叶斯优化、遗传算法、自动调参工具
- **前置知识**: 优化算法详解

### 第七章：深度学习模型

#### 7.1 卷积神经网络（CNN）
- **文件名**: `cnn.md`
- **内容要点**: CNN 基本原理、卷积操作详解、池化操作、经典架构（LeNet/AlexNet/VGG/ResNet）、残差连接原理、CNN 的变体与发展、案例实践
- **前置知识**: 神经网络入门、反向传播算法详解

#### 7.2 循环神经网络（RNN）
- **文件名**: `rnn.md`
- **内容要点**: RNN 基本原理、序列数据处理、RNN 的梯度问题、时序反向传播（BPTT）、RNN 变体、案例实践
- **前置知识**: 神经网络入门、反向传播算法详解

#### 7.3 LSTM 与 GRU
- **文件名**: `lstm-gru.md`
- **内容要点**: LSTM 门控机制详解、LSTM 解决梯度问题、GRU 简化设计、LSTM 与 GRU 对比、双向 LSTM、案例实践
- **前置知识**: RNN

#### 7.4 Transformer 架构详解
- **文件名**: `transformer.md`
- **内容要点**: Transformer 设计理念、自注意力机制详解、多头注意力、位置编码、编码器-解码器结构、Transformer 变体、案例实践
- **前置知识**: 神经网络入门、注意力机制

#### 7.5 注意力机制详解
- **文件名**: `attention-mechanism.md`
- **内容要点**: 注意力机制起源、软注意力与硬注意力、自注意力、多头注意力、交叉注意力、注意力可视化
- **前置知识**: 神经网络入门

### 第八章：大模型与预训练

#### 8.1 预训练语言模型
- **文件名**: `pretraining-lm.md`
- **内容要点**: 预训练的思想、语言模型基础、BERT（双向编码）、GPT（单向生成）、预训练任务设计、预训练数据与规模
- **前置知识**: Transformer 架构详解

#### 8.2 微调技术详解
- **文件名**: `fine-tuning.md`
- **内容要点**: 全量微调、任务特定微调、Prompt Tuning、Prefix Tuning、LoRA 低秩适配、QLoRA 量化微调、微调策略选择
- **前置知识**: 预训练语言模型

#### 8.3 大模型架构演进
- **文件名**: `llm-architecture.md`
- **内容要点**: GPT 系列演进、LLaMA 架构分析、Mistral 与 MoE、大模型效率优化、长序列处理技术
- **前置知识**: Transformer 架构详解、预训练语言模型

#### 8.4 上下文学习与提示工程
- **文件名**: `prompt-engineering.md`
- **内容要点**: 上下文学习（ICL）原理、Prompt 设计原则、Few-shot 提示、Chain-of-Thought 提示、提示模板设计、提示调优技巧
- **前置知识**: 预训练语言模型

### 第九章：生成模型

#### 9.1 变分自编码器（VAE）
- **文件名**: `vae.md`
- **内容要点**: VAE 基本原理、变分推断基础、ELBO 推导、编码器-解码器结构、VAE 的变体、案例实践
- **前置知识**: 神经网络入门、概率论基础

#### 9.2 生成对抗网络（GAN）
- **文件名**: `gan.md`
- **内容要点**: GAN 基本原理、生成器与判别器、训练策略与博弈、GAN 的训练稳定性、GAN 变体（DCGAN/WGAN/StyleGAN）、案例实践
- **前置知识**: 神经网络入门、优化算法详解

#### 9.3 扩散模型详解
- **文件名**: `diffusion-models.md`
- **内容要点**: 扩散模型原理、前向扩散过程、反向去噪过程、DDPM 算法、条件扩散模型、扩散模型应用（图像/音频/视频）、案例实践
- **前置知识**: 神经网络入门、概率论基础

### 第十章：强化学习

#### 10.1 强化学习基础
- **文件名**: `rl-introduction.md`
- **内容要点**: 强化学习定义、Agent-Environment 模型、状态/动作/奖励、策略与价值函数、马尔可夫决策过程（MDP）、强化学习分类
- **前置知识**: 概率论基础、优化理论基础

#### 10.2 基于价值的方法
- **文件名**: `rl-value-based.md`
- **内容要点**: 动态规划方法、值迭代与策略迭代、Q-Learning、SARSA、DQN 及变体、案例实践
- **前置知识**: 强化学习基础

#### 10.3 基于策略的方法
- **文件名**: `rl-policy-based.md`
- **内容要点**: 策略梯度原理、REINFORCE 算法、Actor-Critic 方法、A2C 与 A3C、PPO 算法详解、案例实践
- **前置知识**: 强化学习基础、基于价值的方法

#### 10.4 强化学习进阶
- **文件名**: `rl-advanced.md`
- **内容要点**: 模型预测与规划、多智能体强化学习、强化学习与 LLM 结合、RLHF（人类反馈强化学习）、强化学习应用案例
- **前置知识**: 基于策略的方法

### 第十一章：模型优化与部署

#### 11.1 模型压缩与加速
- **文件名**: `model-compression.md`
- **内容要点**: 模型压缩动机、知识蒸馏、模型剪枝、模型量化（INT8/INT4）、模型架构优化
- **前置知识**: 神经网络入门

#### 11.2 分布式训练
- **文件名**: `distributed-training.md`
- **内容要点**: 分布式训练概述、数据并行、模型并行、混合并行策略、参数服务器架构、AllReduce 算法、分布式训练框架
- **前置知识**: 反向传播算法详解

#### 11.3 模型部署实践
- **文件名**: `model-deployment.md`
- **内容要点**: 模型部署概述、ONNX 模型转换、TensorRT 加速、模型服务化、推理优化技巧、部署最佳实践
- **前置知识**: 模型压缩与加速

---

## 三、文件组织结构

```
jinsblog/source/_posts/
├── zh-CN/技术文档/
│   ├── Agent/                    # 27 篇中文 Agent 文档
│   │   ├── agent-intro.md
│   │   ├── llm-and-agent.md
│   │   ├── agent-architecture.md
│   │   ├── tool-use.md
│   │   ├── agent-memory.md
│   │   ├── agent-planning.md
│   │   ├── multi-agent-intro.md
│   │   ├── multi-agent-collaboration.md
│   │   ├── multi-agent-competition.md
│   │   ├── hierarchical-agent.md
│   │   ├── agent-frameworks.md
│   │   ├── langchain-agent.md
│   │   ├── autogen-agent.md
│   │   ├── crewai-agent.md
│   │   ├── semantic-kernel.md
│   │   ├── code-assistant-agent.md
│   │   ├── data-analysis-agent.md
│   │   ├── customer-service-agent.md
│   │   ├── research-agent.md
│   │   ├── agent-security.md
│   │   ├── agent-evaluation.md
│   │   ├── agent-debugging.md
│   │   ├── agent-software-engineering.md
│   │   ├── agent-ethics.md
│   │   └── agent-future.md
│   │
│   └── 机器学习/                  # 40 篇中文机器学习文档
│       ├── linear-algebra.md
│       ├── probability-theory.md
│       ├── statistics.md
│       ├── optimization.md
│       ├── ml-introduction.md
│       ├── data-preprocessing.md
│       ├── feature-engineering.md
│       ├── linear-regression.md
│       ├── logistic-regression.md
│       ├── decision-tree.md
│       ├── svm.md
│       ├── bayesian-classifier.md
│       ├── knn.md
│       ├── ensemble-learning.md
│       ├── random-forest.md
│       ├── boosting.md
│       ├── gbdt.md
│       ├── xgboost.md
│       ├── lightgbm.md
│       ├── stacking-blending.md
│       ├── neural-network-intro.md
│       ├── activation-functions.md
│       ├── loss-functions.md
│       ├── backpropagation.md
│       ├── optimization-algorithms.md
│       ├── regularization.md
│       ├── hyperparameter-tuning.md
│       ├── cnn.md
│       ├── rnn.md
│       ├── lstm-gru.md
│       ├── transformer.md
│       ├── attention-mechanism.md
│       ├── pretraining-lm.md
│       ├── fine-tuning.md
│       ├── llm-architecture.md
│       ├── prompt-engineering.md
│       ├── vae.md
│       ├── gan.md
│       ├── diffusion-models.md
│       ├── rl-introduction.md
│       ├── rl-value-based.md
│       ├── rl-policy-based.md
│       ├── rl-advanced.md
│       ├── model-compression.md
│       ├── distributed-training.md
│       └── model-deployment.md
│
└── en/tech-docs/
    ├── agent/                    # 27 篇英文 Agent 文档
    │   └── (对应中文版本的英文翻译)
    │
    └── machine-learning/         # 40 篇英文机器学习文档
        └── (对应中文版本的英文翻译)
```

---

## 四、执行计划

### 第一轮：创建中文文档
- 在 `jinsblog/source/_posts/zh-CN/技术文档/Agent/` 创建 27 篇 Agent 文档
- 在 `jinsblog/source/_posts/zh-CN/技术文档/机器学习/` 创建 40 篇机器学习文档
- 每篇文章包含完整的 frontmatter 和正文内容

### 第二轮：创建英文版本
- 在 `jinsblog/source/_posts/en/tech-docs/agent/` 创建对应的英文版本
- 在 `jinsblog/source/_posts/en/tech-docs/machine-learning/` 创建对应的英文版本
- 通过 `translation` 字段关联双语

### 第三轮：验证与生成
- 运行 `hexo generate` 生成静态文件
- 检查 `docs/` 目录输出结构
- 验证页面渲染正确

---

## 五、文档统计

| 系列 | 章节数 | 文章数 | 双语总数 |
|------|--------|--------|----------|
| AI Agent | 7 | 27 | 54 |
| 机器学习 | 11 | 40 | 80 |
| **总计** | **18** | **67** | **134** |

---

## 六、Frontmatter 格式示例

```yaml
---
title: 文章标题
date: 2026-05-10
categories: [技术文档, Agent]
lang: zh-CN
translation: /2026/05/10/en/tech-docs/agent/article-name/
tags: [Agent, LLM, 工具调用]
---
```
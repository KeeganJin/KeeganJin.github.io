---
title: 机器学习概述
date: 2025-10-12
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 入门, 概述]
---

## 机器学习定义与发展历史

### 什么是机器学习

机器学习是让计算机从数据中学习规律，而不需要显式编程的技术。

**Tom Mitchell 定义**：
"对于某类任务T和性能度量P，如果一个计算机程序在T上以P衡量的性能随着经验E而自我完善，则称该程序对T和P进行了学习。"

### 机器学习 vs 传统编程

| 传统编程 | 机器学习 |
|----------|----------|
| 输入：数据 + 规则 | 输入：数据 + 结果 |
| 输出：结果 | 输出：规则 |
| 人编写规则 | 系统学习规则 |

### 发展历史

| 时期 | 发展 |
|------|------|
| 1950s | 逻辑理论家、感知机提出 |
| 1960s | 神经网络、贝叶斯推理 |
| 1970s | 符号学习、决策树 |
| 1980s | 专家系统、反向传播算法 |
| 1990s | SVM、随机森林、Boosting |
| 2000s | 贝叶斯网络、深度学习萌芽 |
| 2010s | 深度学习爆发、AlphaGo |
| 2020s | 大模型时代、Transformer |

## 学习类型：监督、无监督、半监督、强化

### 监督学习（Supervised Learning）

有标签数据，学习输入到输出的映射。

**定义**：给定训练数据 $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$，学习函数 $f: \mathbf{x} \to y$

**任务类型**：

| 任务 | 输出类型 | 示例 |
|------|----------|------|
| 分类 | 类别标签 | 图像分类、垃圾邮件检测 |
| 回归 | 连续数值 | 房价预测、股票预测 |

**常见算法**：
- 线性回归、逻辑回归
- 决策树、随机森林
- SVM
- 神经网络

### 无监督学习（Unsupervised Learning）

无标签数据，发现数据内在结构。

**任务类型**：

| 任务 | 目标 | 示例 |
|------|------|------|
| 聚类 | 分组相似数据 | 客户分群、图像分割 |
| 降维 | 减少特征维度 | PCA、特征压缩 |
| 密度估计 | 估计数据分布 | 生成模型 |

**常见算法**：
- K-means聚类
- 层次聚类
- PCA
- 自编码器

### 半监督学习（Semi-supervised Learning）

结合少量标签数据和大量无标签数据。

**场景**：标签数据获取成本高，无标签数据易获取。

**方法**：
- 自训练（Self-training）
- 协训练（Co-training）
- 图方法

### 强化学习（Reinforcement Learning）

通过与环境交互学习最优策略。

**核心要素**：
- Agent：学习决策的智能体
- Environment：环境
- State：环境状态
- Action：Agent动作
- Reward：即时奖励

**目标**：最大化长期累积奖励。

**常见算法**：
- Q-learning
- SARSA
- Policy Gradient
- Actor-Critic

## 机器学习工作流程

### 标准流程

```
1. 数据收集
   ↓
2. 数据预处理
   ↓
3. 特征工程
   ↓
4. 模型选择
   ↓
5. 模型训练
   ↓
6. 模型评估
   ↓
7. 模型优化
   ↓
8. 模型部署
```

### 各阶段详解

#### 1. 数据收集

确定数据需求，收集相关数据：
- 内部数据源
- 外部数据源
- 公开数据集
- 实时数据流

#### 2. 数据预处理

清洗和准备数据：
- 缺失值处理
- 异常值检测
- 数据标准化
- 数据编码

#### 3. 特征工程

构建有效的特征：
- 特征选择
- 特征提取
- 特征构造
- 降维

#### 4. 模型选择

根据任务选择合适模型：
- 任务类型（分类/回归/聚类）
- 数据规模
- 计算资源
- 性能要求

#### 5. 模型训练

使用训练数据拟合模型：
- 参数初始化
- 优化算法选择
- 训练过程监控
- 超参数调整

#### 6. 模型评估

评估模型性能：
- 验证集评估
- 测试集评估
- 交叉验证
- 多指标评估

#### 7. 模型优化

改进模型性能：
- 超参数调优
- 模型结构调整
- 数据增强
- 集成方法

#### 8. 模型部署

将模型应用于实际：
- 模型打包
- 服务部署
- 监控维护
- 版本管理

## 模型评估指标

### 分类评估指标

#### 准确率（Accuracy）

正确预测的比例：
$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$

#### 精确率（Precision）

预测为正例中实际为正例的比例：
$Precision = \frac{TP}{TP + FP}$

#### 召回率（Recall）

实际正例中被预测为正例的比例：
$Recall = \frac{TP}{TP + FN}$

#### F1分数

精确率和召回率的调和平均：
$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

#### 混淆矩阵

|  | 预测正 | 预测负 |
|---|--------|--------|
| 实际正 | TP | FN |
| 实际负 | FP | TN |

#### ROC曲线和AUC

ROC曲线：不同阈值下的TPR vs FPR
AUC：ROC曲线下的面积

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score

y_true = [0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 0, 1, 0, 0]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
```

### 回归评估指标

#### 均方误差（MSE）

$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

#### 均方根误差（RMSE）

$RMSE = \sqrt{MSE}$

#### 平均绝对误差（MAE）

$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

#### R²分数

$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

## 过拟合与欠拟合

### 过拟合（Overfitting）

模型在训练数据上表现很好，但在测试数据上表现差。

**原因**：
- 模型过于复杂
- 训练数据太少
- 训练时间过长

**表现**：
- 训练误差很小
- 测试误差很大
- 训练-测试误差差距大

**解决方案**：
- 增加数据
- 简化模型
- 正则化
- 早停
- 交叉验证

### 欠拟合（Underfitting）

模型在训练数据和测试数据上都表现差。

**原因**：
- 模型过于简单
- 特征不足
- 训练时间不足

**表现**：
- 训练误差很大
- 测试误差也很大

**解决方案**：
- 增加模型复杂度
- 添加更多特征
- 增加训练时间
- 减少正则化强度

### 模型复杂度与误差关系

```
误差
  │
  │  训练误差（持续下降）
  │  ↘
  │   ↘
  │    ↘──────────
  │
  │  测试误差（先降后升）
  │  ↘    ↗
  │   ↘  ↗
  │    ↘↗
  │     ↗────────
  │
  └────────────────── 模型复杂度
        最优复杂度
```

## 机器学习应用领域

### 主要应用领域

| 领域 | 应用 |
|------|------|
| 计算机视觉 | 图像分类、目标检测、人脸识别 |
| 自然语言处理 | 文本分类、机器翻译、对话系统 |
| 推荐系统 | 商品推荐、内容推荐 |
| 金融 | 信用评分、风险预测、量化交易 |
| 医疗 | 诊断辅助、药物发现 |
| 自动驾驶 | 环境感知、路径规划 |
| 游戏 | AI对手、策略优化 |
| 工业 | 预测性维护、质量控制 |

## 机器学习工具与框架

### 主要框架

| 框架 | 特点 |
|------|------|
| Scikit-learn | Python经典机器学习库 |
| TensorFlow | Google深度学习框架 |
| PyTorch | Facebook深度学习框架 |
| Keras | 高级深度学习API |
| XGBoost | 高性能梯度提升框架 |
| LightGBM | 微软梯度提升框架 |

### 开发环境

```python
# 常用开发环境示例
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 数据加载
data = pd.read_csv('data.csv')
X = data.drop('label', axis=1)
y = data['label']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 数据预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 模型训练
model = LogisticRegression()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

## 总结

机器学习是让计算机从数据中自动学习的技术。主要学习类型包括：监督学习（有标签）、无监督学习（无标签）、半监督学习（混合）、强化学习（交互学习）。

机器学习工作流程包括：数据收集、预处理、特征工程、模型选择、训练、评估、优化、部署。关键概念是过拟合与欠拟合，需要在模型复杂度和泛化能力之间取得平衡。

## 延伸阅读

- [线性代数基础](/2026/05/10/zh-CN/技术文档/机器学习/linear-algebra/)
- [概率论基础](/2026/05/10/zh-CN/技术文档/机器学习/probability-theory/)
- [数据预处理](/2026/05/10/zh-CN/技术文档/机器学习/data-preprocessing/)
- [特征工程](/2026/05/10/zh-CN/技术文档/机器学习/feature-engineering/)
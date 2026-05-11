---
title: 集成学习概述
date: 2026-04-20
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 集成学习, Boosting]
---

## 集成学习核心思想

### 什么是集成学习

集成学习通过组合多个模型来提升整体性能。

**核心思想**：多个模型的组合往往比单个模型表现更好。

**类比**：就像专家团队比单个专家更能做出好的决策。

### 集成学习的基本框架

```
基学习器 → 集成策略 → 最终预测
   ↑
 数据/特征处理
```

**组成要素**：
1. **基学习器**：单个模型
2. **集成策略**：组合方式
3. **多样性来源**：如何保证模型差异

### 集成学习的优势

| 优势 | 描述 |
|------|------|
| 提升准确率 | 组合往往优于单个模型 |
| 提高稳定性 | 减少模型方差 |
| 减少过拟合 | 多模型平衡决策 |
| 处理复杂问题 | 不同模型捕捉不同模式 |

## 偏差与方差分解

### 偏差-方差分解

模型误差可以分解为：

$E[(y - \hat{f}(\mathbf{x}))^2] = Bias^2 + Variance + Irreducible Error$

**各部分含义**：

| 组成 | 描述 |
|------|------|
| 偏差 | 预测值与真实值的系统偏差 |
| 方差 | 模型对数据变化的敏感度 |
| 不可约误差 | 数据本身的噪声 |

### 偏差与方差的权衡

**不同模型的偏差-方差特征**：

| 模型复杂度 | 偏差 | 方差 |
|------------|------|------|
| 简单模型 | 高 | 低 |
| 复杂模型 | 低 | 高 |

### 集成学习如何降低误差

| 集成方法 | 主要作用 |
|----------|----------|
| Bagging | 降低方差 |
| Boosting | 降低偏差和方差 |
| Stacking | 综合降低 |

### Bagging降低方差

**原理**：
- 多个模型的平均预测方差小于单个模型
- 独立模型方差：$\sigma^2$
- K个独立模型平均方差：$\sigma^2/K$

**实际中**模型不完全独立，方差降低幅度有限。

### Boosting降低偏差

**原理**：
- 逐步修正模型错误
- 每个新模型专注于前序模型的不足
- 最终模型组合更准确

## 集成策略分类

### 并行集成（Bagging）

**特点**：
- 多个模型独立训练
- 可以并行计算
- 主要降低方差

**代表**：
- Bagging
- 随机森林

```
训练数据 → 模型1 → 预测1
   ↓      → 模型2 → 预测2    → 组合 → 最终预测
   ↓      → 模型3 → 预测3
```

### 串行集成（Boosting）

**特点**：
- 模型按顺序训练
- 后序模型依赖前序模型
- 主要降低偏差

**代表**：
- AdaBoost
- GBDT
- XGBoost

```
训练数据 → 模型1 → 残差 → 模型2 → 残差 → 模型3 → 组合 → 最终预测
```

### 混合集成（Stacking）

**特点**：
- 使用不同类型的基学习器
- 元模型组合基学习器预测
- 层叠结构

```
训练数据 → 模型A → 预测A → 
          → 模型B → 预测B → 元模型 → 最终预测
          → 模型C → 预测C →
```

### 集成策略对比

| 方面 | Bagging | Boosting | Stacking |
|------|---------|----------|----------|
| 训练方式 | 并行 | 串行 | 两阶段 |
| 主要作用 | 降低方差 | 降低偏差 | 综合优化 |
| 基学习器 | 同类 | 同类 | 可不同类 |
| 计算复杂度 | 低 | 高 | 中 |

## 集成学习的优势

### 为什么集成学习有效

**多样性原则**：
- 模型之间要有差异
- 不同模型犯不同的错误
- 组合可以互补

**统计原因**：
- 训练数据有限，单个模型可能陷入局部最优
- 多个模型从不同角度学习

**计算原因**：
- 搜索空间大，单个模型难以找到最优
- 多个模型可以覆盖更多搜索空间

**表示原因**：
- 单个模型的表示能力有限
- 组合可以扩展表示能力

### 多样性来源

| 来源 | 方法 |
|------|------|
| 数据扰动 | Bootstrap采样、交叉验证 |
| 特征扰动 | 特征子集选择 |
| 算法扰动 | 不同算法、不同参数 |
| 输出扰动 | 输出编码变换 |

### 强弱学习器

**弱学习器**：性能略好于随机猜测
**强学习器**：性能明显优于随机猜测

**Boosting理论**：
- 多个弱学习器可以组合成强学习器
- 这是Boosting算法的理论基础

## 集成学习的关键问题

### 基学习器的选择

| 考虑因素 | 建议 |
|----------|------|
| 算法类型 | 决策树是常用选择 |
| 模型复杂度 | Bagging选复杂模型，Boosting选简单模型 |
| 稳定性 | 不稳定模型（如决策树）适合Bagging |
| 计算成本 | 考虑训练和预测效率 |

### 集成规模

**多少个基学习器**？

| 集成方法 | 建议 |
|----------|------|
| Bagging | 通常50-500个 |
| Boosting | 通常100-1000个 |
| Stacking | 通常3-10个不同类型 |

**过多的问题**：
- 计算成本增加
- 可能边际效益递减
- 可能增加方差（Boosting）

### 组合策略

**组合方式**：

| 方法 | 描述 |
|------|------|
| 平均 | 回归任务，预测值平均 |
| 投票 | 分类任务，多数投票 |
| 加权平均 | 根据模型性能加权 |
| 加权投票 | 根据模型性能加权投票 |
| 学习 | 元模型学习组合（Stacking） |

```python
# 平均组合
y_pred = np.mean(predictions, axis=0)

# 投票组合
from scipy.stats import mode
y_pred = mode(predictions, axis=0)[0]

# 加权平均
weights = [0.3, 0.5, 0.2]  # 根据各模型性能设定
y_pred = np.average(predictions, axis=0, weights=weights)
```

## Python集成学习框架

### sklearn中的集成学习

```python
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

# Bagging
bagging = BaggingClassifier(n_estimators=100)

# 随机森林
rf = RandomForestClassifier(n_estimators=100)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=50)

# GBDT
gbdt = GradientBoostingClassifier(n_estimators=100)

# 投票集成
voting = VotingClassifier(
    estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier())],
    voting='hard'  # 或 'soft'
)

# Stacking
stacking = StackingClassifier(
    estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier())],
    final_estimator=LogisticRegression()
)
```

### 简单集成示例

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# 数据
X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 多个模型
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier()
}

# 训练和预测
predictions = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    print(f"{name}: {accuracy_score(y_test, y_pred):.4f}")

# 投票集成
y_pred_vote = np.mean(predictions, axis=0) >= 0.5  # 简单投票
print(f"\n投票集成: {accuracy_score(y_test, y_pred_vote):.4f}")
```

## 总结

集成学习通过组合多个模型提升性能。核心内容包括：
- 集成学习思想：组合优于单个模型
- 偏差-方差分解：Bagging降低方差，Boosting降低偏差
- 集成策略：并行（Bagging）、串行（Boosting）、混合（Stacking）
- 多样性来源：数据、特征、算法扰动
- 组合策略：平均、投票、学习

集成学习是提升模型性能的重要技术。

## 延伸阅读

- [决策树](/2026/05/10/zh-CN/技术文档/机器学习/decision-tree/)
- [随机森林](/2026/05/10/zh-CN/技术文档/机器学习/random-forest/)
- [Boosting基础](/2026/05/10/zh-CN/技术文档/机器学习/boosting/)
- [GBDT梯度提升树](/2026/05/10/zh-CN/技术文档/机器学习/gbdt/)
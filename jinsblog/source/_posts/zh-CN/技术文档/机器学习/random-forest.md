---
title: 随机森林
date: 2026-04-12
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 集成学习, 随机森林]
---

## Bagging原理

### Bootstrap采样

Bootstrap是一种重采样技术：

**过程**：
从n个样本中有放回地随机抽取n个样本（允许重复）。

**性质**：
- 每个样本被抽中的概率：$\approx 63.2\%$
- 每个样本未被抽中的概率：$\approx 36.8\%$

```python
import numpy as np

def bootstrap_sample(X, y):
    """Bootstrap采样"""
    n = len(X)
    indices = np.random.choice(n, size=n, replace=True)
    return X[indices], y[indices]
```

### Bagging算法

Bagging（Bootstrap Aggregating）：

**流程**：
```
1. 对训练数据进行Bootstrap采样，生成多个子集
2. 在每个子集上训练一个基学习器
3. 对所有基学习器的预测进行平均/投票
```

**降低方差**：
- 假设K个独立模型的方差为 $\sigma^2$
- 平均后方差为 $\sigma^2/K$
- 实际模型不完全独立，方差降低有限

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=1.0,  # Bootstrap样本数
    bootstrap=True,
    random_state=42
)
bagging.fit(X_train, y_train)
```

## 随机森林算法

### 随机森林的提出

随机森林是Bagging的扩展，由Leo Breiman提出。

**关键创新**：在Bagging的基础上增加了特征随机选择。

### 随机森林算法流程

```
for i in 1 to K:
    1. Bootstrap采样，生成训练子集
    2. 构建决策树：
       - 在每个节点，随机选择m个特征
       - 从m个特征中选择最优分裂
       - 不剪枝，完全生长
3. 组合K棵树的预测（投票/平均）
```

### 特征随机选择

**随机特征数m的选择**：
- 分类：$m = \sqrt{p}$（p为总特征数）
- 回归：$m = p/3$

**作用**：
- 增加树之间的多样性
- 降低相关性
- 提高方差降低效果

### 随机森林 vs Bagging

| 方面 | Bagging | 随机森林 |
|------|---------|----------|
| 数据扰动 | Bootstrap | Bootstrap |
| 特征扰动 | 无 | 随机选择特征 |
| 树相关性 | 较高 | 较低 |
| 性能 | 较好 | 更好 |

### 随机森林的工作原理

**双随机性**：
1. **样本随机**：Bootstrap采样
2. **特征随机**：每个节点随机选择特征

**效果**：
- 每棵树看到不同的数据和特征组合
- 树之间差异更大
- 组合效果更好

```python
from sklearn.ensemble import RandomForestClassifier

# 随机森林
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # 特征随机选择
    bootstrap=True,
    random_state=42
)
rf.fit(X_train, y_train)
```

## 特征随机选择

### max_features参数

| 参数值 | 含义 |
|--------|------|
| 'sqrt' | $\sqrt{p}$ 个特征（分类默认） |
| 'log2' | $\log_2(p)$ 个特征 |
| 'auto' | 同 'sqrt' |
| 整数 | 指定特征数 |
| 浮点数 | 特征比例 |
| None | 使用所有特征 |

```python
# 不同特征数的效果
for max_features in ['sqrt', 'log2', 0.5, None]:
    rf = RandomForestClassifier(n_estimators=100, max_features=max_features)
    rf.fit(X_train, y_train)
    print(f"max_features={max_features}: 准确率={rf.score(X_test, y_test):.4f}")
```

### 特征随机的作用

| 作用 | 描述 |
|------|------|
| 降低相关性 | 树之间更独立 |
| 增加多样性 | 不同树关注不同特征 |
| 提高效率 | 每次分裂只考虑部分特征 |
| 减少过拟合 | 防止某些特征主导 |

## 袋外数据（OOB）评估

### OOB数据

Bootstrap采样中未被抽中的样本（约36.8%）可以作为验证数据。

**OOB评估**：
- 对每棵树，用OOB样本评估该树
- 对每个样本，用未包含该样本的树进行预测
- 组合预测，计算OOB误差

### OOB的优势

| 优势 | 描述 |
|------|------|
| 无需额外验证集 | 利用采样留下的数据 |
| 无偏估计 | 等效于K折交叉验证 |
| 高效 | 不需要额外计算 |

```python
# 使用OOB评估
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # 启用OOB评估
    random_state=42
)
rf.fit(X_train, y_train)

print(f"OOB分数: {rf.oob_score_:.4f}")
print(f"测试分数: {rf.score(X_test, y_test):.4f}")
```

### OOB用于参数选择

```python
# 用OOB选择最优参数
n_estimators_range = [10, 50, 100, 200]
oob_scores = []

for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)
    print(f"n_estimators={n}: OOB分数={rf.oob_score_:.4f}")

best_n = n_estimators_range[np.argmax(oob_scores)]
print(f"\n最优n_estimators: {best_n}")
```

## 随机森林的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 高准确率 | 通常比单个决策树更准确 |
| 处理高维数据 | 不需要特征选择 |
| 处理缺失值 | 内置缺失值处理 |
| 特征重要性 | 自动计算特征重要性 |
| 不易过拟合 | 多树组合平衡决策 |
| 并行训练 | 各树独立训练 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 计算成本 | 多棵树训练和预测 |
| 内存占用 | 存储多棵树 |
| 不够可解释 | 比单棵树更难解释 |
| 实时预测慢 | 需遍历所有树 |

### 与其他算法对比

| 方面 | 决策树 | 随机森林 | GBDT |
|------|--------|----------|------|
| 训练速度 | 快 | 中 | 慢 |
| 预测速度 | 快 | 中 | 中 |
| 准确率 | 中 | 高 | 高 |
| 可解释性 | 高 | 低 | 低 |
| 过拟合风险 | 高 | 低 | 中 |

## 案例实践

### 分类随机森林

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 数据
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB分数: {rf.oob_score_:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))
```

### 回归随机森林

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 回归数据
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林回归
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_features=1/3,
    oob_score=True,
    random_state=42
)
rf_reg.fit(X_train, y_train)

# 预测
y_pred = rf_reg.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"OOB分数: {rf_reg.oob_score_:.4f}")
```

### 特征重要性分析

```python
import matplotlib.pyplot as plt

# 特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("特征重要性排名:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {iris.feature_names[idx]}: {importances[idx]:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [iris.feature_names[i] for i in indices])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()
```

### 参数调优

```python
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2', 0.5],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# 网格搜索
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"最优参数: {grid_search.best_params_}")
print(f"最优分数: {grid_search.best_score_:.4f}")

# 使用最优参数
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print(f"测试准确率: {accuracy_score(y_test, y_pred):.4f}")
```

### 单棵树 vs 随机森林

```python
from sklearn.tree import DecisionTreeClassifier

# 单棵决策树
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# 随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("单棵决策树:")
print(f"  训练准确率: {dt.score(X_train, y_train):.4f}")
print(f"  测试准确率: {dt.score(X_test, y_test):.4f}")

print("\n随机森林:")
print(f"  训练准确率: {rf.score(X_train, y_train):.4f}")
print(f"  测试准确率: {rf.score(X_test, y_test):.4f}")
```

## 总结

随机森林是基于Bagging的集成学习算法。核心内容包括：
- Bagging原理：Bootstrap采样降低方差
- 随机森林算法：增加特征随机选择
- 特征随机选择：增加树之间的多样性
- OOB评估：利用未采样数据进行模型评估
- 特征重要性：自动计算特征重要性

随机森林简单高效，是常用的集成学习方法。

## 延伸阅读

- [集成学习概述](/2026/05/10/zh-CN/技术文档/机器学习/ensemble-learning/)
- [决策树](/2026/05/10/zh-CN/技术文档/机器学习/decision-tree/)
- [Boosting基础](/2026/05/10/zh-CN/技术文档/机器学习/boosting/)
- [XGBoost算法详解](/2026/05/10/zh-CN/技术文档/机器学习/xgboost/)
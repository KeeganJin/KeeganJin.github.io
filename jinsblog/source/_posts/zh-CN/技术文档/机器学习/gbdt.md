---
title: GBDT梯度提升树
date: 2025-11-19
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 集成学习, GBDT]
---

## GBDT基本原理

### GBDT的概念

GBDT（Gradient Boosting Decision Tree）是一种基于梯度提升的决策树集成算法。

**核心思想**：
- 每棵树拟合前序模型的残差（负梯度）
- 逐步迭代，累积预测
- 最终模型是所有树的加和

### GBDT与AdaBoost的区别

| 方面 | AdaBoost | GBDT |
|------|----------|------|
| 损失函数 | 指数损失 | 可选择任意损失 |
| 样本权重 | 调整权重 | 不调整权重 |
| 新树目标 | 分类错误的样本 | 拟合负梯度 |
| 基学习器 | 决策树桩 | 一般决策树 |

### GBDT的数学框架

**加法模型**：
$F_M(\mathbf{x}) = \sum_{m=1}^{M} f_m(\mathbf{x})$

**目标**：最小化损失函数
$\min \sum_{i=1}^{n} L(y_i, F_M(\mathbf{x}_i))$

### 前向分步算法

```
初始化 F_0(x) = argmin_c Σ L(y_i, c)

for m = 1 to M:
    计算负梯度（伪残差）:
        r_{mi} = -∂L(y_i, F_{m-1}(x_i))/∂F_{m-1}(x_i)
    
    用 (x_i, r_{mi}) 训练回归树 f_m(x)
    
    计算叶节点最优值
    
    更新模型:
        F_m(x) = F_{m-1}(x) + f_m(x)
```

## 损失函数选择

### 常见损失函数

| 任务 | 损失函数 | 负梯度 |
|------|----------|--------|
| 回归 | MSE | $y - F(x)$ |
| 回归 | MAE | sign(y - F(x)) |
| 回归 | Huber | 见下文 |
| 分类 | 对数损失 | 见下文 |

### MSE损失

$L(y, F) = \frac{1}{2}(y - F)^2$

**负梯度（伪残差）**：
$r = y - F$

即拟合残差本身。

### 对数损失（分类）

$L(y, F) = -y\ln p - (1-y)\ln(1-p)$

其中 $p = \sigma(F) = \frac{1}{1+e^{-F}}$

**负梯度**：
$r = y - p$

### Huber损失

结合MSE和MAE的优点：

$L(y, F) = \begin{cases} \frac{1}{2}(y-F)^2 & |y-F| \leq \delta \\ \delta|y-F| - \frac{1}{2}\delta^2 & |y-F| > \delta \end{cases}$

**特点**：对异常值不敏感。

```python
from sklearn.ensemble import GradientBoostingRegressor

# 不同损失函数
gbdt_mse = GradientBoostingRegressor(loss='squared_error')
gbdt_mae = GradientBoostingRegressor(loss='absolute_error')
gbdt_huber = GradientBoostingRegressor(loss='huber')
```

## 负梯度拟合

### 为什么叫"梯度提升"

GBDT拟合的是损失函数的负梯度方向，相当于沿梯度方向下降。

**负梯度的含义**：
- 表示当前模型的"不足"
- 新树的目标是弥补这个"不足"

### 伪残差

负梯度称为"伪残差"（pseudo-residual）：

$r_{mi} = -\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}$

对于MSE损失：
$r_{mi} = y_i - F_{m-1}(\mathbf{x}_i)$

即真实残差。

### 负梯度拟合过程

1. 计算当前模型预测的负梯度
2. 用负梯度作为目标值训练新树
3. 新树的叶节点输出最优值
4. 将新树加入模型

## GBDT回归与分类

### GBDT回归

**算法流程**：
```
1. 初始化 F_0(x) = mean(y)
2. for m = 1 to M:
   a. 计算残差 r_i = y_i - F_{m-1}(x_i)
   b. 用残差训练回归树 f_m(x)
   c. 计算叶节点均值作为输出
   d. F_m(x) = F_{m-1}(x) + η f_m(x)  (η是学习率)
```

```python
from sklearn.ensemble import GradientBoostingRegressor

# GBDT回归
gbdt_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbdt_reg.fit(X_train, y_train)

y_pred = gbdt_reg.predict(X_test)
```

### GBDT分类

**二分类**：
- 使用对数损失
- 初始化 F_0(x) = ln(p/(1-p))（对数几率）
- 叶节点输出需要经过sigmoid

**多分类**：
- 构建K组树（K为类别数）
- 每组树预测一个类别的得分
- 使用softmax输出概率

```python
from sklearn.ensemble import GradientBoostingClassifier

# GBDT分类
gbdt_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbdt_clf.fit(X_train, y_train)

y_pred = gbdt_clf.predict(X_test)
y_prob = gbdt_clf.predict_proba(X_test)
```

### 叶节点最优值

对于回归树叶节点 $R_j$：

**MSE损失下最优值**：
$c_j = \frac{\sum_{i \in R_j} r_{mi}}{|R_j|}$

即叶节点样本残差的均值。

## GBDT的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 高精度 | 通常比随机森林更准确 |
| 灵活 | 可选择多种损失函数 |
| 自动特征选择 | 内置特征重要性 |
| 可解释性 | 可分析每棵树的贡献 |
| 处理混合数据 | 可处理数值和类别特征 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 训练慢 | 串行训练，无法并行 |
| 调参复杂 | 多个超参数需要调优 |
| 易过拟合 | 需要控制树深度和数量 |
| 对异常值敏感 | MSE损失敏感 |

### 与随机森林对比

| 方面 | GBDT | 随机森林 |
|------|------|----------|
| 训练方式 | 串行 | 并行 |
| 基学习器 | 回归树 | 分类/回归树 |
| 组合方式 | 加和 | 平均/投票 |
| 计算效率 | 低 | 高 |
| 准确率 | 通常更高 | 较高 |
| 过拟合风险 | 较高 | 较低 |

## 案例实践

### GBDT回归示例

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 数据
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GBDT回归
gbdt_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=10,
    random_state=42
)
gbdt_reg.fit(X_train, y_train)

# 预测
y_pred = gbdt_reg.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# 特征重要性
importances = gbdt_reg.feature_importances_
for name, imp in zip(data.feature_names, importances):
    print(f"{name}: {imp:.4f}")
```

### GBDT分类示例

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 数据
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GBDT分类
gbdt_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbdt_clf.fit(X_train, y_train)

# 预测
y_pred = gbdt_clf.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))
```

### 学习曲线分析

```python
import matplotlib.pyplot as plt

# 记录训练过程
train_scores = []
test_scores = []

for i, y_pred_train in enumerate(gbdt_reg.staged_predict(X_train)):
    train_scores.append(r2_score(y_train, y_pred_train))

for i, y_pred_test in enumerate(gbdt_reg.staged_predict(X_test)):
    test_scores.append(r2_score(y_test, y_pred_test))

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_scores)), train_scores, label='Train')
plt.plot(range(len(test_scores)), test_scores, label='Test')
plt.xlabel('Number of Trees')
plt.ylabel('R²')
plt.legend()
plt.title('GBDT Learning Curve')
plt.show()
```

### 参数调优

```python
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [2, 3, 5],
    'min_samples_split': [2, 5, 10]
}

# 网格搜索
grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2'
)
grid_search.fit(X_train, y_train)

print(f"最优参数: {grid_search.best_params_}")
print(f"最优R²: {grid_search.best_score_:.4f}")

# 使用最优参数
best_gbdt = grid_search.best_estimator_
y_pred = best_gbdt.predict(X_test)
print(f"测试R²: {r2_score(y_test, y_pred):.4f}")
```

### 早停策略

```python
# 早停防止过拟合
gbdt_early = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    validation_fraction=0.2,  # 验证比例
    n_iter_no_change=10,  # 无改进容忍次数
    tol=0.01,  # 改进阈值
    random_state=42
)
gbdt_early.fit(X_train, y_train)

print(f"实际使用树数: {gbdt_early.n_estimators_}")
```

## 总结

GBDT是基于梯度提升的决策树集成算法。核心内容包括：
- GBDT基本原理：拟合负梯度，逐步迭代
- 损失函数选择：MSE、MAE、Huber、对数损失
- 负梯度拟合：沿梯度方向下降
- GBDT回归与分类：不同任务的实现差异
- 参数调优：学习率、树深度、迭代次数

GBDT是XGBoost、LightGBM等算法的基础。

## 延伸阅读

- [Boosting基础](/2026/05/10/zh-CN/技术文档/机器学习/boosting/)
- [决策树](/2026/05/10/zh-CN/技术文档/机器学习/decision-tree/)
- [XGBoost算法详解](/2026/05/10/zh-CN/技术文档/机器学习/xgboost/)
- [LightGBM算法详解](/2026/05/10/zh-CN/技术文档/机器学习/lightgbm/)
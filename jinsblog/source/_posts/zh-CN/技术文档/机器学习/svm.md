---
title: 支持向量机（SVM）
date: 2026-04-21
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, SVM, 分类算法]
---

## SVM基本原理

### 支持向量机概念

支持向量机（Support Vector Machine, SVM）是一种寻找最优分类超平面的算法。

**核心思想**：找到使两类样本间隔最大的分类超平面。

**超平面方程**：
$\mathbf{w}^T\mathbf{x} + b = 0$

### SVM的几何理解

**决策边界**：超平面 $\mathbf{w}^T\mathbf{x} + b = 0$

**间隔边界**：
- 正类边界：$\mathbf{w}^T\mathbf{x} + b = 1$
- 负类边界：$\mathbf{w}^T\mathbf{x} + b = -1$

**间隔宽度**：
$\text{margin} = \frac{2}{\|\mathbf{w}\|}$

### SVM的目标

最大化间隔：
$\max_{\mathbf{w}, b} \frac{2}{\|\mathbf{w}\|}$

等价于：
$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$

约束条件：
$y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, ..., n$

## 最大间隔与支持向量

### 支持向量

支持向量是距离超平面最近的样本点，满足：
$y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$

**特点**：
- 支持向量决定了超平面
- 移除其他样本不影响超平面
- 支持向量数量通常较少

### 最大间隔分类器的推导

**原问题**：
$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$
$\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$

**拉格朗日函数**：
$L(\mathbf{w}, b, \alpha) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^{n}\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1]$

**KKT条件**：
$\nabla_\mathbf{w} L = \mathbf{w} - \sum_i \alpha_i y_i \mathbf{x}_i = 0$
$\nabla_b L = -\sum_i \alpha_i y_i = 0$

得到：
$\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$

### 对偶问题

**对偶形式**：
$\max_\alpha \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$
$\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$

**解的性质**：
- 大多数 $\alpha_i = 0$（非支持向量）
- $\alpha_i > 0$ 对应支持向量

### 分类决策函数

$f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b = \sum_{i=1}^{n}\alpha_i y_i \mathbf{x}_i^T \mathbf{x} + b$

预测：
$\hat{y} = \text{sign}(f(\mathbf{x}))$

## 核函数详解

### 线性不可分问题

当数据线性不可分时，可以通过核函数将数据映射到高维空间，使其线性可分。

**映射函数**：$\phi(\mathbf{x})$

**问题**：直接计算 $\phi(\mathbf{x})$ 可能计算成本很高。

### 核技巧

核函数避免了显式计算映射：

$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$

**决策函数变为**：
$f(\mathbf{x}) = \sum_{i=1}^{n}\alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$

### 常见核函数

| 核函数 | 公式 | 特点 |
|--------|------|------|
| 线性核 | $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z}$ | 线性可分数据 |
| 多项式核 | $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^d$ | 非线性边界 |
| RBF核（高斯核） | $K(\mathbf{x}, \mathbf{z}) = e^{-\gamma\|\mathbf{x}-\mathbf{z}\|^2}$ | 最常用，灵活 |
| Sigmoid核 | $K(\mathbf{x}, \mathbf{z}) = \tanh(\alpha\mathbf{x}^T\mathbf{z} + c)$ | 类神经网络 |

### RBF核详解

$K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma\|\mathbf{x}-\mathbf{z}\|^2\right)$

**参数 $\gamma$**：
- $\gamma$ 大：每个样本影响范围小，决策边界复杂
- $\gamma$ 小：每个样本影响范围大，决策边界平滑

```python
from sklearn.svm import SVC

# 不同核函数的SVM
svm_linear = SVC(kernel='linear')
svm_poly = SVC(kernel='poly', degree=3)
svm_rbf = SVC(kernel='rbf', gamma=0.1)
```

### 核函数的选择指南

| 数据特点 | 推荐核函数 |
|----------|------------|
| 线性可分 | 线性核 |
| 低维非线性 | 多项式核 |
| 高维/未知 | RBF核 |
| 样本少特征多 | 线性核 |
| 样本多特征少 | RBF核 |

## 软间隔与松弛变量

### 软间隔问题

当数据存在噪声或无法完全分开时，使用软间隔允许一些样本错误分类。

**引入松弛变量 $\xi_i$**：
$y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i$
$\xi_i \geq 0$

### 目标函数

$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i$

其中 $C$ 是惩罚参数：
- $C$ 大：更严格要求正确分类（可能过拟合）
- $C$ 小：允许更多错误分类（更平滑边界）

### 对偶问题（软间隔）

$\max_\alpha \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$
$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$

```python
# 不同C值的对比
for C in [0.1, 1.0, 10.0, 100.0]:
    svm = SVC(kernel='rbf', C=C)
    svm.fit(X_train, y_train)
    print(f"C={C}: 训练准确率={svm.score(X_train, y_train):.4f}, 测试准确率={svm.score(X_test, y_test):.4f}")
```

## SMO算法

### SMO算法原理

SMO（Sequential Minimal Optimization）是求解SVM对偶问题的高效算法。

**核心思想**：
- 每次只优化两个变量 $\alpha_i$ 和 $\alpha_j$
- 其他变量固定
- 两个变量有解析解

### SMO算法流程

```
1. 选择两个变量 α_i, α_j（启发式选择）
2. 计算这两个变量的最优值
3. 更新 α_i, α_j
4. 更新偏置 b
5. 检查收敛条件
6. 重复直到所有变量满足KKT条件
```

### 变量选择策略

**第一个变量选择**：
- 选择违反KKT条件最严重的变量

**第二个变量选择**：
- 选择使目标函数增长最大的变量
- 通常选择与第一个变量对应样本距离最大的

### SMO算法的优势

| 优势 | 描述 |
|------|------|
| 解析解 | 两个变量问题有精确解 |
| 无需矩阵计算 | 避免大规模矩阵运算 |
| 收敛快 | 实践中收敛效率高 |
| 易实现 | 算法逻辑简单 |

## 多分类SVM

### 一对多（One-vs-Rest）

训练K个二分类SVM，每个区分一个类别与其他类别。

**预测**：选择决策函数值最大的类别。

```python
from sklearn.svm import SVC

# sklearn默认使用OvR
svm = SVC(kernel='rbf', decision_function_shape='ovr')
svm.fit(X_train, y_train)
```

### 一对一（One-vs-One）

训练 $\frac{K(K-1)}{2}$ 个二分类SVM，每个区分两个类别。

**预测**：投票决定最终类别。

```python
# OvO模式
svm = SVC(kernel='rbf', decision_function_shape='ovo')
svm.fit(X_train, y_train)
```

### 方法比较

| 方法 | 分类器数量 | 训练时间 | 预测时间 |
|------|------------|----------|----------|
| OvR | K | 较慢 | 快 |
| OvO | K(K-1)/2 | 快 | 较慢 |

## SVR（支持向量回归）

### SVR原理

支持向量回归是SVM的回归版本。

**目标**：找到使预测误差在阈值 $\epsilon$ 内的超平面。

**约束**：
$|y_i - \mathbf{w}^T\mathbf{x}_i - b| \leq \epsilon$

**目标函数**：
$\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i (\xi_i + \xi_i^*)$

### $\epsilon$-不敏感损失

当误差小于 $\epsilon$ 时损失为0：
$\ell(y, \hat{y}) = \max(0, |y - \hat{y}| - \epsilon)$

```python
from sklearn.svm import SVR

# SVR回归
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)
```

### SVR参数

| 参数 | 描述 |
|------|------|
| C | 惩罚系数，控制拟合程度 |
| epsilon | 不敏感区域宽度 |
| kernel | 核函数类型 |

## 案例实践

### 二分类SVM示例

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 生成数据
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                            n_clusters_per_class=1, random_state=42)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（SVM对尺度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)

# 预测
y_pred = svm.predict(X_test_scaled)

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 支持向量信息
print(f"\n支持向量数量: {len(svm.support_vectors_)}")
print(f"支持向量占比: {len(svm.support_vectors_) / len(X_train):.2%}")
```

### 决策边界可视化

```python
import matplotlib.pyplot as plt

def plot_svm_decision_boundary(X, y, svm, scaler=None):
    """绘制SVM决策边界"""
    if scaler:
        X = scaler.transform(X)
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    
    # 绘制支持向量
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='r', label='Support Vectors')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('SVM Decision Boundary')
    plt.show()

plot_svm_decision_boundary(X, y, svm, scaler)
```

### 参数调优

```python
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['rbf', 'poly', 'linear']
}

# 网格搜索
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"最优参数: {grid_search.best_params_}")
print(f"最优得分: {grid_search.best_score_:.4f}")

# 使用最优模型
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
print(f"测试准确率: {accuracy_score(y_test, y_pred):.4f}")
```

### SVR回归示例

```python
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# 加载回归数据
data = fetch_california_housing()
X, y = data.data[:500], data.target[:500]  # 取部分数据

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# SVR训练
svr = SVR(kernel='rbf', C=10, epsilon=0.1)
svr.fit(X_train_scaled, y_train_scaled)

# 预测
y_pred_scaled = svr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# 评估
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

## SVM的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 泛化能力强 | 最大间隔原则防止过拟合 |
| 高维有效 | 在高维空间表现良好 |
| 核函数灵活 | 可处理非线性问题 |
| 理论基础扎实 | 有完善的数学理论 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 大数据慢 | 训练复杂度O(n²)到O(n³) |
| 参数敏感 | C、γ等参数需要调优 |
| 核函数选择 | 需要经验选择核函数 |
| 不直接输出概率 | 需要额外处理输出概率 |

## 总结

支持向量机是强大的分类和回归模型。核心内容包括：
- SVM基本原理：寻找最大间隔分类超平面
- 支持向量：决定超平面的关键样本
- 核函数：将非线性问题映射到高维空间
- 软间隔：允许一定程度的错误分类
- SMO算法：高效求解对偶问题
- SVR：支持向量回归版本

SVM在中小规模数据上表现优秀，是经典的机器学习算法。

## 延伸阅读

- [线性代数基础](/2026/05/10/zh-CN/技术文档/机器学习/linear-algebra/)
- [优化理论基础](/2026/05/10/zh-CN/技术文档/机器学习/optimization/)
- [逻辑回归](/2026/05/10/zh-CN/技术文档/机器学习/logistic-regression/)
- [核方法与核函数](/2026/05/10/zh-CN/技术文档/机器学习/kernel-methods/)
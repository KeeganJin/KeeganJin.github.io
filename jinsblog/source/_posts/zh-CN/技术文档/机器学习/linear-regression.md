---
title: 线性回归
date: 2026-04-29
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 线性回归, 监督学习]
---

## 线性回归模型推导

### 线性回归基本概念

线性回归是机器学习中最基础的回归模型，用于预测连续数值。

**模型形式**：
$y = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b$

其中：
- $\mathbf{x}$ 是特征向量
- $\mathbf{w}$ 是权重向量
- $b$ 是偏置项
- $y$ 是预测值

### 线性回归的假设

1. **线性关系**：特征与目标之间存在线性关系
2. **独立性**：样本之间相互独立
3. **同方差性**：误差方差相同
4. **正态性**：误差服从正态分布

### 线性回归的几何理解

**向量空间视角**：
- 特征向量构成一个向量空间
- 线性回归寻找该空间中距离目标向量最近的点
- 这个最近点是目标向量在特征空间的投影

## 最小二乘法求解

### 目标函数

最小二乘法通过最小化预测值与真实值的平方误差求解：

$J(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{2}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i - b)^2$

矩阵形式：
$J(\mathbf{w}) = \frac{1}{2}(\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w})$

### 求解方法

#### 解析解（闭式解）

对目标函数求导并令导数为零：

$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y}) = 0$

得到最优解：
$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

**前提条件**：$\mathbf{X}^T\mathbf{X}$ 可逆（特征不线性相关）

```python
import numpy as np

def linear_regression_closed(X, y):
    """最小二乘法解析解"""
    # 添加偏置项
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # 解析解
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    return w

# 示例
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
w = linear_regression_closed(X, y)
print(f"权重: {w}")
```

#### 梯度下降求解

当特征维度高或数据量大时，使用梯度下降：

$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \frac{\partial J}{\partial \mathbf{w}}$

梯度：
$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y})$

```python
def linear_regression_gradient(X, y, learning_rate=0.01, iterations=1000):
    """梯度下降求解线性回归"""
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    w = np.zeros(n + 1)
    
    for i in range(iterations):
        gradient = X_b.T.dot(X_b.dot(w) - y) / m
        w = w - learning_rate * gradient
        
        if i % 100 == 0:
            loss = np.sum((X_b.dot(w) - y) ** 2) / (2 * m)
            print(f"迭代 {i}, 损失: {loss:.4f}")
    
    return w
```

### 最小二乘法的概率解释

假设误差服从正态分布：$y_i = \mathbf{w}^T\mathbf{x}_i + \epsilon_i$, $\epsilon_i \sim N(0, \sigma^2)$

最大似然估计：
$L(\mathbf{w}) = \prod_{i=1}^{n} P(y_i|\mathbf{x}_i, \mathbf{w}) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right)$

最大化似然等价于最小化：
$\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2$

即最小二乘法。

## 正则化：L1（Lasso）、L2（Ridge）

### 正则化的作用

正则化防止过拟合，通过约束权重实现：

- **L2正则化（Ridge）**：权重衰减
- **L1正则化（Lasso）**：产生稀疏解

### Ridge回归（L2正则化）

**目标函数**：
$J(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$

**解析解**：
$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$

**特点**：
- 所有权重都减小，但不为零
- 解决 $\mathbf{X}^T\mathbf{X}$ 不可逆问题
- 等价于约束 $\|\mathbf{w}\|^2 \leq t$

```python
from sklearn.linear_model import Ridge

# Ridge回归
ridge = Ridge(alpha=1.0)  # alpha 即 λ
ridge.fit(X_train, y_train)

print(f"权重: {ridge.coef_}")
print(f"偏置: {ridge.intercept_}")
```

### Lasso回归（L1正则化）

**目标函数**：
$J(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda\|\mathbf{w}\|_1$

**特点**：
- 产生稀疏解（部分权重为零）
- 自动进行特征选择
- 等价于约束 $\|\mathbf{w}\|_1 \leq t$

**求解方法**：
- 坐标下降法
- LARS算法

```python
from sklearn.linear_model import Lasso

# Lasso回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

print(f"权重: {lasso.coef_}")
print(f"非零特征数: {np.sum(lasso.coef_ != 0)}")
```

### Elastic Net（L1 + L2）

结合L1和L2正则化：

$J(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|^2$

```python
from sklearn.linear_model import ElasticNet

# Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio 控制L1比例
elastic.fit(X_train, y_train)
```

### 正则化参数选择

通过交叉验证选择正则化系数：

```python
from sklearn.linear_model import RidgeCV, LassoCV

# Ridge CV
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge_cv.fit(X_train, y_train)
print(f"最优alpha: {ridge_cv.alpha_}")

# Lasso CV
lasso_cv = LassoCV(alphas=[0.01, 0.1, 1.0], cv=5)
lasso_cv.fit(X_train, y_train)
print(f"最优alpha: {lasso_cv.alpha_}")
```

## 多元线性回归

### 多元线性回归模型

当有多个特征时：

$y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$

**矩阵形式**：
$\mathbf{y} = \mathbf{X}\mathbf{w} + b$

### 多元线性回归的特点

1. **特征交互**：各特征独立贡献
2. **维度诅咒**：特征过多时性能下降
3. **特征相关**：特征相关时权重不稳定

### 特征标准化

多元回归前需要标准化特征：

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 训练
model = LinearRegression()
model.fit(X_scaled, y_train)
```

## 模型评估指标

### 回归评估指标

| 指标 | 公式 | 描述 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 均方误差 |
| RMSE | $\sqrt{MSE}$ | 均方根误差 |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 平均绝对误差 |
| R² | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 决定系数 |

### R²分数解释

$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

- $R^2 = 1$：完美拟合
- $R^2 = 0$：模型等同于均值预测
- $R^2 < 0$：模型比均值预测更差

### Adjusted R²

考虑特征数量：
$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

## 案例实践

### 波士顿房价预测示例

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
data = fetch_california_housing()
X, y = data.data, data.target

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 普通线性回归
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Ridge回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Lasso回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# 评估
print("普通线性回归:")
print(f"  R²: {r2_score(y_test, y_pred_lr):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")

print("Ridge回归:")
print(f"  R²: {r2_score(y_test, y_pred_ridge):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")

print("Lasso回归:")
print(f"  R²: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso)):.4f}")
print(f"  非零特征: {np.sum(lasso.coef_ != 0)}")
```

### 权重可视化

```python
import matplotlib.pyplot as plt

# 权重对比
feature_names = data.feature_names

plt.figure(figsize=(12, 6))
x = range(len(feature_names))

plt.bar(x - 0.2, lr.coef_, width=0.2, label='Linear Regression')
plt.bar(x, ridge.coef_, width=0.2, label='Ridge')
plt.bar(x + 0.2, lasso.coef_, width=0.2, label='Lasso')

plt.xticks(x, feature_names)
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.legend()
plt.title('权重对比')
plt.show()
```

## 总结

线性回归是机器学习最基础的回归模型。核心内容包括：
- 线性回归模型推导：线性关系假设
- 最小二乘法求解：解析解和梯度下降
- 正则化：Ridge（L2）防止过拟合，Lasso（L1）产生稀疏解
- 多元线性回归：特征标准化的重要性
- 模型评估：MSE、RMSE、MAE、R²

线性回归简单但应用广泛，是理解更复杂模型的基础。

## 延伸阅读

- [机器学习概述](/2026/05/10/zh-CN/技术文档/机器学习/ml-introduction/)
- [线性代数基础](/2026/05/10/zh-CN/技术文档/机器学习/linear-algebra/)
- [优化理论基础](/2026/05/10/zh-CN/技术文档/机器学习/optimization/)
- [逻辑回归](/2026/05/10/zh-CN/技术文档/机器学习/logistic-regression/)
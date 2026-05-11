---
title: 逻辑回归
date: 2026-05-01
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 逻辑回归, 分类算法]
---

## 逻辑回归原理

### 逻辑回归基本概念

逻辑回归是用于分类任务的线性模型，通过Sigmoid函数将线性输出转换为概率。

**模型形式**：
$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$

**预测规则**：
- $P(y=1|\mathbf{x}) \geq 0.5$ → 预测为正类（y=1）
- $P(y=1|\mathbf{x}) < 0.5$ → 颞测为负类（y=0）

### 逻辑回归与线性回归的区别

| 方面 | 线性回归 | 逻辑回归 |
|------|----------|----------|
| 任务类型 | 回归 | 分类 |
| 输出范围 | 任意实数 | [0, 1] |
| 输出含义 | 预测值 | 概率 |
| 激活函数 | 无 | Sigmoid |
| 损失函数 | MSE | 交叉熵 |

### 逻辑回归的决策边界

决策边界是线性的：
$\mathbf{w}^T\mathbf{x} + b = 0$

对于二维特征，决策边界是一条直线：
$w_1 x_1 + w_2 x_2 + b = 0$

## Sigmoid函数

### Sigmoid函数定义

$\sigma(z) = \frac{1}{1 + e^{-z}}$

### Sigmoid函数性质

| 性质 | 描述 |
|------|------|
| 范围 | 输出在 (0, 1) 之间 |
| 单调性 | 单调递增 |
| 导数 | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ |
| 中心点 | $\sigma(0) = 0.5$ |
| 极限 | $\sigma(+\infty) = 1$, $\sigma(-\infty) = 0$ |

### Sigmoid函数的导数

导数的特殊形式：
$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$

这一性质使得梯度计算非常高效。

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid函数可视化
z = np.linspace(-10, 10, 100)
sig = sigmoid(z)

plt.figure(figsize=(10, 6))
plt.plot(z, sig)
plt.axhline(0.5, color='r', linestyle='--', label='y=0.5')
plt.axvline(0, color='g', linestyle='--', label='x=0')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid函数')
plt.legend()
plt.grid(True)
plt.show()
```

### Sigmoid函数的优缺点

**优点**：
- 输出范围有限，适合表示概率
- 导数计算简单
- 函数平滑连续

**缺点**：
- 当输入很大或很小时，梯度接近零（梯度消失）
- 输出不是零中心的
- 指数运算计算成本较高

## 概率解释与最大似然估计

### 逻辑回归的概率模型

假设：
$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})$
$P(y=0|\mathbf{x}) = 1 - \sigma(\mathbf{w}^T\mathbf{x})$

合并表示：
$P(y|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})^y (1-\sigma(\mathbf{w}^T\mathbf{x}))^{1-y}$

### 最大似然估计（MLE）

**似然函数**：
$L(\mathbf{w}) = \prod_{i=1}^{n} P(y_i|\mathbf{x}_i, \mathbf{w})$

**对数似然**：
$\ln L(\mathbf{w}) = \sum_{i=1}^{n} [y_i \ln \hat{y}_i + (1-y_i) \ln(1-\hat{y}_i)]$

其中 $\hat{y}_i = \sigma(\mathbf{w}^T\mathbf{x}_i)$

### 交叉熵损失函数

最大化对数似然等价于最小化交叉熵损失：

$J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n} [y_i \ln \hat{y}_i + (1-y_i) \ln(1-\hat{y}_i)]$

**单个样本的损失**：
$\ell(y, \hat{y}) = -y \ln \hat{y} - (1-y) \ln(1-\hat{y})$

- 当 $y=1$：损失 = $-\ln \hat{y}$
- 当 $y=0$：损失 = $-\ln(1-\hat{y})$

### 梯度计算

对权重求梯度：
$\frac{\partial J}{\partial w_j} = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i) x_{ij}$

梯度向量：
$\nabla J = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$

```python
def logistic_regression_gradient(X, y, w):
    """计算梯度"""
    m = len(y)
    h = sigmoid(X.dot(w))
    gradient = X.T.dot(h - y) / m
    return gradient
```

### 梯度下降求解

```python
def logistic_regression_fit(X, y, learning_rate=0.01, iterations=1000):
    """逻辑回归梯度下降训练"""
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # 添加偏置
    w = np.zeros(n + 1)
    
    for i in range(iterations):
        h = sigmoid(X_b.dot(w))
        gradient = X_b.T.dot(h - y) / m
        w = w - learning_rate * gradient
        
        if i % 100 == 0:
            loss = -np.mean(y * np.log(h) + (1-y) * np.log(1-h))
            print(f"迭代 {i}, 损失: {loss:.4f}")
    
    return w
```

## 多分类扩展

### 一对多（One-vs-Rest, OvR）

训练K个二分类器，每个分类器区分一个类别与其他所有类别。

**预测**：选择概率最高的类别。

```python
from sklearn.linear_model import LogisticRegression

# OvR多分类
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)
```

### 一对一（One-vs-One, OvO）

训练 $\frac{K(K-1)}{2}$ 个二分类器，每个分类器区分两个类别。

**预测**：投票决定最终类别。

### Softmax回归（多类逻辑回归）

直接输出多类概率：

$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T\mathbf{x}}}$

**Softmax损失函数**（交叉熵）：
$J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \ln P(y_i=k|\mathbf{x}_i)$

```python
# Softmax多分类
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
```

## 正则化方法

### 逻辑回归中的正则化

**L2正则化**：
$J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n} [y_i \ln \hat{y}_i + (1-y_i) \ln(1-\hat{y}_i)] + \frac{\lambda}{2n}\|\mathbf{w}\|^2$

**L1正则化**：
$J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n} [y_i \ln \hat{y}_i + (1-y_i) \ln(1-\hat{y}_i)] + \frac{\lambda}{n}\|\mathbf{w}\|_1$

```python
from sklearn.linear_model import LogisticRegression

# L2正则化（默认）
model_l2 = LogisticRegression(penalty='l2', C=1.0)  # C = 1/λ

# L1正则化
model_l1 = LogisticRegression(penalty='l1', C=1.0, solver='saga')

# Elastic Net
model_en = LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, solver='saga')
```

### 正则化参数C

sklearn中使用 `C` 参数，C是正则化系数的倒数：
- C越大，正则化越弱
- C越小，正则化越强

```python
# 正则化强度对比
for C in [0.01, 0.1, 1.0, 10.0]:
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)
    print(f"C={C}, 非零权重数: {np.sum(model.coef_ != 0)}")
```

## 案例实践

### 二分类示例

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 生成数据
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练逻辑回归
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 查看概率
print("\n前5个样本的概率:")
print(y_prob[:5])
```

### 多分类示例

```python
from sklearn.datasets import load_iris

# 加载鸢尾花数据
iris = load_iris()
X, y = iris.data, iris.target

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# OvR多分类
model_ovr = LogisticRegression(multi_class='ovr')
model_ovr.fit(X_train_scaled, y_train)

# Softmax多分类
model_softmax = LogisticRegression(multi_class='multinomial')
model_softmax.fit(X_train_scaled, y_train)

print("OvR准确率:", accuracy_score(y_test, model_ovr.predict(X_test_scaled)))
print("Softmax准确率:", accuracy_score(y_test, model_softmax.predict(X_test_scaled)))
```

### 决策边界可视化

```python
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, model):
    """绘制决策边界"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# 使用两个特征可视化
X_2d, y_2d = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)
model = LogisticRegression()
model.fit(X_2d, y_2d)
plot_decision_boundary(X_2d, y_2d, model)
```

## 逻辑回归的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 简单高效 | 计算成本低，训练快 |
| 可解释性强 | 权重表示特征重要性 |
| 输出概率 | 提供预测置信度 |
| 不易过拟合 | 加正则化效果更好 |
| 适合线性可分 | 对线性边界效果好 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 线性边界限制 | 无法处理复杂非线性 |
| 特征依赖 | 需要合适的特征工程 |
| 对异常值敏感 | 异常值影响大 |
| 多分类复杂 | 多分类需要扩展 |

## 总结

逻辑回归是经典的二分类算法。核心内容包括：
- 逻辑回归原理：线性模型加Sigmoid激活
- Sigmoid函数：将线性输出转换为概率
- 最大似然估计：推导交叉熵损失函数
- 多分类扩展：OvR、OvO、Softmax
- 正则化方法：L1产生稀疏解，L2防止过拟合

逻辑回归简单但应用广泛，是理解更复杂模型的基础。

## 延伸阅读

- [线性回归](/2026/05/10/zh-CN/技术文档/机器学习/linear-regression/)
- [概率论基础](/2026/05/10/zh-CN/技术文档/机器学习/probability-theory/)
- [支持向量机](/2026/05/10/zh-CN/技术文档/机器学习/svm/)
- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
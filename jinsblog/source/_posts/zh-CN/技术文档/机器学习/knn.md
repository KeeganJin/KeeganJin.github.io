---
title: K近邻（KNN）
date: 2025-10-21
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, KNN, 分类算法]
---

## KNN基本原理

### K近邻概念

K近邻（K-Nearest Neighbors, KNN）是一种基于实例的学习算法。

**核心思想**：
- 找到距离测试样本最近的K个训练样本
- 根据这K个邻居的类别进行预测

**分类决策**：投票决定类别
**回归决策**：取K个邻居的平均值

### KNN的工作流程

```
1. 计算测试样本与所有训练样本的距离
2. 按距离排序，选择最近的K个样本
3. 统计K个邻居的类别
4. 分类：多数投票决定类别
5. 回归：计算邻居的平均值
```

### KNN的特点

| 特点 | 描述 |
|------|------|
| 惰性学习 | 不显式学习模型，存储训练数据 |
| 非参数方法 | 不假设数据分布 |
| 简单直观 | 算法逻辑简单 |
| 计算成本高 | 每次预测需计算所有距离 |

### KNN的几何理解

在特征空间中，KNN的决策边界由邻居分布决定：
- K小：决策边界复杂（可能过拟合）
- K大：决策边界平滑（可能欠拟合）

## 距离度量方法

### 欧氏距离

最常用的距离度量：

$d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{i=1}^{n}(x_i - z_i)^2}$

```python
import numpy as np

def euclidean_distance(x, z):
    """欧氏距离"""
    return np.sqrt(np.sum((x - z) ** 2))
```

### 曼哈顿距离

沿坐标轴方向的距离：

$d(\mathbf{x}, \mathbf{z}) = \sum_{i=1}^{n}|x_i - z_i|$

```python
def manhattan_distance(x, z):
    """曼哈顿距离"""
    return np.sum(np.abs(x - z))
```

### 切比雪夫距离

各坐标差的最大值：

$d(\mathbf{x}, \mathbf{z}) = \max_i |x_i - z_i|$

```python
def chebyshev_distance(x, z):
    """切比雪夫距离"""
    return np.max(np.abs(x - z))
```

### 闵可夫斯基距离

通用距离度量：

$d(\mathbf{x}, \mathbf{z}) = \left(\sum_{i=1}^{n}|x_i - z_i|^p\right)^{1/p}$

- $p=1$：曼哈顿距离
- $p=2$：欧氏距离
- $p=\infty$：切比雪夫距离

```python
def minkowski_distance(x, z, p=2):
    """闵可夫斯基距离"""
    return np.power(np.sum(np.power(np.abs(x - z), p)), 1/p)
```

### 余弦距离

用于文本等高维数据：

$d(\mathbf{x}, \mathbf{z}) = 1 - \frac{\mathbf{x}^T\mathbf{z}}{\|\mathbf{x}\|\|\mathbf{z}\|}$

```python
def cosine_distance(x, z):
    """余弦距离"""
    cosine_sim = np.dot(x, z) / (np.linalg.norm(x) * np.linalg.norm(z))
    return 1 - cosine_sim
```

### 汉明距离

用于离散特征，计算不同位置的数目：

$d(\mathbf{x}, \mathbf{z}) = \sum_{i=1}^{n} \mathbb{1}[x_i \neq z_i]$

```python
def hamming_distance(x, z):
    """汉明距离"""
    return np.sum(x != z)
```

### 距离度量选择指南

| 场景 | 推荐距离 |
|------|----------|
| 连续特征，低维 | 欧氏距离 |
| 连续特征，高维 | 曼哈顿距离 |
| 文本、稀疏数据 | 余弦距离 |
| 离散特征 | 汉明距离 |
| 不确定 | 尝试多种距离 |

```python
from sklearn.neighbors import KNeighborsClassifier

# 不同距离度量
for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric, p=3 if metric=='minkowski' else 2)
    knn.fit(X_train, y_train)
    print(f"{metric}: 准确率={knn.score(X_test, y_test):.4f}")
```

## K值选择策略

### K值的影响

**K值对决策边界的影响**：

| K值 | 决策边界 | 风险 |
|-----|----------|------|
| K=1 | 非常复杂 | 过拟合 |
| K小 | 较复杂 | 可能过拟合 |
| K大 | 较平滑 | 可能欠拟合 |
| K=N | 极平滑 | 总是预测多数类 |

### K=1的情况

最近邻分类：
- 决策边界是训练样本的 Voronoi 图边界
- 对噪声敏感
- 容易过拟合

### 最优K值选择

**交叉验证法**：

```python
from sklearn.model_selection import cross_val_score

# 测试不同K值
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

# 最优K
best_k = k_values[np.argmax(cv_scores)]
print(f"最优K值: {best_k}")
print(f"最高交叉验证分数: {max(cv_scores):.4f}")
```

**可视化K值选择**：

```python
import matplotlib.pyplot as plt

plt.plot(k_values, cv_scores)
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('K值选择')
plt.scatter(best_k, max(cv_scores), color='red', s=100, label=f'Best K={best_k}')
plt.legend()
plt.show()
```

### K值选择原则

| 数据特点 | K值建议 |
|----------|---------|
| 样本少 | 较小K值 |
| 样本多 | 较大K值 |
| 噪声多 | 较大K值 |
| 类别不平衡 | 考虑加权投票 |

### 经验法则

- 通常选择奇数K（避免平票）
- 一般 $K \leq \sqrt{N}$（N为样本数）
- 通过交叉验证确定最优K

## 权重分配方法

### 均权投票

所有邻居权重相同：
- 统计K个邻居中各类别的数量
- 选择数量最多的类别

### 距离加权投票

邻居权重与距离相关：
$w_i = \frac{1}{d_i}$ 或 $w_i = \frac{1}{d_i^2}$

**加权投票**：
$\hat{y} = \arg\max_c \sum_{i \in N_k} w_i \mathbb{1}[y_i = c]$

```python
from sklearn.neighbors import KNeighborsClassifier

# 均权
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# 距离加权
knn_distance = KNeighborsClassifier(n_neighbors=5, weights='distance')

knn_uniform.fit(X_train, y_train)
knn_distance.fit(X_train, y_train)

print(f"均权准确率: {knn_uniform.score(X_test, y_test):.4f}")
print(f"加权准确率: {knn_distance.score(X_test, y_test):.4f}")
```

### 自定义权重函数

```python
def custom_weights(distances):
    """自定义权重函数"""
    # 距离小于阈值的权重高
    threshold = 1.0
    weights = np.where(distances < threshold, 1.0, 0.5)
    return weights

knn_custom = KNeighborsClassifier(n_neighbors=5, weights=custom_weights)
knn_custom.fit(X_train, y_train)
```

### 加权的优势

| 优势 | 描述 |
|------|------|
| 考虑距离信息 | 近邻更重要 |
| 减少噪声影响 | 远距离样本影响小 |
| 更合理的决策 | 综合考虑距离和类别 |

## KD树优化

### KNN的计算问题

朴素KNN每次预测需要计算与所有训练样本的距离：
- 时间复杂度：O(n × d)（n样本数，d特征数）
- 对大规模数据效率低

### KD树

KD树是一种空间划分数据结构，加速最近邻搜索。

**构建过程**：
```
1. 选择方差最大的特征作为分裂轴
2. 选择该特征的中位数作为分裂点
3. 将数据分为两部分
4. 对子集递归构建子树
```

### KD树的结构

```
              (5, 3)
             /      \
         (2, 4)    (8, 7)
         /    \    /    \
      (1,2) (3,4) (6,5) (9,6)
```

### KD树搜索

**最近邻搜索**：
```
1. 从根节点开始，沿树向下找到包含查询点的叶节点
2. 计算叶节点中最近邻
3. 回溯检查是否有更近的点
4. 利用超平面剪枝
```

### KD树的效率

| 情况 | 时间复杂度 |
|------|------------|
| 低维数据 | O(log n) |
| 高维数据 | 接近 O(n) |

### Ball树

对高维数据更有效：

```python
from sklearn.neighbors import KNeighborsClassifier

# KD树
knn_kd = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')

# Ball树
knn_ball = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')

# 暴力搜索
knn_brute = KNeighborsClassifier(n_neighbors=5, algorithm='brute')

# 自动选择
knn_auto = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
```

### 算法选择指南

| 数据特点 | 推荐算法 |
|----------|----------|
| 小数据（<30样本） | brute |
| 低维（<20特征） | kd_tree |
| 高维（>20特征） | ball_tree |
| 不确定 | auto |

## 案例实践

### KNN分类示例

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化（KNN对尺度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)

# 预测
y_pred = knn.predict(X_test_scaled)
y_prob = knn.predict_proba(X_test_scaled)

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))
```

### K值选择实验

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# 不同K值的交叉验证分数
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores)
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('K值对准确率的影响')
plt.grid(True)
plt.show()

# 最优K
best_k = k_range[np.argmax(k_scores)]
print(f"最优K值: {best_k}")
```

### 决策边界可视化

```python
def plot_knn_decision_boundary(X, y, k):
    """绘制KNN决策边界"""
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)
    
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title(f'KNN Decision Boundary (K={k})')
    plt.show()

# 使用前两个特征
X_2d = iris.data[:, :2]
plot_knn_decision_boundary(X_2d, iris.target, k=5)
```

### KNN回归示例

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载回归数据
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data[:500], data.target[:500]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN回归
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train_scaled, y_train)

y_pred = knn_reg.predict(X_test_scaled)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### 算法对比

```python
import time

# 不同搜索算法的效率对比
algorithms = ['brute', 'kd_tree', 'ball_tree']

# 使用较大数据集
X_large, y_large = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_large, y_large, test_size=0.2)

scaler = StandardScaler()
X_train_l_scaled = scaler.fit_transform(X_train_l)
X_test_l_scaled = scaler.transform(X_test_l)

for algo in algorithms:
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
    
    start = time.time()
    knn.fit(X_train_l_scaled, y_train_l)
    y_pred = knn.predict(X_test_l_scaled)
    end = time.time()
    
    print(f"{algo}:")
    print(f"  准确率: {accuracy_score(y_test_l, y_pred):.4f}")
    print(f"  时间: {end - start:.4f}秒")
```

## KNN的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 简单直观 | 算法逻辑简单易懂 |
| 无需训练 | 存储数据即可，无训练过程 |
| 适用于多类 | 天然支持多分类 |
| 可解释性 | 通过邻居解释预测 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 计算慢 | 每次预测计算所有距离 |
| 存储成本 | 需存储所有训练数据 |
| 维度诅咒 | 高维数据效率下降 |
| 尺度敏感 | 需要特征标准化 |
| 不平衡问题 | 类别不平衡时偏向多数类 |

### 维度诅咒

当特征维度很高时：
- 距离概念变得模糊
- 所有样本距离相近
- KD树效率下降

**解决方案**：
- 降维（PCA等）
- 特征选择
- 使用余弦距离

## 总结

K近邻是基于实例的简单分类和回归算法。核心内容包括：
- KNN基本原理：找最近K个邻居投票决定
- 距离度量：欧氏距离、曼哈顿距离、余弦距离等
- K值选择：通过交叉验证选择最优K
- 权重分配：均权或距离加权投票
- KD树优化：加速最近邻搜索

KNN简单但计算成本高，适合小规模低维数据。

## 延伸阅读

- [机器学习概述](/2026/05/10/zh-CN/技术文档/机器学习/ml-introduction/)
- [线性代数基础](/2026/05/10/zh-CN/技术文档/机器学习/linear-algebra/)
- [数据预处理](/2026/05/10/zh-CN/技术文档/机器学习/data-preprocessing/)
- [决策树](/2026/05/10/zh-CN/技术文档/机器学习/decision-tree/)
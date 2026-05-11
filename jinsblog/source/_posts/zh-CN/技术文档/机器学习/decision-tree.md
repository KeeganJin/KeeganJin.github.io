---
title: 决策树
date: 2026-04-17
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 决策树, 分类算法]
---

## 决策树基本概念

### 什么是决策树

决策树是一种基于树状结构的分类和回归模型。

**树结构**：
- **根节点**：包含所有样本
- **内部节点**：特征测试节点
- **分支**：测试结果分支
- **叶节点**：决策结果（类别或值）

**决策过程**：
从根节点开始，沿分支向下，到达叶节点得到预测结果。

### 决策树示例

```
                 年龄？
               /      \
           <=30       >30
            |         |
          收入？     是
         /    \       |
      高      低     买车
       |       |
     买车    不买车
```

### 决策树的优点

| 优点 | 描述 |
|------|------|
| 可解释性强 | 决策过程清晰可见 |
| 不需要特征缩放 | 对特征尺度不敏感 |
| 处理混合数据类型 | 可处理数值和分类特征 |
| 处理缺失值 | 可以处理缺失数据 |
| 计算效率高 | 训练和预测速度快 |

### 决策树的缺点

| 缺点 | 描述 |
|------|------|
| 易过拟合 | 树太复杂时容易过拟合 |
| 不稳定 | 数据小变化可能产生不同树 |
| 偏斜问题 | 类别不平衡时偏向多数类 |
| 线性边界限制 | 决策边界是轴平行 |

## 信息增益与ID3算法

### 信息熵

信息熵度量数据的混乱程度：

$H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k$

其中 $p_k$ 是第 $k$ 类的比例。

**熵的性质**：
- 熵越大，数据越混乱
- 熵越小，数据越纯
- 当所有样本属于同一类时，熵为0
- 当样本均匀分布在所有类时，熵最大

```python
import numpy as np

def entropy(y):
    """计算信息熵"""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# 示例
y1 = [0, 0, 0, 0]  # 纯数据
y2 = [0, 1, 0, 1]  # 混合数据
y3 = [0, 1, 2, 3]  # 更混乱

print(f"纯数据熵: {entropy(y1):.4f}")  # 0
print(f"混合数据熵: {entropy(y2):.4f}")  # 1.0
print(f"更混乱熵: {entropy(y3):.4f}")  # 2.0
```

### 条件熵

给定特征A后数据集D的熵：

$H(D|A) = \sum_{v \in Values(A)} \frac{|D_v|}{|D|} H(D_v)$

其中 $D_v$ 是特征A取值为v的子集。

### 信息增益

信息增益衡量特征对分类的贡献：

$IG(D, A) = H(D) - H(D|A)$

**选择策略**：选择信息增益最大的特征作为分裂节点。

```python
def information_gain(X, y, feature_idx):
    """计算信息增益"""
    # 父节点熵
    parent_entropy = entropy(y)
    
    # 按特征划分
    values = np.unique(X[:, feature_idx])
    children_entropy = 0
    
    for v in values:
        subset_y = y[X[:, feature_idx] == v]
        weight = len(subset_y) / len(y)
        children_entropy += weight * entropy(subset_y)
    
    return parent_entropy - children_entropy
```

### ID3算法

ID3算法使用信息增益选择分裂特征：

**算法流程**：
```
1. 计算所有特征的信息增益
2. 选择信息增益最大的特征作为当前节点
3. 按特征值划分数据集
4. 对每个子集递归构建子树
5. 直到所有样本属于同一类或无特征可选
```

**ID3特点**：
- 只能处理分类特征
- 偏向于取值多的特征
- 不处理缺失值
- 不进行剪枝

## 信息增益率与C4.5算法

### 信息增益的偏斜问题

信息增益偏向于取值较多的特征。

例如：如果一个特征有唯一值（如ID），信息增益最大，但这个特征没有分类意义。

### 信息增益率

信息增益率修正了偏斜问题：

$IGR(D, A) = \frac{IG(D, A)}{IV(A)}$

其中 $IV(A)$ 是特征的固有值（Intrinsic Value）：
$IV(A) = -\sum_{v \in Values(A)} \frac{|D_v|}{|D|} \log_2 \frac{|D_v|}{|D|}$

```python
def intrinsic_value(X, feature_idx):
    """计算固有值"""
    values, counts = np.unique(X[:, feature_idx], return_counts=True)
    probabilities = counts / len(X)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain_ratio(X, y, feature_idx):
    """计算信息增益率"""
    ig = information_gain(X, y, feature_idx)
    iv = intrinsic_value(X, feature_idx)
    return ig / iv if iv != 0 else 0
```

### C4.5算法

C4.5是ID3的改进版本：

**改进点**：
1. 使用信息增益率选择特征
2. 可以处理连续特征（通过二分法）
3. 可以处理缺失值
4. 引入剪枝策略

**连续特征处理**：
- 找到最优分割点
- 将连续值转换为二值（<= 分割点，> 分割点）

```python
def find_best_split(X_cont, y):
    """找到连续特征的最佳分割点"""
    sorted_indices = np.argsort(X_cont)
    sorted_X = X_cont[sorted_indices]
    sorted_y = y[sorted_indices]
    
    best_gain = -1
    best_split = None
    
    for i in range(len(sorted_X) - 1):
        split = (sorted_X[i] + sorted_X[i+1]) / 2
        left_y = sorted_y[sorted_X <= split]
        right_y = sorted_y[sorted_X > split]
        
        gain = entropy(y) - (len(left_y)/len(y) * entropy(left_y) + 
                             len(right_y)/len(y) * entropy(right_y))
        
        if gain > best_gain:
            best_gain = gain
            best_split = split
    
    return best_split, best_gain
```

## Gini指数与CART算法

### Gini指数

Gini指数衡量数据的不纯度：

$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$

**Gini指数性质**：
- Gini越小，数据越纯
- 当所有样本属于同一类时，Gini=0
- 当样本均匀分布时，Gini最大

```python
def gini(y):
    """计算Gini指数"""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

# 示例
y1 = [0, 0, 0, 0]  # 纯数据
y2 = [0, 1, 0, 1]  # 混合数据

print(f"纯数据Gini: {gini(y1):.4f}")  # 0
print(f"混合数据Gini: {gini(y2):.4f}")  # 0.5
```

### Gini增益

特征A对数据集D的Gini增益：
$Gini\_Gain(D, A) = Gini(D) - \sum_v \frac{|D_v|}{|D|} Gini(D_v)$

选择Gini增益最大的特征分裂。

### CART算法

CART（Classification and Regression Trees）算法：

**特点**：
- 使用Gini指数作为分裂标准
- 构建二叉树（每个节点只有两个分支）
- 可以用于分类和回归
- 支持连续和分类特征

**CART分类树**：
```
1. 计算每个特征的每个可能分割的Gini增益
2. 选择Gini增益最大的分割
3. 将节点分裂为两个子节点
4. 递归构建子树
```

**CART回归树**：
使用方差代替Gini指数：
$Var(D) = \frac{1}{|D|}\sum_{i \in D}(y_i - \bar{y})^2$

选择使方差减小最多的分割。

```python
def variance(y):
    """计算方差（用于回归树）"""
    return np.var(y)

def best_split_regression(X_cont, y):
    """回归树最佳分割点"""
    best_var_reduction = -1
    best_split = None
    
    sorted_indices = np.argsort(X_cont)
    sorted_X = X_cont[sorted_indices]
    sorted_y = y[sorted_indices]
    
    parent_var = variance(y)
    
    for i in range(len(sorted_X) - 1):
        split = (sorted_X[i] + sorted_X[i+1]) / 2
        left_y = y[X_cont <= split]
        right_y = y[X_cont > split]
        
        var_reduction = parent_var - (len(left_y)/len(y) * variance(left_y) +
                                       len(right_y)/len(y) * variance(right_y))
        
        if var_reduction > best_var_reduction:
            best_var_reduction = var_reduction
            best_split = split
    
    return best_split, best_var_reduction
```

## 树的剪枝策略

### 为什么需要剪枝

决策树容易过拟合：
- 树太深会记住训练数据的噪声
- 叶节点样本太少会过拟合

**剪枝类型**：
- **预剪枝**：在构建过程中限制树的成长
- **后剪枝**：构建完整树后再修剪

### 预剪枝方法

| 方法 | 描述 |
|------|------|
| 最大深度 | 限制树的最大深度 |
| 最小样本数 | 节点样本数小于阈值停止分裂 |
| 最小增益 | 信息增益小于阈值停止分裂 |
| 最小叶节点样本 | 叶节点样本数限制 |

```python
from sklearn.tree import DecisionTreeClassifier

# 预剪枝参数
model = DecisionTreeClassifier(
    max_depth=5,           # 最大深度
    min_samples_split=10,  # 分裂最小样本数
    min_samples_leaf=5,    # 叶节点最小样本数
    min_impurity_decrease=0.01  # 最小不纯度减少
)
model.fit(X_train, y_train)
```

### 后剪枝方法

#### 降低错误剪枝（REP）

从叶节点向上，删除子树如果不影响验证集准确率。

#### 代价复杂度剪枝（CCP）

使用复杂度参数 $\alpha$ 平衡树的复杂度和准确率：
$R_\alpha(T) = R(T) + \alpha|T|$

其中 $R(T)$ 是树的错误率，$|T|$ 是叶节点数。

```python
# CCP剪枝
model = DecisionTreeClassifier(ccp_alpha=0.01)
model.fit(X_train, y_train)

# 自动找到最优alpha
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# 选择最优alpha
models = []
for alpha in ccp_alphas:
    model = DecisionTreeClassifier(ccp_alpha=alpha)
    model.fit(X_train, y_train)
    models.append(model)

# 用验证集评估选择最佳模型
```

### 剪枝效果对比

```python
import matplotlib.pyplot as plt

# 不同剪枝强度的树深度
max_depths = [3, 5, 10, None]
for depth in max_depths:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"max_depth={depth}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")
```

## 回归树

### 回归树原理

回归树预测连续值，叶节点输出该节点样本的均值。

**分裂标准**：最小化方差

**预测**：叶节点所有样本目标值的均值

```python
from sklearn.tree import DecisionTreeRegressor

# 回归树
reg = DecisionTreeRegressor(max_depth=5)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
```

### 回归树与分类树的区别

| 方面 | 分类树 | 回归树 |
|------|--------|--------|
| 分裂标准 | Gini/信息增益 | 方差/MSE |
| 叶节点输出 | 类别 | 均值 |
| 目标类型 | 类别标签 | 连续数值 |

## 案例实践

### 分类决策树示例

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练决策树
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 特征重要性
print("\n特征重要性:")
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

### 可视化决策树

```python
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
tree.plot_tree(model, 
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True)
plt.show()
```

### 回归树示例

```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练回归树
reg = DecisionTreeRegressor(max_depth=5)
reg.fit(X_train, y_train)

# 预测
y_pred = reg.predict(X_test)

# 评估
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### 剪枝实验

```python
# 对比不同剪枝参数
params = {
    '无剪枝': {},
    '深度限制': {'max_depth': 5},
    '样本限制': {'min_samples_leaf': 10},
    'CCP剪枝': {'ccp_alpha': 0.01}
}

for name, param in params.items():
    model = DecisionTreeClassifier(**param, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    depth = model.get_depth()
    
    print(f"{name}: 深度={depth}, 训练={train_acc:.4f}, 测试={test_acc:.4f}")
```

## 总结

决策树是直观可解释的分类和回归模型。核心内容包括：
- 信息增益与ID3：信息熵度量数据纯度
- 信息增益率与C4.5：解决偏斜问题
- Gini指数与CART：构建二叉决策树
- 剪枝策略：预剪枝和后剪枝防止过拟合
- 回归树：使用方差作为分裂标准

决策树是集成学习的基础模型（随机森林、GBDT等的基础）。

## 延伸阅读

- [机器学习概述](/2026/05/10/zh-CN/技术文档/机器学习/ml-introduction/)
- [概率论基础](/2026/05/10/zh-CN/技术文档/机器学习/probability-theory/)
- [随机森林](/2026/05/10/zh-CN/技术文档/机器学习/random-forest/)
- [GBDT梯度提升树](/2026/05/10/zh-CN/技术文档/机器学习/gbdt/)
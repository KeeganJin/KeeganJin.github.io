---
title: XGBoost算法详解
date: 2025-11-22
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 集成学习, XGBoost]
---

## XGBoost目标函数推导

### XGBoost的改进

XGBoost在GBDT基础上进行了多项改进：

| 改进点 | 描述 |
|------|------|
| 二阶优化 | 使用二阶泰勒展开 |
| 正则化 | 加入L1/L2正则化 |
| 并行计算 | 特征层面并行 |
| 缺失值处理 | 自动学习缺失值处理 |
| Shrinkage | 学习率收缩 |

### 目标函数定义

XGBoost的目标函数：

$Obj = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$

其中：
- $L$：损失函数（如MSE、对数损失）
- $\Omega$：正则化项
- $f_k$：第k棵树

### 正则化项

$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$

其中：
- $T$：叶节点数量
- $w_j$：叶节点权重
- $\gamma$：叶节点数惩罚
- $\lambda$：L2正则化系数

**作用**：
- $\gamma$ 控制树的复杂度（节点数）
- $\lambda$ 控制叶节点权重的平滑度

### 加法模型

$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)$

目标函数在第t步：
$Obj^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$

## 二阶泰勒展开优化

### 泰勒展开

使用二阶泰勒展开近似损失函数：

$L(y, \hat{y}^{(t-1)} + f_t) \approx L(y, \hat{y}^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t^2(\mathbf{x}_i)$

其中：
- $g_i = \frac{\partial L}{\partial \hat{y}^{(t-1)}}$：一阶导数（梯度）
- $h_i = \frac{\partial^2 L}{\partial \hat{y}^{(t-1)2}}$：二阶导数

### 常见损失函数的导数

| 损失函数 | 一阶导数 $g$ | 二阶导数 $h$ |
|----------|-------------|-------------|
| MSE | $\hat{y} - y$ | 1 |
| 对数损失 | $\sigma(\hat{y}) - y$ | $\sigma(\hat{y})(1-\sigma(\hat{y}))$ |

### 简化后的目标函数

去掉常数项：
$Obj^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t^2(\mathbf{x}_i)] + \Omega(f_t)$

## 结构分数与增益计算

### 叶节点表示

将样本分配到叶节点：
- $I_j$：叶节点j包含的样本索引集合

叶节点输出为 $w_j$，则：
$f_t(\mathbf{x}_i) = w_{q(\mathbf{x}_i)}$

其中 $q(\mathbf{x}_i)$ 是样本$\mathbf{x}_i$所属的叶节点。

### 结构分数

目标函数改写为：
$Obj^{(t)} = \sum_{j=1}^{T} \left[\left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2}\left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2\right] + \gamma T$

定义：
- $G_j = \sum_{i \in I_j} g_i$：叶节点j的一阶导数和
- $H_j = \sum_{i \in I_j} h_i$：叶节点j的二阶导数和

**最优叶节点值**：
$w_j^* = -\frac{G_j}{H_j + \lambda}$

**最优目标值**：
$Obj^* = -\frac{1}{2}\sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T$

### 分裂增益

考虑将节点分裂为左节点L和右节点R：

**增益公式**：
$Gain = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} - \gamma$

**解释**：
- $\frac{G_L^2}{H_L + \lambda}$：左子树得分
- $\frac{G_R^2}{H_R + \lambda}$：右子树得分
- $\frac{(G_L + G_R)^2}{H_L + H_R + \lambda}$：分裂前得分
- $\gamma$：分裂复杂度惩罚

## 节点分裂策略

### 精确分裂算法

遍历所有特征的所有可能分裂点：

```
for each feature:
    for each possible split value:
        calculate Gain
        keep the best split
```

**优点**：精确找到最优分裂
**缺点**：计算量大

### 近似分裂算法

将特征值分桶，只考虑分桶边界作为分裂点：

**分桶策略**：
- Global分桶：在开始时分桶，所有层使用相同的分桶
- Local分桶：每次分裂时分桶，更精确但更慢

```python
# XGBoost分桶参数
import xgboost as xgb

params = {
    'tree_method': 'approx',  # 近似算法
    'sketch_eps': 0.03,  # 分桶精度
}
```

### 稀疏特征分裂

对缺失值的处理：
- 默认方向：学习缺失值应该走左还是右
- 只有非缺失值参与分裂计算

### 加权分位数

使用二阶导数作为权重进行分桶：
- $h_i$ 大的样本更重要
- 保证每个桶内的 $h$ 总和接近

## 正则化与剪枝

### 正则化参数

| 参数 | 含义 |
|------|------|
| $\gamma$ (min_split_loss) | 分裂最小增益 |
| $\lambda$ (reg_lambda) | L2正则化系数 |
| $\alpha$ (reg_alpha) | L1正则化系数 |

```python
params = {
    'gamma': 0.1,  # 最小分裂增益
    'reg_lambda': 1.0,  # L2正则化
    'reg_alpha': 0.0,  # L1正则化
}
```

### 剪枝策略

**预剪枝**：
- 当增益小于 $\gamma$ 时停止分裂
- max_depth：最大深度限制
- min_child_weight：叶节点最小权重和（$H_j$）

**后剪枝**：
- 构建完整树后，从底部剪枝增益小于阈值的节点

### Shrinkage（学习率）

$\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(\mathbf{x})$

**作用**：
- 减少每棵树的影响
- 需要更多树来达到相同效果
- 降低过拟合风险

```python
params = {
    'eta': 0.1,  # 学习率（shrinkage）
    'n_estimators': 100,  # 树数量
}
```

## 并行化与缓存优化

### 特征并行

**思想**：
- 不同线程处理不同特征
- 找到各自特征的最佳分裂
- 合并找到全局最佳分裂

### 数据并行

**思想**：
- 将数据分块
- 每个块独立计算统计量
- 合并统计量进行分裂决策

### 缓存优化

**问题**：
- 按特征值排序后，梯度访问不连续
- 缓存命中率低

**解决**：
- Cache-aware访问模式
- 预取梯度数据

### Out-of-core计算

处理数据超过内存的情况：
- Block压缩：压缩特征值
- Block sharding：数据分片到多个磁盘

## 特征重要性分析

### 特征重要性类型

| 类型 | 描述 |
|------|------|
| weight | 特征被用于分裂的次数 |
| gain | 特征带来的总增益 |
| cover | 特征覆盖的样本数 |

```python
import xgboost as xgb

# 训练模型
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# 特征重要性
importance_weight = model.get_booster().get_score(importance_type='weight')
importance_gain = model.get_booster().get_score(importance_type='gain')
importance_cover = model.get_booster().get_score(importance_type='cover')

print("Weight importance:")
print(importance_weight)
```

### 特征重要性可视化

```python
import matplotlib.pyplot as plt

xgb.plot_importance(model, importance_type='gain')
plt.show()
```

## 调参技巧

### 主要参数分类

| 参数类别 | 参数 |
|----------|------|
| 树参数 | max_depth, min_child_weight, gamma |
| 正则化 | reg_lambda, reg_alpha |
| 学习 | eta, n_estimators |
| 采样 | subsample, colsample_bytree |

### 调参顺序建议

```
1. 固定学习率，调树参数
   - max_depth: 3-10
   - min_child_weight: 1-6
   
2. 调正则化参数
   - gamma: 0-0.5
   - reg_lambda: 0-10
   
3. 调采样参数
   - subsample: 0.5-1
   - colsample: 0.5-1
   
4. 调学习率和树数
   - 降低学习率
   - 增加树数
```

### 调参示例

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

# 网格搜索
grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"最优参数: {grid_search.best_params_}")
```

### 交叉验证早停

```python
# 使用xgboost内置交叉验证
params = {
    'objective': 'binary:logistic',
    'eta': 0.1,
    'max_depth': 5,
}

dtrain = xgb.DMatrix(X_train, label=y_train)

cv_result = xgb.cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=10,
    verbose_eval=False
)

print(f"最优迭代次数: {cv_result.shape[0]}")
```

## 案例实践

### XGBoost分类示例

```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 数据
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost分类
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='multi:softmax',
    num_class=3,
    random_state=42
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 特征重要性
print("\n特征重要性:")
for name, imp in zip(iris.feature_names, model.feature_importances_):
    print(f"{name}: {imp:.4f}")
```

### XGBoost回归示例

```python
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# 数据
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost回归
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='reg:squarederror',
    random_state=42
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### DMatrix接口

```python
# 使用DMatrix（更高效）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'eta': 0.1,
    'max_depth': 5,
}

# 训练
bst = xgb.train(params, dtrain, num_boost_round=100)

# 预测
y_pred = bst.predict(dtest)

# 保存模型
bst.save_model('model.json')

# 加载模型
bst_loaded = xgb.Booster()
bst_loaded.load_model('model.json')
```

### 缺失值处理

```python
# XGBoost自动处理缺失值
X_missing = X_train.copy()
X_missing[::10, 0] = np.nan  # 模拟缺失值

model = xgb.XGBClassifier(missing=np.nan)
model.fit(X_missing, y_train)

# 查看缺失值默认方向
# 缺失值会学习最优的默认方向（左或右）
```

## XGBoost vs GBDT

| 方面 | GBDT | XGBoost |
|------|------|----------|
| 优化方法 | 一阶梯度 | 二阶泰勒展开 |
| 正则化 | 无 | L1/L2正则化 |
| 并行 | 无 | 特征并行 |
| 缺失值 | 需预处理 | 自动处理 |
| 计算 | CPU | CPU/GPU |
| 效率 | 较慢 | 更快 |

## 总结

XGBoost是GBDT的重要改进版本。核心内容包括：
- 目标函数推导：损失函数加正则化
- 二阶泰勒展开：更精确的优化
- 结构分数与增益：分裂决策依据
- 节点分裂策略：精确分裂和近似分裂
- 正则化与剪枝：防止过拟合
- 并行化：特征并行和缓存优化
- 调参技巧：树参数、正则化、学习率

XGBoost在竞赛和工业界广泛应用，是性能优秀的梯度提升框架。

## 延伸阅读

- [GBDT梯度提升树](/2026/05/10/zh-CN/技术文档/机器学习/gbdt/)
- [Boosting基础](/2026/05/10/zh-CN/技术文档/机器学习/boosting/)
- [LightGBM算法详解](/2026/05/10/zh-CN/技术文档/机器学习/lightgbm/)
- [优化理论基础](/2026/05/10/zh-CN/技术文档/机器学习/optimization/)
---
title: Stacking与Blending
date: 2025-11-28
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 集成学习, Stacking]
---

## Stacking原理

### Stacking概念

Stacking是一种分层集成方法，使用元模型组合基模型的预测。

**核心思想**：
- 多个基模型各自预测
- 元模型学习如何组合基模型预测
- 两阶段训练过程

### Stacking架构

```
第一阶段：
训练数据 → 基模型A → 预测A
         → 基模型B → 预测B
         → 基模型C → 预测C

第二阶段：
预测A,B,C → 元模型 → 最终预测
```

### Stacking与投票的区别

| 方面 | 投票 | Stacking |
|------|------|----------|
| 组合方式 | 固定规则（平均/投票） | 学习组合权重 |
| 灵活性 | 固定 | 可学习 |
| 复杂度 | 简单 | 较复杂 |
| 潜在性能 | 较好 | 可能更好 |

### 元模型的作用

元模型学习最优的组合方式：
- 可以给不同基模型不同权重
- 可以学习非线性组合
- 可以利用基模型的预测概率

## 元模型设计

### 元模型的选择

| 元模型 | 特点 |
|--------|------|
| 线性回归/逻辑回归 | 简单，防止过拟合 |
| 决策树 | 可学习非线性组合 |
| 神经网络 | 强大但易过拟合 |

**建议**：使用简单模型作为元模型，防止过拟合。

### 元模型的输入

**输入形式**：
- 基模型的预测值（分类：概率，回归：预测值）
- 可以加入原始特征（但可能过拟合）

### 元模型设计原则

1. **简单优于复杂**：防止过拟合
2. **使用概率而非类别**：更多信息
3. **基模型多样化**：不同类型模型
4. **避免数据泄露**：正确的训练流程

## 交叉验证Stacking

### 数据泄露问题

直接用基模型预测作为元模型输入会导致数据泄露：
- 基模型在训练数据上预测
- 元模型看到基模型已见过的数据
- 过拟合风险

### 交叉验证Stacking

使用K折交叉验证避免数据泄露：

```
1. 将训练数据分为K折
2. 对每个基模型：
   for fold k:
     用其他K-1折训练基模型
     用第k折预测，生成预测_k
   组合所有预测，生成完整预测向量
3. 用基模型预测作为元模型输入
4. 元模型在训练数据上训练
5. 测试时：基模型用全部训练数据预测，元模型组合
```

### 交叉验证Stacking流程

```python
import numpy as np
from sklearn.model_selection import KFold

def stacking_cv_predictions(X_train, y_train, X_test, base_models, n_folds=5):
    """交叉验证生成Stacking预测"""
    n_samples = X_train.shape[0]
    n_models = len(base_models)
    
    # 存储训练数据的预测（用于元模型）
    train_predictions = np.zeros((n_samples, n_models))
    
    # 存储测试数据的预测
    test_predictions = np.zeros((X_test.shape[0], n_models))
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for i, model in enumerate(base_models):
        # 测试数据的预测需要平均
        test_preds_fold = np.zeros((X_test.shape[0], n_folds))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            
            # 训练基模型
            model.fit(X_fold_train, y_fold_train)
            
            # 验证集预测
            train_predictions[val_idx, i] = model.predict(X_fold_val)
            
            # 测试集预测
            test_preds_fold[:, fold] = model.predict(X_test)
        
        # 测试预测平均
        test_predictions[:, i] = test_preds_fold.mean(axis=1)
    
    return train_predictions, test_predictions
```

### sklearn Stacking实现

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 定义基模型
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression())
]

# Stacking
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),  # 元模型
    cv=5,  # 交叉验证折数
    stack_method='predict_proba'  # 使用概率
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
```

## Blending方法

### Blending原理

Blending是简化版的Stacking，使用固定验证集而非交叉验证。

**流程**：
```
1. 将训练数据分为训练集和验证集（如70:30）
2. 基模型在训练集上训练，预测验证集和测试集
3. 元模型在验证集预测上训练
4. 测试时用基模型测试预测，元模型组合
```

### Blending vs Stacking

| 方面 | Stacking | Blending |
|------|----------|----------|
| 验证数据 | K折交叉验证 | 固定验证集 |
| 计算成本 | 较高 | 较低 |
| 数据利用 | 更充分 | 验证集数据未用于基模型 |
| 过拟合风险 | 较低 | 较高 |

### Blending实现

```python
def blending_predictions(X_train, y_train, X_test, base_models, val_ratio=0.3):
    """Blending方法"""
    from sklearn.model_selection import train_test_split
    
    # 划分训练集和验证集
    X_blend_train, X_val, y_blend_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=42
    )
    
    n_models = len(base_models)
    
    # 验证集预测
    val_predictions = np.zeros((X_val.shape[0], n_models))
    
    # 测试集预测
    test_predictions = np.zeros((X_test.shape[0], n_models))
    
    for i, model in enumerate(base_models):
        model.fit(X_blend_train, y_blend_train)
        val_predictions[:, i] = model.predict(X_val)
        test_predictions[:, i] = model.predict(X_test)
    
    # 元模型训练
    meta_model.fit(val_predictions, y_val)
    
    # 最终预测
    y_pred = meta_model.predict(test_predictions)
    
    return y_pred
```

## 多层Stacking

### 多层Stacking结构

使用多级元模型：

```
第一层：
基模型A, B, C → 预测A, B, C

第二层：
预测A, B, C → 元模型D, E → 预测D, E

第三层：
预测D, E → 元模型F → 最终预测
```

### 多层Stacking的风险

| 风险 | 描述 |
|------|------|
| 过拟合 | 层数越多越容易过拟合 |
| 数据泄露 | 需要每层都用交叉验证 |
| 复杂度 | 参数更多，调参困难 |

### 多层Stacking实现

```python
from sklearn.ensemble import StackingClassifier

# 第一层基模型
layer1_estimators = [
    ('rf', RandomForestClassifier()),
    ('lr', LogisticRegression())
]

# 第一层Stacking
layer1 = StackingClassifier(
    estimators=layer1_estimators,
    final_estimator=SVC()
)

# 第二层（可以添加更多模型）
layer2_estimators = [
    ('stack1', layer1),
    ('svm', SVC())
]

layer2 = StackingClassifier(
    estimators=layer2_estimators,
    final_estimator=LogisticRegression()
)
```

## 案例实践

### Stacking分类示例

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 数据
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 基模型
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gbdt', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

# Stacking
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='predict_proba'
)

stacking.fit(X_train, y_train)

# 预测
y_pred = stacking.predict(X_test)

print(f"Stacking准确率: {accuracy_score(y_test, y_pred):.4f}")

# 对比各基模型
print("\n各基模型准确率:")
for name, model in estimators:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.4f}")
```

### Stacking回归示例

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 回归数据
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data[:500], data.target[:500]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 基模型
estimators = [
    ('rf', RandomForestRegressor(n_estimators=50)),
    ('gbdt', GradientBoostingRegressor(n_estimators=50)),
    ('ridge', Ridge())
]

# Stacking回归
stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5
)

stacking_reg.fit(X_train, y_train)

y_pred = stacking_reg.predict(X_test)

print(f"Stacking RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):4f}")
print(f"Stacking R²: {r2_score(y_test, y_pred):.4f}")

# 对比各基模型
print("\n各基模型表现:")
for name, model in estimators:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    print(f"{name}: R²={r2:.4f}")
```

### 手动实现Stacking

```python
from sklearn.model_selection import KFold

def manual_stacking(X_train, y_train, X_test, base_models, meta_model, n_folds=5):
    """手动实现Stacking"""
    n_samples = X_train.shape[0]
    n_models = len(base_models)
    
    # 交叉验证生成元模型输入
    meta_features = np.zeros((n_samples, n_models))
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for i, (name, model) in enumerate(base_models):
        for train_idx, val_idx in kfold.split(X_train):
            clone = clone(model)
            clone.fit(X_train[train_idx], y_train[train_idx])
            meta_features[val_idx, i] = clone.predict(X_train[val_idx])
    
    # 训练元模型
    meta_model.fit(meta_features, y_train)
    
    # 生成测试数据的元特征
    test_meta_features = np.zeros((X_test.shape[0], n_models))
    for i, (name, model) in enumerate(base_models):
        model.fit(X_train, y_train)
        test_meta_features[:, i] = model.predict(X_test)
    
    # 最终预测
    y_pred = meta_model.predict(test_meta_features)
    
    return y_pred
```

### Blending实现示例

```python
def blending(X_train, y_train, X_test, base_models, meta_model, val_size=0.3):
    """Blending实现"""
    # 划分验证集
    X_blend, X_val, y_blend, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )
    
    n_models = len(base_models)
    
    # 基模型预测
    val_preds = np.zeros((X_val.shape[0], n_models))
    test_preds = np.zeros((X_test.shape[0], n_models))
    
    for i, (name, model) in enumerate(base_models):
        model.fit(X_blend, y_blend)
        val_preds[:, i] = model.predict(X_val)
        test_preds[:, i] = model.predict(X_test)
    
    # 元模型训练和预测
    meta_model.fit(val_preds, y_val)
    y_pred = meta_model.predict(test_preds)
    
    return y_pred

# 使用Blending
base_models = [
    ('rf', RandomForestClassifier()),
    ('lr', LogisticRegression())
]
meta_model = LogisticRegression()

y_pred = blending(X_train, y_train, X_test, base_models, meta_model)
```

## Stacking最佳实践

### 基模型选择原则

| 原则 | 描述 |
|------|------|
| 多样性 | 选择不同类型的模型 |
| 性能好 | 基模型性能要有一定水平 |
| 互补性 | 模型擅长不同方面 |

### 元模型选择建议

| 建议 | 原因 |
|------|------|
| 使用简单模型 | 防止过拟合 |
| 线性模型优先 | 训练数据有限 |
| 避免复杂神经网络 | 容易过拟合 |

### 防止过拟合

| 方法 | 描述 |
|------|------|
| 交叉验证 | 避免数据泄露 |
| 简单元模型 | 防止元模型过拟合 |
| 特征选择 | 只使用预测，不加原始特征 |

### 其他注意事项

1. **stack_method选择**：
   - predict_proba：使用概率（推荐）
   - predict：使用预测类别
   - predict_log_proba：使用对数概率

2. **passthrough参数**：
   - True：元模型输入包含原始特征
   - False：只使用基模型预测

```python
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    passthrough=False,  # 不使用原始特征
    stack_method='predict_proba'
)
```

## 总结

Stacking是使用元模型组合基模型的集成方法。核心内容包括：
- Stacking原理：两阶段训练，元模型学习组合
- 元模型设计：简单模型防止过拟合
- 交叉验证Stacking：避免数据泄露
- Blending方法：简化版Stacking
- 多层Stacking：多级元模型结构
- 最佳实践：多样性基模型，简单元模型

Stacking是灵活的集成方法，可以进一步提升模型性能。

## 延伸阅读

- [集成学习概述](/2026/05/10/zh-CN/技术文档/机器学习/ensemble-learning/)
- [随机森林](/2026/05/10/zh-CN/技术文档/机器学习/random-forest/)
- [GBDT梯度提升树](/2026/05/10/zh-CN/技术文档/机器学习/gbdt/)
- [模型评估指标](/2026/05/10/zh-CN/技术文档/机器学习/ml-introduction/)
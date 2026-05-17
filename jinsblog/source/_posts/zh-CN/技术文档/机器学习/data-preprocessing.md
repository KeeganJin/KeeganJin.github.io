---
title: 数据预处理
date: 2025-11-01
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 数据预处理, 数据清洗]
---

## 数据清洗的重要性

数据质量直接影响机器学习模型的性能。真实数据通常存在各种问题：
- 数据不完整（缺失值）
- 数据不准确（异常值）
- 数据格式不一致
- 数据冗余

数据清洗是机器学习流程的第一步，占据大量工作时间。

## 缺失值处理

### 缺失值类型

| 类型 | 描述 |
|------|------|
| 完全随机缺失 | 缺失与其他变量无关 |
| 随机缺失 | 缺失与观察到的变量相关 |
| 非随机缺失 | 缺失与缺失值本身相关 |

### 缺失值检测

```python
import pandas as pd
import numpy as np

# 检测缺失值
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

# 检查缺失值数量
print(data.isnull().sum())

# 检查缺失值比例
print(data.isnull().sum() / len(data))
```

### 缺失值处理方法

#### 1. 删除法

**删除缺失样本**：
```python
# 删除含有缺失值的行
data_dropna = data.dropna()

# 删除含有缺失值的列
data_dropcol = data.dropna(axis=1)

# 删除全为缺失值的行
data_dropna_all = data.dropna(how='all')
```

**适用场景**：
- 缺失比例小（<5%）
- 缺失非关键信息
- 数据量充足

**缺点**：可能丢失有用信息

#### 2. 填充法

**均值填充**：
```python
# 数值列用均值填充
data['A'] = data['A'].fillna(data['A'].mean())
```

**中位数填充**：
```python
# 中位数填充（抗异常值）
data['A'] = data['A'].fillna(data['A'].median())
```

**众数填充**：
```python
# 分类数据用众数填充
data['B'] = data['B'].fillna(data['B'].mode()[0])
```

**固定值填充**：
```python
# 用固定值填充
data['C'] = data['C'].fillna(0)
# 或用特定标记
data['C'] = data['C'].fillna('Unknown')
```

**前后值填充**：
```python
# 用前一个值填充
data['A'] = data['A'].fillna(method='ffill')

# 用后一个值填充
data['A'] = data['A'].fillna(method='bfill')
```

**插值法**：
```python
# 线性插值
data['A'] = data['A'].interpolate(method='linear')

# 其他插值方法
data['A'] = data['A'].interpolate(method='polynomial', order=2)
```

#### 3. 预测法

用其他特征预测缺失值：

```python
from sklearn.linear_model import LinearRegression

# 用其他列预测缺失值
def impute_with_model(data, target_col):
    # 分离有缺失和无缺失的数据
    missing = data[data[target_col].isnull()]
    not_missing = data[~data[target_col].isnull()]
    
    # 用其他列作为特征
    feature_cols = [c for c in data.columns if c != target_col]
    
    # 训练模型
    model = LinearRegression()
    model.fit(not_missing[feature_cols], not_missing[target_col])
    
    # 预测缺失值
    predicted = model.predict(missing[feature_cols])
    
    # 填充
    data.loc[data[target_col].isnull(), target_col] = predicted
    
    return data
```

#### 4. 多重插补

创建多个完整数据集：

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=0)
data_imputed = imputer.fit_transform(data)
```

## 异常值检测与处理

### 异常值定义

偏离正常范围的数据点。可能是：
- 测量错误
- 数据录入错误
- 真实的极端值

### 异常值检测方法

#### 1. 统计方法

**基于标准差**：
```python
def detect_outliers_std(data, threshold=3):
    """基于标准差的异常检测"""
    mean = np.mean(data)
    std = np.std(data)
    
    z_scores = np.abs((data - mean) / std)
    outliers = data[z_scores > threshold]
    
    return outliers
```

**基于四分位距（IQR）**：
```python
def detect_outliers_iqr(data, multiplier=1.5):
    """基于IQR的异常检测"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return outliers, lower_bound, upper_bound
```

#### 2. 可视化方法

**箱线图**：
```python
import matplotlib.pyplot as plt

plt.boxplot(data)
plt.show()
# 异常值显示为圆点
```

**散点图**：
```python
plt.scatter(range(len(data)), data)
plt.show()
```

#### 3. 机器学习方法

**Isolation Forest**：
```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1)
outliers = clf.fit_predict(data)
# -1 表示异常值
```

**DBSCAN聚类**：
```python
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=3, min_samples=2)
labels = clustering.fit_predict(data.reshape(-1, 1))
# -1 标签表示异常点
```

### 异常值处理方法

#### 1. 删除异常值

```python
def remove_outliers(data, col):
    outliers, lower, upper = detect_outliers_iqr(data[col])
    data_clean = data[(data[col] >= lower) & (data[col] <= upper)]
    return data_clean
```

#### 2. 替换异常值

```python
def replace_outliers(data, col, method='median'):
    outliers, lower, upper = detect_outliers_iqr(data[col])
    
    if method == 'median':
        replacement = data[col].median()
    elif method == 'mean':
        replacement = data[col].mean()
    elif method == 'boundary':
        replacement_lower = lower
        replacement_upper = upper
    
    # 替换为边界值
    data.loc[data[col] < lower, col] = lower
    data.loc[data[col] > upper, col] = upper
    
    return data
```

#### 3. 保留异常值

有时异常值有意义，不应删除：
- 金融欺诈检测（异常是目标）
- 设备故障预测
- 医学诊断

## 数据标准化与归一化

### 为什么需要标准化

- 不同特征量纲不同
- 影响梯度下降效率
- 影响某些算法效果（如KNN、SVM）
- 神经网络训练稳定性

### 标准化方法

#### Z-score标准化

$x_{scaled} = \frac{x - \mu}{\sigma}$

结果：均值为0，标准差为1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

#### Min-Max归一化

$x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}$

结果：值在[0, 1]范围

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
```

#### 其他标准化方法

| 方法 | 公式 | 适用场景 |
|------|------|----------|
| MaxAbs | $x / |x_{max}|$ | 稀疏数据 |
| Robust | $(x - Q2) / (Q3 - Q1)$ | 有异常值 |
| Log | $\log(x)$ | 大范围数据 |
| Power | $x^\lambda$ | 特定分布 |

```python
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

maxabs_scaler = MaxAbsScaler()
robust_scaler = RobustScaler()

data_maxabs = maxabs_scaler.fit_transform(data)
data_robust = robust_scaler.fit_transform(data)
```

### 标准化注意事项

1. **训练测试分离**：用训练数据拟合scaler，转换测试数据
```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 用训练数据的参数
```

2. **保存scaler**：部署时需要用相同的scaler
3. **反标准化**：预测结果可能需要反标准化
```python
y_pred_original = scaler.inverse_transform(y_pred_scaled)
```

## 特征编码

### 分类特征编码

#### One-Hot编码

每个类别创建一个二进制列：

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(data[['category']])

# 或用pandas
encoded = pd.get_dummies(data['category'])
```

**优点**：
- 不假设类别顺序
- 适合大多数算法

**缺点**：
- 高维度问题（类别多时）
- 稀疏矩阵

#### Label编码

每个类别映射为整数：

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded = encoder.fit_transform(data['category'])
```

**适用场景**：
- 类别有序（如低/中/高）
- 树模型（能处理整数编码）

**注意**：不适合线性模型，会引入错误顺序关系

#### Target编码

用目标变量的统计量编码：

```python
def target_encode(data, category_col, target_col):
    """目标编码"""
    means = data.groupby(category_col)[target_col].mean()
    encoded = data[category_col].map(means)
    return encoded
```

**注意**：
- 可能过拟合
- 需要平滑处理

#### 二进制编码

类别转换为二进制表示：

```python
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['category'])
encoded = encoder.fit_transform(data)
```

### 编码选择指南

| 场景 | 推荐编码 |
|------|----------|
| 类别少（<10） | One-Hot |
| 类别有序 | Label |
| 类别多（>10） | Binary/Target |
| 树模型 | Label |
| 神经网络 | One-Hot |
| 高基数分类 | Target（需平滑） |

## 数据划分策略

### 训练/验证/测试划分

标准划分比例：
- 训练集：60-80%
- 验证集：10-20%
- 测试集：10-20%

```python
from sklearn.model_selection import train_test_split

# 先划分训练和测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 再从训练集中划分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
```

### 划分注意事项

#### 随机划分

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
```

#### 分层划分

保持类别比例：

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

#### 时间序列划分

时间数据按时间划分：

```python
# 时间序列不能随机划分
train_size = int(len(data) * 0.8)
train = data[:train_size]
test = data[train_size:]
```

#### 交叉验证

```python
from sklearn.model_selection import KFold, StratifiedKFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

## 数据预处理最佳实践

### 流程建议

1. **先了解数据**：EDA探索数据特征
2. **先处理缺失**：缺失值影响其他处理
3. **再处理异常**：异常值影响标准化
4. **最后标准化**：标准化在编码后

### 代码示例

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 定义数值列和分类列
numeric_features = ['age', 'income']
categorical_features = ['gender', 'city']

# 数值处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类处理管道
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 组合处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 完整流程
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# 直接使用
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

## 总结

数据预处理是机器学习的关键步骤。主要内容包括：
- 缺失值处理：删除、填充、预测、多重插补
- 异常值处理：检测（统计/可视化/机器学习）、删除或替换
- 标准化：Z-score、Min-Max、Robust等
- 特征编码：One-Hot、Label、Target、Binary
- 数据划分：训练/验证/测试、分层、时间序列、交叉验证

数据预处理直接影响模型性能，需要根据数据特点和任务需求选择合适的处理方法。

## 延伸阅读

- [机器学习概述](/2026/05/10/zh-CN/技术文档/机器学习/ml-introduction/)
- [特征工程](/2026/05/10/zh-CN/技术文档/机器学习/feature-engineering/)
- [统计学基础](/2026/05/10/zh-CN/技术文档/机器学习/statistics/)
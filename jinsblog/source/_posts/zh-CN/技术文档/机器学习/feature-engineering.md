---
title: 特征工程
date: 2025-11-04
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 特征工程, 特征选择]
---

## 特征工程的重要性

特征工程是将原始数据转换为更能代表问题本质的特征的过程。好的特征工程可以显著提升模型性能。

Andrew Ng：**"Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering."**

### 特征工程的内容

| 内容 | 描述 |
|------|------|
| 特征选择 | 选择最有用的特征 |
| 特征提取 | 从原始数据提取新特征 |
| 特征构造 | 创建新的特征 |
| 特征变换 | 改变特征形式 |

## 特征选择方法

### 为什么需要特征选择

- 减少过拟合
- 提高模型性能
- 加快训练速度
- 提高可解释性

### 特征选择方法分类

| 类型 | 方法 | 特点 |
|------|------|------|
| 过滤法 | 统计指标筛选 | 快速，独立于模型 |
| 包装法 | 模型评估筛选 | 准确，计算成本高 |
| 嵌入法 | 模型内置筛选 | 高效，依赖模型 |

### 过滤法（Filter）

基于统计指标选择特征，不依赖模型。

#### 方差过滤

删除方差过小的特征（信息量少）：

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)
```

#### 相关系数过滤

选择与目标相关性高的特征：

```python
import pandas as pd

# 计算相关系数
correlations = data.corr()['target'].abs().sort_values(ascending=False)

# 选择相关性最高的特征
selected_features = correlations[correlations > 0.5].index.tolist()
```

#### 卡方检验

分类任务中特征与目标的独立性检验：

```python
from sklearn.feature_selection import chi2, SelectKBest

selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)
```

#### 信息增益/互信息

衡量特征与目标的信息关系：

```python
from sklearn.feature_selection import mutual_info_classif

scores = mutual_info_classif(X, y)
selected_features = [i for i, score in enumerate(scores) if score > threshold]
```

### 包装法（Wrapper）

使用模型评估特征子集。

#### 递归特征消除（RFE）

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# 被选中的特征
selected_features = rfe.support_
```

#### 前向特征选择

逐步添加特征：

```python
from sklearn.feature_selection import SequentialFeatureSelector

sfs = SequentialFeatureSelector(model, n_features_to_select=10, direction='forward')
X_selected = sfs.fit_transform(X, y)
```

#### 后向特征消除

逐步删除特征：

```python
sfs = SequentialFeatureSelector(model, n_features_to_select=10, direction='backward')
X_selected = sfs.fit_transform(X, y)
```

### 嵌入法（Embedded）

模型训练过程中自动选择特征。

#### L1正则化（Lasso）

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X, y)

# 非零系数的特征被选中
selected_features = model.coef_ != 0
```

#### 树模型特征重要性

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

# 特征重要性
importances = model.feature_importances_
selected_features = [i for i, imp in enumerate(importances) if imp > threshold]
```

## 特征提取技术

### 主成分分析（PCA）

将高维数据投影到低维空间：

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # 保留95%方差
X_pca = pca.fit_transform(X)

print(f"原始维度: {X.shape[1]}")
print(f"PCA后维度: {X_pca.shape[1]}")
```

**PCA原理**：
- 找到数据方差最大的方向（主成分）
- 投影到这些方向上
- 保留主要信息，减少维度

**适用场景**：
- 高维数据降维
- 去除噪声
- 可视化（降到2维或3维）

### 线性判别分析（LDA）

有监督降维，最大化类别分离：

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

**LDA vs PCA**：
- PCA：无监督，最大化方差
- LDA：有监督，最大化类别分离

### t-SNE

非线性降维，适合可视化：

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30)
X_tsne = tsne.fit_transform(X)
```

### 自动编码器

神经网络非线性降维：

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 编码器
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)

# 解码器
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(X.shape[1], activation='sigmoid')(decoded)

# 自编码器
autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=100)

# 提取特征
X_encoded = encoder.predict(X)
```

## 特征构造策略

### 数值特征构造

#### 数学变换

```python
# 对数变换（处理大范围数据）
data['log_income'] = np.log(data['income'])

# 幂变换
data['square_age'] = data['age'] ** 2

# 根号变换
data['sqrt_price'] = np.sqrt(data['price'])
```

#### 统计特征

```python
# 统计聚合
data['mean_per_category'] = data.groupby('category')['value'].transform('mean')
data['std_per_category'] = data.groupby('category')['value'].transform('std')
data['count_per_category'] = data.groupby('category')['value'].transform('count')
```

#### 时间特征

```python
# 从时间戳提取特征
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
data['weekday'] = data['date'].dt.weekday
data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)
```

### 分类特征构造

#### 组合特征

```python
# 组合两个分类特征
data['gender_age'] = data['gender'] + '_' + data['age_group']

# 或用乘法交互
data['gender_age_interaction'] = data['gender'] * data['age']
```

#### 计数特征

```python
# 类别出现次数
data['category_count'] = data.groupby('category')['id'].transform('count')

# 类别占比
data['category_ratio'] = data['category_count'] / len(data)
```

### 文本特征构造

#### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
tfidf_features = vectorizer.fit_transform(data['text'])
```

#### 词嵌入

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data['text'])
```

#### 文本统计特征

```python
# 文本长度
data['text_length'] = data['text'].str.len()

# 词数
data['word_count'] = data['text'].str.split().str.len()

# 特殊字符数
data['special_char_count'] = data['text'].str.count('[!@#$%^&*]')
```

### 组合/交互特征

```python
# 加法交互
data['age_income'] = data['age'] + data['income']

# 乘法交互
data['age_income_product'] = data['age'] * data['income']

# 比率
data['income_per_age'] = data['income'] / data['age']

# 差值
data['price_diff'] = data['price'] - data['base_price']
```

## 特征重要性评估

### 模型内置重要性

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

importances = model.feature_importances_

# 可视化
import matplotlib.pyplot as plt
plt.bar(range(len(importances)), importances)
plt.xticks(range(len(importances)), feature_names)
plt.show()
```

### SHAP值

更细粒度的特征重要性分析：

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 可视化
shap.summary_plot(shap_values, X)
```

### Permutation Importance

不依赖模型的重要性评估：

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)

importances = result.importances_mean
```

## 特征工程最佳实践

### 流程建议

```
原始数据 → 数据清洗 → 特征理解 → 特征构造 → 特征选择 → 特征变换 → 模型训练
```

### 关键原则

1. **理解数据**：了解特征含义和业务背景
2. **理解问题**：特征要服务于预测目标
3. **迭代实验**：特征工程需要不断迭代
4. **避免过拟合**：特征构造要合理，不引入未来信息
5. **保持简单**：复杂特征不一定更好

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| 特征过多 | 特征选择、降维 |
| 特征无关 | 相关性分析、特征选择 |
| 特征冗余 | 去除重复、PCA |
| 特征分布异常 | 变换、标准化 |
| 特征缺失 | 填充、创建缺失指示 |

## 总结

特征工程是机器学习中最重要的环节之一。主要内容包括：
- 特征选择：过滤法、包装法、嵌入法
- 特征提取：PCA、LDA、t-SNE、自动编码器
- 特征构造：数学变换、统计特征、时间特征、组合特征、文本特征
- 特征重要性评估：模型内置、SHAP、Permutation

好的特征工程需要理解数据和问题，持续迭代实验。特征工程直接影响模型性能。

## 延伸阅读

- [机器学习概述](/2026/05/10/zh-CN/技术文档/机器学习/ml-introduction/)
- [数据预处理](/2026/05/10/zh-CN/技术文档/机器学习/data-preprocessing/)
- [线性代数基础](/2026/05/10/zh-CN/技术文档/机器学习/linear-algebra/)
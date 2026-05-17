---
title: 贝叶斯分类器
date: 2025-10-24
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 贝叶斯, 分类算法]
---

## 贝叶斯分类原理

### 贝叶斯定理

贝叶斯定理描述了条件概率之间的关系：

$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

在分类任务中：
$P(y=c|\mathbf{x}) = \frac{P(\mathbf{x}|y=c)P(y=c)}{P(\mathbf{x})$

其中：
- $P(y=c|\mathbf{x})$：后验概率（给定特征，样本属于类别c的概率）
- $P(\mathbf{x}|y=c)$：似然（类别c中观察到特征$\mathbf{x}$的概率）
- $P(y=c)$：先验概率（类别c的整体概率）
- $P(\mathbf{x})$：证据（特征的总体概率）

### 贝叶斯分类决策

**决策规则**：选择后验概率最大的类别

$\hat{y} = \arg\max_c P(y=c|\mathbf{x})$

由于 $P(\mathbf{x})$ 对所有类别相同，可以忽略：

$\hat{y} = \arg\max_c P(\mathbf{x}|y=c)P(y=c)$

### 贝叶斯分类的优势

| 优势 | 描述 |
|------|------|
| 理论基础扎实 | 基于概率论 |
| 可处理不确定性 | 输出概率而非硬分类 |
| 可增量学习 | 新数据可更新概率 |
| 计算简单 | 某些情况下计算高效 |

## 朴素贝叶斯

### 朴素贝叶斯的"朴素"假设

假设所有特征相互独立：
$P(\mathbf{x}|y=c) = P(x_1|y=c) \times P(x_2|y=c) \times ... \times P(x_n|y=c)$

**后验概率**：
$P(y=c|\mathbf{x}) = \frac{P(y=c)\prod_{j=1}^{n}P(x_j|y=c)}{P(\mathbf{x})}$

**决策函数**：
$\hat{y} = \arg\max_c P(y=c)\prod_{j=1}^{n}P(x_j|y=c)$

### 为什么叫"朴素"

现实中特征通常不独立，所以这个假设是"朴素"的（naive）。

但朴素贝叶斯在实践中表现良好，因为：
- 分类只需要概率的相对大小，不需要精确值
- 简化的计算成本大大降低

### 参数估计

#### 先验概率估计

$P(y=c) = \frac{N_c}{N}$

其中 $N_c$ 是类别c的样本数，$N$ 是总样本数。

#### 条件概率估计

**连续特征**：假设服从某种分布（如正态分布）

$P(x_j|y=c) = \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left(-\frac{(x_j - \mu_c)^2}{2\sigma_c^2}\right)$

**离散特征**：频率估计

$P(x_j=v|y=c) = \frac{N_{c,v}}{N_c}$

### 朴素贝叶斯的类型

| 类型 | 特征假设 | 适用场景 |
|------|----------|----------|
| GaussianNB | 特征服从正态分布 | 连续特征 |
| MultinomialNB | 特征服从多项分布 | 文本分类 |
| BernoulliNB | 特征服从伯努利分布 | 二值特征 |

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

# 高斯朴素贝叶斯
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)  # 输出概率

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
```

### Multinomial Naive Bayes

常用于文本分类：

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 文本特征提取
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(text_train)
X_test_vec = vectorizer.transform(text_test)

# 多项朴素贝叶斯
mnb = MultinomialNB(alpha=1.0)  # alpha是平滑参数
mnb.fit(X_train_vec, y_train)

y_pred = mnb.predict(X_test_vec)
```

### Bernoulli Naive Bayes

用于二值特征：

```python
from sklearn.naive_bayes import BernoulliNB

# 伯努利朴素贝叶斯
bnb = BernoulliNB(alpha=1.0, binarize=0.0)  # binarize阈值
bnb.fit(X_train_binary, y_train)

y_pred = bnb.predict(X_test_binary)
```

### Laplace平滑

防止零概率问题：

$P(x_j=v|y=c) = \frac{N_{c,v} + \alpha}{N_c + \alpha K}$

其中 $\alpha$ 是平滑参数（通常为1），$K$ 是特征取值数。

```python
# 不同平滑参数
for alpha in [0.0, 0.1, 1.0, 10.0]:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train_vec, y_train)
    print(f"alpha={alpha}: 准确率={mnb.score(X_test_vec, y_test):.4f}")
```

## 贝叶斯网络

### 贝叶斯网络概念

贝叶斯网络是有向无环图（DAG），表示变量之间的依赖关系。

**组成**：
- **节点**：随机变量
- **边**：变量间的依赖关系
- **条件概率表（CPT）**：每个节点的条件概率

### 贝叶斯网络的优势

| 优势 | 描述 |
|------|------|
| 表示依赖 | 不需要朴素假设 |
| 可解释性 | 图结构清晰 |
| 不确定性推理 | 处理不完整数据 |
| 知识整合 | 结合专家知识 |

### 贝叶斯网络的推理

**目标**：计算某些变量的后验概率。

**推理类型**：
- **精确推理**：计算精确后验概率
- **近似推理**：蒙特卡洛方法等

### 贝叶斯网络的构建

**方法**：
1. **专家构建**：由专家设计网络结构
2. **数据学习**：从数据学习网络结构和参数

**结构学习**：
- 找到最优的DAG结构
- 评分函数：BIC、AIC等
- 搜索算法：贪心搜索、遗传算法等

### 贝叶斯网络示例

```python
# 使用pgmpy库构建贝叶斯网络
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# 定义网络结构
model = BayesianNetwork([('A', 'C'), ('B', 'C')])

# 定义条件概率表
cpd_a = TabularCPD('A', 2, [[0.3], [0.7]])
cpd_b = TabularCPD('B', 2, [[0.4], [0.6]])
cpd_c = TabularCPD('C', 2, 
                   [[0.1, 0.2, 0.3, 0.4],
                    [0.9, 0.8, 0.7, 0.6]],
                   evidence=['A', 'B'], evidence_card=[2, 2])

# 添加CPD
model.add_cpds(cpd_a, cpd_b, cpd_c)

# 推理
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
result = infer.query(['C'], evidence={'A': 1, 'B': 0})
print(result)
```

## 参数估计

### 最大似然估计（MLE）

从数据估计参数：

$\hat{\theta}_{MLE} = \arg\max_\theta P(D|\theta)$

**朴素贝叶斯MLE**：
- 先验：$\hat{P}(y=c) = \frac{N_c}{N}$
- 条件概率：$\hat{P}(x_j=v|y=c) = \frac{N_{c,v}}{N_c}$

### 最大后验估计（MAP）

考虑参数的先验：

$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta P(D|\theta)P(\theta)$

**朴素贝叶斯MAP**（Laplace平滑）：
$\hat{P}(x_j=v|y=c) = \frac{N_{c,v} + \alpha}{N_c + \alpha K}$

### 贝叶斯参数估计

考虑参数的完整后验分布：

$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$

**预测**：
$P(x_{new}|D) = \int P(x_{new}|\theta)P(\theta|D)d\theta$

## 半朴素贝叶斯

### 半朴素贝叶斯的概念

放宽特征独立假设，考虑部分特征之间的依赖。

### 独依赖估计（ODE）

假设每个特征最多依赖一个其他特征：

$P(\mathbf{x}|y=c) = \prod_{j=1}^{n}P(x_j|y=c, x_{pa(j)})$

其中 $x_{pa(j)}$ 是 $x_j$ 的父特征。

### SPODE算法

假设所有特征都依赖一个"超父"特征：

$P(\mathbf{x}|y=c) = \prod_{j=1}^{n}P(x_j|y=c, x_{super})$

选择最优超父特征：

```python
# 选择超父
best_parent = None
best_accuracy = 0

for parent_idx in range(n_features):
    # 使用该父特征训练ODE模型
    accuracy = evaluate_spode(parent_idx)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_parent = parent_idx
```

### TAN算法（Tree Augmented Naive Bayes）

构建特征依赖树：
- 每个特征最多依赖一个其他特征
- 使用最大加权树构建依赖结构

**步骤**：
1. 计算特征间的条件互信息
2. 构建最大加权树
3. 确定依赖关系方向

### AODE算法

集成多个SPODE模型：
$P(y=c|\mathbf{x}) \propto \sum_{i: N_{c,x_i} \geq m} P(y=c, x_i)\prod_{j=1}^{n}P(x_j|y=c, x_i)$

只使用出现次数足够多的特征作为超父（避免估计不准确）。

## 案例实践

### 朴素贝叶斯文本分类

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 加载文本数据
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

# 特征提取
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(newsgroups.data)

# 训练朴素贝叶斯
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_vec, newsgroups.target)

# 测试
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
X_test_vec = vectorizer.transform(newsgroups_test.data)
y_pred = mnb.predict(X_test_vec)

print(f"准确率: {accuracy_score(newsgroups_test.target, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(newsgroups_test.target, y_pred, target_names=categories))
```

### GaussianNB分类示例

```python
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# 加载鸢尾花数据
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 查看每个类别每个特征的均值和方差
print("各类别特征均值:")
print(gnb.theta_)

print("\n各类别特征方差:")
print(gnb.var_)

# 预测概率
y_prob = gnb.predict_proba(X_test)
print("\n前5个样本概率:")
print(y_prob[:5])
```

### 不同朴素贝叶斯类型对比

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler, Binarizer

# 准备不同特征形式
# GaussianNB: 原始连续特征
# MultinomialNB: 非负特征（需要归一化）
# BernoulliNB: 二值特征

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

binarizer = Binarizer(threshold=0.5)
X_binary = binarizer.fit_transform(X_scaled)

# 对比
models = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB()
}

for name, model in models.items():
    if name == 'GaussianNB':
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
    elif name == 'MultinomialNB':
        model.fit(scaler.transform(X_train), y_train)
        acc = model.score(scaler.transform(X_test), y_test)
    else:
        model.fit(X_binary, y_train)
        acc = model.score(binarizer.transform(scaler.transform(X_test)), y_test)
    
    print(f"{name}: 准确率={acc:.4f}")
```

### 贝叶斯网络示例

```python
# 简化的贝叶斯网络实现
class SimpleBayesianNetwork:
    def __init__(self, structure):
        """structure: dict, {parent: [children]}"""
        self.structure = structure
        self.cpds = {}
    
    def fit(self, X, y):
        """学习条件概率"""
        # 学习先验
        classes, counts = np.unique(y, return_counts=True)
        self.class_prior = counts / len(y)
        self.classes = classes
        
        # 学习条件概率
        for c in classes:
            X_c = X[y == c]
            self.cpds[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0)
            }
    
    def predict(self, X):
        """预测"""
        probs = []
        for c in self.classes:
            prior = self.class_prior[c]
            likelihood = np.prod(
                np.exp(-(X - self.cpds[c]['mean'])**2 / (2*self.cpds[c]['var'])) /
                np.sqrt(2*np.pi*self.cpds[c]['var']),
                axis=1
            )
            probs.append(prior * likelihood)
        
        return self.classes[np.argmax(probs, axis=0)]

# 使用示例
bn = SimpleBayesianNetwork({'class': ['feature1', 'feature2']})
bn.fit(X_train, y_train)
y_pred = bn.predict(X_test)
```

## 贝叶斯分类器的优缺点

### 优点

| 优点 | 描述 |
|------|------|
| 计算高效 | 训练和预测速度快 |
| 样本需求少 | 小样本也能工作 |
| 可增量学习 | 新数据可更新模型 |
| 输出概率 | 提供置信度信息 |
| 处理缺失值 | 可处理部分特征缺失 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 朴素假设 | 特征独立假设通常不成立 |
| 特征相关敏感 | 特征相关时性能下降 |
| 零概率问题 | 需要平滑处理 |

## 总结

贝叶斯分类器是基于概率理论的分类模型。核心内容包括：
- 贝叶斯分类原理：后验概率最大化
- 朴素贝叶斯：特征独立假设，计算简单高效
- 贝叶斯网络：表示变量依赖关系的有向图
- 参数估计：MLE、MAP、贝叶斯估计
- 半朴素贝叶斯：放宽独立假设

朴素贝叶斯虽然假设简单，但在文本分类等场景中表现良好。

## 延伸阅读

- [概率论基础](/2026/05/10/zh-CN/技术文档/机器学习/probability-theory/)
- [统计学基础](/2026/05/10/zh-CN/技术文档/机器学习/statistics/)
- [逻辑回归](/2026/05/10/zh-CN/技术文档/机器学习/logistic-regression/)
- [决策树](/2026/05/10/zh-CN/技术文档/机器学习/decision-tree/)
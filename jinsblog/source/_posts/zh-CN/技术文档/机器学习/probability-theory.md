---
title: 概率论基础
date: 2026-04-10
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 数学基础, 概率论]
---

## 概率基本概念

### 随机事件

**随机试验**：结果不确定但所有可能结果已知的试验。

**样本空间**：所有可能结果的集合，记作Ω。

**随机事件**：样本空间的子集。

### 概率定义

**古典概型**：
$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{有利结果数}}{\text{所有可能结果数}}$

**统计概率**：
$P(A) = \lim_{n \to \infty} \frac{n_A}{n}$

其中 $n_A$ 是事件A发生的次数。

### 概率公理

1. **非负性**：$P(A) \geq 0$
2. **规范性**：$P(\Omega) = 1$
3. **可加性**：若 $A \cap B = \emptyset$，则 $P(A \cup B) = P(A) + P(B)$

### 概率基本性质

- $P(\bar{A}) = 1 - P(A)$
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- 若 $A \subseteq B$，则 $P(A) \leq P(B)$
- $P(\emptyset) = 0$

## 条件概率与贝叶斯定理

### 条件概率

事件A在事件B已发生条件下的概率：
$P(A|B) = \frac{P(A \cap B)}{P(B)}$

### 乘法公式

$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$

### 全概率公式

若 $B_1, B_2, ..., B_n$ 是完备事件组（互斥且穷尽）：
$P(A) = \sum_{i=1}^{n} P(A|B_i) P(B_i)$

### 贝叶斯定理

$P(B_i|A) = \frac{P(A|B_i) P(B_i)}{\sum_{j=1}^{n} P(A|B_j) P(B_j)}$

**理解贝叶斯定理**：
- $P(B_i)$：先验概率（Prior）
- $P(A|B_i)$：似然（Likelihood）
- $P(B_i|A)$：后验概率（Posterior）

### 贝叶斯定理应用

**医学诊断示例**：
- 患病率：$P(D) = 0.01$
- 检测阳性（患病）：$P(+|D) = 0.99$
- 检测阳性（未患病）：$P(+|\bar{D}) = 0.05$

检测结果阳性时，实际患病概率：
$P(D|+) = \frac{P(+|D) P(D)}{P(+|D) P(D) + P(+|\bar{D}) P(\bar{D)}$
$= \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} \approx 0.167$

## 常见概率分布

### 离散分布

#### 伯努利分布

单次试验，成功概率p：
$P(X = 1) = p, \quad P(X = 0) = 1-p$

期望：$E[X] = p$
方差：$Var(X) = p(1-p)$

#### 二项分布

n次独立伯努利试验中成功的次数：
$P(X = k) = C_n^k p^k (1-p)^{n-k}$

期望：$E[X] = np$
方差：$Var(X) = np(1-p)$

```python
import numpy as np
from scipy.stats import binom

# n=10, p=0.5 的二项分布
n, p = 10, 0.5
x = np.arange(0, n+1)
probs = binom.pmf(x, n, p)
```

#### 泊松分布

单位时间内稀有事件发生的次数：
$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$

期望：$E[X] = \lambda$
方差：$Var(X) = \lambda$

#### 几何分布

首次成功所需的试验次数：
$P(X = k) = p(1-p)^{k-1}$

### 连续分布

#### 均匀分布

在区间[a, b]上均匀分布：
$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$

期望：$E[X] = \frac{a+b}{2}$
方差：$Var(X) = \frac{(b-a)^2}{12}$

#### 正态分布（高斯分布）

最重要的连续分布：
$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

记作 $X \sim N(\mu, \sigma^2)$

期望：$E[X] = \mu$
方差：$Var(X) = \sigma^2$

**标准正态分布**：$\mu = 0, \sigma = 1$
$f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$

```python
import numpy as np
from scipy.stats import norm

# 正态分布
mu, sigma = 0, 1
x = np.linspace(-4, 4, 100)
pdf = norm.pdf(x, mu, sigma)
cdf = norm.cdf(x, mu, sigma)
```

#### 指数分布

事件发生间隔时间：
$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$

期望：$E[X] = \frac{1}{\lambda}$
方差：$Var(X) = \frac{1}{\lambda^2}$

## 期望与方差

### 期望（均值）

离散随机变量：
$E[X] = \sum_{i} x_i P(x_i)$

连续随机变量：
$E[X] = \int_{-\infty}^{\infty} x f(x) dx$

### 期望性质

- $E[c] = c$（常数）
- $E[cX] = cE[X]$
- $E[X + Y] = E[X] + E[Y]$
- $E[XY] = E[X]E[Y]$（当X、Y独立）

### 方差

方差度量随机变量的离散程度：
$Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$

标准差：$\sigma = \sqrt{Var(X)}$

### 方差性质

- $Var(c) = 0$
- $Var(cX) = c^2 Var(X)$
- $Var(X + c) = Var(X)$
- $Var(X + Y) = Var(X) + Var(Y)$（当X、Y独立）

## 协方差与相关系数

### 协方差

度量两个随机变量的相关性：
$Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$

### 协方差性质

- $Cov(X, X) = Var(X)$
- $Cov(X, Y) = Cov(Y, X)$
- $Cov(aX, bY) = ab \cdot Cov(X, Y)$
- $Cov(X + Y, Z) = Cov(X, Z) + Cov(Y, Z)$

### 相关系数

标准化协方差，范围[-1, 1]：
$\rho_{XY} = \frac{Cov(X, Y)}{\sqrt{Var(X) Var(Y)}}$

**解释**：
- $\rho = 1$：完全正相关
- $\rho = -1$：完全负相关
- $\rho = 0$：不相关

### 协方差矩阵

对随机向量 $\mathbf{X} = (X_1, X_2, ..., X_n)$：
$\mathbf{\Sigma} = \begin{bmatrix} Var(X_1) & Cov(X_1, X_2) & ... \\ Cov(X_2, X_1) & Var(X_2) & ... \\ \vdots & \vdots & \ddots \end{bmatrix}$

## 大数定律与中心极限定理

### 大数定律

**弱大数定律**：样本均值收敛于期望

$\lim_{n \to \infty} P\left(\left|\frac{S_n}{n} - \mu\right| < \epsilon\right) = 1$

其中 $S_n = X_1 + X_2 + ... + X_n$

**意义**：大量独立重复试验的平均结果稳定于理论期望。

### 中心极限定理

无论原始分布是什么，样本均值在大样本下近似正态分布：

$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \to N(0, 1)$

**意义**：
- 解释了正态分布的普遍性
- 统计推断的理论基础
- 机器学习中很多方法的理论支撑

```python
import numpy as np

# 中心极限定理演示
# 从均匀分布（非正态）采样
population = np.random.uniform(0, 1, 100000)

# 多次采样，每次取样本均值
sample_means = [np.mean(np.random.choice(population, 100)) 
                for _ in range(1000)]

# 样本均值分布接近正态
import matplotlib.pyplot as plt
plt.hist(sample_means, bins=30)
plt.show()
```

## 概率论在机器学习中的应用

### 模型假设

很多机器学习模型基于概率假设：
- 线性回归假设误差服从正态分布
- 贝叶斯分类器基于概率推理
- 生成模型基于概率分布

### 参数估计

- 最大似然估计（MLE）
- 最大后验估计（MAP）
- 贝叶斯估计

### 模型评估

- 概率预测（如softmax输出）
- 信息熵、交叉熵
- KL散度

### 不确定性量化

- 模型预测的概率置信度
- 贝叶斯神经网络
- 不确定性传播

## 总结

概率论是机器学习的重要数学基础。核心概念包括：概率定义与公理、条件概率与贝叶斯定理、常见概率分布（离散和连续）、期望与方差、协方差与相关系数、大数定律与中心极限定理。

贝叶斯定理是机器学习中贝叶斯方法的核心，正态分布是最常用的分布，中心极限定理解释了很多统计方法的理论基础。

## 延伸阅读

- [统计学基础](/2026/05/10/zh-CN/技术文档/机器学习/statistics/)
- [贝叶斯分类器](/2026/05/10/zh-CN/技术文档/机器学习/bayesian-classifier/)
- [损失函数详解](/2026/05/10/zh-CN/技术文档/机器学习/loss-functions/)
---
title: 统计学基础
date: 2026-04-20
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 数学基础, 统计学]
---

## 统计量与统计推断

### 统计学的任务

统计学主要解决两类问题：
1. **描述统计**：描述数据特征
2. **推断统计**：从样本推断总体

### 常用统计量

#### 集中趋势统计量

| 统计量 | 公式 | 描述 |
|--------|------|------|
| 均值 | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | 平均值 |
| 中位数 | 排序后中间值 | 抗异常值 |
| 众数 | 出现最多的值 | 分类数据适用 |

#### 离散程度统计量

| 统计量 | 公式 | 描述 |
|--------|------|------|
| 方差 | $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$ | 数据离散程度 |
| 标准差 | $s = \sqrt{s^2}$ | 方差的平方根 |
| 极差 | $R = x_{max} - x_{min}$ | 最大值减最小值 |
| 四分位距 | $IQR = Q_3 - Q_1$ | 中间50%数据范围 |

#### 形状统计量

**偏度**：衡量分布的不对称性
$Skewness = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^3}{(n-1)s^3}$

- 偏度 > 0：右偏（正偏）
- 偏度 < 0：左偏（负偏）
- 偏度 = 0：对称

**峰度**：衡量分布的尖锐程度
$Kurtosis = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^4}{(n-1)s^4} - 3$

- 峰度 > 0：比正态分布更尖锐
- 峰度 < 0：比正态分布更平坦
- 峰度 = 0：与正态分布相同

```python
import numpy as np
from scipy import stats

data = np.random.normal(0, 1, 1000)

# 统计量计算
mean = np.mean(data)
median = np.median(data)
var = np.var(data)
std = np.std(data)
skew = stats.skew(data)
kurt = stats.kurtosis(data)
```

## 点估计与区间估计

### 点估计

从样本估计总体参数的单个值。

#### 估计量性质

| 性质 | 描述 |
|------|------|
| 无偏性 | $E[\hat{\theta}] = \theta$ |
| 有效性 | 方差最小 |
| 一致性 | $n \to \infty$ 时收敛于真值 |

#### 常见点估计

**均值估计**：
$\hat{\mu} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$

**方差估计**（无偏）：
$\hat{\sigma^2} = s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$

### 区间估计

给出参数估计的范围和置信度。

#### 置信区间

$P(\hat{\theta}_L \leq \theta \leq \hat{\theta}_U) = 1 - \alpha$

$1-\alpha$ 是置信水平（如95%）。

#### 均值的置信区间

当总体方差已知：
$\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$

当总体方差未知（用样本方差）：
$\bar{x} \pm t_{\alpha/2}(n-1) \frac{s}{\sqrt{n}}$

```python
import numpy as np
from scipy import stats

# 样本数据
data = np.random.normal(50, 10, 100)

# 均值的95%置信区间
mean = np.mean(data)
std = np.std(data, ddof=1)
n = len(data)

# 使用t分布（方差未知）
t_value = stats.t.ppf(0.975, n-1)
ci_lower = mean - t_value * std / np.sqrt(n)
ci_upper = mean + t_value * std / np.sqrt(n)

print(f"置信区间: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

## 假设检验

### 基本概念

**假设检验**：根据样本判断关于总体的假设是否成立。

**步骤**：
1. 建立假设（原假设 $H_0$ 和备择假设 $H_1$）
2. 选择检验统计量
3. 确定显著性水平 $\alpha$
4. 计算p值或临界值
5. 做出决策

### 常见假设检验

#### t检验（均值检验）

**单样本t检验**：检验样本均值是否等于某值
$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$

**双样本t检验**：检验两组均值是否相等
$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$

```python
from scipy import stats

# 单样本t检验
data = np.random.normal(50, 10, 100)
t_stat, p_value = stats.ttest_1samp(data, 50)

# 双样本t检验
group1 = np.random.normal(50, 10, 50)
group2 = np.random.normal(55, 10, 50)
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"p值: {p_value}")
print(f"结论: {'拒绝H0' if p_value < 0.05 else '接受H0'}")
```

#### 卡方检验（分类变量）

检验分类变量的独立性或分布拟合。

```python
# 卡方检验
from scipy.stats import chi2_contingency

# 观测频数表
observed = np.array([[10, 20], [30, 40]])
chi2, p_value, dof, expected = chi2_contingency(observed)
```

#### F检验（方差比较）

检验两组方差是否相等。

### p值解读

| p值 | 结论 |
|------|------|
| p < 0.01 | 强证据拒绝H0 |
| p < 0.05 | 中等证据拒绝H0 |
| p < 0.1 | 弱证据拒绝H0 |
| p > 0.1 | 无足够证据拒绝H0 |

### 检验错误

| 错误类型 | 描述 | 概率 |
|----------|------|------|
| 第一类错误 | H0真但拒绝 | $\alpha$ |
| 第二类错误 | H0假但接受 | $\beta$ |
| 检验功效 | H0假时正确拒绝 | $1-\beta$ |

## 参数估计方法（MLE、MAP）

### 最大似然估计（MLE）

找到使似然函数最大的参数值。

**似然函数**：
$L(\theta) = P(X|\theta) = \prod_{i=1}^{n} P(x_i|\theta)$

**对数似然**（更易计算）：
$\ln L(\theta) = \sum_{i=1}^{n} \ln P(x_i|\theta)$

**MLE求解**：
$\hat{\theta}_{MLE} = \arg\max_\theta L(\theta)$

#### MLE示例：正态分布参数

对于正态分布样本：
$L(\mu, \sigma) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}$

求解得：
- $\hat{\mu}_{MLE} = \bar{x}$（样本均值）
- $\hat{\sigma^2}_{MLE} = \frac{1}{n}\sum(x_i - \bar{x})^2$（样本方差）

```python
import numpy as np

def mle_normal(data):
    """正态分布MLE"""
    mu = np.mean(data)
    sigma2 = np.var(data)  # MLE方差（不是无偏估计）
    return mu, np.sqrt(sigma2)

data = np.random.normal(5, 2, 1000)
mu_mle, sigma_mle = mle_normal(data)
print(f"MLE估计: mu={mu_mle:.2f}, sigma={sigma_mle:.2f}")
```

### 最大后验估计（MAP）

考虑参数的先验分布。

**贝叶斯公式**：
$P(\theta|X) = \frac{P(X|\theta) P(\theta)}{P(X)}$

**MAP估计**：
$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta|X) = \arg\max_\theta P(X|\theta) P(\theta)$

MAP = MLE + 先验

#### MAP示例：正态分布均值

假设 $\mu$ 的先验为 $N(\mu_0, \sigma_0^2)$：
$\hat{\mu}_{MAP} = \frac{\sigma^2}{\sigma^2 + n\sigma_0^2}\mu_0 + \frac{n\sigma_0^2}{\sigma^2 + n\sigma_0^2}\bar{x}$

当 $n \to \infty$，MAP趋近于MLE。

### MLE vs MAP vs 贝叶斯估计

| 方法 | 特点 |
|------|------|
| MLE | 不使用先验，只看数据 |
| MAP | 使用先验，给出单点估计 |
| 贝叶斯估计 | 使用先验，给出参数分布 |

## 贝叶斯统计简介

### 贝叶斯框架

**贝叶斯统计**将参数视为随机变量，有概率分布。

$P(\theta|X) = \frac{P(X|\theta) P(\theta)}{P(X)}$

- $P(\theta)$：先验分布
- $P(X|\theta)$：似然
- $P(\theta|X)$：后验分布

### 贝叶斯推断流程

```
先验 P(θ) → 观察数据 X → 后验 P(θ|X)
              ↑
              似然 P(X|θ)
```

### 贝叶斯估计

贝叶斯估计使用后验分布：
- **后验均值**：$\hat{\theta} = E[\theta|X]$
- **后验中位数**：后验分布的中位数
- **后验众数**：后验分布最大值（等于MAP）

### 贝叶斯预测

预测新数据：
$P(x_{new}|X) = \int P(x_{new}|\theta) P(\theta|X) d\theta$

## 统计学在机器学习中的应用

### 模型评估

- 交叉验证的统计解释
- 模型比较的统计检验
- A/B测试

### 参数估计

- 模型参数的MLE估计
- 贝叶斯模型的MAP估计
- 概率模型的参数推断

### 不确定性量化

- 参数估计的置信区间
- 预测的不确定性
- 贝叶斯神经网络

### 特征选择

- 特征显著性检验
- 相关性检验
- 统计特征选择方法

## 总结

统计学是机器学习的理论基础之一。核心内容包括：统计量与统计推断、点估计与区间估计、假设检验、参数估计方法（MLE/MAP）、贝叶斯统计。

MLE是最常用的参数估计方法，MAP考虑先验知识。贝叶斯统计将参数视为随机变量，提供更完整的概率框架。

## 延伸阅读

- [概率论基础](/2026/05/10/zh-CN/技术文档/机器学习/probability-theory/)
- [贝叶斯分类器](/2026/05/10/zh-CN/技术文档/机器学习/bayesian-classifier/)
- [超参数调优](/2026/05/10/zh-CN/技术文档/机器学习/hyperparameter-tuning/)
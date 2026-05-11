---
title: 线性代数基础
date: 2026-04-28
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 数学基础, 线性代数]
---

## 向量与向量运算

### 向量的定义

向量是线性代数的基本概念，表示一组有序的数值。

**定义**：n维向量是一个n元有序数组，可表示为：
$\mathbf{v} = (v_1, v_2, ..., v_n)$

或列向量形式：
$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$

### 向量运算

#### 向量加法

$\mathbf{a} + \mathbf{b} = (a_1 + b_1, a_2 + b_2, ..., a_n + b_n)$

#### 向量数乘

$c\mathbf{v} = (cv_1, cv_2, ..., cv_n)$

#### 向量内积（点积）

$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + ... + a_nb_n$

**几何意义**：
$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}| |\mathbf{b}| \cos\theta$

其中θ是两向量夹角。

#### 向量外积（叉积）

仅适用于三维向量：
$\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_2b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1 \end{bmatrix}$

#### 向量范数

**L2范数（欧几里得范数）**：
$|\mathbf{v}|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}$

**L1范数**：
$|\mathbf{v}|_1 = \sum_{i=1}^{n} |v_i|$

**无穷范数**：
$|\mathbf{v}|_\infty = \max_i |v_i|$

## 矩阵与矩阵运算

### 矩阵的定义

矩阵是一个二维数组，表示为：
$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n} \\ a_{21} & a_{22} & ... & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & ... & a_{mn} \end{bmatrix}$

记作 $m \times n$ 矩阵，有m行n列。

### 特殊矩阵

| 类型 | 描述 |
|------|------|
| 方阵 | m = n 的矩阵 |
| 单位矩阵 | $I_{ij} = 1$ 若 $i=j$，否则为0 |
| 零矩阵 | 所有元素为0 |
| 对角矩阵 | 非对角元素全为0 |
| 对称矩阵 | $A^T = A$ |
| 正交矩阵 | $A^T A = I$ |

### 矩阵运算

#### 矩阵加法

$\mathbf{A} + \mathbf{B} = \begin{bmatrix} a_{11}+b_{11} & ... & a_{1n}+b_{1n} \\ \vdots & \ddots & \vdots \\ a_{m1}+b_{m1} & ... & a_{mn}+b_{mn} \end{bmatrix}$

要求矩阵形状相同。

#### 矩阵数乘

$c\mathbf{A} = \begin{bmatrix} ca_{11} & ... & ca_{1n} \\ \vdots & \ddots & \vdots \\ ca_{m1} & ... & ca_{mn} \end{bmatrix}$

#### 矩阵乘法

$\mathbf{A}$ 是 $m \times k$，$\mathbf{B}$ 是 $k \times n$，则：
$\mathbf{AB} = \mathbf{C} \quad (m \times n)$

其中：
$c_{ij} = \sum_{l=1}^{k} a_{il} b_{lj}$

**注意**：矩阵乘法不满足交换律：$\mathbf{AB} \neq \mathbf{BA}$

#### 矩阵转置

$\mathbf{A}^T_{ij} = \mathbf{A}_{ji}$

行列互换。

#### 矩阵求逆

若 $\mathbf{A}$ 可逆（非奇异），则存在 $\mathbf{A}^{-1}$ 使得：
$\mathbf{A} \mathbf{A}^{-1} = \mathbf{A}^{-1} \mathbf{A} = \mathbf{I}$

## 矩阵分解（LU、QR、SVD）

### LU 分解

将矩阵分解为下三角矩阵L和上三角矩阵U：
$\mathbf{A} = \mathbf{LU}$

**应用**：
- 解线性方程组
- 计算行列式
- 求矩阵逆

**步骤**：
```
A = LU
其中 L 是下三角（对角线为1）
U 是上三角
```

### QR 分解

将矩阵分解为正交矩阵Q和上三角矩阵R：
$\mathbf{A} = \mathbf{QR}$

**应用**：
- 最小二乘问题
- 解线性方程组
- 特征值计算

**Gram-Schmidt方法**：
将A的列向量正交化得到Q，R = Q^T A

### SVD 分解（奇异值分解）

最重要的矩阵分解，适用于任意矩阵：
$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$

其中：
- $\mathbf{U}$：$m \times m$ 正交矩阵（左奇异向量）
- $\mathbf{\Sigma}$：$m \times n$ 对角矩阵（奇异值）
- $\mathbf{V}$：$n \times n$ 正交矩阵（右奇异向量）

**奇异值性质**：
- $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r > 0$，其余为0
- r是矩阵的秩
- 奇异值反映矩阵的"能量"分布

**应用**：
- 主成分分析（PCA）
- 数据降维
- 图像压缩
- 推荐系统
- 自然语言处理

**Python实现**：
```python
import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = np.linalg.svd(A)

# 重构
A_reconstructed = U @ np.diag(S) @ Vt
```

## 线性变换与空间

### 线性变换定义

线性变换是保持向量加法和数乘的映射：
$T(\mathbf{a} + \mathbf{b}) = T(\mathbf{a}) + T(\mathbf{b})$
$T(c\mathbf{a}) = cT(\mathbf{a})$

矩阵可以表示线性变换：
$T(\mathbf{x}) = \mathbf{A}\mathbf{x}$

### 常见线性变换

| 变换 | 矩阵表示 |
|------|----------|
| 缩放 | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$ |
| 旋转 | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ |
| 平移 | 需要齐次坐标 |
| 反射 | $\begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$（关于y轴） |
| 剪切 | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$ |

### 向量空间

向量空间（线性空间）是满足以下性质的向量集合：
- 加法封闭：$\mathbf{u} + \mathbf{v} \in V$
- 数乘封闭：$c\mathbf{u} \in V$
- 含有零向量
- 满足加法和数乘的八条公理

**子空间**：向量空间的子集，自身也是向量空间。

**基**：一组线性无关的向量，可以生成整个空间。

**维数**：基中向量的个数。

## 特征值与特征向量

### 定义

对于方阵 $\mathbf{A}$，若存在非零向量 $\mathbf{v}$ 和数 $\lambda$ 使得：
$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$

则 $\lambda$ 是特征值，$\mathbf{v}$ 是对应的特征向量。

### 特征方程

$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$

求解此方程得到所有特征值。

### 特征值性质

- n×n矩阵有n个特征值（可能重复）
- 特征值之和 = 矩阵迹（trace）
- 特征值之积 = 矩阵行列式
- 对称矩阵的特征值都是实数

### 特征分解

若 $\mathbf{A}$ 有n个线性无关的特征向量：
$\mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^{-1}$

其中：
- $\mathbf{P}$：特征向量组成的矩阵
- $\mathbf{\Lambda}$：特征值组成的对角矩阵

**应用**：
- 矩阵幂计算：$\mathbf{A}^n = \mathbf{P} \mathbf{\Lambda}^n \mathbf{P}^{-1}$
- PCA
- 系统稳定性分析

### Python计算

```python
import numpy as np
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值:", eigenvalues)
print("特征向量:", eigenvectors)
```

## 张量简介

### 张量的定义

张量是多维数组的泛化：
- 0阶张量：标量（单个数值）
- 1阶张量：向量
- 2阶张量：矩阵
- n阶张量：n维数组

### 张量运算

```python
import torch

# 创建张量
t = torch.tensor([[1, 2], [3, 4]])  # 2阶张量

# 张量运算
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])

# 加法
t_add = t1 + t2

# 点积
dot = torch.dot(t1, t2)

# 矩阵乘法
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.matmul(A, B)
```

### 张量在深度学习中的应用

- 神经网络权重存储为张量
- 输入数据（图像、文本）表示为张量
- GPU加速张量计算

## 线性代数在机器学习中的应用

### 数据表示

- 数据集：m×n矩阵（m样本，n特征）
- 单个样本：n维向量
- 权重：向量或矩阵

### 模型计算

- 线性回归：$\mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}$
- 神经网络：大量矩阵乘法
- PCA：特征分解或SVD

### 优化求解

- 梯度：向量
- 参数更新：向量运算
- 正则化：范数计算

## 总结

线性代数是机器学习的数学基础。核心概念包括：向量与向量运算、矩阵与矩阵运算、矩阵分解（LU/QR/SVD）、线性变换、特征值与特征向量、张量。

掌握线性代数对于理解机器学习算法、神经网络结构、优化方法至关重要。

## 延伸阅读

- [概率论基础](/2026/05/10/zh-CN/技术文档/机器学习/probability-theory/)
- [优化理论基础](/2026/05/10/zh-CN/技术文档/机器学习/optimization/)
- [线性回归](/2026/05/10/zh-CN/技术文档/机器学习/linear-regression/)
- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
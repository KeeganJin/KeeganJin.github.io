---
title: 优化理论基础
date: 2026-05-09
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 数学基础, 优化理论]
---

## 优化问题分类

### 优化问题的定义

优化问题寻找使目标函数最优的参数：
$\min_\mathbf{x} f(\mathbf{x})$

可能带有约束条件：
$g_i(\mathbf{x}) \leq 0, \quad h_j(\mathbf{x}) = 0$

### 优化问题分类

| 类型 | 描述 | 特点 |
|------|------|------|
| 无约束优化 | 无约束条件 | 相对简单 |
| 约束优化 | 有约束条件 | 更复杂 |
| 凸优化 | 目标函数和约束凸 | 有全局最优 |
| 非凸优化 | 目标函数或约束非凸 | 可能局部最优 |
| 连续优化 | 变量连续 | 经典优化方法 |
| 离散优化 | 变量离散 | 组合优化 |

### 机器学习中的优化

- **参数学习**：最小化损失函数
- **超参数优化**：寻找最优超参数
- **结构优化**：优化模型结构

## 凸优化基础

### 凸集定义

集合 $\mathcal{C}$ 是凸集，如果对任意 $x, y \in \mathcal{C}$：
$\lambda x + (1-\lambda)y \in \mathcal{C}, \quad 0 \leq \lambda \leq 1$

直观理解：集合内任意两点的连线仍在集合内。

### 凸函数定义

函数 $f$ 是凸函数，如果：
$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$

**等价条件**：对于可微函数，若 $\nabla^2 f(x) \geq 0$（Hessian矩阵半正定），则 $f$ 是凸函数。

### 凸优化的重要性

**凸优化问题性质**：
- 局部最优 = 全局最优
- 有有效的求解算法
- 收敛性有保证

**非凸优化问题**：
- 可能有多个局部最优
- 求解困难
- 可能陷入局部最优

### 常见凸函数

| 函数 | 条件 |
|------|------|
| $ax + b$ | 任意（线性函数既是凸也是凹） |
| $x^2$ | 任意 |
| $e^x$ | 任意 |
| $-\log x$ | $x > 0$ |
| $|x|$ | 任意 |
| $\|\mathbf{x}\|^2$ | 任意 |

## 梯度与方向导数

### 方向导数

函数 $f$ 在点 $\mathbf{x}$ 沿方向 $\mathbf{d}$ 的变化率：
$\nabla_{\mathbf{d}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{d}) - f(\mathbf{x})}{h}$

### 梯度定义

梯度是方向导数的特殊情况：
$\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right]^T$

### 梯度与方向导数的关系

$\nabla_{\mathbf{d}} f(\mathbf{x}) = \nabla f(\mathbf{x})^T \mathbf{d}$

### 梯度的几何意义

- 梯度指向函数增长最快的方向
- 梯度模 $\|\nabla f\|$ 是最大变化率
- 梯度方向是最陡上升方向
- 负梯度方向是最陡下降方向

### Hessian矩阵

二阶导数组成的矩阵：
$\mathbf{H} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & ... \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & ... \\ \vdots & \vdots & \ddots \end{bmatrix}$

**Hessian的意义**：
- $\mathbf{H}$ 正定：局部最小值
- $\mathbf{H}$ 负定：局部最大值
- $\mathbf{H}$ 半正定：凸函数

## 无约束优化方法

### 梯度下降法

沿负梯度方向迭代更新：
$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)$

其中 $\alpha$ 是学习率（步长）。

**算法流程**：
```
1. 初始化 x_0
2. 计算梯度 g_k = ∇f(x_k)
3. 更新 x_{k+1} = x_k - α g_k
4. 检查收敛条件
5. 重复步骤2-4
```

**收敛条件**：
- $\|\nabla f(\mathbf{x}_k)\| < \epsilon$
- $|f(\mathbf{x}_{k+1}) - f(\mathbf{x}_k)| < \epsilon$

### 学习率选择

| 学习率 | 效果 |
|--------|------|
| 过大 | 可能震荡、发散 |
| 过小 | 收敛慢 |
| 适中 | 快速稳定收敛 |

**学习率策略**：
- 固定学习率
- 递减学习率：$\alpha_k = \alpha_0 / k$
- 自适应学习率（Adam等）

### 最速下降法

每步选择最优步长：
$\alpha_k = \arg\min_\alpha f(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k))$

### 牛顿法

使用二阶信息（Hessian）：
$\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}^{-1} \nabla f(\mathbf{x}_k)$

**优点**：收敛快（二次收敛）
**缺点**：计算Hessian和逆矩阵成本高

### 拟牛顿法

不直接计算Hessian，用近似矩阵：
- BFGS算法
- L-BFGS（有限内存BFGS）

## 约束优化方法

### 等式约束优化

问题形式：
$\min_\mathbf{x} f(\mathbf{x}) \quad \text{s.t.} \quad h(\mathbf{x}) = 0$

#### 拉格朗日乘子法

引入拉格朗日函数：
$L(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda h(\mathbf{x})$

求解：
$\nabla_\mathbf{x} L = 0, \quad \nabla_\lambda L = 0$

### 不等式约束优化

问题形式：
$\min_\mathbf{x} f(\mathbf{x}) \quad \text{s.t.} \quad g(\mathbf{x}) \leq 0$

#### KKT条件

最优解满足KKT条件：
1. $\nabla f(\mathbf{x}) + \mu \nabla g(\mathbf{x}) = 0$
2. $g(\mathbf{x}) \leq 0$
3. $\mu \geq 0$
4. $\mu g(\mathbf{x}) = 0$（互补松弛）

### 一般约束优化

$\min_\mathbf{x} f(\mathbf{x})$
$\text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \quad h_j(\mathbf{x}) = 0$

**拉格朗日函数**：
$L(\mathbf{x}, \mu, \lambda) = f(\mathbf{x}) + \sum_i \mu_i g_i(\mathbf{x}) + \sum_j \lambda_j h_j(\mathbf{x})$

**完整KKT条件**：
1. $\nabla_\mathbf{x} L = 0$
2. $g_i(\mathbf{x}) \leq 0, \quad h_j(\mathbf{x}) = 0$
3. $\mu_i \geq 0$
4. $\mu_i g_i(\mathbf{x}) = 0$

## 拉格朗日乘子法详解

### 直观理解

拉格朗日乘子法寻找目标函数梯度与约束函数梯度平行的点。

**几何解释**：
- 约束 $h(\mathbf{x}) = 0$ 定义一个曲面
- 目标函数在该曲面上的最小值点，梯度必与曲面平行
- 即 $\nabla f = \lambda \nabla h$

### 求解步骤

1. 构造拉格朗日函数 $L$
2. 对所有变量求偏导并令为0
3. 解方程组得到候选解
4. 验证最优性

### 示例

问题：$\min x^2 + y^2$，约束 $x + y = 1$

拉格朗日函数：
$L = x^2 + y^2 + \lambda(x + y - 1)$

求导：
$\frac{\partial L}{\partial x} = 2x + \lambda = 0$
$\frac{\partial L}{\partial y} = 2y + \lambda = 0$
$\frac{\partial L}{\partial \lambda} = x + y - 1 = 0$

解得：$x = y = 0.5$，$\lambda = -1$

## 优化在机器学习中的应用

### 损失函数最小化

机器学习的核心优化问题：
$\min_\mathbf{w} L(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n} \ell(\mathbf{w}, \mathbf{x}_i, y_i)$

### 常见优化场景

| 场景 | 问题 |
|------|------|
| 线性回归 | 最小二乘问题 |
| 逻辑回归 | 最大似然估计 |
| SVM | 约束凸优化 |
| 神经网络 | 非凸优化 |
| 正则化 | 约束优化形式 |

### 正则化与约束优化

正则化可以看作约束优化：

**L2正则化**：
$\min_\mathbf{w} L(\mathbf{w}) + \lambda\|\mathbf{w}\|^2$

等价于：
$\min_\mathbf{w} L(\mathbf{w}) \quad \text{s.t.} \quad \|\mathbf{w}\|^2 \leq r$

**L1正则化**：
$\min_\mathbf{w} L(\mathbf{w}) + \lambda\|\mathbf{w}\|_1$

等价于：
$\min_\mathbf{w} L(\mathbf{w}) \quad \text{s.t.} \quad \|\mathbf{w}\|_1 \leq r$

## Python优化实现

### 无约束优化

```python
from scipy.optimize import minimize

# 定义目标函数
def f(x):
    return x[0]**2 + x[1]**2

# 初始点
x0 = [1.0, 1.0]

# 优化
result = minimize(f, x0, method='BFGS')

print(f"最优解: {result.x}")
print(f"最优值: {result.fun}")
```

### 约束优化

```python
from scipy.optimize import minimize

def f(x):
    return x[0]**2 + x[1]**2

# 约束条件
constraints = [
    {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1}  # x + y = 1
]

result = minimize(f, x0, method='SLSQP', constraints=constraints)
```

## 总结

优化理论是机器学习的数学基础。核心内容包括：优化问题分类、凸优化基础、梯度与方向导数、无约束优化方法（梯度下降、牛顿法）、约束优化方法（拉格朗日乘子法、KKT条件）。

凸优化问题有全局最优且可高效求解。拉格朗日乘子法将约束优化转化为无约束问题。机器学习中的参数学习本质上都是优化问题。

## 延伸阅读

- [线性代数基础](/2026/05/10/zh-CN/技术文档/机器学习/linear-algebra/)
- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [优化算法详解](/2026/05/10/zh-CN/技术文档/机器学习/optimization-algorithms/)
- [正则化技术](/2026/05/10/zh-CN/技术文档/机器学习/regularization/)
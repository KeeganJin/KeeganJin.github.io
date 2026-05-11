---
title: 超参数调优
date: 2026-04-25
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 神经网络, 超参数调优]
---

## 超参数分类

### 什么是超参数

超参数是在训练前设定的参数，不是通过训练学习的。

**与模型参数的区别**：
| 类型 | 描述 | 设置方式 |
|------|------|----------|
| 模型参数 | 通过训练学习 | 自动学习 |
| 超参数 | 训练前设定 | 手动或自动调优 |

### 超参数的重要性

超参数直接影响模型性能：
- 影响训练收敛
- 影响最终准确率
- 影响计算效率

### 常见超参数分类

| 类别 | 超参数 |
|------|--------|
| 网络结构 | 层数、神经元数、激活函数 |
| 训练参数 | 学习率、批量大小、迭代次数 |
| 正则化 | Dropout率、权重衰减 |
| 优化器 | 优化器类型、动量参数 |

### 网络结构超参数

| 参数 | 影响 |
|------|------|
| 层数 | 模型复杂度 |
| 神经元数 | 表达能力 |
| 激活函数 | 非线性能力 |
| 连接方式 | 信息流动 |

### 训练参数

| 参数 | 影响 |
|------|------|
| 学习率 | 收敛速度、稳定性 |
| 批量大小 | 梯度稳定性、效率 |
| 迭代次数 | 训练程度 |

### 正则化参数

| 参数 | 影响 |
|------|------|
| Dropout率 | 过拟合程度 |
| 权重衰减 | 参数复杂度 |
| 早停轮数 | 训练时机 |

## 网格搜索

### 网格搜索原理

穷举搜索超参数组合：

**流程**：
```
1. 定义参数范围和步长
2. 生成所有组合
3. 对每个组合训练评估
4. 选择最佳组合
```

### 网格搜索实现

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# 网格搜索
model = RandomForestClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"最优参数: {grid_search.best_params_}")
print(f"最优分数: {grid_search.best_score_:.4f}")
```

### 网格搜索的优缺点

| 优点 | 缺点 |
|------|------|
| 简单直观 | 计算量大 |
| 保证找到最佳 | 可能组合太多 |
| 可并行计算 | 维度高时不可行 |

### 网格搜索适用场景

- 参数数量少（<5）
- 每个参数候选值少（<10）
- 有充足计算资源

## 随机搜索

### 随机搜索原理

随机采样超参数组合：

**流程**：
```
1. 定义参数范围（可以是分布）
2. 随机采样若干组合
3. 对每个组合训练评估
4. 选择最佳组合
```

### 随机搜索 vs 网格搜索

网格搜索：固定步长穷举
随机搜索：随机采样

**优势**：高维空间更高效

### 随机搜索实现

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# 定义参数分布
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'learning_rate': uniform(0.01, 0.1)
}

# 随机搜索
random_search = RandomizedSearchCV(
    model, 
    param_distributions,
    n_iter=50,  # 采样次数
    cv=5,
    scoring='accuracy'
)
random_search.fit(X_train, y_train)

print(f"最优参数: {random_search.best_params_}")
```

### 随机搜索的优缺点

| 优点 | 缺点 |
|------|------|
| 高维高效 | 不保证全局最优 |
| 可控制计算成本 | 可能错过最佳 |
| 参数范围灵活 | 需要足够采样 |

### 随机搜索适用场景

- 参数数量多
- 参数范围大
- 需要快速探索

## 贝叶斯优化

### 贝叶斯优化原理

使用概率模型指导搜索：

**流程**：
```
1. 建立代理模型（如Gaussian Process）
2. 评估若干初始点
3. 根据模型选择下一个评估点
4. 更新模型
5. 重复直到满足条件
```

### 贝叶斯优化的关键组件

| 组件 | 描述 |
|------|------|
| 代理模型 | 估计目标函数分布 |
| 采集函数 | 选择下一个评估点 |
| 目标函数 | 超参数评估函数 |

### 采集函数

常用采集函数：
- Expected Improvement (EI)
- Probability of Improvement (PI)
- Lower Confidence Bound (LCB)

**EI公式**：
$EI(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]$

其中 $f(x^+)$ 是当前最佳值。

### 贝叶斯优化实现

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# 定义搜索空间
search_space = {
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(5, 30),
    'learning_rate': Real(0.01, 0.1),
    'optimizer': Categorical(['adam', 'sgd'])
}

# 贝叶斯优化
bayes_search = BayesSearchCV(
    model,
    search_space,
    n_iter=50,
    cv=5,
    scoring='accuracy'
)
bayes_search.fit(X_train, y_train)

print(f"最优参数: {bayes_search.best_params_}")
```

### 贝叶斯优化的优缺点

| 优点 | 缺点 |
|------|------|
| 高效（样本少） | 计算代理模型成本 |
| 适合高维空间 | 可能陷入局部最优 |
| 可利用历史信息 | 对代理模型敏感 |

### 贝叶斯优化适用场景

- 训练成本高
- 参数空间复杂
- 需要精准调优

### Hyperopt库示例

```python
from hyperopt import fmin, tpe, hp, Trials

# 定义搜索空间
space = {
    'learning_rate': hp.loguniform('learning_rate', -5, -1),
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
    'n_layers': hp.choice('n_layers', [2, 3, 4]),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256])
}

# 定义目标函数
def objective(params):
    model = create_model(params)
    score = evaluate_model(model)
    return -score  # 返回负值（最小化）

# 贝叶斯优化
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
print(f"最优参数: {best}")
```

## 遗传算法

### 遗传算法原理

模拟生物进化过程：

**流程**：
```
1. 初始化种群（随机超参数）
2. 评估适应度（模型性能）
3. 选择（保留优秀个体）
4. 交叉（组合参数）
5. 变异（随机变化）
6. 重复直到满足条件
```

### 遗传算法的关键操作

| 操作 | 描述 |
|------|------|
| 选择 | 选择适应度高的个体 |
| 交叉 | 交换部分参数 |
| 变异 | 随机改变参数 |

### 遗传算法实现

```python
import numpy as np

class GeneticOptimizer:
    def __init__(self, param_bounds, population_size=20, generations=50):
        self.bounds = param_bounds
        self.pop_size = population_size
        self.generations = generations
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            individual = {}
            for param, bounds in self.bounds.items():
                individual[param] = np.random.uniform(bounds[0], bounds[1])
            population.append(individual)
        return population
    
    def evaluate_fitness(self, individual):
        """评估适应度"""
        model = create_model(individual)
        return evaluate_model(model)
    
    def select(self, population, fitnesses):
        """选择（轮盘赌）"""
        probs = fitnesses / np.sum(fitnesses)
        selected = np.random.choice(len(population), size=self.pop_size, p=probs)
        return [population[i] for i in selected]
    
    def crossover(self, parent1, parent2):
        """交叉"""
        child = {}
        for param in self.bounds.keys():
            if np.random.rand() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def mutate(self, individual, mutation_rate=0.1):
        """变异"""
        for param, bounds in self.bounds.items():
            if np.random.rand() < mutation_rate:
                individual[param] = np.random.uniform(bounds[0], bounds[1])
        return individual
    
    def optimize(self):
        """优化"""
        population = self.initialize_population()
        
        for gen in range(self.generations):
            fitnesses = np.array([self.evaluate_fitness(ind) for ind in population])
            
            # 选择
            population = self.select(population, fitnesses)
            
            # 交叉和变异
            new_population = []
            for i in range(0, self.pop_size, 2):
                child1 = self.crossover(population[i], population[i+1])
                child2 = self.crossover(population[i], population[i+1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population
            best_fitness = np.max(fitnesses)
            print(f"Generation {gen}: Best fitness = {best_fitness}")
        
        return population[np.argmax(fitnesses)]
```

### 遗传算法的优缺点

| 优点 | 缺点 |
|------|------|
| 全局搜索能力 | 计算量大 |
| 不需要梯度信息 | 参数多时慢 |
| 可处理复杂空间 | 收敛不确定 |

## 自动调参工具

### Optuna

```python
import optuna

def objective(trial):
    # 定义参数
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    n_layers = trial.suggest_int('n_layers', 2, 5)
    
    model = create_model(learning_rate, dropout, n_layers)
    score = train_and_evaluate(model)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"最优参数: {study.best_params}")
print(f"最优值: {study.best_value}")
```

### Keras Tuner

```python
import keras_tuner as kt

def build_model(hp):
    model = tf.keras.Sequential()
    
    # 动态层数
    for i in range(hp.Int('n_layers', 2, 5)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'units_{i}', 32, 128),
            activation='relu'
        ))
    
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy'
    )
    
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=50
)

tuner.search(X_train, y_train, validation_split=0.2, epochs=10)
best_model = tuner.get_best_models()[0]
```

### Ray Tune

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    model = create_model(config)
    for epoch in range(config['epochs']):
        train_one_epoch(model)
        val_loss = evaluate(model)
        tune.report(val_loss=val_loss)

config = {
    'learning_rate': tune.loguniform(1e-5, 1e-1),
    'dropout': tune.uniform(0.1, 0.5),
    'batch_size': tune.choice([32, 64, 128])
}

analysis = tune.run(
    train_model,
    config=config,
    scheduler=ASHAScheduler(),
    num_samples=50
)

print(f"最优配置: {analysis.best_config}")
```

### 工具对比

| 工具 | 特点 |
|------|------|
| Optuna | 灵活，支持多种算法 |
| Keras Tuner | Keras/TensorFlow专用 |
| Ray Tune | 分布式，大规模搜索 |
| Hyperopt | 简单，贝叶斯优化 |

## 案例实践

### 综合调参示例

```python
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# 数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 定义模型构建函数
def build_model(hp):
    model = keras.Sequential()
    
    # 层数
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', 32, 256),
            activation='relu'
        ))
        model.add(keras.layers.Dropout(
            rate=hp.Float(f'dropout_{i}', 0.1, 0.5)
        ))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # 学习率
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 使用Keras Tuner
import keras_tuner as kt

tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=50,
    directory='tuner_results'
)

# 搜索
tuner.search(X_train, y_train, epochs=20, validation_split=0.2)

# 获取最佳模型
best_model = tuner.get_best_models()[0]
best_params = tuner.get_best_hyperparameters()[0]

print("最优参数:")
print(best_params.values)

# 评估
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"测试准确率: {test_acc:.4f}")
```

### 学习率调优

```python
# 学习率范围测试
import matplotlib.pyplot as plt

def find_learning_rate(model, X, y, min_lr=1e-5, max_lr=10):
    """学习率范围测试"""
    losses = []
    lrs = []
    
    lr = min_lr
    factor = (max_lr / min_lr) ** (1/100)
    
    for i in range(100):
        # 训练一步
        loss = train_step(model, X, y, lr)
        losses.append(loss)
        lrs.append(lr)
        lr *= factor
    
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.show()
    
    # 建议学习率：损失下降最快点之前
    return optimal_lr

lr = find_learning_rate(model, X_train, y_train)
print(f"建议学习率: {lr}")
```

### 批量大小调优

```python
# 批量大小对比
batch_sizes = [16, 32, 64, 128, 256]

for bs in batch_sizes:
    model = create_model()
    history = model.fit(X_train, y_train, 
                       batch_size=bs,
                       epochs=20,
                       validation_split=0.2)
    
    print(f"Batch size {bs}:")
    print(f"  训练时间: {history.history['time']}")
    print(f"  最终准确率: {history.history['val_accuracy'][-1]:.4f}")
```

## 超参数调优最佳实践

### 调优优先级

| 优先级 | 参数 |
|--------|------|
| 高 | 学习率、网络结构 |
| 中 | 批量大小、正则化 |
| 低 | 详细参数 |

### 调优策略

| 策略 | 描述 |
|------|------|
| 粗调 | 先大范围快速搜索 |
| 精调 | 在粗调结果附近精细调优 |
| 分层 | 按重要性分层调参 |

### 避免常见错误

| 错误 | 解决 |
|------|------|
| 搜索空间太小 | 扩大范围 |
| 评估不充分 | 增加交叉验证 |
| 过拟合验证集 | 使用测试集最终验证 |

### 调优效率建议

| 建议 | 描述 |
|------|------|
| 并行化 | 多任务并行搜索 |
| 早停 | 不好的配置提前停止 |
| 增量搜索 | 利用已有结果 |

## 总结

超参数调优是提升模型性能的关键。核心内容包括：
- 超参数分类：网络结构、训练参数、正则化参数
- 网格搜索：穷举搜索，适合小空间
- 随机搜索：随机采样，适合大空间
- 贝叶斯优化：智能搜索，高效精准
- 遗传算法：进化搜索，全局探索
- 自动调参工具：Optuna、Keras Tuner、Ray Tune

合理的调优策略可以显著提升模型性能，但需要平衡搜索效率和计算成本。

## 延伸阅读

- [正则化技术](/2026/05/10/zh-CN/技术文档/机器学习/regularization/)
- [优化算法详解](/2026/05/10/zh-CN/技术文档/机器学习/optimization-algorithms/)
- [模型评估指标](/2026/05/10/zh-CN/技术文档/机器学习/ml-introduction/)
- [机器学习概述](/2026/05/10/zh-CN/技术文档/机器学习/ml-introduction/)
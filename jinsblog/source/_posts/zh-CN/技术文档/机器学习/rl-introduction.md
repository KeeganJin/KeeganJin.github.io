---
title: 强化学习基础
date: 2026-01-17
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 强化学习]
---

## 强化学习定义

### 什么是强化学习

强化学习是通过与环境交互学习最优行为的机器学习方法。

**核心要素**：
- Agent（智能体）：学习者
- Environment（环境）：交互对象
- Reward（奖励）：反馈信号

### 与其他学习方法的区别

| 学习类型 | 数据来源 | 特点 |
|----------|----------|------|
| 监督学习 | 标注数据 | 输入→输出映射 |
| 无监督学习 | 无标注数据 | 结构发现 |
| 强化学习 | 环境交互 | 试错学习 |

### 强化学习的特点

| 特点 | 描述 |
|------|------|
| 序列决策 | 行动影响未来 |
| 延迟奖励 | 奖励可能延迟 |
| 试错学习 | 通过尝试学习 |
| 平衡探索利用 | 新尝试vs已知最优 |

## Agent-Environment模型

### 基本框架

```
Agent            Environment
  │                   │
  │──Action──────────>│
  │                   │
  │<──State, Reward──│
  │                   │
```

### 交互循环

1. Agent观察状态 $s_t$
2. Agent选择行动 $a_t$
3. Environment执行行动
4. Environment返回新状态 $s_{t+1}$ 和奖励 $r_t$
5. Agent根据反馈学习

### 状态与行动

**状态（State）**：
- Agent对环境的观察
- 包含决策所需信息

**行动（Action）**：
- Agent可执行的操作
- 可能有限或无限

### 奖励（Reward）

**定义**：环境给Agent的反馈信号。

**特点**：
- 稀疏或密集
- 即时或延迟
- 正或负

**设计原则**：
- 反映目标
- 易于学习
- 避免歧义

## 状态、动作、奖励

### 状态空间

| 类型 | 描述 |
|------|------|
| 离散 | 有限状态（如网格位置） |
| 连续 | 无限状态（如机器人位置） |

### 动作空间

| 类型 | 描述 |
|------|------|
| 离散 | 有限动作（如上下左右） |
| 连续 | 无限动作（如力的大小） |

### 奖励设计

**原则**：
- 与目标一致
- 适当的稀疏性
- 避免局部最优

```python
# 网格世界奖励示例
def get_reward(state, action, next_state):
    if next_state == goal:
        return 10  # 到达目标
    if next_state in obstacles:
        return -10  # 碰到障碍
    return -1  # 每步惩罚（鼓励快速到达）
```

## 策略与价值函数

### 策略（Policy）

策略定义Agent如何选择行动：
$\pi(a|s)$

**类型**：
| 类型 | 定义 | 特点 |
|------|------|------|
| 确定性策略 | $\pi(s) = a$ | 固定行动 |
| 随机策略 | $\pi(a|s) = P(a|s)$ | 行动概率 |

### 价值函数（Value Function）

**状态价值函数**：
$V^\pi(s) = \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t r_t | s_0 = s]$

**状态-行动价值函数**（Q函数）：
$Q^\pi(s, a) = \mathbb{E}_\pi[\sum_{t=0}^\infty \gamma^t r_t | s_0 = s, a_0 = a]$

### 折扣因子（Discount Factor）

$\gamma \in [0, 1]$

**作用**：
- 控制未来奖励的重要性
- $\gamma$接近0：重视即时奖励
- $\gamma$接近1：重视长期奖励

### Bellman方程

**Bellman期望方程**：
$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[r(s,a,s') + \gamma V^\pi(s')]$

**Bellman最优方程**：
$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[r(s,a,s') + \gamma V^*(s')]$

$Q^*(s, a) = \sum_{s'} P(s'|s,a)[r(s,a,s') + \gamma \max_{a'} Q^*(s', a')]$

## 马尔可夫决策过程（MDP）

### MDP定义

马尔可夫决策过程描述强化学习环境：

$MDP = (S, A, P, R, \gamma)$

**组成**：
- S：状态空间
- A：动作空间
- P：转移概率 $P(s'|s,a)$
- R：奖励函数 $R(s,a,s')$
- $\gamma$：折扣因子

### 马尔可夫性质

**定义**：未来只依赖当前状态，不依赖历史。

$P(s_{t+1}|s_t, a_t, s_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$

**意义**：简化决策过程。

### MDP示例

**网格世界**：
```python
# 状态：位置 (x, y)
states = [(0, 0), (0, 1), ..., (3, 3)]

# 动作：上、下、左、右
actions = ['up', 'down', 'left', 'right']

# 转移概率：确定性
def transition(state, action):
    if action == 'up':
        return (state[0], state[1] + 1)
    ...

# 奖励
def reward(state, next_state):
    if next_state == goal:
        return 10
    return -1
```

### 部分可观测MDP（POMDP）

**定义**：Agent只能观测部分状态信息。

**组成**：
- MDP元素
- 观测空间O
- 观测概率 $P(o|s)$

## 强化学习分类

### 基于价值的方法

学习价值函数，间接推导策略：
- Value Iteration
- Q-Learning
- DQN

**特点**：
- 学习V或Q函数
- 策略由价值推导
- 适合离散动作

### 基于策略的方法

直接学习策略：
- Policy Gradient
- REINFORCE
- PPO

**特点**：
- 直接优化策略
- 可处理连续动作
- 可学习随机策略

### Actor-Critic方法

结合价值和策略：
- 学习价值函数（Critic）
- 学习策略（Actor）
- 如A2C、A3C

### 模型基础分类

| 类型 | 描述 |
|------|------|
| Model-based | 学习环境模型 |
| Model-free | 不学习环境模型 |

| 方法 | 代表算法 |
|------|----------|
| Model-based Value | Dyna-Q |
| Model-free Value | Q-Learning, DQN |
| Model-free Policy | REINFORCE, PPO |
| Actor-Critic | A2C, A3C |

## 探索与利用

### 探索-利用困境

**探索（Exploration）**：尝试新行动，发现可能更好的策略。

**利用（Exploitation）**：使用已知最优行动，最大化即时奖励。

### 平衡策略

| 策略 | 描述 |
|------|------|
| ε-greedy | 以ε概率随机探索 |
| Softmax | 按概率探索 |
| UCB | 不确定性探索 |
| Thompson采样 | 贝叶斯探索 |

### ε-greedy策略

```python
def epsilon_greedy(q_values, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)  # 随机探索
    else:
        return argmax(q_values)  # 利用最优
```

### 探索衰减

探索概率随时间衰减：
$\epsilon_t = \epsilon_0 \cdot decay^t$

## 强化学习工作流程

### 标准流程

```
1. 初始化策略/价值函数
2. 循环：
   a. 选择行动（根据策略）
   b. 执行行动，观察奖励和新状态
   c. 更新策略/价值函数
3. 评估策略性能
```

### 训练过程

```python
def train_agent(env, agent, episodes):
    for episode in range(episodes):
        state = env.reset()
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            if done:
                break
```

### 评估指标

| 指标 | 描述 |
|------|------|
| 累计奖励 | 一轮总奖励 |
| 成功率 | 目标达成比例 |
| 学习速度 | 达到性能所需轮数 |

## 案例实践

### 简单环境实现

```python
import numpy as np

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.goal = (size-1, size-1)
        self.state = (0, 0)
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        
        if action == 0:  # up
            y = min(y + 1, self.size - 1)
        elif action == 1:  # down
            y = max(y - 1, 0)
        elif action == 2:  # left
            x = max(x - 1, 0)
        elif action == 3:  # right
            x = min(x + 1, self.size - 1)
        
        self.state = (x, y)
        
        reward = 10 if self.state == self.goal else -1
        done = self.state == self.goal
        
        return self.state, reward, done

# 使用
env = GridWorld()
state = env.reset()
next_state, reward, done = env.step(3)  # right
```

### 简单Agent实现

```python
class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    
    def select_action(self, state):
        return random.randint(0, self.n_actions - 1)
    
    def update(self, state, action, reward, next_state):
        # 不学习
        pass

# 测试
agent = RandomAgent(4)
total_reward = 0

for episode in range(10):
    state = env.reset()
    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
```

### Q表初始化

```python
class QAgent:
    def __init__(self, n_states, n_actions):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = 0.1  # 学习率
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 0.1  # 探索率
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        # Q-Learning更新
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - 
            self.q_table[state, action]
        )
```

### 使用OpenAI Gym

```python
import gym

# 创建环境
env = gym.make('CartPole-v1')

# 观察
state = env.reset()
print(f"State space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# 交互
for t in range(100):
    action = env.action_space.sample()  # 随机动作
    next_state, reward, done, info = env.step(action)
    
    if done:
        break

env.close()
```

## 强化学习应用领域

### 游戏AI

| 应用 | 描述 |
|------|------|
| Atari游戏 | DQN |
| 围棋 | AlphaGo |
| Dota 2 | OpenAI Five |
| 麻将 | Suphx |

### 机器人控制

| 应用 | 描述 |
|------|------|
| 运动控制 | 步行、跳跃 |
| 操作控制 | 抓取、放置 |
| 导航 | 路径规划 |

### 自动驾驶

| 应用 | 描述 |
|------|------|
| 车辆控制 | 加速、转向 |
| 路径规划 | 决策导航 |

### 推荐系统

| 应用 | 描述 |
|------|------|
| 内容推荐 | 用户交互优化 |
| 广告投放 | 点击率优化 |

### 资源管理

| 应用 | 描述 |
|------|------|
| 数据中心 | 能源优化 |
| 网络控制 | 流量调度 |

## 强化学习挑战

### 样本效率

**问题**：需要大量交互才能学习。

**解决**：
- Model-based方法
- 经验回放
- 算法改进

### 稳定性

**问题**：训练不稳定，性能波动。

**解决**：
- 目标网络
- 适当的学习率
- 正则化

### 探索难题

**问题**：稀疏奖励难以探索。

**解决**：
- 奖励塑形
- 好奇心驱动
- 层次化探索

### 安全性

**问题**：学习过程可能有危险行为。

**解决**：
- 安全约束
- 模拟训练
- 人监督

## 总结

强化学习是通过交互学习的机器学习方法。核心内容包括：
- Agent-Environment模型：交互框架
- 状态、动作、奖励：核心要素
- 策略与价值函数：决策表示
- MDP：数学框架
- 学习类型分类：基于价值、基于策略、Actor-Critic
- 探索与利用：平衡困境

强化学习适合序列决策问题，应用广泛。

## 延伸阅读

- [基于价值的方法](/2026/05/10/zh-CN/技术文档/机器学习/rl-value-based/)
- [基于策略的方法](/2026/05/10/zh-CN/技术文档/机器学习/rl-policy-based/)
- [概率论基础](/2026/05/10/zh-CN/技术文档/机器学习/probability-theory/)
- [优化理论基础](/2026/05/10/zh-CN/技术文档/机器学习/optimization/)
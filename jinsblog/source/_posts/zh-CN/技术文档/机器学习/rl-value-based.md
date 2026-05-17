---
title: 基于价值的方法
date: 2026-01-20
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 强化学习, Q-Learning, DQN]
---

## 动态规划方法

### 动态规划基础

动态规划是强化学习的理论基础。

**条件**：
- 完全已知MDP模型
- 状态空间有限

### 值迭代（Value Iteration）

直接求解最优价值函数：

$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$

**算法**：
```
初始化 V(s) = 0
重复：
    对所有状态 s：
        V(s) = max_a Σ P(s'|s,a)[R + γ V(s')]
直到收敛
```

```python
def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.n_states)
    
    while True:
        delta = 0
        for s in range(env.n_states):
            v = V[s]
            V[s] = max(sum(env.P(s, a, s_next) * (env.R(s, a, s_next) + gamma * V[s_next])
                         for s_next in range(env.n_states))
                       for a in range(env.n_actions))
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V
```

### 策略迭代（Policy Iteration）

交替评估和改进策略：

**流程**：
```
1. 策略评估：计算当前策略的V^π
2. 策略改进：基于V^π改进策略
3. 重复直到策略不变
```

```python
def policy_iteration(env, gamma=0.99):
    policy = np.random.choice(env.n_actions, env.n_states)
    V = np.zeros(env.n_states)
    
    while True:
        # 策略评估
        while True:
            delta = 0
            for s in range(env.n_states):
                v = V[s]
                a = policy[s]
                V[s] = sum(env.P(s, a, s_next) * (env.R(s, a, s_next) + gamma * V[s_next])
                          for s_next in range(env.n_states))
                delta = max(delta, abs(v - V[s]))
            if delta < 1e-8:
                break
        
        # 策略改进
        policy_stable = True
        for s in range(env.n_states):
            old_action = policy[s]
            policy[s] = np.argmax([sum(env.P(s, a, s_next) * (env.R(s, a, s_next) + gamma * V[s_next])
                                   for s_next in range(env.n_states))
                                  for a in range(env.n_actions)])
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy, V
```

### 值迭代vs策略迭代

| 方面 | 值迭代 | 策略迭代 |
|------|--------|----------|
| 收敛速度 | 可能慢 | 通常快 |
| 每轮计算 | 单次更新 | 多次评估 |
| 适用 | 大状态空间 | 小状态空间 |

## Q-Learning算法

### Q-Learning原理

学习最优Q函数，无需环境模型。

**更新公式**：
$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

### Q-Learning算法

```
初始化 Q(s,a) = 0
对每轮：
    初始化状态 s
    while 未终止：
        选择行动 a（ε-greedy）
        执行 a，观察 r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        s ← s'
```

### Q-Learning特点

| 特点 | 描述 |
|------|------|
| Model-free | 不需环境模型 |
| Off-policy | 可用任何数据学习 |
| 收敛性 | 保证收敛最优Q |

### Q-Learning实现

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
    
    def train(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            
            while True:
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)
                
                state = next_state
                if done:
                    break

# 使用
agent = QLearning(n_states=16, n_actions=4)
agent.train(env, episodes=1000)
```

### 探索策略

ε-greedy探索：
```python
def epsilon_greedy(q_values, epsilon):
    if random.random() < epsilon:
        return random.choice(range(len(q_values)))
    return np.argmax(q_values)

# 衰减探索
epsilon_decay = lambda episode: max(0.01, 0.1 * 0.995 ** episode)
```

## SARSA算法

### SARSA原理

同策略（On-policy）学习：
$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$

### SARSA vs Q-Learning

| 方面 | Q-Learning | SARSA |
|------|------------|-------|
| 学习方式 | Off-policy | On-policy |
| 目标 | max Q(s',a') | Q(s',a') |
| 收敛 | 最优策略 | 当前策略 |

### SARSA实现

```python
class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
    
    def train(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            action = self.select_action(state)
            
            while True:
                next_state, reward, done = env.step(action)
                next_action = self.select_action(next_state)
                
                self.update(state, action, reward, next_state, next_action)
                
                state = next_state
                action = next_action
                if done:
                    break
```

## DQN及其变体

### DQN原理

用神经网络近似Q函数：
$Q(s,a; \theta)$

**创新点**：
- 经验回放
- 目标网络

### 经验回放

**作用**：
- 打破数据相关性
- 提高样本利用效率

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
```

### 目标网络

**作用**：稳定训练目标

**方法**：
- 主网络：Q(s,a; θ)
- 目标网络：Q(s,a; θ')
- 定期更新θ' ← θ

### DQN实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(10000)
        
        self.gamma = 0.99
        self.epsilon = 0.1
        self.batch_size = 32
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
```

### Double DQN

**问题**：DQN可能高估Q值。

**解决**：用策略网络选择行动，目标网络评估。

$Q_{target} = r + \gamma Q_{target}(s', argmax_a Q_{policy}(s', a))$

```python
# Double DQN目标
with torch.no_grad():
    next_actions = self.policy_net(next_states).argmax(1)
    max_next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
```

### Dueling DQN

**结构**：分离状态价值和行动优势。

$Q(s,a) = V(s) + A(s,a) - \frac{1}{|A|}\sum_{a'} A(s,a')$

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # 状态价值
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 行动优势
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        value = self.value(shared)
        advantage = self.advantage(shared)
        
        return value + advantage - advantage.mean(1, keepdim=True)
```

### Prioritized Experience Replay

**思想**：优先学习重要样本。

**优先级**：TD误差大小

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = []
        self.priorities = []
        self.capacity = capacity
        self.alpha = alpha
    
    def push(self, state, action, reward, next_state, done, priority):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority ** self.alpha)
    
    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights
```

## 案例实践

### Q-Learning网格世界

```python
def train_qlearning_grid():
    env = GridWorld(size=5)
    agent = QLearning(n_states=25, n_actions=4)
    
    rewards_history = []
    
    for episode in range(500):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards_history.append(total_reward)
    
    return agent, rewards_history
```

### DQN CartPole

```python
import gym

def train_dqn_cartpole():
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    for episode in range(500):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        if episode % 10 == 0:
            agent.update_target()
        
        print(f"Episode {episode}: Reward {total_reward}")
    
    return agent
```

### 可视化训练过程

```python
import matplotlib.pyplot as plt

def plot_training(rewards_history):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()

# 绘制滑动平均
def plot_smooth_rewards(rewards, window=50):
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.show()
```

### 策略可视化

```python
def visualize_policy(q_table, env_size):
    """可视化最优策略"""
    policy = np.argmax(q_table, axis=1)
    
    arrows = ['↑', '↓', '←', '→']
    grid = np.zeros((env_size, env_size), dtype=str)
    
    for s in range(len(policy)):
        x, y = s // env_size, s % env_size
        grid[x, y] = arrows[policy[s]]
    
    print(grid)
```

## 基于价值方法总结

### 优点

| 优点 | 描述 |
|------|------|
| 算法成熟 | 理论基础完善 |
| 收敛保证 | 有收敛性证明 |
| 易实现 | 相对简单 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 离散动作 | 不适合连续动作 |
| 状态有限 | 表格方法受限 |
| 探索难题 | 可能陷入局部最优 |

## 总结

基于价值的方法是强化学习的基础。核心内容包括：
- 动态规划：值迭代、策略迭代
- Q-Learning：Model-free学习最优Q
- SARSA：On-policy学习
- DQN：神经网络近似Q函数
- DQN变体：Double DQN、Dueling DQN、Prioritized Replay

基于价值的方法适合离散动作空间，理论基础成熟。

## 延伸阅读

- [强化学习基础](/2026/05/10/zh-CN/技术文档/机器学习/rl-introduction/)
- [基于策略的方法](/2026/05/10/zh-CN/技术文档/机器学习/rl-policy-based/)
- [强化学习进阶](/2026/05/10/zh-CN/技术文档/机器学习/rl-advanced/)
- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
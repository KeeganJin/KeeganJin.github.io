---
title: 基于策略的方法
date: 2026-04-16
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 强化学习, Policy Gradient, PPO]
---

## 策略梯度原理

### 直接策略优化

基于策略的方法直接优化策略：
$\pi_\theta(a|s)$

**目标**：最大化期望奖励
$J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^t r_t]$

### 策略梯度定理

$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$

**解读**：
- 增大高奖励行动的概率
- 减小低奖励行动的概率

### 与基于价值方法的区别

| 方面 | 基于价值 | 基于策略 |
|------|----------|----------|
| 学习目标 | 价值函数 | 策略参数 |
| 策略获取 | argmax Q | 直接参数 |
| 连续动作 | 困难 | 自然支持 |
| 随机策略 | 不支持 | 支持 |

### 策略参数化

**离散动作**：
- Softmax输出概率

$\pi_\theta(a|s) = \frac{\exp(f_\theta(s,a))}{\sum_{a'}\exp(f_\theta(s,a'))}$

**连续动作**：
- 输出分布参数（如正态分布的μ、σ）

$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$

```python
# 离散策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)

# 连续策略网络
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mu = self.mu_net(state)
        std = torch.exp(self.log_std)
        return mu, std
```

## REINFORCE算法

### REINFORCE原理

蒙特卡洛策略梯度：

$\nabla_\theta J \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \cdot G_t^i$

其中 $G_t$ 是从时刻t的累计奖励。

### REINFORCE算法

```
对每轮：
    用当前策略采样轨迹
    计算每步的累计奖励G
    更新参数：θ ← θ + α Σ ∇logπ(a|s)·G
```

### REINFORCE实现

```python
class REINFORCE:
    def __init__(self, policy_net, lr=0.01, gamma=0.99):
        self.policy = policy_net
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob
    
    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def update(self, log_probs, returns):
        returns = torch.FloatTensor(returns)
        
        # 标准化（可选）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, env, episodes):
        for episode in range(episodes):
            log_probs = []
            rewards = []
            
            state = env.reset()
            while True:
                action, log_prob = self.select_action(state)
                next_state, reward, done = env.step(action)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                
                state = next_state
                if done:
                    break
            
            returns = self.compute_returns(rewards)
            self.update(log_probs, returns)
```

### REINFORCE的缺点

| 缺点 | 描述 |
|------|------|
| 高方差 | 轨迹随机性大 |
| 样本效率低 | 每轨迹只用一次 |
| 学习慢 | 需大量样本 |

## Actor-Critic方法

### Actor-Critic原理

结合策略梯度和价值学习：
- Actor：策略网络，选择行动
- Critic：价值网络，评估行动

**优势**：降低方差，提高样本效率。

### Actor-Critic框架

$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)]$

其中 $A(s,a)$ 是优势函数。

### 优势函数

$A(s,a) = Q(s,a) - V(s)$

**作用**：衡量行动相对好坏。

### A2C（Advantage Actor-Critic）

同步更新Actor和Critic：

```python
class A2C:
    def __init__(self, policy_net, value_net, lr=0.01, gamma=0.99):
        self.policy = policy_net
        self.value = value_net
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算TD目标
        with torch.no_grad():
            values_next = self.value(next_states)
            td_target = rewards + self.gamma * values_next * (1 - dones)
        
        # Critic损失
        values = self.value(states)
        critic_loss = nn.MSELoss()(values, td_target)
        
        # Actor损失
        probs = self.policy(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)))
        advantage = td_target - values.detach()
        actor_loss = -(log_probs * advantage).mean()
        
        # 更新
        self.optimizer_policy.zero_grad()
        actor_loss.backward()
        self.optimizer_policy.step()
        
        self.optimizer_value.zero_grad()
        critic_loss.backward()
        self.optimizer_value.step()
```

### A3C（异步A2C）

多进程异步训练：
- 多个Agent并行探索
- 异步更新参数
- 提高样本效率

## PPO算法详解

### PPO原理

Proximal Policy Optimization限制策略更新幅度：

**目标函数**：
$L(\theta) = \mathbb{E}[\min(r(\theta) \cdot A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \cdot A)]$

其中 $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$。

### PPO的动机

**问题**：策略更新过大可能崩溃。

**解决**：限制新旧策略的差异。

### Clip函数

$\text{clip}(r, 1-\epsilon, 1+\epsilon)$

**效果**：限制概率比在范围内。

### PPO实现

```python
class PPO:
    def __init__(self, policy_net, value_net, lr=3e-4, gamma=0.99, epsilon=0.2):
        self.policy = policy_net
        self.value = value_net
        self.optimizer = optim.Adam(list(policy_net.parameters()) + 
                                    list(value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
    
    def compute_advantages(self, rewards, values, dones):
        advantages = []
        G = 0
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            advantages.insert(0, G - v)
        return advantages
    
    def update(self, states, actions, log_probs_old, rewards, dones, epochs=10):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        
        values = self.value(states).detach()
        advantages = self.compute_advantages(rewards, values.numpy(), dones)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            probs = self.policy(states)
            log_probs_new = torch.log(probs.gather(1, actions.unsqueeze(1)))
            
            ratio = torch.exp(log_probs_new - log_probs_old.unsqueeze(1))
            
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages.unsqueeze(1)
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            values = self.value(states)
            critic_loss = nn.MSELoss()(values, torch.FloatTensor(
                self.compute_returns(rewards, dones)))
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

### PPO特点

| 特点 | 描述 |
|------|------|
| 稳定 | 限制策略更新幅度 |
| 高效 | 多轮使用样本 |
| 简单 | 相对易实现 |

## TRPO算法

### TRPO原理

Trust Region Policy Optimization：
$\max_\theta \mathbb{E}[A(s,a) \frac{\pi_\theta}{\pi_{\theta_{old}}}]$
约束：$\mathbb{E}[KL(\pi_{\theta_{old}}, \pi_\theta)] \leq \delta$

### TRPO vs PPO

| 方面 | TRPO | PPO |
|------|------|------|
| 约束 | KL约束 | Clip近似 |
| 实现 | 复杂 | 简单 |
| 性能 | 类似 | 类似 |

## 案例实践

### REINFORCE CartPole

```python
import gym

def train_reinforce_cartpole():
    env = gym.make('CartPole-v1')
    
    policy = PolicyNetwork(state_dim=4, action_dim=2)
    agent = REINFORCE(policy, lr=0.01)
    
    rewards_history = []
    
    for episode in range(500):
        log_probs = []
        rewards = []
        
        state = env.reset()
        while True:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        returns = agent.compute_returns(rewards)
        agent.update(log_probs, returns)
        
        total_reward = sum(rewards)
        rewards_history.append(total_reward)
        
        if episode % 50 == 0:
            print(f"Episode {episode}: Reward {total_reward}")
    
    return agent, rewards_history
```

### PPO连续控制

```python
import gym

def train_ppo_continuous():
    env = gym.make('Pendulum-v0')
    
    # 连续策略
    policy = GaussianPolicy(state_dim=3, action_dim=1)
    value = nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    agent = PPO(policy, value)
    
    for episode in range(1000):
        states, actions, rewards, log_probs = collect_trajectory(env, policy)
        agent.update(states, actions, log_probs, rewards)
```

### 训练监控

```python
def monitor_training(agent, env, eval_episodes=10):
    """评估策略性能"""
    total_rewards = []
    
    for _ in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)

def plot_training_curve(rewards):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['Raw', 'Smoothed'])
    plt.show()
```

## 基于策略方法总结

### 优点

| 优点 | 描述 |
|------|------|
| 连续动作 | 自然支持 |
| 随机策略 | 可学习概率策略 |
| 收敛稳定 | 通常更稳定 |
| 无价值偏差 | 直接优化策略 |

### 缺点

| 缺点 | 描述 |
|------|------|
| 高方差 | 需技巧降低 |
| 样本效率 | 可能较低 |
| 局部最优 | 可能陷入 |

### 选择建议

| 场景 | 推荐 |
|------|------|
| 连续动作 | PPO、SAC |
| 离散动作 | PPO、A2C |
| 需稳定 | PPO |
| 样本效率 | Actor-Critic |

## 总结

基于策略的方法直接优化策略。核心内容包括：
- 策略梯度：直接优化策略参数
- REINFORCE：蒙特卡洛策略梯度
- Actor-Critic：结合价值和策略
- A2C/A3C：同步/异步Actor-Critic
- PPO：限制策略更新，稳定训练
- TRPO：KL约束优化

基于策略的方法适合连续动作空间，可学习随机策略。

## 延伸阅读

- [强化学习基础](/2026/05/10/zh-CN/技术文档/机器学习/rl-introduction/)
- [基于价值的方法](/2026/05/10/zh-CN/技术文档/机器学习/rl-value-based/)
- [强化学习进阶](/2026/05/10/zh-CN/技术文档/机器学习/rl-advanced/)
- [优化算法详解](/2026/05/10/zh-CN/技术文档/机器学习/optimization-algorithms/)
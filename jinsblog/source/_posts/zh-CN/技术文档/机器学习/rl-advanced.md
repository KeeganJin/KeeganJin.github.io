---
title: 强化学习进阶
date: 2026-04-14
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 强化学习, RLHF, 多智能体]
---

## 模型预测与规划

### Model-based强化学习

学习环境模型，用于规划：

**环境模型**：
- 状态转移模型：$P(s'|s,a)$
- 奖励模型：$R(s,a)$

### Dyna算法

结合Model-free和Model-based：

```
1. 用真实经验更新Q（Q-Learning）
2. 存储经验到记忆
3. 用记忆模拟经验，多次更新Q
```

```python
class DynaQ:
    def __init__(self, n_states, n_actions, n_planning=10):
        self.q_table = np.zeros((n_states, n_actions))
        self.model = {}  # 存储(s,a)->(r,s')
        self.n_planning = n_planning
    
    def update(self, state, action, reward, next_state):
        # 直接学习
        self.q_learning_update(state, action, reward, next_state)
        
        # 存储模型
        self.model[(state, action)] = (reward, next_state)
        
        # 模拟规划
        for _ in range(self.n_planning):
            s, a = random.choice(list(self.model.keys()))
            r, s_next = self.model[(s, a)]
            self.q_learning_update(s, a, r, s_next)
```

### 世界模型

学习完整的环境模型：
- 状态预测
- 奖励预测
- 用于高效规划

```python
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # 状态预测
        self.state_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        
        # 奖励预测
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        next_state = self.state_net(x)
        reward = self.reward_net(x)
        return next_state, reward
```

### MBPO

Model-Based Policy Optimization：
- 学习世界模型
- 在模型上短时规划
- 结合真实经验

## 多智能体强化学习

### 多智能体系统

多个Agent在同一环境交互。

**类型**：
| 类型 | 描述 |
|------|------|
| 合作 | 共同目标 |
| 竞争 | 对抗目标 |
| 混合 | 合作+竞争 |

### 多智能体挑战

| 挑战 | 描述 |
|------|------|
| 非平稳 | 其他Agent策略变化 |
| 信用分配 | 团队奖励如何分配 |
| 可扩展性 | Agent数量多时 |

### 独立学习

每个Agent独立学习：

```python
class IndependentLearners:
    def __init__(self, n_agents, state_dim, action_dim):
        self.agents = [DQNAgent(state_dim, action_dim) 
                       for _ in range(n_agents)]
    
    def act(self, states):
        actions = [agent.select_action(s) 
                   for agent, s in zip(self.agents, states)]
        return actions
    
    def update(self, experiences):
        for agent, exp in zip(self.agents, experiences):
            agent.update(*exp)
```

### QMIX

合作多智能体：
- 集中训练，分散执行
- 混合价值函数

### MADDPG

多Agent DDPG：
- 每个Agent有自己的Actor
- Critic看全局状态

```python
class MADDPG:
    def __init__(self, n_agents, state_dim, action_dim):
        self.actors = [ActorNetwork(state_dim, action_dim) 
                       for _ in range(n_agents)]
        self.critics = [CriticNetwork(n_agents * state_dim, 
                                       n_agents * action_dim)
                        for _ in range(n_agents)]
    
    def act(self, observations):
        actions = [actor(obs) for actor, obs in zip(self.actors, observations)]
        return actions
    
    def update(self, batch):
        # 每个Agent用自己的Critic评估全局
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            # 计算Q_i(o, a_1, ..., a_n)
            full_states = torch.cat(batch.states)
            full_actions = torch.cat(batch.actions)
            q_value = critic(full_states, full_actions)
            
            # 更新Actor
            action_i = actor(batch.observations[i])
            loss = -q_value.mean()
            
            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
```

### 自我对弈

Agent与自己博弈：
- 两个相同策略对弈
- 不断改进

## RLHF人类反馈强化学习

### RLHF原理

用人类反馈训练LLM：
- 人类评价模型输出
- 学习奖励模型
- 用RL优化策略

### RLHF流程

```
1. 预训练模型（SFT）
2. 收集人类比较数据
3. 训练奖励模型
4. 用PPO优化策略
```

### 奖励模型学习

```python
class RewardModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # 基于LLM
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, text):
        hidden = self.model.encode(text)
        reward = self.reward_head(hidden)
        return reward

def train_reward_model(reward_model, comparisons):
    """
    comparisons: [(text_a, text_b, winner)]
    winner: 0 if a better, 1 if b better
    """
    for text_a, text_b, winner in comparisons:
        r_a = reward_model(text_a)
        r_b = reward_model(text_b)
        
        # Bradley-Terry模型
        prob_a = torch.sigmoid(r_a - r_b)
        loss = -torch.log(prob_a) if winner == 0 else -torch.log(1 - prob_a)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### PPO训练

```python
def rlhf_ppo_train(policy, reward_model, prompts):
    for prompt in prompts:
        # 生成回复
        response = policy.generate(prompt)
        
        # 计算奖励
        reward = reward_model(prompt + response)
        
        # PPO更新
        # KL惩罚防止偏离太远
        ref_log_prob = reference_model.log_prob(response)
        policy_log_prob = policy.log_prob(response)
        kl_penalty = kl_divergence(policy_log_prob, ref_log_prob)
        
        # 最终奖励
        final_reward = reward - kl_coef * kl_penalty
        
        # PPO损失
        ppo_loss = compute_ppo_loss(policy, response, final_reward)
        
        optimizer.zero_grad()
        ppo_loss.backward()
        optimizer.step()
```

### RLHF应用

| 应用 | 描述 |
|------|------|
| ChatGPT | OpenAI对话模型 |
| Claude | Anthropic助手 |
| LLaMA-2-chat | Meta对话模型 |

## 强化学习应用案例

### 游戏AI

**AlphaGo**：
- 策略网络+价值网络
- 蒙特卡洛树搜索
- 自我对弈训练

**AlphaZero**：
- 不需人类知识
- 纯自我对弈
- 适用于多种游戏

### 推荐系统

```python
class RecommenderRL:
    def __init__(self, user_features, item_features):
        self.state_dim = user_feature_dim
        self.action_dim = num_items
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
    
    def get_state(self, user_history):
        # 用户历史编码为状态
        return encode_history(user_history)
    
    def recommend(self, user_state):
        probs = self.policy(user_state)
        return torch.multinomial(probs, 10)
    
    def update(self, user_state, action, feedback):
        # 反馈为奖励（点击/购买）
        reward = feedback
        # 策略梯度更新
        ...
```

### 机器人控制

```python
def train_robot_control():
    env = RobotEnv()
    
    # SAC算法（适合连续控制）
    agent = SAC(state_dim=robot_state_dim, 
                action_dim=robot_action_dim)
    
    for episode in range(10000):
        state = env.reset()
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            if done:
                break
```

### 资源管理

数据中心冷却控制：
- 状态：温度、负载
- 动作：风扇转速、空调
- 奖励：能耗最小化

## 强化学习前沿

### Offline RL

从历史数据学习，不交互：

**挑战**：
- 分布偏移
- 数据质量

**方法**：
- Conservative Q-Learning
- Batch RL

### 分布式RL

大规模并行训练：
- IMPALA
- Ape-X
- RLlib

### 安全RL

考虑安全约束：
- Constrained Policy Optimization
- Safe RL with constraints

```python
def constrained_update(policy, cost_limit):
    """带约束的策略更新"""
    # 主目标：最大化奖励
    reward_objective = compute_reward_gradient()
    
    # 安全约束：成本不超过限制
    cost_objective = compute_cost_gradient()
    
    # Lagrange优化
    while estimated_cost > cost_limit:
        increase_lambda()
        update_with_constraint(reward_objective, cost_objective, lambda)
```

### 层次化RL

分层决策：
- 高层：设定子目标
- 低层：执行子任务

```python
class HierarchicalAgent:
    def __init__(self):
        self.high_policy = HighLevelPolicy()  # 选择子目标
        self.low_policy = LowLevelPolicy()    # 执行子目标
    
    def act(self, state):
        # 高层定期决策
        if should_decide_goal():
            goal = self.high_policy(state)
        
        # 低层执行
        action = self.low_policy(state, goal)
        return action
```

## 强化学习最佳实践

### 环境设计

| 建议 | 描述 |
|------|------|
| 合理奖励 | 与目标一致 |
| 适当稀疏 | 平衡学习难度 |
| 清晰终止 | 明确结束条件 |

### 网络设计

| 建议 | 描述 |
|------|------|
| 适当规模 | 不太大 |
| 稳定架构 | 经典结构 |
| 正则化 | 防过拟合 |

### 训练技巧

| 技巧 | 描述 |
|------|------|
| 学习率 | 适当衰减 |
| 探索 | 逐渐减少 |
| 监控 | 跟踪指标 |
| 调试 | 可视化决策 |

### 常见问题

| 问题 | 解决 |
|------|------|
| 不收敛 | 检查奖励、学习率 |
| 性能差 | 增探索、调网络 |
| 过拟合 | 数据多样性 |

## 总结

强化学习进阶涵盖高级主题。核心内容包括：
- Model-based方法：学习环境模型用于规划
- 多智能体RL：合作、竞争、混合场景
- RLHF：人类反馈训练LLM
- 应用案例：游戏AI、推荐、机器人
- 前沿方向：Offline RL、安全RL、层次化RL

强化学习应用广泛，持续发展。

## 延伸阅读

- [强化学习基础](/2026/05/10/zh-CN/技术文档/机器学习/rl-introduction/)
- [基于价值的方法](/2026/05/10/zh-CN/技术文档/机器学习/rl-value-based/)
- [基于策略的方法](/2026/05/10/zh-CN/技术文档/机器学习/rl-policy-based/)
- [大模型架构演进](/2026/05/10/zh-CN/技术文档/机器学习/llm-architecture/)
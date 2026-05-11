---
title: 分布式训练
date: 2026-04-19
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 分布式训练, 并行策略]
---

## 分布式训练概述

### 训练规模增长

| 模型 | 参数量 | GPU需求 |
|------|--------|----------|
| BERT | 110M | 1-4 GPU |
| GPT-2 | 1.5B | 8-16 GPU |
| GPT-3 | 175B | 千级GPU |
| LLaMA | 65B | 百级GPU |

### 分布式训练需求

| 需求 | 描述 |
|------|------|
| 模型过大 | 单GPU内存不足 |
| 训练加速 | 多GPU并行计算 |
| 大数据量 | 加速数据处理 |

### 分布式训练类型

| 类型 | 描述 |
|------|------|
| 数据并行 | 多GPU处理不同数据 |
| 模型并行 | 模型分布到多GPU |
| 混合并行 | 结合数据和模型并行 |

## 数据并行

### 数据并行原理

每个GPU持有完整模型副本，处理不同数据批次。

**流程**：
```
数据分为N份 → 每GPU处理一份 → 梯度聚合 → 更新参数
```

### 前向传播

各GPU独立计算：
- 每GPU获得不同数据子集
- 独立前向传播
- 计算损失

### 反向传播与梯度同步

各GPU计算梯度后同步：
- AllReduce聚合梯度
- 平均梯度
- 同步更新参数

### 数据并行实现

```python
import torch
import torch.distributed as dist
import torch.nn.parallel.DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """初始化分布式"""
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12345',
        world_size=world_size,
        rank=rank
    )

def train_ddp(rank, world_size):
    setup_distributed(rank, world_size)
    
    # 模型
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 数据（分布式采样）
    dataset = MyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 保证数据随机
        
        for batch in loader:
            batch = batch.to(rank)
            output = model(batch)
            loss = compute_loss(output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    dist.destroy_process_group()
```

### PyTorch DDP

```python
import torch.multiprocessing as mp

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)

if __name__ == '__main__':
    main()
```

### 分布式采样器

保证数据分片正确：

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)
```

## 模型并行

### 模型并行原理

模型分片到多GPU：
- 适用于超大模型
- 单GPU内存不足时

### 层间并行

模型不同层分布到不同GPU：

```
GPU 0: Layer 1-3
GPU 1: Layer 4-6
GPU 2: Layer 7-9
```

```python
class ModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # GPU 0上的层
        self.layer1 = nn.Linear(1024, 512).to('cuda:0')
        self.layer2 = nn.Linear(512, 256).to('cuda:0')
        
        # GPU 1上的层
        self.layer3 = nn.Linear(256, 128).to('cuda:1')
        self.layer4 = nn.Linear(128, 10).to('cuda:1')
    
    def forward(self, x):
        x = x.to('cuda:0')
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = x.to('cuda:1')  # 跨GPU传输
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
```

### 层内并行

单层内部并行计算：

**张量并行**：
- 分割矩阵乘法
- 多GPU并行计算

```python
class ColumnParallelLinear(nn.Module):
    """列并行线性层"""
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        
        # 每GPU持有部分权重
        self.weight = nn.Parameter(
            torch.randn(out_features // world_size, in_features)
        )
    
    def forward(self, x):
        # 每GPU计算部分输出
        output = F.linear(x, self.weight)
        
        # AllGather收集完整输出
        full_output = all_gather(output)
        return full_output
```

### Megatron-LM张量并行

Transformer层张量并行：
- 并行计算注意力
- 并行计算MLP

```python
class ParallelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, world_size):
        super().__init__()
        self.world_size = world_size
        
        # 分割注意力头
        self.num_heads_per_gpu = num_heads // world_size
        
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, world_size
        )
        self.out_proj = RowParallelLinear(
            hidden_size, hidden_size, world_size
        )
    
    def forward(self, x):
        qkv = self.qkv_proj(x)  # 并行投影
        
        # 分割注意力计算
        q, k, v = split_qkv(qkv)
        attn_output = attention(q, k, v)
        
        output = self.out_proj(attn_output)  # 并行合并
        return output
```

### Pipeline并行

流水线执行不同层：

```
时刻t: GPU0处理batch1, GPU1处理batch0
时刻t+1: GPU0处理batch2, GPU1处理batch1
```

**减少GPU空闲**。

## 混合并行策略

### 3D并行

结合三种并行：
- 数据并行
- 张量并行
- Pipeline并行

**示例**：
- 64 GPU
- 数据并行：8组
- 张量并行：4 GPU
- Pipeline并行：2段

### 并行策略选择

| 模型规模 | 推荐 |
|----------|------|
| 小（<1B） | 数据并行 |
| 中（1B-10B） | 数据+张量并行 |
| 大（>10B） | 3D并行 |

### 并行效率分析

| 策略 | 通信量 | 内存效率 |
|------|--------|----------|
| 数据并行 | 高 | 低（完整副本） |
| 张量并行 | 高 | 高 |
| Pipeline | 低 | 高 |

## 参数服务器架构

### 参数服务器原理

中心化参数管理：
- Worker：计算梯度
- Server：聚合更新参数

### 架构组成

```
Worker 1 ──→ Parameter Server ──← Worker 2
Worker 3 ──→                   ←── Worker 4
```

### 同步更新

所有Worker完成后再更新：

```python
class ParameterServer:
    def __init__(self, model):
        self.model = model
        self gradients = []
    
    def receive_gradient(self, gradient, worker_id):
        self.gradients.append(gradient)
        
        if len(self.gradients) == num_workers:
            # 平均梯度
            avg_gradient = sum(self.gradients) / num_workers
            self.update_model(avg_gradient)
            self.gradients = []
            
            # 发送新参数
            broadcast_parameters()
```

### 异步更新

Worker完成立即更新：

```python
class AsyncParameterServer:
    def receive_gradient(self, gradient):
        # 立即更新
        self.model.update(gradient)
        broadcast_parameters()
```

### 同步vs异步

| 方面 | 同步 | 异步 |
|------|------|------|
| 收敛性 | 稳定 | 可能波动 |
| 效率 | 可能等待 | 高效 |
| 适用 | 小规模 | 大规模 |

## AllReduce算法

### Ring AllReduce

环形通信高效聚合：

```
GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0
```

**步骤**：
1. Scatter-reduce：分段聚合
2. All-gather：广播完整结果

### NCCL实现

NVIDIA NCCL高效AllReduce：

```python
import torch.distributed as dist

# AllReduce
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# AllGather
dist.all_gather(tensor_list, tensor)

# Broadcast
dist.broadcast(tensor, src=0)
```

### 通信优化

| 技术 | 描述 |
|------|------|
| 梯度压缩 | 减少通信量 |
| 异步通信 | 计算通信重叠 |
| 梯度累积 | 减少通信频率 |

## 分布式训练框架

### PyTorch Distributed

原生分布式支持：
- DDP：数据并行
- RPC：远程调用
- Distributed：底层通信

### DeepSpeed

微软深度学习优化库：
- ZeRO：内存优化
- 混合精度训练
- Pipeline并行

```python
import deepspeed

# DeepSpeed配置
ds_config = {
    "train_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    },
    "fp16": {"enabled": True}
}

# 初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# 训练
for batch in train_loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### Megatron-LM

NVIDIA大模型训练框架：
- 张量并行
- Pipeline并行
- 混合精度

### Ray

通用分布式框架：

```python
import ray

ray.init()

@ray.remote
def train_worker(config):
    model = create_model(config)
    train(model)
    return model.results

# 并行执行
results = [train_worker.remote(config) for config in configs]
ray.get(results)
```

## 案例实践

### 多GPU训练示例

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # 初始化
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    
    # 模型
    model = MyModel().cuda()
    model = DDP(model, device_ids=[local_rank])
    
    # 数据
    dataset = MyDataset()
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(100):
        for batch in loader:
            batch = batch.cuda()
            output = model(batch)
            loss = F.cross_entropy(output, batch.labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

### DeepSpeed ZeRO训练

```python
def train_with_deepspeed():
    ds_config = {
        "train_batch_size": 128,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu"},
            "offload_optimizer": {"device": "cpu"}
        },
        "fp16": {"enabled": True}
    }
    
    model_engine = deepspeed.initialize(
        model=model,
        config_params=ds_config
    )
    
    for epoch in range(epochs):
        for batch in loader:
            loss = model_engine(batch)
            model_engine.backward(loss)
            model_engine.step()
```

### 分布式推理

```python
def distributed_inference(model, data):
    """分布式推理"""
    # 分割数据
    chunks = split_data(data, world_size)
    
    # 各GPU处理部分数据
    local_chunk = chunks[rank]
    local_results = model(local_chunk)
    
    # 收集结果
    all_results = all_gather(local_results)
    return combine_results(all_results)
```

### 训练监控

```python
import wandb

def setup_monitoring():
    """训练监控"""
    wandb.init(project="distributed-training")
    
    # 记录指标
    wandb.log({
        "loss": loss,
        "learning_rate": lr,
        "gpu_memory": torch.cuda.memory_allocated()
    })
```

## 分布式训练最佳实践

### 数据加载优化

| 建议 | 描述 |
|------|------|
| 预取数据 | 异步加载 |
| 分布式采样 | 保证数据分片 |
| 内存映射 | 大数据处理 |

### 通信优化

| 建议 | 描述 |
|------|------|
| 梯度累积 | 减少通信 |
| 计算重叠 | 异步通信 |
| 混合精度 | 减少数据量 |

### 内存管理

| 建议 | 描述 |
|------|------|
| ZeRO优化 | 分片参数 |
| 激活checkpoint | 减少内存 |
| CPU卸载 | 利用CPU内存 |

## 分布式训练挑战

### 通信瓶颈

**问题**：通信延迟高。

**解决**：
- 高速网络
- 梯度压缩
- 计算通信重叠

### 同步开销

**问题**：等待其他GPU。

**解决**：
- 异步更新
- Pipeline并行
- 负载均衡

### 容错性

**问题**：节点故障。

**解决**：
- 检查点保存
- 故障恢复
- 弹性训练

## 总结

分布式训练是大模型训练的关键。核心内容包括：
- 数据并行：多GPU处理不同数据
- 模型并行：模型分片到多GPU
- 混合并行：3D并行策略
- 参数服务器：中心化参数管理
- AllReduce：高效梯度聚合
- 训练框架：DeepSpeed、Megatron-LM

分布式训练使大规模模型训练成为可能。

## 延伸阅读

- [模型压缩与加速](/2026/05/10/zh-CN/技术文档/机器学习/model-compression/)
- [大模型架构演进](/2026/05/10/zh-CN/技术文档/机器学习/llm-architecture/)
- [反向传播算法详解](/2026/05/10/zh-CN/技术文档/机器学习/backpropagation/)
- [模型部署实践](/2026/05/10/zh-CN/技术文档/机器学习/model-deployment/)
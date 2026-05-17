---
title: 模型部署实践
date: 2026-02-23
categories: [技术文档, 机器学习]
lang: zh-CN
tags: [机器学习, 模型部署, 推理优化, ONNX]
---

## 模型部署概述

### 部署流程

```
训练模型 → 模型导出 → 模型优化 → 部署 → 监控
```

### 部署场景

| 场景 | 描述 |
|------|------|
| 云端部署 | 高性能服务器 |
| 边缘部署 | 移动/嵌入式设备 |
| 本地部署 | 本地应用 |
| 浏览器部署 | Web应用 |

### 部署挑战

| 挑战 | 描述 |
|------|------|
| 性能要求 | 推理延迟 |
| 资源限制 | 内存/存储 |
| 环境差异 | 不同硬件 |
| 可维护性 | 版本管理 |

## ONNX模型转换

### ONNX概述

Open Neural Network Exchange：
- 跨框架模型格式
- 标准化模型表示
- 广泛支持

### ONNX优势

| 优势 | 描述 |
|------|------|
| 跨框架 | PyTorch、TensorFlow等 |
| 跨平台 | Windows、Linux、Mac |
| 优化支持 | TensorRT等优化器 |

### PyTorch转ONNX

```python
import torch
import torch.onnx

def export_to_onnx(model, input_sample, output_path):
    """导出为ONNX"""
    model.eval()
    
    torch.onnx.export(
        model,
        input_sample,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")

# 示例
model = MyModel()
dummy_input = torch.randn(1, 3, 224, 224)
export_to_onnx(model, dummy_input, "model.onnx")
```

### ONNX验证

```python
import onnx

def validate_onnx(model_path):
    """验证ONNX模型"""
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid")
    
    # 打印模型信息
    print(onnx.helper.printable_graph(model.graph))

validate_onnx("model.onnx")
```

### ONNX Runtime推理

```python
import onnxruntime as ort

def inference_onnx(model_path, input_data):
    """ONNX推理"""
    session = ort.InferenceSession(model_path)
    
    # 获取输入名称
    input_name = session.get_inputs()[0].name
    
    # 推理
    outputs = session.run(None, {input_name: input_data})
    
    return outputs

# 使用
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = inference_onnx("model.onnx", input_data)
```

### TensorFlow转ONNX

```python
import tf2onnx

def convert_tf_to_onnx(saved_model_path, output_path):
    """TensorFlow SavedModel转ONNX"""
    model_proto, external_tensor_storage = tf2onnx.convert.from_saved_model(
        saved_model_path,
        output_path=output_path,
        opset=12
    )
    
    print(f"Converted to {output_path}")
```

## TensorRT加速

### TensorRT概述

NVIDIA高性能推理引擎：
- 针对GPU优化
- 支持量化
- 层融合优化

### TensorRT优化

| 优化 | 描述 |
|------|------|
| 层融合 | 合并层减少计算 |
| 精度校准 | FP16/INT8优化 |
| 内核调优 | 自动选择最优内核 |
| 动态批处理 | 自动批处理 |

### ONNX转TensorRT

```python
import tensorrt as trt

def build_engine(onnx_path, max_batch_size=32):
    """构建TensorRT引擎"""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    with trt.Builder(TRT_LOGGER) as builder:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # 解析ONNX
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
        
        # 配置
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # FP16模式
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # 构建引擎
        engine = builder.build_engine(network, config)
        
        return engine

# 保存引擎
engine = build_engine("model.onnx")
with open("model.trt", "wb") as f:
    f.write(engine.serialize())
```

### TensorRT推理

```python
import pycuda.driver as cuda
import pycuda.autoinit

def trt_inference(engine, input_data):
    """TensorRT推理"""
    context = engine.create_execution_context()
    
    # 分配内存
    input_binding = engine.get_binding_index('input')
    output_binding = engine.get_binding_index('output')
    
    input_shape = engine.get_binding_shape(input_binding)
    output_shape = engine.get_binding_shape(output_binding)
    
    # GPU内存
    input_mem = cuda.mem_alloc(input_data.nbytes)
    output_mem = cuda.mem_alloc(np.prod(output_shape) * 4)
    
    # 传输数据
    cuda.memcpy_htod(input_mem, input_data)
    
    # 执行
    context.execute_v2([int(input_mem), int(output_mem)])
    
    # 取回结果
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, output_mem)
    
    return output
```

### TensorRT量化

```python
def calibrate_int8(engine, calibration_data):
    """INT8校准"""
    calibrator = MyCalibrator(calibration_data)
    
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    
    return builder.build_engine(network, config)
```

## 模型服务化

### 服务化架构

```
客户端 → API网关 → 模型服务 → 模型推理
```

### Flask服务

```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
model = torch.load('model.pt')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_tensor = preprocess(data['input'])
    
    with torch.no_grad():
        output = model(input_tensor)
    
    result = postprocess(output)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI服务

```python
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort

app = FastAPI()
session = ort.InferenceSession("model.onnx")

class InputData(BaseModel):
    features: list

@app.post("/predict")
async def predict(data: InputData):
    input_array = np.array(data.features).astype(np.float32)
    outputs = session.run(None, {'input': input_array})
    return {"prediction": outputs[0].tolist()}

# 运行: uvicorn server:app --host 0.0.0.0 --port 8000
```

### TorchServe

PyTorch官方模型服务器：

```python
# 打包模型
torch-model-archiver --model-name mymodel \
    --version 1.0 \
    --model-file model.py \
    --serialized-file model.pt \
    --handler custom_handler.py

# 启动服务
torchserve --start --model-store model_store --models mymodel.mar
```

### Triton推理服务器

NVIDIA高性能推理服务器：

```yaml
# config.pbtxt
name: "mymodel"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
```

```bash
# 启动
tritonserver --model-repository=/models
```

## 推理优化技巧

### 批处理优化

```python
def batch_inference(model, requests, batch_size=32):
    """批量推理"""
    results = []
    
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        batch_tensor = stack_batch(batch)
        
        outputs = model(batch_tensor)
        results.extend(split_outputs(outputs))
    
    return results
```

### 动态批处理

```python
class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, timeout=0.1):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.queue = []
    
    def add_request(self, input_data):
        self.queue.append(input_data)
        
        if len(self.queue) >= self.max_batch_size:
            return self.process_batch()
        
        return None
    
    def process_batch(self):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        
        return self.model(stack_batch(batch))
```

### 异步推理

```python
import asyncio

async def async_inference(model, inputs):
    """异步推理"""
    loop = asyncio.get_event_loop()
    
    # 在后台线程执行
    output = await loop.run_in_executor(
        None,
        model,
        inputs
    )
    
    return output

async def process_requests(requests):
    results = await asyncio.gather(
        *[async_inference(model, req) for req in requests]
    )
    return results
```

### 预热推理

```python
def warmup(model, input_shape, iterations=10):
    """模型预热"""
    dummy_input = torch.randn(input_shape)
    
    for _ in range(iterations):
        with torch.no_grad():
            model(dummy_input)
    
    print("Warmup complete")
```

### 内存优化

```python
def optimized_inference(model, inputs):
    """内存优化推理"""
    # 使用torch.inference_mode
    with torch.inference_mode():
        output = model(inputs)
    
    return output

# 或使用no_grad
def inference_no_grad(model, inputs):
    with torch.no_grad():
        output = model(inputs)
    return output
```

## 案例实践

### 完整部署流程

```python
class ModelDeployment:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.warmup()
    
    def load_model(self, path):
        """加载模型"""
        if path.endswith('.onnx'):
            return ort.InferenceSession(path)
        elif path.endswith('.pt'):
            model = torch.load(path)
            model.eval()
            return model
    
    def warmup(self):
        """预热"""
        dummy = self.create_dummy_input()
        self.inference(dummy)
    
    def inference(self, input_data):
        """推理"""
        if isinstance(self.model, ort.InferenceSession):
            return self.model.run(None, {'input': input_data})
        else:
            with torch.inference_mode():
                return self.model(input_data)
    
    def serve(self, port=8000):
        """启动服务"""
        app = FastAPI()
        
        @app.post("/predict")
        async def predict(data: InputData):
            result = self.inference(data.features)
            return {"prediction": result}
        
        uvicorn.run(app, host="0.0.0.0", port=port)
```

### LLM部署

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def deploy_llm(model_name):
    """部署LLM"""
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 推理函数
    def generate(prompt, max_length=100):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True
            )
        
        return tokenizer.decode(outputs[0])
    
    return generate

# 服务化
@app.post("/generate")
async def generate_text(request):
    result = generate(request.prompt)
    return {"text": result}
```

### 量化推理部署

```python
def deploy_quantized(model_path):
    """部署量化模型"""
    from transformers import BitsAndBytesConfig
    
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=config,
        device_map="auto"
    )
    
    return model
```

### 边缘部署

```python
def deploy_to_mobile(model, output_path):
    """移动端部署"""
    # 转换为移动格式
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    
    # 或ONNX
    export_to_onnx(model, dummy_input, "mobile.onnx")
    
    # 量化
    quantized = torch.quantization.quantize_dynamic(
        scripted_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized
```

## 部署最佳实践

### 模型优化

| 建议 | 描述 |
|------|------|
| 模型量化 | 减少计算和存储 |
| 层融合 | 减少层间开销 |
| 算子优化 | 使用高效算子 |

### 服务优化

| 建议 | 描述 |
|------|------|
| 批处理 | 提高吞吐量 |
| 异步处理 | 非阻塞响应 |
| 缓存 | 缓存重复请求 |

### 可靠性

| 建议 | 描述 |
|------|------|
| 健康检查 | 监控服务状态 |
| 优雅退出 | 处理中断 |
| 错误处理 | 异常恢复 |

### 监控

```python
import prometheus_client

# 指标
REQUEST_COUNT = prometheus_client.Counter('request_count', 'Request count')
LATENCY = prometheus_client.Histogram('latency', 'Request latency')

@app.post("/predict")
async def predict(data):
    REQUEST_COUNT.inc()
    
    start = time.time()
    result = inference(data)
    latency = time.time() - start
    
    LATENCY.observe(latency)
    return result
```

## 部署工具对比

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| Flask | 简单 | 小型应用 |
| FastAPI | 高性能 | 现代应用 |
| TorchServe | PyTorch官方 | PyTorch模型 |
| Triton | NVIDIA | GPU推理 |
| ONNX Runtime | 跨平台 | ONNX模型 |
| TensorRT | 高性能 | NVIDIA GPU |

## 总结

模型部署是模型应用的最后一步。核心内容包括：
- ONNX转换：跨框架模型格式
- TensorRT加速：GPU推理优化
- 模型服务化：API服务部署
- 推理优化：批处理、异步、预热
- 部署实践：完整部署流程

模型部署需要考虑性能、资源和可维护性。

## 延伸阅读

- [模型压缩与加速](/2026/05/10/zh-CN/技术文档/机器学习/model-compression/)
- [分布式训练](/2026/05/10/zh-CN/技术文档/机器学习/distributed-training/)
- [神经网络入门](/2026/05/10/zh-CN/技术文档/机器学习/neural-network-intro/)
- [大模型架构演进](/2026/05/10/zh-CN/技术文档/机器学习/llm-architecture/)
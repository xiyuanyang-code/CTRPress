# Metrics

- Language Model Metrics：
    - PPL
    - Position-wise PPL (front_ppl)
    - Position-wise PPL (middle_ppl)
    - Position-wise PPL (back_ppl)
- Time Efficiency
    - Prefilling time
    - time to first token
    - time per token
    - generation time
    - throughput
- Memory Efficiency
    - peak memory usage
    - kv cache size

> 具体在实验中要记录的指标就这些，其他和 baseline 比较的指标我们在处理数据的时候统一计算

## 一、语言模型质量指标

### 1.1 Perplexity (PPL) - 困惑度

**含义：**
困惑度是语言模型评估中最核心的指标，表示模型对文本的"困惑程度"。PPL 越低，表示模型对文本的理解越好，预测能力越强。

**直观理解：**
- PPL = 10 表示模型在每个位置平均有 10 个等可能的选择
- PPL = 1 表示模型能完美预测下一个 token
- PPL 越低越好

**计算公式：**
```
PPL = exp(Loss)
Loss = -1/N * Σ log P(x_t | x_{<t})

其中：
- N: 序列长度
- P(x_t | x_{<t}): 模型预测第 t 个 token 的概率
```

**测量方法：**
```python
import torch

def calculate_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    ppl = torch.exp(loss)
    return ppl.item()
```

**适用场景：**
- 评估压缩后的语言建模能力
- 比较不同压缩方法的质量损失
- WikiText 和 PG-19 的标准评估指标

**优缺点：**
- ✅ 直观易懂，是标准评估指标
- ✅ 与语言建模目标直接相关
- ❌ 对长尾分布敏感
- ❌ 不直接反映下游任务性能


### 1.6 Position-wise PPL - 按位置困惑度

**含义：**
将文本分成前、中、后三段，分别计算 PPL，分析压缩对不同位置的影响。

**测量方法：**
```python
def calculate_position_ppl(model, tokenizer, text, n_segments=3):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]
    segment_len = seq_len // n_segments

    position_ppls = []
    for i in range(n_segments):
        start = i * segment_len
        end = (i + 1) * segment_len if i < n_segments - 1 else seq_len

        segment_inputs = {
            "input_ids": inputs["input_ids"][:, start:end],
            "attention_mask": inputs["attention_mask"][:, start:end]
        }

        with torch.no_grad():
            outputs = model(**segment_inputs, labels=segment_inputs["input_ids"])
        ppl = torch.exp(outputs.loss).item()
        position_ppls.append(ppl)

    return position_ppls  # [front_ppl, middle_ppl, back_ppl]
```

**适用场景：**
- 分析压缩对文本不同部分的影响
- 检测位置偏差
- 评估长距离依赖的保持程度

---

## 二、性能效率指标

### 2.1 Prefilling Time - 预填充时间

**含义：**
处理输入上下文（prompt）所需的时间。这是 KV cache 压缩主要优化的阶段。

**测量方法：**
```python
from time import time
import torch

def measure_prefilling_time(model, inputs, press=None):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    start = time()
    with torch.no_grad():
        if press:
            with press(model):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
    torch.cuda.synchronize()
    elapsed = time() - start

    return elapsed
```

**适用场景：**
- 评估压缩带来的预填充加速效果
- 比较不同压缩方法的预填充效率
- 长上下文处理的关键指标

---

### 2.2 Generation Time - 生成时间

**含义：**
生成指定数量 token 所需的总时间。

**测量方法：**
```python
def measure_generation_time(model, inputs, max_new_tokens=100, press=None):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    start = time()
    with torch.no_grad():
        if press:
            with press(model):
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    elapsed = time() - start

    return elapsed, outputs.shape[1] - inputs["input_ids"].shape[1]
```

**适用场景：**
- 评估解码阶段的效率
- 计算吞吐量的基础
- 评估 KV cache 压缩对生成的影响

---

### 2.3 Time Per Token - 每 Token 生成时间

**含义：**
生成每个 token 的平均时间，是衡量解码效率的关键指标。

**计算公式：**
```
Time Per Token = Generation Time / Number of Generated Tokens
```

**测量方法：**
```python
def measure_time_per_token(model, inputs, max_new_tokens=100, press=None, n_runs=5):
    times = []
    for _ in range(n_runs):
        elapsed, n_tokens = measure_generation_time(model, inputs, max_new_tokens, press)
        times.append(elapsed / n_tokens * 1000)  # 转换为毫秒

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times)
    }
```

**适用场景：**
- 评估实时生成的延迟
- 比较不同压缩方法的解码效率
- 用户体验的关键指标

---

### 2.4 Throughput - 吞吐量

**含义：**
每秒生成的 token 数量，反映模型的处理能力。

**计算公式：**
```
Throughput = Number of Generated Tokens / Generation Time (tokens/second)
```

**测量方法：**
```python
def measure_throughput(model, inputs, max_new_tokens=100, press=None):
    elapsed, n_tokens = measure_generation_time(model, inputs, max_new_tokens, press)
    throughput = n_tokens / elapsed
    return throughput
```

**适用场景：**
- 评估批量处理能力
- 服务器部署的容量规划
- 成本效益分析

---


## 三、内存效率指标

### 3.1 Peak Memory Usage - 峰值内存使用

**含义：**
推理过程中 GPU 内存的最大使用量，包括模型权重、KV cache 和其他开销。

**测量方法：**
```python
def measure_peak_memory(model, inputs, press=None):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # 记录初始内存
    initial_memory = torch.cuda.max_memory_allocated()

    with torch.no_grad():
        if press:
            with press(model):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    # 记录峰值内存
    peak_memory = torch.cuda.max_memory_allocated()

    return {
        "initial_mb": initial_memory / 1024**2,
        "peak_mb": peak_memory / 1024**2,
        "total_mb": (peak_memory - initial_memory) / 1024**2
    }
```

**适用场景：**
- 评估内存需求
- 确定可支持的最大 batch size
- 硬件资源规划

---

### 3.2 KV Cache Size - KV 缓存大小

**含义：**
KV 缓存占用的内存大小，是长上下文推理的主要内存消耗。

**测量方法：**
```python
def measure_kv_cache_size(cache):
    """测量 KV cache 的大小"""
    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
        # DynamicCache
        size = 0
        for key in cache.key_cache:
            size += key.element_size() * key.nelement()
        for value in cache.value_cache:
            size += value.element_size() * value.nelement()
        return size
    else:
        # 其他类型的 cache
        return 0

def measure_cache_size_after_compression(model, inputs, press):
    """测量压缩后的 cache 大小"""
    from transformers import DynamicCache

    torch.cuda.empty_cache()
    cache = DynamicCache()

    with torch.no_grad():
        with press(model):
            outputs = model(**inputs, past_key_values=cache)

    cache_size = measure_kv_cache_size(cache)
    return cache_size / 1024**3  # 转换为 GB
```

**适用场景：**
- 评估压缩对 KV cache 的压缩效果
- 内存优化的关键指标
- 长上下文处理能力评估



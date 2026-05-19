# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Compression Evaluator

核心指标测量：
- Language Model Metrics: PPL, Position-wise PPL
- Time Efficiency: Prefilling time, TTFT, Time per token, Generation time, Throughput
- Memory Efficiency: KV cache size
"""

import torch
import numpy as np
from time import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    # Language Model Metrics
    ppl: float
    front_ppl: float
    middle_ppl: float
    back_ppl: float

    # Time Efficiency (seconds)
    prefilling_time: float
    ttft: float
    time_per_token: float
    generation_time: float
    throughput: float

    # Memory Efficiency (GB)
    kv_cache_size: float


class KVCacheEvaluator:
    """KV Cache 压缩评估器"""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 prompt_length: int = 1024, eval_length: int = 128):
        """
        初始化评估器

        Args:
            model_path: 模型路径
            device: 设备
            prompt_length: prompt 部分的 token 数（用于 prefill + 压缩）
            eval_length: eval 部分的 token 数（用于计算 PPL / 生成）
        """
        self.device = device
        self.prompt_length = prompt_length
        self.eval_length = eval_length
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _get_language_model(self):
        """获取 backbone language model，适配不同架构"""
        from transformers import GPTNeoXForCausalLM
        if isinstance(self.model, GPTNeoXForCausalLM):
            return self.model.gpt_neox
        return self.model.model

    def calculate_ppl(self, text: str, press=None) -> float:
        """
        计算困惑度

        1. 用前 prompt_length 个 token 做 prefill（带压缩），构建 KV cache
        2. 用压缩后的 KV cache 计算接下来 eval_length 个 token 的 PPL

        Args:
            text: 输入文本
            press: KV cache 压缩方法

        Returns:
            PPL 值
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        total_needed = self.prompt_length + self.eval_length
        if seq_len < total_needed:
            raise ValueError(
                f"Input has {seq_len} tokens, need at least {total_needed} "
                f"(prompt={self.prompt_length} + eval={self.eval_length})"
            )

        context_ids = input_ids[:, :self.prompt_length]
        eval_ids = input_ids[:, self.prompt_length:self.prompt_length + self.eval_length]

        with torch.no_grad():
            cache = DynamicCache()

            # Step 1: Prefill context（带或不带压缩）
            lm = self._get_language_model()
            if press:
                with press(self.model):
                    lm(input_ids=context_ids, past_key_values=cache)
            else:
                lm(input_ids=context_ids, past_key_values=cache)

            # Step 2: 用 cache 计算评估部分的 PPL
            position_ids = torch.arange(self.prompt_length, self.prompt_length + self.eval_length, device=self.device).unsqueeze(0)
            outputs = self.model(
                input_ids=eval_ids,
                past_key_values=cache,
                position_ids=position_ids,
                labels=eval_ids,
            )
            loss = outputs.loss

        ppl = torch.exp(loss).item()
        return ppl

    def calculate_position_ppl(self, text: str, press=None) -> Tuple[float, float, float]:
        """
        计算按位置的困惑度（前、中、后）

        对 prompt_length 的 context 做 prefill + 压缩，然后对 eval_length
        等分为三段分别计算 PPL：
        - Front: eval 前 1/3
        - Middle: eval 中间 1/3
        - Back: eval 后 1/3

        Args:
            text: 输入文本
            press: KV cache 压缩方法

        Returns:
            (front_ppl, middle_ppl, back_ppl)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        total_needed = self.prompt_length + self.eval_length
        if seq_len < total_needed:
            raise ValueError(
                f"Input has {seq_len} tokens, need at least {total_needed} "
                f"(prompt={self.prompt_length} + eval={self.eval_length})"
            )

        context_ids = input_ids[:, :self.prompt_length]
        eval_ids = input_ids[:, self.prompt_length:self.prompt_length + self.eval_length]
        segment_len = self.eval_length // 3

        position_ppls = []
        for i in range(3):
            seg_start = i * segment_len
            seg_end = (i + 1) * segment_len if i < 2 else self.eval_length

            seg_eval_ids = eval_ids[:, seg_start:seg_end]

            with torch.no_grad():
                cache = DynamicCache()

                # Prefill context（带或不带压缩）
                lm = self._get_language_model()
                if press:
                    with press(self.model):
                        lm(input_ids=context_ids, past_key_values=cache)
                else:
                    lm(input_ids=context_ids, past_key_values=cache)

                # 计算该段的 PPL
                abs_start = self.prompt_length + seg_start
                abs_end = self.prompt_length + seg_end
                position_ids = torch.arange(abs_start, abs_end, device=self.device).unsqueeze(0)
                try:
                    outputs = self.model(
                        input_ids=seg_eval_ids,
                        past_key_values=cache,
                        position_ids=position_ids,
                        labels=seg_eval_ids,
                    )
                    ppl = torch.exp(outputs.loss).item()
                except (AssertionError, RuntimeError):
                    ppl = float("nan")

            position_ppls.append(ppl)

        return tuple(position_ppls)

    def measure_prefilling_time(self, text: str, press=None) -> float:
        """
        测量预填充时间

        Args:
            text: 输入文本
            press: KV cache 压缩方法

        Returns:
            预填充时间（秒）
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # 预热
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 测量预填充时间
        start = time()
        with torch.no_grad():
            if press:
                with press(self.model):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        torch.cuda.synchronize()
        elapsed = time() - start

        return elapsed

    def measure_ttft(self, text: str, press=None) -> float:
        """
        测量 Time to First Token (TTFT)

        Args:
            text: 输入文本
            press: KV cache 压缩方法

        Returns:
            TTFT 时间（秒）
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # 预热
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 测量 TTFT（预填充 + 第一个 token 生成）
        start = time()
        with torch.no_grad():
            if press:
                with press(self.model):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False
                )
        torch.cuda.synchronize()
        elapsed = time() - start

        return elapsed

    def measure_generation_time(self, text: str, max_new_tokens: int = 100, press=None) -> Tuple[float, int]:
        """
        测量生成时间

        Args:
            text: 输入文本
            max_new_tokens: 生成的最大 token 数
            press: KV cache 压缩方法

        Returns:
            (生成时间, 生成的 token 数)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # 预热
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 测量生成时间
        start = time()
        with torch.no_grad():
            if press:
                with press(self.model):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
        torch.cuda.synchronize()
        elapsed = time() - start

        n_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        return elapsed, n_generated

    def measure_time_per_token(self, text: str, max_new_tokens: int = 100, press=None) -> float:
        """
        测量每 token 生成时间

        Args:
            text: 输入文本
            max_new_tokens: 生成的最大 token 数
            press: KV cache 压缩方法

        Returns:
            每 token 生成时间（毫秒）
        """
        elapsed, n_tokens = self.measure_generation_time(text, max_new_tokens, press)
        return elapsed / n_tokens * 1000  # 转换为毫秒

    def measure_throughput(self, text: str, max_new_tokens: int = 100, press=None) -> float:
        """
        测量吞吐量

        Args:
            text: 输入文本
            max_new_tokens: 生成的最大 token 数
            press: KV cache 压缩方法

        Returns:
            吞吐量（tokens/second）
        """
        elapsed, n_tokens = self.measure_generation_time(text, max_new_tokens, press)
        throughput = n_tokens / elapsed
        return throughput

    def measure_kv_cache_size(self, text: str, press=None) -> float:
        """
        测量 KV cache 大小

        Args:
            text: 输入文本
            press: KV cache 压缩方法

        Returns:
            KV cache 大小（GB）
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # 清空缓存
        torch.cuda.empty_cache()
        cache = DynamicCache()

        # 测量 KV cache 大小
        with torch.no_grad():
            if press:
                with press(self.model):
                    outputs = self.model(**inputs, past_key_values=cache)
            else:
                outputs = self.model(**inputs, past_key_values=cache)

        # 计算 cache 大小
        cache_size = 0
        for layer in cache.layers:
            cache_size += layer.keys.element_size() * layer.keys.nelement()
            cache_size += layer.values.element_size() * layer.values.nelement()

        return cache_size / 1024**3  # 转换为 GB

    def evaluate(self, text: str, press=None, max_new_tokens: int = 100) -> EvaluationMetrics:
        """
        完整评估

        Args:
            text: 输入文本
            press: KV cache 压缩方法
            max_new_tokens: 生成的最大 token 数

        Returns:
            评估指标
        """
        # Language Model Metrics
        ppl = self.calculate_ppl(text, press)
        front_ppl, middle_ppl, back_ppl = self.calculate_position_ppl(text, press)

        # Time Efficiency
        prefilling_time = self.measure_prefilling_time(text, press)
        ttft = self.measure_ttft(text, press)
        time_per_token = self.measure_time_per_token(text, max_new_tokens, press)
        generation_time, n_tokens = self.measure_generation_time(text, max_new_tokens, press)
        throughput = n_tokens / generation_time

        # Memory Efficiency
        kv_cache_size = self.measure_kv_cache_size(text, press)

        return EvaluationMetrics(
            ppl=ppl,
            front_ppl=front_ppl,
            middle_ppl=middle_ppl,
            back_ppl=back_ppl,
            prefilling_time=prefilling_time,
            ttft=ttft,
            time_per_token=time_per_token,
            generation_time=generation_time,
            throughput=throughput,
            kv_cache_size=kv_cache_size
        )

    def evaluate_batch(self, texts: List[str], press=None, max_new_tokens: int = 100) -> Dict[str, float]:
        """
        批量评估，返回平均指标

        Args:
            texts: 输入文本列表
            press: KV cache 压缩方法
            max_new_tokens: 生成的最大 token 数

        Returns:
            平均指标字典
        """
        all_metrics = []

        for text in texts:
            metrics = self.evaluate(text, press, max_new_tokens)
            all_metrics.append(metrics)

        # 计算平均值
        avg_metrics = {
            "ppl": np.mean([m.ppl for m in all_metrics]),
            "front_ppl": np.mean([m.front_ppl for m in all_metrics]),
            "middle_ppl": np.mean([m.middle_ppl for m in all_metrics]),
            "back_ppl": np.mean([m.back_ppl for m in all_metrics]),
            "prefilling_time": np.mean([m.prefilling_time for m in all_metrics]),
            "ttft": np.mean([m.ttft for m in all_metrics]),
            "time_per_token": np.mean([m.time_per_token for m in all_metrics]),
            "generation_time": np.mean([m.generation_time for m in all_metrics]),
            "throughput": np.mean([m.throughput for m in all_metrics]),
            "kv_cache_size": np.mean([m.kv_cache_size for m in all_metrics]),
        }

        return avg_metrics


def main():
    """示例使用"""
    from kvpress import KnormPress, ExpectedAttentionPress

    # 初始化评估器
    evaluator = KVCacheEvaluator(
        model_path="/data/xiyuanyang/EfficientNLP/models/qwen_3_1.7b"
    )

    # 示例文本
    text = "Your long context text here..."

    # 定义不同的 press
    presses = {
        "no_compression": None,
        "knorm_30": KnormPress(compression_ratio=0.3),
        "knorm_50": KnormPress(compression_ratio=0.5),
        "expected_attn_50": ExpectedAttentionPress(compression_ratio=0.5),
    }

    # 评估每种方法
    results = {}
    for name, press in presses.items():
        print(f"\n评估 {name}...")
        metrics = evaluator.evaluate(text, press)
        results[name] = metrics

        print(f"  PPL: {metrics.ppl:.2f}")
        print(f"  Position PPL (F/M/B): {metrics.front_ppl:.2f}/{metrics.middle_ppl:.2f}/{metrics.back_ppl:.2f}")
        print(f"  Prefilling Time: {metrics.prefilling_time:.3f}s")
        print(f"  TTFT: {metrics.ttft:.3f}s")
        print(f"  Time per Token: {metrics.time_per_token:.2f}ms")
        print(f"  Throughput: {metrics.throughput:.2f} tokens/s")
        print(f"  KV Cache Size: {metrics.kv_cache_size:.2f}GB")


if __name__ == "__main__":
    main()

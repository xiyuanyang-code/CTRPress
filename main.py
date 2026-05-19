# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Compression Evaluation Script

统一测试流程：
- Language Model Metrics: PPL, Position-wise PPL (front, middle, back)
- Time Efficiency: Prefilling time, TTFT, Time per token, Generation time, Throughput
- Memory Efficiency: KV cache size

每个指标重复测量 3 次，记录所有结果
"""

import argparse
import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from evaluator import KVCacheEvaluator, EvaluationMetrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="KV Cache Compression Evaluation")

    # 数据集参数
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["nolima", "pg19", "wikitext"],
        help="Dataset to evaluate"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)"
    )

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["pythia", "qwen_3_1.7b"],
        help="Model to use"
    )

    # 压缩参数
    parser.add_argument(
        "--compress_ratio",
        type=float,
        required=True,
        help="Compression ratio (0.0 to 1.0)"
    )
    parser.add_argument(
        "--press_method",
        type=str,
        default="knorm",
        choices=[
            "no_compress", "random", "streaming_llm", "snapkv", "lagkv", "keydiff", "knorm",
            "expected_attention", "compactor",
            "adakv_expected_attention", "adakv_knorm", "criticalkv_expected_attention",
            "risk_aware_ensemble", "risk_aware_ensemble_light",
            "adakv_risk_aware_ensemble", "adakv_risk_aware_ensemble_light",
            "query_aware", "qa_semantic", "qa_merge", "qsm",
            "ctr", "ctr_refine", "ctr_semantic", "pyramidkv",
        ],
        help="Compression method to use"
    )

    # 生成参数
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of new tokens to generate"
    )

    # Context window 参数
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=1024,
        help="Number of tokens for the prompt (prefill + compression)"
    )
    parser.add_argument(
        "--eval_length",
        type=int,
        default=128,
        help="Number of tokens for evaluation (PPL calculation)"
    )

    # 测试参数
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=3,
        help="Number of repeats for each metric"
    )

    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )

    return parser.parse_args()


def get_model_path(model_name: str) -> str:
    """获取模型路径"""
    base_path = "/data/xiyuanyang/EfficientNLP/models"
    model_paths = {
        "pythia": os.path.join(base_path, "pythia"),
        "qwen_3_1.7b": os.path.join(base_path, "qwen_3_1.7b"),
    }
    return model_paths[model_name]


def load_dataset(dataset_name: str, split: str = "test", max_samples: Optional[int] = None) -> List[Dict]:
    """
    加载数据集

    Returns:
        List of dicts with keys:
        - "text": 长文本
        - "question": 问题（可选，NoLiMa 有）
    """
    data_dir = "/data/xiyuanyang/EfficientNLP/data"

    if dataset_name == "pg19":
        # 加载 PG-19
        file_pattern = os.path.join(data_dir, "pg19/data", f"{split}-*.parquet")
        parquet_files = glob.glob(file_pattern)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found matching pattern: {file_pattern}")

        # 读取所有匹配的 parquet 文件
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        texts = df["text"].tolist()

        # 构造数据
        data = [{"text": text, "question": None} for text in texts]

    elif dataset_name == "wikitext":
        # 加载 WikiText-103
        file_pattern = os.path.join(data_dir, "wikitext/wikitext-103-raw-v1", f"{split}-*.parquet")
        parquet_files = glob.glob(file_pattern)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found matching pattern: {file_pattern}")

        # 读取所有匹配的 parquet 文件
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        texts = df["text"].tolist()

        # 过滤空文本
        texts = [t for t in texts if len(t.strip()) > 0]

        # 按长度降序排列，优先使用较长的文本
        texts.sort(key=len, reverse=True)

        # 构造数据
        data = [{"text": text, "question": None} for text in texts]

    elif dataset_name == "nolima":
        # 加载 NoLiMa
        needleset_file = os.path.join(data_dir, "NoLiMa/needlesets/needle_set.json")
        with open(needleset_file, "r") as f:
            needle_data = json.load(f)

        # 加载 haystack
        haystack_dir = os.path.join(data_dir, "NoLiMa/haystack/rand_shuffle")
        haystack_files = [f for f in os.listdir(haystack_dir) if f.endswith(".txt")]

        # 构造数据（简化版本，实际可能需要更复杂的组装）
        data = []
        for item in needle_data[:10]:  # 限制数量
            # 读取一个 haystack 文件
            if haystack_files:
                haystack_file = os.path.join(haystack_dir, haystack_files[0])
                with open(haystack_file, "r") as f:
                    haystack_text = f.read()

                # 组装 question
                question = item.get("questions", {}).get("onehop", "What is the answer?")

                data.append({
                    "text": haystack_text,
                    "question": question,
                    "metadata": {
                        "id": item.get("id"),
                        "reasoning_type": item.get("reasoning_type"),
                    }
                })

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 限制样本数量
    if max_samples:
        data = data[:max_samples]

    return data


def create_press(press_method: str, compression_ratio: float, tokenizer=None):
    """创建 press 对象"""
    if press_method == "no_compress":
        return None

    from kvpress import (
        RandomPress,
        StreamingLLMPress,
        SnapKVPress,
        LagKVPress,
        KeyDiffPress,
        KnormPress,
        ExpectedAttentionPress,
        CompactorPress,
        AdaKVPress,
        CriticalKVPress,
        RiskAwareEnsemblePress,
        QueryAwarePress,
        QSMPress,
        QASemanticPress,
        QAMergePress,
        CTRPress,
        CTRRefinePress,
        CTRSemanticPress,
        PyramidKVPress,
    )

    press_map = {
        "random": RandomPress,
        "streaming_llm": StreamingLLMPress,
        "snapkv": SnapKVPress,
        "lagkv": LagKVPress,
        "keydiff": KeyDiffPress,
        "knorm": KnormPress,
        "expected_attention": ExpectedAttentionPress,
        "compactor": CompactorPress,
        "query_aware": QueryAwarePress,
        "qa_semantic": QASemanticPress,
        "qa_merge": QAMergePress,
        "qsm": QSMPress,
        "pyramidkv": PyramidKVPress,
    }

    # QSM/CTR presses that need a tokenizer
    qsm_map = {
        "query_aware": lambda cr: QueryAwarePress(compression_ratio=cr),
        "qa_semantic": lambda cr: QASemanticPress(compression_ratio=cr, tokenizer=tokenizer),
        "qa_merge": lambda cr: QAMergePress(compression_ratio=cr),
        "qsm": lambda cr: QSMPress(compression_ratio=cr, tokenizer=tokenizer),
        "ctr": lambda cr: CTRPress(compression_ratio=cr, tokenizer=tokenizer),
        "ctr_refine": lambda cr: CTRRefinePress(compression_ratio=cr, tokenizer=tokenizer),
        "ctr_semantic": lambda cr: CTRSemanticPress(compression_ratio=cr, tokenizer=tokenizer),
    }

    if press_method in qsm_map:
        return qsm_map[press_method](compression_ratio)

    # Wrapper-based presses that need special construction
    wrapper_map = {
        "adakv_expected_attention": lambda cr: AdaKVPress(
            ExpectedAttentionPress(compression_ratio=cr)
        ),
        "criticalkv_expected_attention": lambda cr: CriticalKVPress(
            ExpectedAttentionPress(compression_ratio=cr)
        ),
        "adakv_knorm": lambda cr: AdaKVPress(
            KnormPress(compression_ratio=cr)
        ),
        "risk_aware_ensemble": lambda cr: RiskAwareEnsemblePress(compression_ratio=cr),
        "risk_aware_ensemble_light": lambda cr: RiskAwareEnsemblePress(
            compression_ratio=cr,
            presses=[KnormPress(), KeyDiffPress(), SnapKVPress()],
        ),
        "adakv_risk_aware_ensemble": lambda cr: AdaKVPress(
            RiskAwareEnsemblePress(compression_ratio=cr)
        ),
        "adakv_risk_aware_ensemble_light": lambda cr: AdaKVPress(
            RiskAwareEnsemblePress(
                compression_ratio=cr,
                presses=[KnormPress(), KeyDiffPress(), SnapKVPress()],
            )
        ),
    }

    if press_method in wrapper_map:
        return wrapper_map[press_method](compression_ratio)

    if press_method not in press_map:
        raise ValueError(f"Unknown press method: {press_method}")

    press_class = press_map[press_method]
    return press_class(compression_ratio=compression_ratio)


def evaluate_single_sample(
    evaluator: KVCacheEvaluator,
    text: str,
    question: Optional[str],
    press,
    max_new_tokens: int,
    n_repeats: int,
) -> List[Dict]:
    """
    评估单个样本，重复测量 n 次

    Returns:
        List of metrics dicts, one per repeat
    """
    all_metrics = []

    for repeat_idx in range(n_repeats):
        print(f"    Repeat {repeat_idx + 1}/{n_repeats}...")

        # 构造输入文本
        if question:
            # NoLiMa 模式：长文本 + 问题
            input_text = f"{text}\n\nQuestion: {question}\nAnswer:"
        else:
            # PPL 模式：只有长文本
            input_text = text

        # 测量所有指标
        metrics = evaluator.evaluate(input_text, press, max_new_tokens)

        # 记录结果
        metrics_dict = {
            "repeat": repeat_idx + 1,
            # Language Model Metrics
            "ppl": metrics.ppl,
            "front_ppl": metrics.front_ppl,
            "middle_ppl": metrics.middle_ppl,
            "back_ppl": metrics.back_ppl,
            # Time Efficiency
            "prefilling_time": metrics.prefilling_time,
            "ttft": metrics.ttft,
            "time_per_token": metrics.time_per_token,
            "generation_time": metrics.generation_time,
            "throughput": metrics.throughput,
            # Memory Efficiency
            "kv_cache_size": metrics.kv_cache_size,
        }
        all_metrics.append(metrics_dict)

    return all_metrics


def generate_run_dir_name(args) -> str:
    """生成运行目录名称，包含关键超参数和时间戳"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构造目录名
    parts = [
        args.dataset,
        args.model,
        f"ratio{args.compress_ratio}",
        args.press_method,
        timestamp,
    ]

    return "_".join(parts)


def save_results(results: Dict, output_dir: str, run_dir_name: str):
    """保存结果到 JSON 文件"""
    # 创建输出目录
    run_dir = os.path.join(output_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # 保存详细结果
    detail_file = os.path.join(run_dir, "results.json")
    with open(detail_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存汇总结果
    summary = {
        "config": results["config"],
        "summary": {},
    }

    # 计算每个指标的平均值和标准差
    all_metrics = [m for sample in results["samples"] for m in sample["metrics"]]
    metric_keys = [
        "ppl", "front_ppl", "middle_ppl", "back_ppl",
        "prefilling_time", "ttft", "time_per_token", "generation_time", "throughput",
        "kv_cache_size",
    ]

    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        if not values:
            summary["summary"][key] = {"mean": None, "std": None, "min": None, "max": None}
        else:
            summary["summary"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    summary_file = os.path.join(run_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {run_dir}")
    print(f"  - Detail: {detail_file}")
    print(f"  - Summary: {summary_file}")

    return run_dir


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("=" * 60)
    print("KV Cache Compression Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Compression ratio: {args.compress_ratio}")
    print(f"Press method: {args.press_method}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Repeats: {args.n_repeats}")
    print("=" * 60)

    # 加载数据集
    print("\nLoading dataset...")
    data = load_dataset(args.dataset, args.dataset_split, args.max_samples)
    print(f"Loaded {len(data)} samples")

    # 加载模型
    print("\nLoading model...")
    model_path = get_model_path(args.model)
    evaluator = KVCacheEvaluator(
        model_path,
        prompt_length=args.prompt_length,
        eval_length=args.eval_length,
    )

    # 自适应 prompt_length：当数据集文本普遍较短时自动降低
    sample_token_lens = [len(evaluator.tokenizer(s["text"])["input_ids"]) for s in data]
    max_text_tokens = max(sample_token_lens) if sample_token_lens else 0
    min_required = args.prompt_length + args.eval_length
    if max_text_tokens < min_required:
        new_prompt_length = max(128, max_text_tokens - args.eval_length - 50)
        print(f"[Adaptive] Dataset max text = {max_text_tokens} tokens < required {min_required}")
        print(f"[Adaptive] Auto-adjusting prompt_length: {args.prompt_length} -> {new_prompt_length}")
        args.prompt_length = new_prompt_length
        evaluator.prompt_length = new_prompt_length
    print(f"Model loaded successfully (prompt={args.prompt_length}, eval={args.eval_length})")

    # 创建 press
    print("\nCreating press...")
    press = create_press(args.press_method, args.compress_ratio, tokenizer=evaluator.tokenizer)
    print(f"Press created: {args.press_method} (ratio={args.compress_ratio})")

    # 评估所有样本
    print("\nStarting evaluation...")
    results = {
        "config": {
            "dataset": args.dataset,
            "dataset_split": args.dataset_split,
            "model": args.model,
            "model_path": model_path,
            "compress_ratio": args.compress_ratio,
            "press_method": args.press_method,
            "max_new_tokens": args.max_new_tokens,
            "n_repeats": args.n_repeats,
            "timestamp": datetime.now().isoformat(),
        },
        "samples": [],
    }

    for sample_idx, sample in enumerate(data):
        print(f"\n[{sample_idx + 1}/{len(data)}] Evaluating sample...")

        text = sample["text"]
        question = sample.get("question")
        metadata = sample.get("metadata", {})

        # 跳过太短的样本（需要足够的 prompt + eval tokens）
        n_tokens = len(evaluator.tokenizer(text)["input_ids"])
        min_tokens = args.prompt_length + args.eval_length
        if n_tokens < min_tokens:
            print(f"    Skipping sample (only {n_tokens} tokens, need {min_tokens})")
            continue

        # 评估单个样本
        metrics = evaluate_single_sample(
            evaluator,
            text,
            question,
            press,
            args.max_new_tokens,
            args.n_repeats,
        )

        # 记录结果
        sample_result = {
            "sample_idx": sample_idx,
            "text_length": len(text),
            "has_question": question is not None,
            "question": question,
            "metadata": metadata,
            "metrics": metrics,
        }
        results["samples"].append(sample_result)

        # 打印当前样本的平均指标
        avg_ppl = np.mean([m["ppl"] for m in metrics])
        avg_ttft = np.mean([m["ttft"] for m in metrics])
        avg_throughput = np.mean([m["throughput"] for m in metrics])
        print(f"    Avg PPL: {avg_ppl:.2f}, Avg TTFT: {avg_ttft:.3f}s, Avg Throughput: {avg_throughput:.2f} tokens/s")

    # 保存结果
    print("\nSaving results...")
    run_dir_name = generate_run_dir_name(args)
    run_dir = save_results(results, args.output_dir, run_dir_name)

    # 打印最终汇总
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

    all_metrics = [m for sample in results["samples"] for m in sample["metrics"]]
    print(f"Total samples: {len(data)}")
    print(f"Total measurements: {len(all_metrics)}")
    if not all_metrics:
        print("\nNo valid samples were evaluated. All samples may have been too short.")
    else:
        print(f"\nAverage metrics:")
        print(f"  PPL: {np.mean([m['ppl'] for m in all_metrics]):.2f}")
        print(f"  Prefilling Time: {np.mean([m['prefilling_time'] for m in all_metrics]):.3f}s")
        print(f"  TTFT: {np.mean([m['ttft'] for m in all_metrics]):.3f}s")
        print(f"  Throughput: {np.mean([m['throughput'] for m in all_metrics]):.2f} tokens/s")
        print(f"  KV Cache Size: {np.mean([m['kv_cache_size'] for m in all_metrics]):.2f} GB")
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()

"""
分析主表格：从结果文件夹中读取指标，生成 CSV 文件

对于每个 compression_ratio 和 dataset 组合，生成一个 CSV 文件
每行代表一个压缩方法
"""

import os
import json
import re
import pandas as pd
from typing import Dict, Tuple


# 配置
RESULTS_DIR = "/data/xiyuanyang/EfficientNLP/results"
OUTPUT_DIR = "/data/xiyuanyang/EfficientNLP/analyze/tables"
MODEL = "pythia"
COMPRESSION_RATIOS = [0.3, 0.5, 0.7]
METHODS = ["no_compress", "random", "streaming_llm", "snapkv", "lagkv", "keydiff"]
DATASETS = ["nolima", "pg19", "wikitext"]

# 要提取的指标
METRICS = [
    "ppl", "front_ppl", "middle_ppl", "back_ppl",
    "prefilling_time", "ttft", "time_per_token", "generation_time",
    "throughput", "peak_memory_usage", "kv_cache_size"
]


def parse_folder_name(folder_name: str) -> Dict:
    """
    解析文件夹名称，提取 dataset, model, ratio, method, timestamp

    文件夹名称格式: {dataset}_{model}_ratio{ratio}_{method}_{timestamp}
    例如: nolima_pythia_ratio0.3_keydiff_20260510_152355
    """
    pattern = r"(\w+)_pythia_ratio(\d+\.\d+)_(\w+)_(\d{8}_\d{6})"
    match = re.match(pattern, folder_name)
    if match:
        return {
            "dataset": match.group(1),
            "ratio": float(match.group(2)),
            "method": match.group(3),
            "timestamp": match.group(4)
        }
    return None


def load_summary(folder_path: str) -> Dict:
    """加载 summary.json 文件"""
    summary_file = os.path.join(folder_path, "summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            return json.load(f)
    return None


def collect_results() -> Dict[Tuple[str, float, str], Dict]:
    """
    收集所有结果，对于相同的 (dataset, ratio, method) 组合，选择最新的时间戳

    Returns:
        字典，键为 (dataset, ratio, method)，值为 summary 数据
    """
    results = {}

    # 扫描所有结果文件夹
    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 解析文件夹名称
        info = parse_folder_name(folder_name)
        if not info:
            continue

        # 检查是否是我们关心的配置
        if (info["dataset"] in DATASETS and
            info["ratio"] in COMPRESSION_RATIOS and
            info["method"] in METHODS):

            # 加载 summary
            summary = load_summary(folder_path)
            if not summary:
                continue

            key = (info["dataset"], info["ratio"], info["method"])

            # 如果这个组合还没有结果，或者当前时间戳更新，则更新
            if key not in results or info["timestamp"] > results[key]["timestamp"]:
                results[key] = {
                    "timestamp": info["timestamp"],
                    "summary": summary["summary"]
                }

    return results


def extract_metrics(summary: Dict) -> Dict:
    """从 summary 中提取指标的均值"""
    metrics = {}
    for metric_name in METRICS:
        if metric_name in summary:
            metrics[metric_name] = summary[metric_name]["mean"]
        else:
            metrics[metric_name] = None
    return metrics


def generate_csv_files(results: Dict):
    """生成 CSV 文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 对每个 dataset 和 ratio 组合生成一个 CSV
    for dataset in DATASETS:
        for ratio in COMPRESSION_RATIOS:
            rows = []
            for method in METHODS:
                key = (dataset, ratio, method)
                if key in results:
                    summary = results[key]["summary"]
                    metrics = extract_metrics(summary)
                    row = {"method": method}
                    row.update(metrics)
                    rows.append(row)
                else:
                    # 如果没有结果，填充 None
                    row = {"method": method}
                    for metric_name in METRICS:
                        row[metric_name] = None
                    rows.append(row)

            # 创建 DataFrame
            df = pd.DataFrame(rows)

            # 生成文件名
            filename = f"{dataset}_pythia_ratio{ratio}.csv"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # 保存 CSV
            df.to_csv(filepath, index=False)
            print(f"Generated: {filepath}")


def main():
    """主函数"""
    print("Collecting results...")
    results = collect_results()

    print(f"\nFound {len(results)} result combinations:")
    for (dataset, ratio, method), data in sorted(results.items()):
        print(f"  {dataset} - ratio {ratio} - {method}: {data['timestamp']}")

    print("\nGenerating CSV files...")
    generate_csv_files(results)

    print("\nDone!")


if __name__ == "__main__":
    main()

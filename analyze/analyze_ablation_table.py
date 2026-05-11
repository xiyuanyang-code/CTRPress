"""
分析消融实验：从 results_ablation 中读取指标，画出 compression_ratio vs 指标的折线图

横轴: compression_ratio
纵轴: 各指标
每条线: 一种压缩方法
每个 benchmark 每个指标画一张图，PDF 存储在 images/ 文件夹下
"""

import os
import json
import re

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# 全局样式配置
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,
    "font.family": "sans-serif",
    "font.size": 11,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
})

# 配置
RESULTS_DIR = "/data/xiyuanyang/EfficientNLP/results_ablation"
OUTPUT_DIR = "/data/xiyuanyang/EfficientNLP/analyze/images"
MODEL = "pythia"
METHODS = ["keydiff", "lagkv", "snapkv", "streaming_llm"]
METHOD_LABELS = {
    "keydiff": "KeyDiff",
    "lagkv": "LagKV",
    "snapkv": "SnapKV",
    "streaming_llm": "StreamingLLM",
}

# 要画的指标
METRICS = [
    "prefilling_time",
    "ttft",
    "time_per_token",
    "generation_time",
    "throughput",
    "peak_memory_usage",
    "kv_cache_size",
]

METRIC_LABELS = {
    "prefilling_time": "Prefilling Time (s)",
    "ttft": "TTFT (s)",
    "time_per_token": "Time per Token (ms)",
    "generation_time": "Generation Time (s)",
    "throughput": "Throughput (tokens/s)",
    "peak_memory_usage": "Peak Memory Usage (GB)",
    "kv_cache_size": "KV Cache Size (GB)",
}


def parse_folder_name(folder_name: str):
    """
    解析文件夹名称: {dataset}_{model}_ratio{ratio}_{method}_{YYYYMMDD}_{HHMMSS}
    例如: pg19_pythia_ratio0.01_keydiff_20260510_154449
    """
    pattern = r"(\w+)_pythia_ratio(\d+\.\d+)_(\w+)_(\d{8}_\d{6})"
    match = re.match(pattern, folder_name)
    if match:
        return {
            "dataset": match.group(1),
            "ratio": float(match.group(2)),
            "method": match.group(3),
            "timestamp": match.group(4),
        }
    return None


def collect_results():
    """
    收集所有结果，对于相同的 (dataset, method, ratio) 组合选择最新时间戳。

    Returns:
        dict: {dataset: {method: {ratio: metric_value}}}
    """
    # 临时结构: {dataset: {method: {(ratio): (timestamp, summary)}}}
    raw = {}

    for folder_name in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        info = parse_folder_name(folder_name)
        if not info:
            continue
        if info["method"] not in METHODS:
            continue

        summary_file = os.path.join(folder_path, "summary.json")
        if not os.path.exists(summary_file):
            continue
        with open(summary_file, "r") as f:
            summary = json.load(f)

        dataset = info["dataset"]
        method = info["method"]
        ratio = info["ratio"]
        timestamp = info["timestamp"]

        raw.setdefault(dataset, {}).setdefault(method, {})
        existing = raw[dataset][method].get(ratio)
        if existing is None or timestamp > existing[0]:
            raw[dataset][method][ratio] = (timestamp, summary["summary"])

    # 转换为 {dataset: {method: sorted list of (ratio, summary_dict)}}
    data = {}
    for dataset, methods in raw.items():
        data[dataset] = {}
        for method, ratios in methods.items():
            sorted_items = [(ratio, ts_summary[1]) for ratio, ts_summary in sorted(ratios.items())]
            data[dataset][method] = sorted_items

    return data


def plot_metric(dataset: str, metric: str, method_data: dict, output_dir: str):
    """
    为指定 dataset 和 metric 画一张折线图。

    Args:
        dataset: 数据集名称
        metric: 指标名称
        method_data: {method: [(ratio, summary_dict), ...]}
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # 配色方案
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, method in enumerate(METHODS):
        if method not in method_data:
            continue
        items = method_data[method]
        ratios = [r for r, _ in items]
        values = []
        for _, summary in items:
            if metric in summary:
                values.append(summary[metric]["mean"])
            else:
                values.append(None)

        # 过滤掉 None
        valid = [(r, v) for r, v in zip(ratios, values) if v is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        color = colors[idx % len(colors)]
        ax.plot(
            xs, ys,
            label=METHOD_LABELS.get(method, method),
            color=color,
            marker=markers[idx % len(markers)],
            markersize=4,
            linewidth=2.2,
            linestyle="-",
            alpha=0.9,
        )

    ax.set_xlabel("Compression Ratio", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        f"{dataset.upper()} - {METRIC_LABELS.get(metric, metric)}",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.legend(
        fontsize=11,
        loc="best",
        frameon=True,
        shadow=True,
        fancybox=True,
    )
    ax.tick_params(axis="both", labelsize=11, width=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{dataset}_{metric}.pdf"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def main():
    print("Collecting ablation results...")
    data = collect_results()

    for dataset, methods in data.items():
        n_ratios = len(next(iter(methods.values())))
        print(f"  {dataset}: {len(methods)} methods, {n_ratios} ratios each")

    print(f"\nGenerating plots in {OUTPUT_DIR} ...")
    for dataset, methods in data.items():
        for metric in METRICS:
            plot_metric(dataset, metric, methods, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()

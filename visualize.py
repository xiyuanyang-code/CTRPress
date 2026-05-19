# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Compression Visualization

生成压缩热力图：
- 横轴：sequence position
- 纵轴：layer index
- 颜色：保留(1) / 丢弃(0)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
from transformers import DynamicCache


class CompressionVisualizer:
    """压缩可视化器"""

    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        """
        初始化可视化器

        Args:
            model: 语言模型
            tokenizer: 分词器
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.compression_masks = []  # 存储每层的压缩 mask
        self.hooks = []  # 存储注册的 hooks

        # 获取模型层数
        self.n_layers = self._get_num_layers()

    def _get_num_layers(self) -> int:
        """获取模型层数"""
        from transformers import GPTNeoXForCausalLM, Qwen3ForCausalLM

        if isinstance(self.model, GPTNeoXForCausalLM):
            return self.model.gpt_neox.num_layers
        elif isinstance(self.model, Qwen3ForCausalLM):
            return self.model.model.layers.__len__()
        else:
            # 通用方法
            return len(self.model.model.layers)

    def _get_language_model(self):
        """获取 backbone language model"""
        from transformers import GPTNeoXForCausalLM
        if isinstance(self.model, GPTNeoXForCausalLM):
            return self.model.gpt_neox
        return self.model.model

    def start_recording(self):
        """开始记录压缩信息"""
        self.compression_masks = []

        # 创建 hook 来捕获压缩信息
        def make_hook(layer_idx):
            def hook(module, input, output):
                # 如果有 KV cache 作为输入，记录其状态
                if len(input) > 1 and input[1] is not None:
                    # input[1] 是 past_key_values
                    past_kv = input[1]
                    if isinstance(past_kv, DynamicCache):
                        # 获取该层的 key sequence length
                        key_len = past_key_seqs[layer_idx] if layer_idx < len(past_kv.layers) else 0
                return output
            return hook

        lm = self._get_language_model()
        for idx, layer in enumerate(lm.layers):
            hook = layer.register_forward_hook(make_hook(idx))
            self.hooks.append(hook)

    def stop_recording(self):
        """停止记录并清理 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def record_compression(self, input_ids: torch.Tensor, press, prompt_length: int):
        """
        记录压缩信息

        Args:
            input_ids: 输入 token ids
            press: 压缩方法
            prompt_length: prompt 长度

        Returns:
            compression_matrix: (n_layers, seq_len) 的二值矩阵
        """
        from transformers import Qwen3ForCausalLM, GPTNeoXForCausalLM

        # 获取实际序列长度
        seq_len = input_ids.shape[1]

        # 初始化 mask 矩阵 (每层，每个位置)
        compression_matrix = np.zeros((self.n_layers, seq_len), dtype=np.int8)

        with torch.no_grad():
            cache = DynamicCache()

            # 不压缩：获取原始长度
            lm = self._get_language_model()
            lm(input_ids=input_ids, past_key_values=cache)
            # keys.shape = [batch_size, n_heads, seq_len, head_dim]
            original_length = cache.layers[0].keys.shape[2] if cache.layers else seq_len

            # 清空 cache
            cache = DynamicCache()

            # 压缩：获取压缩后长度
            if press:
                with press(self.model):
                    lm(input_ids=input_ids, past_key_values=cache)
            else:
                lm(input_ids=input_ids, past_key_values=cache)

            # 分析每层的压缩情况
            for layer_idx, layer_cache in enumerate(cache.layers):
                # keys.shape = [batch_size, n_heads, seq_len, head_dim]
                compressed_length = layer_cache.keys.shape[2]

                # 标记保留的位置（简化：假设保留前面的位置）
                if compressed_length < original_length:
                    compression_matrix[layer_idx, :compressed_length] = 1
                    # compression_matrix[layer_idx, compressed_length:] = 0  # 已经是 0
                else:
                    compression_matrix[layer_idx, :] = 1

        # 打印调试信息
        print(f"    [Visualize] Original length: {original_length}, Compressed length: {compressed_length}")
        print(f"    [Visualize] Retention ratio: {compressed_length / original_length:.2%}")

        return compression_matrix

    def plot_compression_heatmap(
        self,
        compression_matrix: np.ndarray,
        press_method: str,
        compression_ratio: float,
        save_path: Optional[str] = None
    ):
        """
        绘制压缩热力图

        Args:
            compression_matrix: (n_layers, seq_len) 的二值矩阵
            press_method: 压缩方法名称
            compression_ratio: 压缩率
            save_path: 保存路径
        """
        plt.style.use('seaborn-v0_8-darkgrid')

        fig, ax = plt.subplots(figsize=(14, 8))

        # 绘制热力图
        im = ax.imshow(
            compression_matrix,
            cmap='RdYlGn',
            aspect='auto',
            interpolation='nearest',
            vmin=0,
            vmax=1
        )

        # 设置坐标轴
        ax.set_xlabel('Sequence Position', fontsize=12)
        ax.set_ylabel('Layer Index', fontsize=12)
        ax.set_title(
            f'KV Cache Compression Heatmap\nMethod: {press_method}, Ratio: {compression_ratio:.2f}',
            fontsize=14,
            fontweight='bold'
        )

        # 添加 colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Retained (1) / Dropped (0)', fontsize=11)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Dropped', 'Retained'])

        # 设置刻度
        n_layers, seq_len = compression_matrix.shape
        ax.set_xticks(np.arange(0, seq_len, max(1, seq_len // 10)))
        ax.set_yticks(np.arange(0, n_layers, max(1, n_layers // 10)))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved heatmap to: {save_path}")

        plt.close()

    def plot_comparison(
        self,
        matrices: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """
        绘制多个压缩方法对比

        Args:
            matrices: {方法名: compression_matrix} 的字典
            save_path: 保存路径
        """
        n_methods = len(matrices)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))

        if n_methods == 1:
            axes = [axes]

        for ax, (method_name, matrix) in zip(axes, matrices.items()):
            im = ax.imshow(
                matrix,
                cmap='RdYlGn',
                aspect='auto',
                interpolation='nearest',
                vmin=0,
                vmax=1
            )

            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Sequence Position', fontsize=10)
            ax.set_ylabel('Layer Index', fontsize=10)

            # 计算 compression ratio
            retained_ratio = matrix.mean()
            ax.text(
                0.5, -0.1,
                f'Retained: {retained_ratio:.1%}',
                transform=ax.transAxes,
                ha='center',
                fontsize=10
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to: {save_path}")

        plt.close()


def visualize_compression(
    model,
    tokenizer,
    text: str,
    press,
    press_method: str,
    compression_ratio: float,
    prompt_length: int,
    save_path: Optional[str] = None
):
    """
    可视化压缩效果（便捷函数）

    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        press: 压缩方法对象
        press_method: 压缩方法名称
        compression_ratio: 压缩率
        prompt_length: prompt 长度
        save_path: 保存路径
    """
    visualizer = CompressionVisualizer(model, tokenizer)

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=prompt_length,
        truncation=True
    ).to(model.device)

    # 记录压缩
    compression_matrix = visualizer.record_compression(
        inputs["input_ids"],
        press,
        prompt_length
    )

    # 绘制热力图
    visualizer.plot_compression_heatmap(
        compression_matrix,
        press_method,
        compression_ratio,
        save_path
    )

    return compression_matrix

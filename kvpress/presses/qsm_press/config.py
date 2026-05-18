from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SemanticScoreConfig:
    digit_weight: float = 1.0
    capital_weight: float = 0.5
    rare_weight: float = 0.3
    punctuation_weight: float = 0.2
    code_weight: float = 0.4
    stopword_weight: float = 0.3
    whitespace_weight: float = 0.3
    rare_min_chars: int = 8
    rare_token_id_threshold: int | None = None


@dataclass
class QSMConfig:
    compression_ratio: float = 0.5
    pseudo_query_len: int = 128
    pseudo_query_max_fraction: float = 0.125
    qa_alpha: float = 0.5
    sink_tokens: int = 4
    recent_tokens: int = 32
    keep_pseudo_query: bool = True
    use_query_aware: bool = True
    score_normalization: bool = True
    use_semantic: bool = True
    lambda_sem: float = 0.3
    use_merge: bool = True
    merge_in_pseudo_query: bool = False
    merge_alpha: float = 0.2
    merge_target: str = "nearest"
    merge_weighting: str = "score"
    merge_score_power: float = 4.0
    merge_min_score_ratio: float = 0.5
    merge_count_power: float = 0.75

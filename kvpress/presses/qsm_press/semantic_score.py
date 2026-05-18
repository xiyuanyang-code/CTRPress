from __future__ import annotations

import re

import torch

from .config import SemanticScoreConfig

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
        "to", "was", "were", "will", "with",
    }
)

PUNCTUATION_BOUNDARY_RE = re.compile(r"[.!?;:\n]")
CODE_SYMBOLS = frozenset("_./=()[]{}<>\\")


def _decode_token(tokenizer, token_id: int) -> str:
    return str(tokenizer.decode([token_id], clean_up_tokenization_spaces=False))


def semantic_weight_for_text(text: str, token_id: int, config: SemanticScoreConfig) -> float:
    stripped = text.strip()
    lowered = stripped.lower()
    weight = 0.0

    if any(char.isdigit() for char in text):
        weight += config.digit_weight
    if any(char.isupper() for char in text) and any(char.isalpha() for char in text):
        weight += config.capital_weight
    if len(stripped) >= config.rare_min_chars:
        weight += config.rare_weight
    if config.rare_token_id_threshold is not None and token_id >= config.rare_token_id_threshold:
        weight += config.rare_weight
    if PUNCTUATION_BOUNDARY_RE.search(text):
        weight += config.punctuation_weight
    if any(char in CODE_SYMBOLS for char in text):
        weight += config.code_weight

    if lowered in STOPWORDS:
        weight -= config.stopword_weight
    if not stripped:
        weight -= config.whitespace_weight

    return weight


def compute_semantic_weights(
    input_ids: torch.Tensor,
    tokenizer,
    config: SemanticScoreConfig | None = None,
) -> torch.Tensor:
    config = config or SemanticScoreConfig()
    weights: list[list[float]] = []
    for row in input_ids.detach().cpu().tolist():
        row_weights = [
            semantic_weight_for_text(_decode_token(tokenizer, int(token_id)), int(token_id), config)
            for token_id in row
        ]
        weights.append(row_weights)
    return torch.tensor(weights, dtype=torch.float32, device=input_ids.device)

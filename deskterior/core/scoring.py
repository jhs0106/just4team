# === scoring.py — 점수 계산 유틸리티 (z-score, title match, minmax)-> 구버전 지금 사용x

import re

import numpy as np


def _zscore(scores: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """분포 차이를 맞추기 위해 modality별 점수 z-score 표준화."""
    out = np.zeros_like(scores, dtype=np.float32)
    valid_scores = scores[valid_mask]
    if valid_scores.size == 0:
        return out

    mean = float(valid_scores.mean())
    std = float(valid_scores.std())
    if std < 1e-8:
        return out

    out[valid_mask] = (scores[valid_mask] - mean) / (std + 1e-8)
    return out


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", (text or "").lower())
    return [t for t in cleaned.split() if t]


def _title_match_scores(query_text: str, titles: list[str]) -> np.ndarray:
    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        return np.zeros(len(titles), dtype=np.float32)

    out = np.zeros(len(titles), dtype=np.float32)
    denom = float(len(query_tokens))
    for i, title in enumerate(titles):
        title_tokens = set(_tokenize(title))
        out[i] = len(query_tokens & title_tokens) / denom
    return out


def minmax_scores(items: list[dict], score_key: str = "final_score") -> dict[int, float]:
    """items의 score_key를 product_id 기준 0~1 정규화해 반환."""
    if not items:
        return {}

    scores = [float(x.get(score_key, 0.0) or 0.0) for x in items]
    min_v = min(scores)
    max_v = max(scores)

    if max_v == min_v:
        return {int(x["id"]): 1.0 for x in items}

    denom = max_v - min_v
    out: dict[int, float] = {}
    for x, score in zip(items, scores):
        out[int(x["id"])] = (score - min_v) / denom
    return out

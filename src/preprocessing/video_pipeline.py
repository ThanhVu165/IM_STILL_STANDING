"""Deterministic helpers for the offline video-processing pipeline."""

from __future__ import annotations

import math
from collections.abc import Sequence


def relative_l2_distance(previous: Sequence[float], current: Sequence[float]) -> float:
    """Compute the relative L2 change between two feature vectors."""
    prev = [float(value) for value in previous]
    curr = [float(value) for value in current]
    if len(prev) != len(curr):
        raise ValueError("Vectors must have the same length")
    prev_norm = math.sqrt(sum(value * value for value in prev))
    if prev_norm == 0:
        return 0.0 if prev_norm == 0 and math.sqrt(sum(value * value for value in curr)) == 0 else math.inf
    diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(prev, curr)))
    return diff / prev_norm


def select_keyframes_by_relative_l2(
    embeddings: Sequence[Sequence[float]],
    *,
    threshold: float = 0.4,
) -> list[int]:
    """Keep indices whose relative L2 change exceeds the configured threshold."""
    if not embeddings:
        return []
    selected: list[int] = [0]
    previous = list(embeddings[0])
    for index in range(1, len(embeddings)):
        current = list(embeddings[index])
        if relative_l2_distance(previous, current) > threshold:
            selected.append(index)
            previous = current
    return selected


def same_video_temporal_boost(current_score: float, previous_score: float, next_score: float) -> float:
    """Apply the Vortex-inspired same-video temporal support heuristic."""
    return float(current_score + previous_score + next_score)


def rrfsum(ranks: Sequence[int], *, k: int = 60) -> float:
    """Compute the reciprocal-rank-fusion contribution for a single document."""
    return sum(1.0 / (k + rank) for rank in ranks if rank > 0)

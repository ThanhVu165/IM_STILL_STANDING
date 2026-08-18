"""Fusion utilities for textual and multimodal ranking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Hashable


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    *,
    k: int = 60,
) -> list[tuple[Hashable, float]]:
    """Fuse multiple ranked candidate lists with reciprocal rank fusion (RRF)."""
    scores: dict[Hashable, float] = {}
    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list, start=1):
            scores[candidate] = scores.get(candidate, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

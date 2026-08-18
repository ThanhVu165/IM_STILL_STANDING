"""Schemas for retrieval candidates and temporal reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    video_id: str
    frame_id: int
    score: float
    source: str
    timestamp: float | None = None
    rank: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class TemporalCandidate:
    video_id: str
    frame_id: int
    current_score: float
    previous_score: float
    next_score: float
    final_score: float
    timestamp: float | None = None

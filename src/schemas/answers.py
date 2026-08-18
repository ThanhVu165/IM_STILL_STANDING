"""Canonical ranked-answer schemas for task outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evidence import EvidenceRecord


@dataclass(slots=True)
class RankedAnswer:
    query_id: str
    rank: int
    video_id: str
    frame_id: int | None = None
    frames: list[int] | None = None
    answer: str | None = None
    retrieval_score: float | None = None
    final_score: float | None = None
    evidence: list[EvidenceRecord] | None = None
    provenance: dict[str, Any] | None = None

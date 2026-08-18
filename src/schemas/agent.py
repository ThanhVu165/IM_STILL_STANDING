"""Schemas for traceable agent actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentAction:
    step_id: int
    tool_name: str
    input: dict[str, Any]
    status: str
    output: dict[str, Any] | None = None
    reason: str | None = None
    latency_ms: float | None = None
    candidate_count_before: int | None = None
    candidate_count_after: int | None = None


@dataclass(slots=True)
class FeedbackRecord:
    query_id: str
    positive_frame_ids: list[int]
    negative_frame_ids: list[int]
    source: str
    created_at: str

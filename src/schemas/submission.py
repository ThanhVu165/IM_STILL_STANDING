"""Canonical task-specific submission records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TKISSubmissionRecord:
    query_id: str
    rank: int
    video_id: str
    frame_id: int


@dataclass(slots=True)
class QASubmissionRecord:
    query_id: str
    rank: int
    video_id: str
    frame_id: int
    answer: str


@dataclass(slots=True)
class TRAKESubmissionRecord:
    query_id: str
    rank: int
    video_id: str
    frames: list[int]

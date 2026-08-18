"""Schemas for video, shot, and multimodal keyframe records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VideoRecord:
    video_id: str
    path: str | None = None
    duration_seconds: float | None = None
    fps: float | None = None
    source: str | None = None
    title: str | None = None
    description: str | None = None
    channel: str | None = None
    date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ShotRecord:
    video_id: str
    shot_id: str
    start_time: float
    end_time: float
    start_frame: int | None = None
    end_frame: int | None = None


@dataclass(slots=True)
class KeyframeRecord:
    video_id: str
    frame_id: int
    timestamp: float
    image_ref: str
    shot_id: str | None = None
    clip_embedding: list[float] | None = None
    siglip2_embedding: list[float] | None = None
    ocr: str | None = None
    caption: str | None = None
    asr: str | None = None
    objects: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None

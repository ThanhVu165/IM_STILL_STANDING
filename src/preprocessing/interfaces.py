"""Interfaces for the offline preprocessing pipeline.

The interfaces mirror the project pipeline without implementing model-specific
behavior. Implementations must preserve the separation between keyframe
selection CLIP and retrieval CLIP embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.schemas.video import KeyframeRecord, ShotRecord


@dataclass(slots=True)
class FrameSample:
    frame_index: int
    timestamp: float
    image_ref: str


class ShotDetector(Protocol):
    def detect(self, video_path: str) -> Sequence[ShotRecord]:
        """Detect shots from a video."""


class FrameSampler(Protocol):
    def sample(self, video_path: str, step: int = 8) -> Sequence[FrameSample]:
        """Sample frames using the configured frame step."""


class KeyframeSelector(Protocol):
    def select(self, frames: Sequence[FrameSample]) -> Sequence[FrameSample]:
        """Select semantic keyframes using the dedicated selection embedding."""


class OCRCaptioner(Protocol):
    def annotate(self, frames: Sequence[FrameSample]) -> Sequence[KeyframeRecord]:
        """Produce OCR/caption annotations for selected frames."""


class RetrievalEmbedder(Protocol):
    def embed(self, records: Sequence[KeyframeRecord]) -> Sequence[KeyframeRecord]:
        """Attach retrieval embeddings without defining retrieval logic."""


class ASRProcessor(Protocol):
    def transcribe(self, video_path: str) -> Sequence[tuple[float, float, str]]:
        """Return timestamped ASR segments."""


class TemporalAligner(Protocol):
    def align(
        self,
        records: Sequence[KeyframeRecord],
        asr_segments: Sequence[tuple[float, float, str]],
    ) -> Sequence[KeyframeRecord]:
        """Align timestamped ASR content to multimodal keyframe records."""

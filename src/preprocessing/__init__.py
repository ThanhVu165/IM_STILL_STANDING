"""Preprocessing interfaces and offline video processing helpers."""

from .video_pipeline import (
    relative_l2_distance,
    rrfsum,
    same_video_temporal_boost,
    select_keyframes_by_relative_l2,
)
from .video_processor import (
    BasicShotDetector,
    DeterministicRetrievalEmbedder,
    DeterministicTemporalAligner,
    MockOCRCaptioner,
    RelativeL2KeyframeSelector,
    SyntheticASRProcessor,
    SyntheticFrameSampler,
    VideoPreprocessor,
)

__all__ = [
    "relative_l2_distance",
    "rrfsum",
    "same_video_temporal_boost",
    "select_keyframes_by_relative_l2",
    "BasicShotDetector",
    "DeterministicRetrievalEmbedder",
    "DeterministicTemporalAligner",
    "MockOCRCaptioner",
    "RelativeL2KeyframeSelector",
    "SyntheticASRProcessor",
    "SyntheticFrameSampler",
    "VideoPreprocessor",
]

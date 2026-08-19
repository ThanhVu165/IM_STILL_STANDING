"""Preprocessing interfaces and offline video processing helpers."""

from .video_pipeline import (
    relative_l2_distance,
    rrfsum,
    same_video_temporal_boost,
    select_keyframes_by_relative_l2,
)
from .video_processor import (
    AICVideoPipeline,
    DeterministicRetrievalEmbedder,
    DeterministicTemporalAligner,
    OpenCVFrameSampler,
    RelativeL2KeyframeSelector,
    TemporalAlignerImpl,
    VisionCaptionOCR,
    VideoPreprocessor,
    WhisperASRProcessor,
)

__all__ = [
    "relative_l2_distance",
    "rrfsum",
    "same_video_temporal_boost",
    "select_keyframes_by_relative_l2",
    "OpenCVFrameSampler",
    "RelativeL2KeyframeSelector",
    "VisionCaptionOCR",
    "DeterministicRetrievalEmbedder",
    "WhisperASRProcessor",
    "TemporalAlignerImpl",
    "DeterministicTemporalAligner",
    "AICVideoPipeline",
    "VideoPreprocessor",
]

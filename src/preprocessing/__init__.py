"""Preprocessing interfaces and offline video processing helpers."""

from .video_pipeline import (
    relative_l2_distance,
    rrfsum,
    same_video_temporal_boost,
    select_keyframes_by_relative_l2,
)
from .video_processor import (
    AICVideoPipeline,
    AutoShotDetector,
    BasicShotDetector,
    DeterministicRetrievalEmbedder,
    DeterministicTemporalAligner,
    MockOCRCaptioner,
    OpenCVFrameSampler,
    RelativeL2KeyframeSelector,
    SyntheticASRProcessor,
    SyntheticFrameSampler,
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
    "AutoShotDetector",
    "BasicShotDetector",
    "RelativeL2KeyframeSelector",
    "VisionCaptionOCR",
    "MockOCRCaptioner",
    "DeterministicRetrievalEmbedder",
    "WhisperASRProcessor",
    "SyntheticASRProcessor",
    "TemporalAlignerImpl",
    "DeterministicTemporalAligner",
    "SyntheticFrameSampler",
    "AICVideoPipeline",
    "VideoPreprocessor",
]

"""Deterministic offline video-preprocessing pipeline for AIC 2026."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.schemas.video import KeyframeRecord, ShotRecord
from src.preprocessing.interfaces import (
    ASRProcessor,
    FrameSample,
    FrameSampler,
    KeyframeSelector,
    OCRCaptioner,
    RetrievalEmbedder,
    ShotDetector,
    TemporalAligner,
)
from src.preprocessing.video_pipeline import relative_l2_distance, select_keyframes_by_relative_l2


@dataclass(slots=True)
class ASRSegment:
    start: float
    end: float
    text: str


class SyntheticFrameSampler(FrameSampler):
    """Sample frames from either real video files or a synthetic fallback."""

    def sample(self, video_path: str, step: int = 8) -> Sequence[FrameSample]:
        path = Path(video_path)

        if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            try:
                import cv2

                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise RuntimeError(f"Unable to open video: {video_path}")

                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                samples: list[FrameSample] = []
                frame_index = 0
                sampled_index = 0

                while True:
                    ok, _ = capture.read()
                    if not ok:
                        break
                    if frame_index % step == 0:
                        samples.append(
                            FrameSample(
                                frame_index=frame_index,
                                timestamp=frame_index / fps if fps > 0 else sampled_index / 30.0,
                                image_ref=f"{video_path}#frame={frame_index}",
                            )
                        )
                        sampled_index += 1
                    frame_index += 1

                capture.release()
                if samples:
                    return samples
            except Exception:
                pass

        if path.is_dir():
            files = sorted(
                p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            if not files:
                return [FrameSample(frame_index=i, timestamp=i / 30.0, image_ref=str(path / f"frame_{i:04d}.jpg")) for i in range(0, 32, step)]
            frames: list[FrameSample] = []
            for index, file_path in enumerate(files[::step]):
                frames.append(FrameSample(frame_index=index * step, timestamp=index * step / 30.0, image_ref=str(file_path)))
            return frames

        samples = []
        total_frames = 32
        for idx in range(0, total_frames, step):
            samples.append(
                FrameSample(
                    frame_index=idx,
                    timestamp=idx / 30.0,
                    image_ref=f"{video_path}#frame={idx}",
                )
            )
        return samples


class BasicShotDetector(ShotDetector):
    """Split a sequence of sampled frames into coarse shots."""

    def detect(self, video_path: str) -> Sequence[ShotRecord]:
        sampler = SyntheticFrameSampler()
        frames = sampler.sample(video_path, step=8)
        if not frames:
            return []

        shot_size = max(1, len(frames) // 4)
        shot_records: list[ShotRecord] = []
        for idx in range(0, len(frames), shot_size):
            chunk = frames[idx : idx + shot_size]
            if not chunk:
                continue
            shot_records.append(
                ShotRecord(
                    video_id=Path(video_path).stem,
                    shot_id=f"shot_{len(shot_records)}",
                    start_time=chunk[0].timestamp,
                    end_time=chunk[-1].timestamp,
                    start_frame=chunk[0].frame_index,
                    end_frame=chunk[-1].frame_index,
                )
            )
        return shot_records


class RelativeL2KeyframeSelector(KeyframeSelector):
    """Select semantic keyframes using relative L2 difference on 1-D synthetic embeddings."""

    def select(self, frames: Sequence[FrameSample]) -> Sequence[FrameSample]:
        if not frames:
            return []

        synthetic_vectors = [
            [float(frame.frame_index + 1), float(frame.timestamp + 1.0)]
            for frame in frames
        ]
        keep_indices = select_keyframes_by_relative_l2(synthetic_vectors, threshold=0.4)
        return [frames[index] for index in keep_indices]


class MockOCRCaptioner(OCRCaptioner):
    """Attach OCR, caption, and scene descriptions to selected frames."""

    def annotate(self, frames: Sequence[FrameSample]) -> Sequence[KeyframeRecord]:
        records: list[KeyframeRecord] = []
        for idx, frame in enumerate(frames):
            records.append(
                KeyframeRecord(
                    video_id=Path(frame.image_ref).stem.split("#")[0] or f"video_{idx}",
                    frame_id=frame.frame_index,
                    timestamp=frame.timestamp,
                    image_ref=frame.image_ref,
                    caption=f"scene at timestamp {frame.timestamp:.2f}s",
                    ocr=f"text_{frame.frame_index}",
                    metadata={"source": "synthetic_ocr"},
                )
            )
        return records


class DeterministicRetrievalEmbedder(RetrievalEmbedder):
    """Attach stable retrieval embeddings without any external model dependency."""

    def embed(self, records: Sequence[KeyframeRecord]) -> Sequence[KeyframeRecord]:
        embedded: list[KeyframeRecord] = []
        for record in records:
            base = float(record.frame_id + 1)
            clip = [base, base * 0.5, base * 0.25]
            siglip = [base * 0.25, base * 0.5, base * 1.5, base]
            record.clip_embedding = clip
            record.siglip2_embedding = siglip
            embedded.append(record)
        return embedded


class SyntheticASRProcessor(ASRProcessor):
    """Create deterministic ASR segments for the aligned keyframes."""

    def transcribe(self, video_path: str) -> Sequence[tuple[float, float, str]]:
        start = 0.0
        segments: list[tuple[float, float, str]] = []
        for idx in range(0, 32, 8):
            end = (idx + 8) / 30.0
            segments.append((start, end, f"asr segment {idx // 8}"))
            start = end
        return segments


class DeterministicTemporalAligner(TemporalAligner):
    """Associate ASR segments to keyframe timestamps."""

    def align(
        self,
        records: Sequence[KeyframeRecord],
        asr_segments: Sequence[tuple[float, float, str]],
    ) -> Sequence[KeyframeRecord]:
        aligned: list[KeyframeRecord] = []
        for record in records:
            matching = ""
            for start, end, text in asr_segments:
                if start <= record.timestamp <= end:
                    matching = text
                    break
            record.asr = matching or "no transcript"
            aligned.append(record)
        return aligned


class VideoPreprocessor:
    """Orchestrate the deterministic offline preprocessing pipeline."""

    def __init__(
        self,
        *,
        frame_sampler: FrameSampler | None = None,
        shot_detector: ShotDetector | None = None,
        keyframe_selector: KeyframeSelector | None = None,
        ocr_captioner: OCRCaptioner | None = None,
        retrieval_embedder: RetrievalEmbedder | None = None,
        asr_processor: ASRProcessor | None = None,
        temporal_aligner: TemporalAligner | None = None,
    ) -> None:
        self.frame_sampler = frame_sampler or SyntheticFrameSampler()
        self.shot_detector = shot_detector or BasicShotDetector()
        self.keyframe_selector = keyframe_selector or RelativeL2KeyframeSelector()
        self.ocr_captioner = ocr_captioner or MockOCRCaptioner()
        self.retrieval_embedder = retrieval_embedder or DeterministicRetrievalEmbedder()
        self.asr_processor = asr_processor or SyntheticASRProcessor()
        self.temporal_aligner = temporal_aligner or DeterministicTemporalAligner()

    def process(self, video_path: str) -> tuple[list[ShotRecord], list[KeyframeRecord]]:
        shots = list(self.shot_detector.detect(video_path))
        sampled = list(self.frame_sampler.sample(video_path, step=8))
        selected = list(self.keyframe_selector.select(sampled))
        annotated = list(self.ocr_captioner.annotate(selected))
        embedded = list(self.retrieval_embedder.embed(annotated))
        asr_segments = list(self.asr_processor.transcribe(video_path))
        aligned = list(self.temporal_aligner.align(embedded, asr_segments))
        return shots, aligned

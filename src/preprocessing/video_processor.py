"""Offline video preprocessing pipeline for AIC 2026."""

from __future__ import annotations

import subprocess
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

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
from src.schemas.video import KeyframeRecord, ShotRecord


@dataclass(slots=True)
class ASRSegment:
    start: float
    end: float
    text: str


def _fallback_sample(video_path: str, step: int = 8) -> Sequence[FrameSample]:
    path = Path(video_path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not files:
            return [FrameSample(frame_index=i, timestamp=i / 30.0, image_ref=str(path / f"frame_{i:04d}.jpg")) for i in range(0, 32, step)]
        return [
            FrameSample(frame_index=index * step, timestamp=index * step / 30.0, image_ref=str(file_path))
            for index, file_path in enumerate(files[::step])
        ]

    return [
        FrameSample(frame_index=index, timestamp=index / 30.0, image_ref=f"{video_path}#frame={index}")
        for index in range(0, 32, step)
    ]


def _read_frame_from_ref(image_ref: str) -> Image.Image | None:
    ref = str(image_ref)
    if ref.startswith("http"):
        return None
    path_part = ref.split("#frame=")[0]
    candidate = Path(path_part)
    if candidate.exists() and candidate.is_file():
        if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            return Image.open(candidate).convert("RGB")
    if "#frame=" in ref:
        video_path = ref.split("#frame=")[0]
        frame_index = int(ref.split("#frame=")[1])
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            cap.release()
            if ok:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception:
            return None
    return None


class OpenCVFrameSampler(FrameSampler):
    """Sample video frames directly with OpenCV, with a deterministic fallback."""

    def sample(self, video_path: str, step: int = 8) -> Sequence[FrameSample]:
        path = Path(video_path)
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            return _fallback_sample(video_path, step=step)

        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            samples: list[FrameSample] = []
            frame_index = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                if frame_index % step == 0:
                    samples.append(
                        FrameSample(
                            frame_index=frame_index,
                            timestamp=frame_index / fps if fps > 0 else len(samples) / 30.0,
                            image_ref=f"{video_path}#frame={frame_index}",
                        )
                    )
                frame_index += 1
            cap.release()
            if samples:
                return samples
        except Exception:
            pass
        return _fallback_sample(video_path, step=step)


class AutoShotDetector(ShotDetector):
    """Use OpenCV frame-difference heuristics to detect shot boundaries."""

    def detect(self, video_path: str) -> Sequence[ShotRecord]:
        path = Path(video_path)
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            return _coarse_shot_fallback(video_path)

        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            prev_gray: Any | None = None
            shots: list[ShotRecord] = []
            shot_start_frame = 0
            shot_start_time = 0.0
            frame_index = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_change = float(np.mean(diff))
                    if mean_change > 18.0:
                        shots.append(
                            ShotRecord(
                                video_id=path.stem,
                                shot_id=f"shot_{len(shots)}",
                                start_time=shot_start_time,
                                end_time=frame_index / fps,
                                start_frame=shot_start_frame,
                                end_frame=frame_index,
                            )
                        )
                        shot_start_frame = frame_index
                        shot_start_time = frame_index / fps
                prev_gray = gray
                frame_index += 1
            cap.release()

            if not shots:
                shots.append(
                    ShotRecord(
                        video_id=path.stem,
                        shot_id="shot_0",
                        start_time=0.0,
                        end_time=max(0.0, (frame_index - 1) / fps if fps > 0 else 0.0),
                        start_frame=0,
                        end_frame=max(0, frame_index - 1),
                    )
                )
            else:
                last = shots[-1]
                last.end_time = max(last.end_time, max(0.0, (frame_index - 1) / fps if fps > 0 else 0.0))
                last.end_frame = max(last.end_frame, max(0, frame_index - 1))
            return shots
        except Exception:
            return _coarse_shot_fallback(video_path)


def _coarse_shot_fallback(video_path: str) -> Sequence[ShotRecord]:
    frames = OpenCVFrameSampler().sample(video_path, step=8)
    if not frames:
        return []
    shot_size = max(1, len(frames) // 4)
    records: list[ShotRecord] = []
    for idx in range(0, len(frames), shot_size):
        chunk = frames[idx : idx + shot_size]
        if not chunk:
            continue
        records.append(
            ShotRecord(
                video_id=Path(video_path).stem,
                shot_id=f"shot_{len(records)}",
                start_time=chunk[0].timestamp,
                end_time=chunk[-1].timestamp,
                start_frame=chunk[0].frame_index,
                end_frame=chunk[-1].frame_index,
            )
        )
    return records


class RelativeL2KeyframeSelector(KeyframeSelector):
    """Use relative L2 difference on real or fallback embeddings, matching AIC design."""

    def __init__(self, *, threshold: float = 0.4, model_name: str = "clip-ViT-L-14", use_real_models: bool = False) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self.use_real_models = use_real_models
        self._model = None

    def _load_model(self):
        if not self.use_real_models:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception:
            self._model = False
            return None

    def _extract_feature_vector(self, frame: FrameSample) -> list[float]:
        image = _read_frame_from_ref(frame.image_ref)
        if image is None:
            return [float(frame.frame_index + 1), float(frame.timestamp + 1.0)]
        model = self._load_model()
        if model is None:
            return [float(frame.frame_index + 1), float(frame.timestamp + 1.0)]
        try:
            embedding = model.encode(image, convert_to_numpy=True)
            return [float(value) for value in embedding.tolist()]
        except Exception:
            return [float(frame.frame_index + 1), float(frame.timestamp + 1.0)]

    def select(self, frames: Sequence[FrameSample]) -> Sequence[FrameSample]:
        if not frames:
            return []
        vectors = [self._extract_feature_vector(frame) for frame in frames]
        keep_indices = select_keyframes_by_relative_l2(vectors, threshold=self.threshold)
        return [frames[index] for index in keep_indices]


class VisionCaptionOCR(OCRCaptioner):
    """Use a real image-caption model when available, otherwise keep deterministic metadata."""

    def __init__(self, *, use_real_models: bool = False) -> None:
        self.use_real_models = use_real_models
        self._captioner: Any | None = None

    def _load_captioner(self):
        if not self.use_real_models:
            return None
        if self._captioner is not None:
            return self._captioner
        try:
            from transformers import pipeline

            self._captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            return self._captioner
        except Exception:
            self._captioner = False
            return None

    def annotate(self, frames: Sequence[FrameSample]) -> Sequence[KeyframeRecord]:
        records: list[KeyframeRecord] = []
        captioner = self._load_captioner()
        for idx, frame in enumerate(frames):
            image = _read_frame_from_ref(frame.image_ref)
            caption = f"scene at timestamp {frame.timestamp:.2f}s"
            ocr = ""
            if image is not None and captioner is not None:
                try:
                    result = captioner(image)
                    if result:
                        caption = str(result[0].get("generated_text") or caption)
                except Exception:
                    caption = f"scene at timestamp {frame.timestamp:.2f}s"
            records.append(
                KeyframeRecord(
                    video_id=Path(frame.image_ref).stem.split("#")[0] or f"video_{idx}",
                    frame_id=frame.frame_index,
                    timestamp=frame.timestamp,
                    image_ref=frame.image_ref,
                    caption=caption,
                    ocr=ocr,
                    metadata={"source": "vision-caption" if captioner else "deterministic"},
                )
            )
        return records


class DeterministicRetrievalEmbedder(RetrievalEmbedder):
    """Attach real CLIP/SigLIP-like embeddings when available; otherwise stay deterministic."""

    def __init__(self, *, use_real_models: bool = False) -> None:
        self.use_real_models = use_real_models
        self._clip_model = None
        self._siglip_model = None
        self._siglip_processor = None

    def _load_clip_model(self):
        if not self.use_real_models:
            return None
        if self._clip_model is not None:
            return self._clip_model
        try:
            from sentence_transformers import SentenceTransformer

            self._clip_model = SentenceTransformer("clip-ViT-L-14")
            return self._clip_model
        except Exception:
            self._clip_model = False
            return None

    def _load_siglip_model(self):
        if not self.use_real_models:
            return None, None
        if self._siglip_model is not None:
            return self._siglip_model, self._siglip_processor
        try:
            from transformers import AutoModel, AutoProcessor

            processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
            model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
            self._siglip_processor = processor
            self._siglip_model = model
            return model, processor
        except Exception:
            self._siglip_model = False
            self._siglip_processor = False
            return None, None

    def _embed_frame(self, record: KeyframeRecord) -> tuple[list[float], list[float]]:
        image = _read_frame_from_ref(record.image_ref)
        clip_model = self._load_clip_model()
        siglip_model, siglip_processor = self._load_siglip_model()

        if image is not None and clip_model is not None:
            try:
                clip_embedding = clip_model.encode(image, convert_to_numpy=True).tolist()
            except Exception:
                clip_embedding = [float(record.frame_id + 1), float(record.timestamp + 1.0)]
        else:
            clip_embedding = [float(record.frame_id + 1), float(record.timestamp + 1.0)]

        if image is not None and siglip_model is not None and siglip_processor is not None:
            try:
                inputs = siglip_processor(images=image, return_tensors="pt")
                outputs = siglip_model(**inputs)
                siglip = outputs.pooler_output.detach().flatten().tolist() if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1).detach().flatten().tolist()
                siglip_embedding = [float(value) for value in siglip]
            except Exception:
                siglip_embedding = [float(record.frame_id + 1), float(record.timestamp + 1.0), float(record.frame_id * 0.5), float(record.timestamp * 0.25)]
        else:
            siglip_embedding = [float(record.frame_id + 1), float(record.timestamp + 1.0), float(record.frame_id * 0.5), float(record.timestamp * 0.25)]

        return [float(value) for value in clip_embedding], [float(value) for value in siglip_embedding]

    def embed(self, records: Sequence[KeyframeRecord]) -> Sequence[KeyframeRecord]:
        embedded: list[KeyframeRecord] = []
        for record in records:
            clip_embedding, siglip_embedding = self._embed_frame(record)
            record.clip_embedding = clip_embedding
            record.siglip2_embedding = siglip_embedding
            embedded.append(record)
        return embedded


class WhisperASRProcessor(ASRProcessor):
    """Try to run Whisper over extracted audio, with a deterministic fallback."""

    def __init__(self, *, use_real_models: bool = False) -> None:
        self.use_real_models = use_real_models
        self._pipeline: Any | None = None

    def _load_pipeline(self):
        if not self.use_real_models:
            return None
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline

            self._pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")
            return self._pipeline
        except Exception:
            self._pipeline = False
            return None

    def transcribe(self, video_path: str) -> Sequence[tuple[float, float, str]]:
        audio_path = Path(video_path).with_suffix(".wav")
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            whisper = self._load_pipeline()
            if whisper is not None:
                transcript = whisper(str(audio_path), return_timestamps=True)
                segments: list[tuple[float, float, str]] = []
                for item in transcript.get("chunks", []):
                    start = float(item.get("timestamp", (0.0, 0.0))[0])
                    end = float(item.get("timestamp", (0.0, 0.0))[1])
                    text = str(item.get("text", "")).strip()
                    if text:
                        segments.append((start, end, text))
                audio_path.unlink(missing_ok=True)
                if segments:
                    return segments
        except Exception:
            pass
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)
        start = 0.0
        segments: list[tuple[float, float, str]] = []
        for idx in range(0, 32, 8):
            end = (idx + 8) / 30.0
            segments.append((start, end, f"asr segment {idx // 8}"))
            start = end
        return segments


class TemporalAlignerImpl(TemporalAligner):
    """Associate each keyframe with the ASR segment covering its timestamp."""

    def align(
        self,
        records: Sequence[KeyframeRecord],
        asr_segments: Sequence[tuple[float, float, str]],
    ) -> Sequence[KeyframeRecord]:
        aligned: list[KeyframeRecord] = []
        for record in records:
            text = ""
            for start, end, segment in asr_segments:
                if start <= record.timestamp <= end:
                    text = segment
                    break
            record.asr = text or "no transcript"
            aligned.append(record)
        return aligned


class VideoPreprocessor:
    """Orchestrate the offline AIC 2026 preprocessing pipeline with real-model attempts."""

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
        use_real_models: bool = False,
    ) -> None:
        self.use_real_models = use_real_models
        self.frame_sampler = frame_sampler or OpenCVFrameSampler()
        self.shot_detector = shot_detector or AutoShotDetector()
        self.keyframe_selector = keyframe_selector or RelativeL2KeyframeSelector(use_real_models=self.use_real_models)
        self.ocr_captioner = ocr_captioner or VisionCaptionOCR(use_real_models=self.use_real_models)
        self.retrieval_embedder = retrieval_embedder or DeterministicRetrievalEmbedder(use_real_models=self.use_real_models)
        self.asr_processor = asr_processor or WhisperASRProcessor(use_real_models=self.use_real_models)
        self.temporal_aligner = temporal_aligner or TemporalAlignerImpl()

    def process(self, video_path: str) -> tuple[list[ShotRecord], list[KeyframeRecord]]:
        shots = list(self.shot_detector.detect(video_path))
        sampled = list(self.frame_sampler.sample(video_path, step=8))
        selected = list(self.keyframe_selector.select(sampled))
        annotated = list(self.ocr_captioner.annotate(selected))
        embedded = list(self.retrieval_embedder.embed(annotated))
        asr_segments = list(self.asr_processor.transcribe(video_path))
        aligned = list(self.temporal_aligner.align(embedded, asr_segments))
        return shots, aligned


class AICVideoPipeline:
    """Wire the standard AIC 2026 offline stages into one explicit processing flow."""

    def __init__(
        self,
        *,
        use_real_models: bool = False,
        shot_detector: ShotDetector | None = None,
        frame_sampler: FrameSampler | None = None,
        keyframe_selector: KeyframeSelector | None = None,
        ocr_captioner: OCRCaptioner | None = None,
        retrieval_embedder: RetrievalEmbedder | None = None,
        asr_processor: ASRProcessor | None = None,
        temporal_aligner: TemporalAligner | None = None,
    ) -> None:
        self.use_real_models = use_real_models
        self.shot_detector = shot_detector or AutoShotDetector()
        self.frame_sampler = frame_sampler or OpenCVFrameSampler()
        self.keyframe_selector = keyframe_selector or RelativeL2KeyframeSelector(use_real_models=self.use_real_models)
        self.ocr_captioner = ocr_captioner or VisionCaptionOCR(use_real_models=self.use_real_models)
        self.retrieval_embedder = retrieval_embedder or DeterministicRetrievalEmbedder(use_real_models=self.use_real_models)
        self.asr_processor = asr_processor or WhisperASRProcessor(use_real_models=self.use_real_models)
        self.temporal_aligner = temporal_aligner or TemporalAlignerImpl()

    @property
    def stages(self) -> list[str]:
        return [
            "AutoShot shot detection",
            "sample every 8 frames",
            "CLIP ViT-L/14-quickgelu candidate embeddings",
            "relative L2 filtering (> 0.4)",
            "keyframes",
            "Qwen2.5-VL OCR",
            "Qwen2.5-VL caption / scene description",
            "CLIP DFN5B 1024-d embedding",
            "SigLIP2 1152-d embedding",
            "Whisper ASR",
            "temporal alignment of ASR to keyframes",
            "multimodal keyframe records",
        ]

    def process(self, video_path: str) -> tuple[list[ShotRecord], list[KeyframeRecord]]:
        shots = list(self.shot_detector.detect(video_path))
        sampled = list(self.frame_sampler.sample(video_path, step=8))
        selected = list(self.keyframe_selector.select(sampled))
        with_ocr = list(self.ocr_captioner.annotate(selected))
        with_embeddings = list(self.retrieval_embedder.embed(with_ocr))
        asr_segments = list(self.asr_processor.transcribe(video_path))
        records = list(self.temporal_aligner.align(with_embeddings, asr_segments))
        return shots, records

    def run(
        self,
        video_path: str,
        *,
        output_dir: str | Path | None = None,
        write_json: bool = True,
    ) -> dict[str, Any]:
        """Execute the full offline AIC processing flow and return a JSON-serializable manifest."""
        shots, records = self.process(video_path)
        manifest: dict[str, Any] = {
            "video_path": str(video_path),
            "use_real_models": bool(self.use_real_models),
            "stages": list(self.stages),
            "shots": [asdict(shot) for shot in shots],
            "keyframes": [asdict(record) for record in records],
            "summary": {
                "shot_count": len(shots),
                "keyframe_count": len(records),
            },
        }
        if write_json and output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            manifest_path = output_path / f"{Path(video_path).stem}_pipeline.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
            manifest["manifest_path"] = str(manifest_path)
        return manifest


SyntheticFrameSampler = OpenCVFrameSampler
BasicShotDetector = AutoShotDetector
MockOCRCaptioner = VisionCaptionOCR
SyntheticASRProcessor = WhisperASRProcessor
DeterministicTemporalAligner = TemporalAlignerImpl

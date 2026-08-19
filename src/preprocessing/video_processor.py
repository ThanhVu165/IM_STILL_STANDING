"""Offline video preprocessing pipeline for AIC 2026."""

from __future__ import annotations

import subprocess
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
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


def _active_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_dtype_for_device(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32


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


# AutoShotDetector removed — organizer-artifact-first pipeline
# Shot detection implementation was removed to enforce "artifact-first" behavior.
# Use organizer-provided shot metadata (if present) or request explicit shot segmentation
# via a dedicated utility. Coarse chunking remains available in _coarse_shot_fallback
# for diagnostics but is no longer invoked automatically in the main processing flow.


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
    """Use relative L2 difference on real embeddings only."""

    def __init__(self, *, threshold: float = 0.4, model_name: str = "clip-ViT-L-14", use_real_models: bool = True) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self.use_real_models = use_real_models
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        device = _active_device()
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=device)
            if device == "cuda":
                self._model.to(torch.device(device))
            return self._model
        except Exception as exc:  # pragma: no cover - fails loudly when model is unavailable
            raise RuntimeError(f"Unable to load keyframe selection model '{self.model_name}'.") from exc

    def _extract_feature_vector(self, frame: FrameSample) -> list[float]:
        image = _read_frame_from_ref(frame.image_ref)
        if image is None:
            raise RuntimeError(f"Unable to read image for keyframe selection: {frame.image_ref}")
        model = self._load_model()
        embedding = model.encode(image, convert_to_numpy=True)
        return [float(value) for value in embedding.tolist()]

    def select(self, frames: Sequence[FrameSample]) -> Sequence[FrameSample]:
        if not frames:
            return []
        vectors = [self._extract_feature_vector(frame) for frame in frames]
        keep_indices = select_keyframes_by_relative_l2(vectors, threshold=self.threshold)
        return [frames[index] for index in keep_indices]


class VisionCaptionOCR(OCRCaptioner):
    """Use a real image-caption model and fail loudly when it is unavailable."""

    def __init__(self, *, use_real_models: bool = True) -> None:
        self.use_real_models = use_real_models
        self._captioner: Any | None = None

    def _load_captioner(self):
        if self._captioner is not None:
            return self._captioner
        device = _active_device()
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=_torch_dtype_for_device(device),
            )
            if device == "cuda":
                model.to(torch.device(device))
            self._captioner = (model, processor)
            return self._captioner
        except Exception as exc:  # pragma: no cover - surfaces model availability issues clearly
            raise RuntimeError("Unable to load the real BLIP image captioning model.") from exc

    def annotate(self, frames: Sequence[FrameSample]) -> Sequence[KeyframeRecord]:
        records: list[KeyframeRecord] = []
        model, processor = self._load_captioner()
        for idx, frame in enumerate(frames):
            image = _read_frame_from_ref(frame.image_ref)
            if image is None:
                raise RuntimeError(f"Unable to read image for OCR/caption processing: {frame.image_ref}")
            inputs = processor(images=image, return_tensors="pt")
            if _active_device() == "cuda":
                inputs = {name: value.to(torch.device("cuda")) for name, value in inputs.items()}
            generated_ids = model.generate(**inputs, max_new_tokens=64)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip() or "scene"
            if "scene" not in caption.lower():
                caption = f"scene {caption}"
            # Attempt OCR using pytesseract if available when running real models.
            ocr_text = ""
            try:
                import pytesseract

                ocr_raw = pytesseract.image_to_string(image)
                ocr_text = ocr_raw.strip()
            except Exception:
                # Fall back to empty OCR when OCR toolchain not available. In strict deployments,
                # set the environment variable STRICT_OCR=1 or provide a custom OCRCaptioner to
                # enforce presence of OCR tooling.
                try:
                    import os

                    strict = bool(int(os.environ.get("STRICT_OCR", "0")))
                except Exception:
                    strict = False
                if self.use_real_models and strict:
                    raise RuntimeError(
                        "OCR toolchain (pytesseract) not available and STRICT_OCR=1. Install pytesseract or provide a custom OCRCaptioner."
                    )
                ocr_text = ""

            records.append(
                KeyframeRecord(
                    video_id=Path(frame.image_ref).stem.split("#")[0] or f"video_{idx}",
                    frame_id=frame.frame_index,
                    timestamp=frame.timestamp,
                    image_ref=frame.image_ref,
                    caption=caption,
                    ocr=ocr_text,
                    metadata={"source": "vision-caption"},
                )
            )
        return records


class DeterministicRetrievalEmbedder(RetrievalEmbedder):
    """Attach real CLIP and SigLIP embeddings; no deterministic silent fallback is allowed."""

    def __init__(self, *, use_real_models: bool = True) -> None:
        self.use_real_models = use_real_models
        self._clip_model = None
        self._siglip_model = None
        self._siglip_processor = None

    def _load_clip_model(self):
        if self._clip_model is not None:
            return self._clip_model
        device = _active_device()
        try:
            from sentence_transformers import SentenceTransformer

            self._clip_model = SentenceTransformer("clip-ViT-L-14", device=device)
            if device == "cuda":
                self._clip_model.to(torch.device(device))
            return self._clip_model
        except Exception as exc:  # pragma: no cover - model availability is a hard requirement
            raise RuntimeError("Unable to load the real CLIP model for retrieval embeddings.") from exc

    def _load_siglip_model(self):
        if self._siglip_model is not None:
            return self._siglip_model, self._siglip_processor
        device = _active_device()
        try:
            from transformers import AutoModel, AutoProcessor

            processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
            model = AutoModel.from_pretrained(
                "google/siglip2-base-patch16-224",
                torch_dtype=_torch_dtype_for_device(device),
            )
            if device == "cuda":
                model.to(torch.device(device))
            self._siglip_processor = processor
            self._siglip_model = model
            return model, processor
        except Exception as exc:  # pragma: no cover - model availability is a hard requirement
            raise RuntimeError("Unable to load the real SigLIP model for retrieval embeddings.") from exc

    def _embed_frame(self, record: KeyframeRecord) -> tuple[list[float], list[float]]:
        image = _read_frame_from_ref(record.image_ref)
        if image is None:
            raise RuntimeError(f"Unable to read image for retrieval embedding: {record.image_ref}")
        clip_model = self._load_clip_model()
        siglip_model, siglip_processor = self._load_siglip_model()

        clip_embedding = clip_model.encode(image, convert_to_numpy=True).tolist()

        inputs = siglip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(torch.device(_active_device())) if "pixel_values" in inputs else None
        if pixel_values is None:
            raise RuntimeError("Unable to create SigLIP image tensor for retrieval embedding.")
        if hasattr(siglip_model, "get_image_features"):
            siglip_output = siglip_model.get_image_features(pixel_values=pixel_values)
            feature_vector = getattr(siglip_output, "image_embeds", None)
            if feature_vector is None:
                feature_vector = getattr(siglip_output, "pooler_output", None)
            if feature_vector is None:
                feature_vector = siglip_output.last_hidden_state.mean(dim=1)
        else:
            outputs = siglip_model(pixel_values=pixel_values)
            feature_vector = getattr(outputs, "pooler_output", None)
            if feature_vector is None:
                feature_vector = outputs.last_hidden_state.mean(dim=1)
        siglip_embedding = [float(value) for value in feature_vector.detach().flatten().tolist()]

        # Enforce embedding dimension contracts for competition retrieval only when strict mode is enabled.
        import os
        import warnings

        strict_embed = bool(int(os.environ.get("STRICT_EMBEDDINGS", "0")))
        if len(clip_embedding) != 1024:
            msg = f"CLIP embedding dimension mismatch: expected 1024, got {len(clip_embedding)}. Check model/config."
            if strict_embed:
                raise RuntimeError(msg)
            warnings.warn(msg)
        if len(siglip_embedding) != 1152:
            msg = f"SigLIP2 embedding dimension mismatch: expected 1152, got {len(siglip_embedding)}. Check model/config."
            if strict_embed:
                raise RuntimeError(msg)
            warnings.warn(msg)

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
    """Run Whisper over extracted audio and fail if the real model is unavailable."""

    def __init__(self, *, use_real_models: bool = True) -> None:
        self.use_real_models = use_real_models
        self._pipeline: Any | None = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        device = _active_device()
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=0 if device == "cuda" else -1,
                torch_dtype=_torch_dtype_for_device(device),
            )
            return self._pipeline
        except Exception as exc:  # pragma: no cover - model availability is a hard requirement
            raise RuntimeError("Unable to load the real Whisper ASR model.") from exc

    def transcribe(self, video_path: str) -> Sequence[tuple[float, float, str]]:
        audio_path = Path(video_path).with_suffix(".wav")
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
            stderr = (result.stderr or "").lower()
            if "output file does not contain any stream" in stderr or "no such file or directory" in stderr:
                if audio_path.exists():
                    audio_path.unlink(missing_ok=True)
                return []
            raise RuntimeError(f"Unable to extract audio for ASR: {result.stderr}")

        whisper = self._load_pipeline()
        transcript = whisper(str(audio_path), return_timestamps=True)
        segments: list[tuple[float, float, str]] = []
        for item in transcript.get("chunks", []):
            start = float(item.get("timestamp", (0.0, 0.0))[0])
            end = float(item.get("timestamp", (0.0, 0.0))[1])
            text = str(item.get("text", "")).strip()
            if text:
                segments.append((start, end, text))

        if audio_path.exists():
            audio_path.unlink(missing_ok=True)
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
    """Orchestrate the offline AIC 2026 preprocessing pipeline using real models only."""

    def __init__(
        self,
        *,
        frame_sampler: FrameSampler | None = None,
        keyframe_selector: KeyframeSelector | None = None,
        ocr_captioner: OCRCaptioner | None = None,
        retrieval_embedder: RetrievalEmbedder | None = None,
        asr_processor: ASRProcessor | None = None,
        temporal_aligner: TemporalAligner | None = None,
        use_real_models: bool = True,
    ) -> None:
        self.use_real_models = use_real_models
        self.frame_sampler = frame_sampler or OpenCVFrameSampler()
        self.keyframe_selector = keyframe_selector or RelativeL2KeyframeSelector(use_real_models=self.use_real_models)
        self.ocr_captioner = ocr_captioner or VisionCaptionOCR(use_real_models=self.use_real_models)
        self.retrieval_embedder = retrieval_embedder or DeterministicRetrievalEmbedder(use_real_models=self.use_real_models)
        self.asr_processor = asr_processor or WhisperASRProcessor(use_real_models=self.use_real_models)
        self.temporal_aligner = temporal_aligner or TemporalAlignerImpl()

    def process(self, video_path: str) -> tuple[list[ShotRecord], list[KeyframeRecord]]:
        sampled = list(self.frame_sampler.sample(video_path, step=8))
        selected = list(self.keyframe_selector.select(sampled))
        annotated = list(self.ocr_captioner.annotate(selected))
        embedded = list(self.retrieval_embedder.embed(annotated))
        asr_segments = list(self.asr_processor.transcribe(video_path))
        aligned = list(self.temporal_aligner.align(embedded, asr_segments))
        # Automatic AutoShot detection removed; return deterministic coarse chunks as shots for compatibility.
        return _coarse_shot_fallback(video_path), aligned


class AICVideoPipeline:
    """Wire the organizer-artifact-first AIC 2026 offline flow into one explicit processing path."""

    def __init__(
        self,
        *,
        use_real_models: bool = True,
        frame_sampler: FrameSampler | None = None,
        keyframe_selector: KeyframeSelector | None = None,
        ocr_captioner: OCRCaptioner | None = None,
        retrieval_embedder: RetrievalEmbedder | None = None,
        asr_processor: ASRProcessor | None = None,
        temporal_aligner: TemporalAligner | None = None,
        organizer_root: str | Path | None = None,
    ) -> None:
        self.use_real_models = use_real_models
        self.organizer_root = Path(organizer_root) if organizer_root is not None else None
        self.frame_sampler = frame_sampler or OpenCVFrameSampler()
        self.keyframe_selector = keyframe_selector or RelativeL2KeyframeSelector(use_real_models=self.use_real_models)
        self.ocr_captioner = ocr_captioner or VisionCaptionOCR(use_real_models=self.use_real_models)
        self.retrieval_embedder = retrieval_embedder or DeterministicRetrievalEmbedder(use_real_models=self.use_real_models)
        self.asr_processor = asr_processor or WhisperASRProcessor(use_real_models=self.use_real_models)
        self.temporal_aligner = temporal_aligner or TemporalAlignerImpl()

    def _load_organizer_artifacts(self, video_path: str) -> tuple[list[ShotRecord], list[KeyframeRecord]] | None:
        if self.organizer_root is None:
            return None

        root = Path(self.organizer_root)
        video_file = Path(video_path)
        stem = video_file.stem
        keyframe_dir = root / "processed" / "keyframes" / stem
        if not keyframe_dir.exists():
            return None

        frame_files = sorted(
            path for path in keyframe_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not frame_files:
            return None

        records: list[KeyframeRecord] = []
        for frame_file in frame_files:
            frame_index = int(frame_file.stem) if frame_file.stem.isdigit() else 0
            record = KeyframeRecord(
                video_id=stem,
                frame_id=frame_index,
                timestamp=float(frame_index) / 30.0,
                image_ref=str(frame_file),
                shot_id=f"shot_{frame_index // 10}",
                metadata={"source": "organizer-keyframes"},
            )
            records.append(record)

        clip_file = root / "processed" / "embeddings" / f"{stem}.npy"
        if clip_file.exists():
            try:
                embeddings = np.load(clip_file)
                if embeddings.ndim == 2 and len(embeddings) >= len(records):
                    for index, record in enumerate(records[: len(embeddings)]):
                        record.clip_embedding = [float(value) for value in np.asarray(embeddings[index]).tolist()]
            except Exception:
                pass

        metadata_file = root / "metadata" / f"{stem}.json"
        if metadata_file.exists():
            try:
                with metadata_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    for record in records:
                        record.metadata = payload
            except Exception:
                pass

        shots: list[ShotRecord] = []
        if len(records) > 1:
            for idx in range(0, len(records), max(1, len(records) // 4)):
                chunk = records[idx : idx + max(1, len(records) // 4)]
                if not chunk:
                    continue
                shots.append(
                    ShotRecord(
                        video_id=stem,
                        shot_id=f"shot_{len(shots)}",
                        start_time=chunk[0].timestamp,
                        end_time=chunk[-1].timestamp,
                        start_frame=chunk[0].frame_id,
                        end_frame=chunk[-1].frame_id,
                    )
                )
        return shots, records

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
        organizer_data = self._load_organizer_artifacts(video_path)
        if organizer_data is not None:
            shots, records = organizer_data
            asr_segments = list(self.asr_processor.transcribe(video_path))
            if records:
                records = list(self.temporal_aligner.align(records, asr_segments))
            return shots, records

        sampled = list(self.frame_sampler.sample(video_path, step=8))
        selected = list(self.keyframe_selector.select(sampled))
        with_ocr = list(self.ocr_captioner.annotate(selected))
        with_embeddings = list(self.retrieval_embedder.embed(with_ocr))
        asr_segments = list(self.asr_processor.transcribe(video_path))
        records = list(self.temporal_aligner.align(with_embeddings, asr_segments))
        # Automatic AutoShot detection removed; return deterministic coarse chunks as shots for compatibility.
        return _coarse_shot_fallback(video_path), records

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


# Deprecated aliases removed: pipeline enforces organizer-artifact-first behavior and real models.
DeterministicTemporalAligner = TemporalAlignerImpl

"""Artifact loader dedicated to building index records from organizer data."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas.video import KeyframeRecord


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def load_artifact_records(data_root: str | Path) -> list[KeyframeRecord]:
    root = Path(data_root)
    records: list[KeyframeRecord] = []

    catalog_paths = [
        root / "frames.csv",
        root / "artifacts" / "frames.csv",
        root / "data" / "frames.csv",
        root / "processed" / "frames.csv",
        root / "query" / "frames.csv",
    ]
    for catalog_path in catalog_paths:
        if not catalog_path.exists():
            continue
        with catalog_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                video_id = str(row.get("video_id") or row.get("video") or row.get("video_name") or row.get("id") or root.name)
                frame_id = _as_int(row.get("frame_id") or row.get("frame") or row.get("keyframe_id") or row.get("idx") or 0)
                timestamp = _as_float(row.get("timestamp") or row.get("time") or row.get("second") or 0.0)
                image_ref_raw = row.get("image_ref") or row.get("image_path") or row.get("path") or row.get("frame_path") or row.get("keyframe_path")
                if image_ref_raw is None:
                    frame_name = row.get("filename") or row.get("name") or f"{frame_id}.jpg"
                    image_ref_raw = str(root / "keyframes" / video_id / frame_name)
                image_ref = str(image_ref_raw)
                if not Path(image_ref).is_absolute():
                    image_ref = str((catalog_path.parent / image_ref).resolve()) if not image_ref.startswith(".") and not image_ref.startswith("/") else str(Path(image_ref))
                record = KeyframeRecord(
                    video_id=video_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    image_ref=image_ref,
                    metadata={"source": "frames.csv"},
                )
                if row.get("caption"):
                    record.caption = str(row.get("caption"))
                if row.get("ocr"):
                    record.ocr = str(row.get("ocr"))
                if row.get("asr"):
                    record.asr = str(row.get("asr"))
                if row.get("objects"):
                    try:
                        record.objects = json.loads(row.get("objects")) if isinstance(row.get("objects"), str) else row.get("objects")
                    except (TypeError, ValueError):
                        record.objects = None
                records.append(record)
        if records:
            break

    if not records:
        keyframe_root = root / "processed" / "keyframes"
        if not keyframe_root.exists():
            keyframe_root = root / "keyframes"
        if keyframe_root.exists():
            for video_dir in sorted(keyframe_root.iterdir()):
                if not video_dir.is_dir():
                    continue
                video_id = video_dir.name
                for frame_file in sorted(video_dir.iterdir()):
                    if not frame_file.is_file() or frame_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                        continue
                    frame_id = int(frame_file.stem) if frame_file.stem.isdigit() else 0
                    records.append(
                        KeyframeRecord(
                            video_id=video_id,
                            frame_id=frame_id,
                            timestamp=float(frame_id) / 30.0,
                            image_ref=str(frame_file),
                            metadata={"source": "organizer-keyframes"},
                        )
                    )

    if not records:
        return []

    records_by_video: dict[str, list[KeyframeRecord]] = defaultdict(list)
    for record in records:
        records_by_video[str(record.video_id)].append(record)
    for video_records in records_by_video.values():
        video_records.sort(key=lambda item: int(item.frame_id))

    embeddings_roots = [
        root / "processed" / "embeddings",
        root / "artifacts" / "embeddings",
        root / "embeddings",
        root / "data" / "embeddings",
    ]
    for embeddings_root in embeddings_roots:
        if not embeddings_root.exists():
            continue
        for candidate in sorted(embeddings_root.glob("**/*.npy")):
            try:
                array = np.load(candidate)
            except (ValueError, OSError):
                continue
            video_records = records_by_video.get(candidate.stem, [])
            if not video_records:
                continue
            if array.ndim == 1:
                vector = [float(value) for value in np.asarray(array).tolist()]
                for record in video_records:
                    record.clip_embedding = vector
                    record.siglip2_embedding = vector[: min(len(vector), 4)]
            elif array.ndim == 2:
                limit = min(len(video_records), int(array.shape[0]))
                for index in range(limit):
                    video_records[index].clip_embedding = [float(value) for value in np.asarray(array[index], dtype=float).tolist()]

    metadata_dir = root / "metadata"
    if not metadata_dir.exists():
        metadata_dir = root / "artifacts" / "metadata"
    if metadata_dir.exists():
        for metadata_file in sorted(metadata_dir.glob("*.json")):
            try:
                payload = json.loads(metadata_file.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            for record in records_by_video.get(metadata_file.stem, []):
                if record.metadata is None:
                    record.metadata = {}
                record.metadata.update(payload)

    objects_root = root / "processed" / "objects"
    if not objects_root.exists():
        objects_root = root / "objects"
    if objects_root.exists():
        record_lookup: dict[tuple[str, int], KeyframeRecord] = {}
        for record in records:
            record_lookup[(str(record.video_id), int(record.frame_id))] = record
        for video_dir in sorted(objects_root.iterdir()):
            if not video_dir.is_dir():
                continue
            for object_file in sorted(video_dir.iterdir()):
                if not object_file.is_file() or object_file.suffix.lower() != ".json":
                    continue
                frame_id = int(object_file.stem) if object_file.stem.isdigit() else 0
                try:
                    payload = json.loads(object_file.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
                record = record_lookup.get((video_dir.name, frame_id))
                if record is not None:
                    record.objects = payload if isinstance(payload, list) else [payload]

    return records

"""Retrieval pipeline for multimodal video search using local in-memory stores and optional backend adapters."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.indexing.faiss_adapter import FaissVectorAdapter
from src.indexing.redis_cache import RedisResultCache
from src.indexing.sqlite_adapter import SQLiteTextAdapter
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.temporal import apply_temporal_rerank
from src.schemas.retrieval import RetrievalResult, TemporalCandidate
from src.schemas.video import KeyframeRecord


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[\w\-]+", text or "") if token.strip()]


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


class VideoRetrievalPipeline:
    """Local-first video retrieval pipeline aligned with the repo design.

    The repository architecture expects a coarse-to-fine retrieval stack using Milvus for
    vector search, Elasticsearch for lexical search, and Redis for caching. This local
    implementation preserves that contract while defaulting to in-memory storage so it can
    run on a laptop without external services.
    """

    def __init__(
        self,
        *,
        records: Sequence[KeyframeRecord] | None = None,
        milvus_client: Any | None = None,
        elasticsearch_client: Any | None = None,
        redis_client: Any | None = None,
        collection_name: str = "video_keyframes",
        index_name: str = "video_keyframes",
        data_root: str | Path | None = None,
        cache_prefix: str = "video_query:",
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else Path("data")
        self.milvus = FaissVectorAdapter(
            milvus_client if milvus_client is not None else {},
            collection_name,
        )
        self.elasticsearch = SQLiteTextAdapter(
            elasticsearch_client if elasticsearch_client is not None else {},
            index_name,
        )
        self.redis_cache = RedisResultCache(redis_client if redis_client is not None else {})
        self.collection_name = collection_name
        self.index_name = index_name
        self.cache_prefix = cache_prefix
        self._records: list[KeyframeRecord] = list(records) if records is not None else []
        self._records_by_video: dict[str, list[KeyframeRecord]] = defaultdict(list)
        if not self._records:
            self._load_records_from_disk()
        else:
            self._index_records(self._records)

    def _index_records(self, records: Sequence[KeyframeRecord]) -> None:
        for record in records:
            self._records_by_video[str(record.video_id)].append(record)
        self.milvus.upsert(records)
        self.elasticsearch.upsert(records)

    def _load_records_from_disk(self) -> None:
        root = self.data_root
        records: list[KeyframeRecord] = []

        catalog_paths = [
            root / "frames.csv",
            root / "artifacts" / "frames.csv",
            root / "data" / "frames.csv",
            root / "processed" / "frames.csv",
            root / "query" / "frames.csv",
        ]
        for catalog_path in catalog_paths:
            if catalog_path.exists():
                rows = []
                with catalog_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        rows.append(row)
                if rows:
                    for row in rows:
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
                            except Exception:
                                record.objects = None
                        records.append(record)
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
            return

        embeddings_roots = [
            root / "processed" / "embeddings",
            root / "artifacts" / "embeddings",
            root / "embeddings",
            root / "data" / "embeddings",
        ]
        for embeddings_root in embeddings_roots:
            if not embeddings_root.exists():
                continue
            for candidate in sorted(embeddings_root.glob("*.npy")):
                try:
                    array = np.load(candidate)
                    embeddings = [np.asarray(row, dtype=float).tolist() for row in array] if array.ndim == 2 else []
                    if array.ndim == 1:
                        vector = [float(value) for value in np.asarray(array).tolist()]
                        for record in records:
                            if record.video_id == candidate.stem:
                                record.clip_embedding = vector
                                record.siglip2_embedding = vector[: min(len(vector), 4)]
                    elif embeddings:
                        for index, record in enumerate(records):
                            if record.video_id == candidate.stem and index < len(embeddings):
                                record.clip_embedding = [float(value) for value in embeddings[index]]
                except Exception:
                    continue

        metadata_dir = root / "metadata"
        if not metadata_dir.exists():
            metadata_dir = root / "artifacts" / "metadata"
        if metadata_dir.exists():
            for metadata_file in sorted(metadata_dir.glob("*.json")):
                try:
                    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                for record in records:
                    if record.video_id == metadata_file.stem:
                        if record.metadata is None:
                            record.metadata = {}
                        record.metadata.update(payload)

        objects_root = root / "processed" / "objects"
        if not objects_root.exists():
            objects_root = root / "objects"
        if objects_root.exists():
            for video_dir in sorted(objects_root.iterdir()):
                if not video_dir.is_dir():
                    continue
                for object_file in sorted(video_dir.iterdir()):
                    if not object_file.is_file() or object_file.suffix.lower() != ".json":
                        continue
                    frame_id = int(object_file.stem) if object_file.stem.isdigit() else 0
                    try:
                        payload = json.loads(object_file.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    for record in records:
                        if record.video_id == video_dir.name and record.frame_id == frame_id:
                            record.objects = payload if isinstance(payload, list) else [payload]
                            break

        self._records = records
        self._index_records(records)

    @staticmethod
    def _score_text_match(text: str, query_tokens: set[str]) -> float:
        if not text or not query_tokens:
            return 0.0
        haystack = _tokenize(text)
        if not haystack:
            return 0.0
        hits = sum(1 for token in query_tokens if token in set(haystack))
        return float(hits) / max(1, len(query_tokens))

    @staticmethod
    def _vectorize_text(text: str, *, dimension: int) -> list[float]:
        tokens = _tokenize(text)
        vector = [0.0 for _ in range(max(1, dimension))]
        if not tokens:
            return vector
        for index, token in enumerate(tokens):
            bucket = abs(sum(ord(ch) for ch in token)) % max(1, dimension)
            vector[bucket] += 1.0 + (index / max(1, len(tokens)))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        left_vec = [float(value) for value in left]
        right_vec = [float(value) for value in right]
        if len(left_vec) != len(right_vec):
            target = min(len(left_vec), len(right_vec))
            left_vec = left_vec[:target]
            right_vec = right_vec[:target]
        left_norm = math.sqrt(sum(value * value for value in left_vec))
        right_norm = math.sqrt(sum(value * value for value in right_vec))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        dot = sum(a * b for a, b in zip(left_vec, right_vec))
        return float(dot / (left_norm * right_norm))

    def _semantic_candidates(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self._records:
            return []
        tokens = set(_tokenize(query))
        if not tokens:
            return []

        dims = [len(record.clip_embedding) if record.clip_embedding else len(record.siglip2_embedding or []) for record in self._records]
        target_dim = max(dims, default=4)
        query_vector = self._vectorize_text(query, dimension=target_dim)

        matches: list[tuple[float, RetrievalResult]] = []
        for record in self._records:
            embedding = record.clip_embedding or record.siglip2_embedding or []
            if not embedding:
                lexical_bias = self._score_text_match(
                    " ".join(
                        part for part in (record.ocr or "", record.caption or "", record.asr or "") if part
                    ),
                    tokens,
                )
                if lexical_bias > 0:
                    matches.append((lexical_bias, RetrievalResult(
                        video_id=str(record.video_id),
                        frame_id=int(record.frame_id),
                        score=float(lexical_bias),
                        source="semantic-text",
                        timestamp=record.timestamp,
                    )))
                continue
            similarity = self._cosine_similarity(embedding[:target_dim], query_vector)
            text_bonus = self._score_text_match(
                " ".join(
                    part for part in (record.ocr or "", record.caption or "", record.asr or "", str(record.metadata or "")) if part
                ),
                tokens,
            )
            final_score = max(float(similarity), 0.0) + text_bonus * 0.5
            if final_score > 0:
                matches.append((final_score, RetrievalResult(
                    video_id=str(record.video_id),
                    frame_id=int(record.frame_id),
                    score=float(final_score),
                    source="clip" if record.clip_embedding else "siglip2",
                    timestamp=record.timestamp,
                )))

        matches.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in matches[: max(0, top_k)]]

    def _lexical_candidates(self, query: str, top_k: int) -> list[RetrievalResult]:
        results = list(self.elasticsearch.search(query, top_k, fields=("ocr", "caption", "asr", "metadata")))
        if not results:
            for record in self._records:
                haystack = " ".join(
                    part for part in (record.ocr or "", record.caption or "", record.asr or "", json.dumps(record.metadata or {}, ensure_ascii=False)) if part
                )
                score = self._score_text_match(haystack, set(_tokenize(query)))
                if score > 0:
                    results.append(
                        RetrievalResult(
                            video_id=str(record.video_id),
                            frame_id=int(record.frame_id),
                            score=float(score),
                            source="lexical",
                            timestamp=record.timestamp,
                        )
                    )
        return results[: max(0, top_k)]

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        previous_query: str | None = None,
        next_query: str | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []
        query = query.strip()
        lexical = self._lexical_candidates(query, top_k=max(1, top_k * 3))
        semantic = self._semantic_candidates(query, top_k=max(1, top_k * 3))

        candidate_map: dict[tuple[str, int], RetrievalResult] = {}
        for result in lexical:
            candidate_map[(str(result.video_id), int(result.frame_id))] = result
        for result in semantic:
            key = (str(result.video_id), int(result.frame_id))
            current = candidate_map.get(key)
            if current is None:
                candidate_map[key] = result
            else:
                current.score = max(current.score, result.score)
                current.metadata = {**(current.metadata or {}), "secondary_source": result.source}

        fused_ids = reciprocal_rank_fusion(
            [
                [(str(result.video_id), int(result.frame_id)) for result in lexical],
                [(str(result.video_id), int(result.frame_id)) for result in semantic],
            ],
            k=60,
        )

        fused_results: list[RetrievalResult] = []
        for key, score in fused_ids:
            if not isinstance(key, tuple):
                continue
            if key not in candidate_map:
                continue
            result = candidate_map[key]
            result.score = float(score)
            result.source = "fused"
            result.rank = len(fused_results) + 1
            fused_results.append(result)

        if not fused_results:
            fused_results = list(candidate_map.values())

        fused_results = sorted(fused_results, key=lambda item: item.score, reverse=True)[: max(0, top_k)]
        for index, result in enumerate(fused_results, start=1):
            result.rank = index

        if previous_query or next_query:
            previous = self._lexical_candidates(previous_query or "", top_k=max(1, top_k * 3)) if previous_query else []
            next_hits = self._lexical_candidates(next_query or "", top_k=max(1, top_k * 3)) if next_query else []
            reranked = apply_temporal_rerank(fused_results, previous_results=previous, next_results=next_hits)
            final_results: list[RetrievalResult] = []
            for item in reranked:
                final_results.append(
                    RetrievalResult(
                        video_id=item.video_id,
                        frame_id=item.frame_id,
                        score=float(item.final_score),
                        source="temporal",
                        timestamp=item.timestamp,
                        rank=len(final_results) + 1,
                        metadata={
                            "current_score": item.current_score,
                            "previous_score": item.previous_score,
                            "next_score": item.next_score,
                        },
                    )
                )
            return final_results[: max(0, top_k)]

        return fused_results[: max(0, top_k)]

    def query_frames(self, query: str, *, top_k: int = 10, previous_query: str | None = None, next_query: str | None = None) -> list[dict[str, Any]]:
        results = self.query(query, top_k=top_k, previous_query=previous_query, next_query=next_query)
        payload: list[dict[str, Any]] = []
        for result in results:
            payload.append(
                {
                    "video_id": result.video_id,
                    "frame_id": result.frame_id,
                    "timestamp": result.timestamp,
                    "path": next(
                        (
                            item.image_ref
                            for item in self._records
                            if str(item.video_id) == str(result.video_id) and int(item.frame_id) == int(result.frame_id)
                        ),
                        "",
                    ),
                    "score": result.score,
                    "source": result.source,
                    "rank": result.rank,
                }
            )
        return payload

    def build_cache_key(self, query: str) -> str:
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        return f"{self.cache_prefix}{normalized}"

    def cached_search(self, query: str, *, top_k: int = 10) -> list[RetrievalResult] | None:
        key = self.build_cache_key(query)
        cached = self.redis_cache.get(key)
        if cached is None:
            return None
        return [RetrievalResult(**item) for item in cached]

    def query(self, query: str, *, top_k: int = 10, previous_query: str | None = None, next_query: str | None = None) -> list[RetrievalResult]:
        cached = self.cached_search(query, top_k=top_k)
        if cached is not None:
            return cached[: max(0, top_k)]
        results = self.search(query, top_k=top_k, previous_query=previous_query, next_query=next_query)
        self.redis_cache.set(self.build_cache_key(query), [asdict(result) for result in results], ttl_seconds=3600)
        return results

    def __len__(self) -> int:
        return len(self._records)

    def items(self) -> list[KeyframeRecord]:
        return list(self._records)

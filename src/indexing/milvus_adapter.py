"""Milvus adapter contract for CLIP DFN5B and SigLIP2 vectors."""

from __future__ import annotations

import math
from typing import Any, Sequence

from src.schemas.retrieval import RetrievalResult
from src.schemas.video import KeyframeRecord


class MilvusAdapter:
    """Backend adapter; the concrete client is injected at construction time."""

    def __init__(self, client: Any, collection_name: str) -> None:
        self._client = client
        self.collection_name = collection_name
        self._documents: list[dict[str, Any]] = []

    def _coerce_record(self, record: KeyframeRecord | dict[str, Any]) -> dict[str, Any]:
        if isinstance(record, KeyframeRecord):
            return {
                "video_id": record.video_id,
                "frame_id": record.frame_id,
                "timestamp": record.timestamp,
                "image_ref": record.image_ref,
                "clip_embedding": record.clip_embedding or [],
                "siglip2_embedding": record.siglip2_embedding or [],
                "ocr": record.ocr,
                "caption": record.caption,
                "asr": record.asr,
                "shot_id": record.shot_id,
                "metadata": record.metadata or {},
            }
        return dict(record)

    def _iter_documents(self) -> list[dict[str, Any]]:
        if isinstance(self._client, dict):
            docs = self._client.get(self.collection_name, [])
            return [dict(doc) for doc in docs]
        if hasattr(self._client, "_documents"):
            return [dict(doc) for doc in getattr(self._client, "_documents")]
        return list(self._documents)

    def upsert(self, records: Sequence[KeyframeRecord]) -> None:
        docs = [self._coerce_record(record) for record in records]
        if hasattr(self._client, "upsert"):
            self._client.upsert(self.collection_name, docs)
            return
        if hasattr(self._client, "insert"):
            self._client.insert(self.collection_name, docs)
            return
        if isinstance(self._client, dict):
            stored = self._client.setdefault(self.collection_name, [])
            existing = {self._key(doc): doc for doc in stored}
            for doc in docs:
                existing[self._key(doc)] = doc
            self._client[self.collection_name] = list(existing.values())
            return
        if hasattr(self._client, "collection"):
            collection = self._client.collection(self.collection_name)
            if hasattr(collection, "upsert"):
                collection.upsert(docs)
                return
            if hasattr(collection, "insert"):
                collection.insert(docs)
                return
        self._documents.extend(docs)

    @staticmethod
    def _key(doc: dict[str, Any]) -> tuple[str, int]:
        video_id = str(doc.get("video_id", ""))
        frame_id = int(doc.get("frame_id", 0))
        return (video_id, frame_id)

    @staticmethod
    def _l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Vectors must have the same length")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        field: str,
    ) -> Sequence[RetrievalResult]:
        query = [float(v) for v in query_vector]
        matches: list[tuple[float, dict[str, Any]]] = []

        for doc in self._iter_documents():
            embedding = doc.get(field)
            if not embedding:
                continue
            distance = self._l2_distance(query, [float(v) for v in embedding])
            similarity = 1.0 / (1.0 + distance)
            matches.append((similarity, doc))

        matches.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievalResult] = []
        for rank, (similarity, doc) in enumerate(matches[: max(0, top_k)], start=1):
            results.append(
                RetrievalResult(
                    video_id=str(doc.get("video_id", "")),
                    frame_id=int(doc.get("frame_id", 0)),
                    score=float(similarity),
                    source=str(field),
                    timestamp=float(doc.get("timestamp")) if doc.get("timestamp") is not None else None,
                    rank=rank,
                    metadata={"collection": self.collection_name},
                )
            )
        return results

"""Local vector indexing adapter built on FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.schemas.retrieval import RetrievalResult
from src.schemas.video import KeyframeRecord

try:  # pragma: no cover - optional dependency path
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback for constrained environments
    faiss = None


class FaissVectorAdapter:
    """Vector adapter that prefers local FAISS indexes and falls back to in-memory search."""

    def __init__(self, client: Any | None = None, collection_name: str = "video_keyframes") -> None:
        self._client = client
        self.collection_name = collection_name
        self._documents: list[dict[str, Any]] = []
        self._index: Any | None = None
        self._dimension: int | None = None
        self._path: Path | None = None

        if isinstance(client, (str, Path)):
            self._path = Path(client)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                self._load_index(self._path)

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

    def _rebuild_index(self) -> None:
        if faiss is None:
            self._index = None
            return
        valid = [
            doc.get("clip_embedding") or doc.get("siglip2_embedding") or []
            for doc in self._documents
            if (doc.get("clip_embedding") or doc.get("siglip2_embedding"))
        ]
        if not valid:
            self._index = None
            self._dimension = None
            return
        dimension = len(valid[0])
        matrix = np.asarray(valid, dtype="float32")
        if matrix.size == 0:
            self._index = None
            return
        matrix = matrix.reshape(len(valid), dimension)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)
        self._index = index
        self._dimension = dimension

    def upsert(self, records: Sequence[KeyframeRecord]) -> None:
        docs = [self._coerce_record(record) for record in records]
        merged = {self._key(doc): doc for doc in self._documents}
        for doc in docs:
            merged[self._key(doc)] = doc
        self._documents = list(merged.values())
        self._rebuild_index()

        if self._path is not None:
            payload = [
                {
                    "video_id": doc.get("video_id"),
                    "frame_id": doc.get("frame_id"),
                    "timestamp": doc.get("timestamp"),
                    "image_ref": doc.get("image_ref"),
                    "clip_embedding": doc.get("clip_embedding") or [],
                    "siglip2_embedding": doc.get("siglip2_embedding") or [],
                }
                for doc in self._documents
            ]
            np.save(self._path, np.asarray(payload, dtype=object))

        if isinstance(self._client, dict):
            self._client[self.collection_name] = list(self._documents)
        elif hasattr(self._client, "upsert"):
            self._client.upsert(self.collection_name, list(self._documents))

    def _load_index(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            raw = np.load(path, allow_pickle=True)
            payload = raw.tolist() if isinstance(raw, np.ndarray) else []
            docs = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                docs.append({
                    "video_id": item.get("video_id"),
                    "frame_id": item.get("frame_id"),
                    "timestamp": item.get("timestamp"),
                    "image_ref": item.get("image_ref"),
                    "clip_embedding": item.get("clip_embedding") or [],
                    "siglip2_embedding": item.get("siglip2_embedding") or [],
                })
            self._documents = docs
            self._rebuild_index()
        except Exception:
            self._documents = []
            self._index = None

    @staticmethod
    def _key(doc: dict[str, Any]) -> tuple[str, int]:
        return (str(doc.get("video_id", "")), int(doc.get("frame_id", 0)))

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        field: str = "clip_embedding",
    ) -> Sequence[RetrievalResult]:
        if not query_vector:
            return []
        records = self._iter_documents()
        if not records:
            return []

        if faiss is not None and self._index is not None and self._dimension is not None:
            query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
            if query.shape[1] != self._dimension:
                query = query[:, : self._dimension]
            faiss.normalize_L2(query)
            limit = max(1, min(top_k, len(records)))
            distances, indices = self._index.search(query, limit)
            results: list[RetrievalResult] = []
            for rank, idx in enumerate(indices[0], start=1):
                if idx == -1:
                    continue
                doc = records[int(idx)]
                results.append(
                    RetrievalResult(
                        video_id=str(doc.get("video_id", "")),
                        frame_id=int(doc.get("frame_id", 0)),
                        score=float(distances[0][rank - 1]),
                        source=str(field),
                        timestamp=float(doc.get("timestamp")) if doc.get("timestamp") is not None else None,
                        rank=rank,
                        metadata={"collection": self.collection_name, "field": field},
                    )
                )
            return results

        matches: list[tuple[float, dict[str, Any]]] = []
        for doc in records:
            embedding = doc.get(field) or doc.get("clip_embedding") or doc.get("siglip2_embedding") or []
            if not embedding:
                continue
            if len(embedding) != len(query_vector):
                length = min(len(embedding), len(query_vector))
                embedding = embedding[:length]
                query = list(query_vector)[:length]
            else:
                query = list(query_vector)
            dot = sum(a * b for a, b in zip(query, embedding))
            norm = np.linalg.norm(query) * np.linalg.norm(embedding)
            score = float(dot / norm) if norm > 0 else 0.0
            if score > 0:
                matches.append((score, doc))
        matches.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievalResult] = []
        for rank, (score, doc) in enumerate(matches[: max(0, top_k)], start=1):
            results.append(
                RetrievalResult(
                    video_id=str(doc.get("video_id", "")),
                    frame_id=int(doc.get("frame_id", 0)),
                    score=float(score),
                    source=str(field),
                    timestamp=float(doc.get("timestamp")) if doc.get("timestamp") is not None else None,
                    rank=rank,
                    metadata={"collection": self.collection_name, "field": field},
                )
            )
        return results

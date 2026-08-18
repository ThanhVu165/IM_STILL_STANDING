"""Elasticsearch adapter contract for text and metadata retrieval."""

from __future__ import annotations

from typing import Any, Sequence

from src.schemas.retrieval import RetrievalResult
from src.schemas.video import KeyframeRecord


class ElasticsearchAdapter:
    """Backend adapter for OCR, captions, ASR, and metadata."""

    def __init__(self, client: Any, index_name: str) -> None:
        self._client = client
        self.index_name = index_name
        self._documents: list[dict[str, Any]] = []

    def _coerce_record(self, record: KeyframeRecord | dict[str, Any]) -> dict[str, Any]:
        if isinstance(record, KeyframeRecord):
            return {
                "video_id": record.video_id,
                "frame_id": record.frame_id,
                "timestamp": record.timestamp,
                "ocr": record.ocr or "",
                "caption": record.caption or "",
                "asr": record.asr or "",
                "image_ref": record.image_ref,
                "metadata": record.metadata or {},
            }
        return dict(record)

    def _iter_documents(self) -> list[dict[str, Any]]:
        if isinstance(self._client, dict):
            docs = self._client.get(self.index_name, [])
            return [dict(doc) for doc in docs]
        if hasattr(self._client, "_documents"):
            return [dict(doc) for doc in getattr(self._client, "_documents")]
        return list(self._documents)

    def upsert(self, records: Sequence[KeyframeRecord]) -> None:
        docs = [self._coerce_record(record) for record in records]
        if hasattr(self._client, "index"):
            for doc in docs:
                self._client.index(self.index_name, doc)
            return
        if hasattr(self._client, "index_documents"):
            self._client.index_documents(self.index_name, docs)
            return
        if isinstance(self._client, dict):
            stored = self._client.setdefault(self.index_name, [])
            existing = {self._key(doc): doc for doc in stored}
            for doc in docs:
                existing[self._key(doc)] = doc
            self._client[self.index_name] = list(existing.values())
            return
        self._documents.extend(docs)

    @staticmethod
    def _key(doc: dict[str, Any]) -> tuple[str, int]:
        return (str(doc.get("video_id", "")), int(doc.get("frame_id", 0)))

    def search(
        self,
        query: str,
        top_k: int,
        *,
        fields: Sequence[str],
    ) -> Sequence[RetrievalResult]:
        if not query:
            return []
        tokens = {part.lower() for part in query.split() if part.strip()}
        field_names = tuple(fields) if fields else ("ocr", "caption", "asr", "metadata")
        hits: list[tuple[float, dict[str, Any]]] = []

        for doc in self._iter_documents():
            score = 0.0
            for field in field_names:
                raw_value = doc.get(field)
                if not raw_value:
                    continue
                if isinstance(raw_value, dict):
                    text = " ".join(str(v) for v in raw_value.values())
                else:
                    text = str(raw_value)
                lowered = text.lower()
                score += sum(1 for token in tokens if token in lowered)
            if score > 0:
                hits.append((float(score), doc))

        hits.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievalResult] = []
        for rank, (score_value, doc) in enumerate(hits[: max(0, top_k)], start=1):
            results.append(
                RetrievalResult(
                    video_id=str(doc.get("video_id", "")),
                    frame_id=int(doc.get("frame_id", 0)),
                    score=float(score_value),
                    source="elasticsearch",
                    timestamp=float(doc.get("timestamp")) if doc.get("timestamp") is not None else None,
                    rank=rank,
                    metadata={"field_hits": field_names},
                )
            )
        return results

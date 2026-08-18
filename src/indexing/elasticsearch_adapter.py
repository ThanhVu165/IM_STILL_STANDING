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

    def upsert(self, records: Sequence[KeyframeRecord]) -> None:
        raise NotImplementedError("Elasticsearch mapping is pending index configuration")

    def search(
        self,
        query: str,
        top_k: int,
        *,
        fields: Sequence[str],
    ) -> Sequence[RetrievalResult]:
        raise NotImplementedError("Elasticsearch search mapping is pending index configuration")

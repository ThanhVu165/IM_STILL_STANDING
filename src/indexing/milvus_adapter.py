"""Milvus adapter contract for CLIP DFN5B and SigLIP2 vectors."""

from __future__ import annotations

from typing import Any, Sequence

from src.schemas.retrieval import RetrievalResult
from src.schemas.video import KeyframeRecord


class MilvusAdapter:
    """Backend adapter; the concrete client is injected at construction time."""

    def __init__(self, client: Any, collection_name: str) -> None:
        self._client = client
        self.collection_name = collection_name

    def upsert(self, records: Sequence[KeyframeRecord]) -> None:
        raise NotImplementedError("Milvus write mapping is pending schema/index configuration")

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        field: str,
    ) -> Sequence[RetrievalResult]:
        raise NotImplementedError("Milvus search mapping is pending schema/index configuration")

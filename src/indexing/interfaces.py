"""Common indexing contracts.

Index implementations must hide backend-specific representations behind these
interfaces so retrieval branches can consume a common candidate representation.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from src.schemas.retrieval import RetrievalResult
from src.schemas.video import KeyframeRecord


class VectorIndex(Protocol):
    def upsert(self, records: Sequence[KeyframeRecord]) -> None:
        """Insert or update vector-bearing keyframe records."""

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        field: str,
    ) -> Sequence[RetrievalResult]:
        """Search one configured vector field and return common candidates."""


class TextIndex(Protocol):
    def upsert(self, records: Sequence[KeyframeRecord]) -> None:
        """Insert or update searchable OCR/caption/ASR/metadata records."""

    def search(
        self,
        query: str,
        top_k: int,
        *,
        fields: Sequence[str],
    ) -> Sequence[RetrievalResult]:
        """Search configured text fields and return common candidates."""


class ResultCache(Protocol):
    def get(self, key: str) -> object | None:
        """Return a cached result when present."""

    def set(self, key: str, value: object, *, ttl_seconds: int | None = None) -> None:
        """Store a result with optional TTL."""

    def delete(self, key: str) -> None:
        """Delete a cached result."""

"""Redis cache adapter boundary for retrieval results."""

from __future__ import annotations

from typing import Any


class RedisResultCache:
    def __init__(self, client: Any) -> None:
        self._client = client

    def get(self, key: str) -> object | None:
        raise NotImplementedError("Redis serialization policy is pending configuration")

    def set(self, key: str, value: object, *, ttl_seconds: int | None = None) -> None:
        raise NotImplementedError("Redis serialization policy is pending configuration")

    def delete(self, key: str) -> None:
        raise NotImplementedError("Redis deletion policy is pending configuration")

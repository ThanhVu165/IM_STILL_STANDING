"""Redis cache adapter boundary for retrieval results."""

from __future__ import annotations

import json
from typing import Any


class RedisResultCache:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._ttl: dict[str, int | None] = {}

    def get(self, key: str) -> object | None:
        if hasattr(self._client, "get"):
            value = self._client.get(key)
            if value is None:
                return None
            if isinstance(value, (str, bytes, bytearray)):
                text = value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            return value
        if isinstance(self._client, dict):
            value = self._client.get(key)
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        return None

    def set(self, key: str, value: object, *, ttl_seconds: int | None = None) -> None:
        payload = value
        if hasattr(self._client, "set"):
            if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
                payload = json.dumps(value)
            self._client.set(key, payload, ex=ttl_seconds)
            return
        if isinstance(self._client, dict):
            self._client[key] = json.dumps(value) if not isinstance(value, str) else value
            self._ttl[key] = ttl_seconds
            return
        raise TypeError("Unsupported Redis client type")

    def delete(self, key: str) -> None:
        if hasattr(self._client, "delete"):
            self._client.delete(key)
            return
        if isinstance(self._client, dict):
            self._client.pop(key, None)
            self._ttl.pop(key, None)
            return
        raise TypeError("Unsupported Redis client type")

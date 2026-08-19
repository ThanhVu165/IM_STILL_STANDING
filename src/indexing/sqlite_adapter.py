"""Local SQLite-backed text index for metadata/OCR/ASR/caption search."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from src.schemas.retrieval import RetrievalResult
from src.schemas.video import KeyframeRecord


class SQLiteTextAdapter:
    """Minimal local text index built on SQLite so the app can run without Elasticsearch."""

    def __init__(self, client: Any | None = None, index_name: str = "video_keyframes", *, db_path: str | Path | None = None) -> None:
        self._client = client
        self.index_name = index_name
        self.db_path = str(Path(db_path) if db_path is not None else Path("data") / "indexes" / f"{index_name}.sqlite")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS keyframes (
                video_id TEXT,
                frame_id INTEGER,
                timestamp REAL,
                image_ref TEXT,
                ocr TEXT,
                caption TEXT,
                asr TEXT,
                metadata_json TEXT,
                PRIMARY KEY (video_id, frame_id)
            )
            """
        )

    def _coerce_record(self, record: KeyframeRecord | dict[str, Any]) -> dict[str, Any]:
        if isinstance(record, KeyframeRecord):
            return {
                "video_id": record.video_id,
                "frame_id": record.frame_id,
                "timestamp": record.timestamp,
                "image_ref": record.image_ref,
                "ocr": record.ocr or "",
                "caption": record.caption or "",
                "asr": record.asr or "",
                "metadata": record.metadata or {},
            }
        return dict(record)

    def _iter_documents(self) -> list[dict[str, Any]]:
        if isinstance(self._client, dict):
            return [dict(doc) for doc in self._client.get(self.index_name, [])]
        rows = self._conn.execute(
            "SELECT video_id, frame_id, timestamp, image_ref, ocr, caption, asr, metadata_json FROM keyframes"
        ).fetchall()
        docs: list[dict[str, Any]] = []
        for row in rows:
            docs.append(
                {
                    "video_id": row[0],
                    "frame_id": row[1],
                    "timestamp": row[2],
                    "image_ref": row[3],
                    "ocr": row[4],
                    "caption": row[5],
                    "asr": row[6],
                    "metadata": json.loads(row[7]) if row[7] else {},
                }
            )
        return docs

    def upsert(self, records: Sequence[KeyframeRecord]) -> None:
        docs = [self._coerce_record(record) for record in records]
        if isinstance(self._client, dict):
            stored = self._client.setdefault(self.index_name, [])
            existing = {(str(doc.get("video_id", "")), int(doc.get("frame_id", 0))): doc for doc in stored}
            for doc in docs:
                existing[(str(doc.get("video_id", "")), int(doc.get("frame_id", 0)))] = doc
            self._client[self.index_name] = list(existing.values())
            return

        for doc in docs:
            self._conn.execute(
                """
                INSERT INTO keyframes (video_id, frame_id, timestamp, image_ref, ocr, caption, asr, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id, frame_id)
                DO UPDATE SET
                    timestamp = excluded.timestamp,
                    image_ref = excluded.image_ref,
                    ocr = excluded.ocr,
                    caption = excluded.caption,
                    asr = excluded.asr,
                    metadata_json = excluded.metadata_json
                """,
                (
                    str(doc.get("video_id", "")),
                    int(doc.get("frame_id", 0)),
                    float(doc.get("timestamp", 0.0) or 0.0),
                    str(doc.get("image_ref") or ""),
                    str(doc.get("ocr") or ""),
                    str(doc.get("caption") or ""),
                    str(doc.get("asr") or ""),
                    json.dumps(doc.get("metadata") or {}, ensure_ascii=False),
                ),
            )
        self._conn.commit()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token.lower() for token in text.replace("-", " ").split() if token.strip()}

    def search(
        self,
        query: str,
        top_k: int,
        *,
        fields: Sequence[str],
    ) -> Sequence[RetrievalResult]:
        if not query or not query.strip():
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []

        docs = self._iter_documents()
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in docs:
            score = 0.0
            for field in fields:
                value = doc.get(field) or ""
                if isinstance(value, dict):
                    text = " ".join(str(fragment) for fragment in value.values())
                else:
                    text = str(value)
                text = text.lower()
                score += sum(1 for token in tokens if token in text)
            if score > 0:
                scored.append((float(score), doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievalResult] = []
        for rank, (score_value, doc) in enumerate(scored[: max(0, top_k)], start=1):
            results.append(
                RetrievalResult(
                    video_id=str(doc.get("video_id", "")),
                    frame_id=int(doc.get("frame_id", 0)),
                    score=float(score_value),
                    source="sqlite",
                    timestamp=float(doc.get("timestamp")) if doc.get("timestamp") is not None else None,
                    rank=rank,
                    metadata={"fields": tuple(fields)},
                )
            )
        return results

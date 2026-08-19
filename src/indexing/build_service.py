"""Index build service. Only responsible for indexing organizer artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.indexing.catalog_loader import load_artifact_records
from src.indexing.faiss_adapter import FaissVectorAdapter
from src.indexing.sqlite_adapter import SQLiteTextAdapter
from src.schemas.video import KeyframeRecord


def build_artifact_index(
    *,
    data_root: str | Path = "data",
    collection_name: str = "video_keyframes",
    index_name: str = "video_keyframes",
    milvus_client: Any | None = None,
    elasticsearch_client: Any | None = None,
) -> list[KeyframeRecord]:
    root = Path(data_root)
    index_root = root / "indexes"
    index_root.mkdir(parents=True, exist_ok=True)

    records = load_artifact_records(root)
    if not records:
        return []

    vector_index = FaissVectorAdapter(
        milvus_client if milvus_client is not None else str(index_root / f"{collection_name}.npy"),
        collection_name,
    )
    text_index = SQLiteTextAdapter(
        elasticsearch_client if elasticsearch_client is not None else None,
        index_name,
        db_path=index_root / f"{index_name}.sqlite",
    )
    vector_index.upsert(records)
    text_index.upsert(records)
    return records

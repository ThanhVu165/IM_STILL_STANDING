"""End-to-end AIC indexing pipeline from video preprocessing to search stores."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.indexing.faiss_adapter import FaissVectorAdapter
from src.indexing.redis_cache import RedisResultCache
from src.indexing.sqlite_adapter import SQLiteTextAdapter
from src.preprocessing.video_processor import AICVideoPipeline
from src.schemas.video import KeyframeRecord, ShotRecord


class VideoIndexingPipeline:
    """Preprocess a video, then index the resulting multimodal keyframes into vector and lexical stores."""

    def __init__(
        self,
        *,
        preprocessor: Any | None = None,
        milvus_client: Any | None = None,
        elasticsearch_client: Any | None = None,
        redis_client: Any | None = None,
        collection_name: str = "video_keyframes",
        index_name: str = "video_keyframes",
        cache_prefix: str = "video_index:",
        use_real_models: bool = True,
    ) -> None:
        self.preprocessor = preprocessor or AICVideoPipeline(use_real_models=use_real_models)
        self.milvus = FaissVectorAdapter(milvus_client if milvus_client is not None else {}, collection_name)
        self.elasticsearch = SQLiteTextAdapter(
            elasticsearch_client if elasticsearch_client is not None else {},
            index_name,
        )
        self.redis_cache = RedisResultCache(redis_client if redis_client is not None else {})
        self.collection_name = collection_name
        self.index_name = index_name
        self.cache_prefix = cache_prefix

    def index_video(self, video_path: str) -> tuple[list[ShotRecord], list[KeyframeRecord]]:
        shots, records = self.preprocessor.process(video_path)
        if records:
            self.milvus.upsert(records)
            self.elasticsearch.upsert(records)
            self.redis_cache.set(
                f"{self.cache_prefix}{Path(video_path).stem}",
                {
                    "video_path": video_path,
                    "shot_count": len(shots),
                    "keyframe_count": len(records),
                    "frame_ids": [record.frame_id for record in records],
                },
                ttl_seconds=3600,
            )
        return shots, records

    def run(
        self,
        video_path: str,
        *,
        output_dir: str | Path | None = None,
        write_json: bool = True,
    ) -> dict[str, Any]:
        """Run end-to-end preprocessing and ingestion into the configured stores."""
        shots, records = self.index_video(video_path)
        manifest: dict[str, Any] = {
            "video_path": str(video_path),
            "shots": [asdict(shot) for shot in shots],
            "keyframes": [asdict(record) for record in records],
            "cache": self.cached_video(video_path),
            "summary": {
                "shot_count": len(shots),
                "keyframe_count": len(records),
            },
        }
        if write_json and output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            json_path = output_path / f"{Path(video_path).stem}_index_manifest.json"
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
            manifest["manifest_path"] = str(json_path)
        return manifest

    def search_video(self, query: str, *, top_k: int = 10) -> list[Any]:
        return list(self.elasticsearch.search(query, top_k, fields=("ocr", "caption", "asr")))

    def cached_video(self, video_path: str) -> dict[str, Any] | None:
        return self.redis_cache.get(f"{self.cache_prefix}{Path(video_path).stem}")

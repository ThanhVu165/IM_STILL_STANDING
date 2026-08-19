"""CLI for the local, repo-aligned video retrieval pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval.pipeline import VideoRetrievalPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local video retrieval using the project’s multimodal index contract.")
    parser.add_argument("query", help="User query text")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum number of ranked results to return")
    parser.add_argument("--data-root", default="data", help="Root folder that contains processed metadata and keyframes")
    parser.add_argument("--build-index", action="store_true", help="Build the local FAISS + SQLite index from the data directory before querying")
    parser.add_argument("--previous-query", default=None, help="Optional previous event query for temporal reranking")
    parser.add_argument("--next-query", default=None, help="Optional next event query for temporal reranking")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    index_root = data_root / "indexes"
    needs_build = args.build_index or not ((index_root / "video_keyframes.npy").exists() and (index_root / "video_keyframes.sqlite").exists())

    pipeline = VideoRetrievalPipeline(
        data_root=data_root,
        load_index_only=not needs_build,
        initialize_from_disk=False if needs_build else True,
    )
    if needs_build:
        pipeline.build_index()

    results = pipeline.query_frames(
        args.query,
        top_k=args.top_k,
        previous_query=args.previous_query,
        next_query=args.next_query,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

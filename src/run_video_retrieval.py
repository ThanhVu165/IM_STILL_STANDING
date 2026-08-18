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
    parser.add_argument("--previous-query", default=None, help="Optional previous event query for temporal reranking")
    parser.add_argument("--next-query", default=None, help="Optional next event query for temporal reranking")
    args = parser.parse_args()

    pipeline = VideoRetrievalPipeline(data_root=Path(args.data_root))
    results = pipeline.query(
        args.query,
        top_k=args.top_k,
        previous_query=args.previous_query,
        next_query=args.next_query,
    )
    print(json.dumps([result.__dict__ for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

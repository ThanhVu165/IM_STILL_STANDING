"""CLI entry point for running the AIC offline video pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.indexing.pipeline import VideoIndexingPipeline
from src.preprocessing.video_processor import AICVideoPipeline


def run_pipeline(
    video_path: str,
    *,
    use_real_models: bool = False,
    index_to_stores: bool = False,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run the end-to-end preprocessing pipeline for a single input video."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    pipeline = AICVideoPipeline(use_real_models=use_real_models)
    manifest = pipeline.run(str(path), output_dir=output_dir, write_json=output_dir is not None)

    if index_to_stores:
        index_pipeline = VideoIndexingPipeline(use_real_models=use_real_models)
        indexed = index_pipeline.run(str(path), output_dir=output_dir, write_json=output_dir is not None)
        manifest["indexed"] = indexed
        manifest["cache"] = indexed.get("cache")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline AIC video processing pipeline.")
    parser.add_argument("video_path", help="Path to the source video file")
    parser.add_argument("--output-dir", default=None, help="Optional folder for JSON manifests")
    parser.add_argument("--use-real-models", action="store_true", help="Attempt to load the real AIC models when available")
    parser.add_argument("--index", action="store_true", help="Index generated keyframes into Milvus/Elasticsearch/Redis when backend clients are configured")
    args = parser.parse_args()

    manifest = run_pipeline(
        args.video_path,
        use_real_models=args.use_real_models,
        index_to_stores=args.index,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "video_path": manifest["video_path"],
        "summary": manifest["summary"],
        "stages": manifest["stages"],
        "manifest_path": manifest.get("manifest_path"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

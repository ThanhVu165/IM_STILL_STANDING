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
    index_to_stores: bool = False,
    output_dir: str | None = None,
    processing_pipeline: AICVideoPipeline | None = None,
    indexing_pipeline: VideoIndexingPipeline | None = None,
) -> dict[str, Any]:
    """Run the artifact-first video processing pipeline for a single input video."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    pipeline = processing_pipeline or AICVideoPipeline(use_real_models=True)
    shots, records = pipeline.process(str(path))
    manifest = pipeline.run(
        str(path),
        output_dir=output_dir,
        write_json=output_dir is not None,
        preprocessed=(shots, records),
    )

    if index_to_stores:
        index_pipeline = indexing_pipeline or VideoIndexingPipeline(use_real_models=True)
        indexed = index_pipeline.run(
            str(path),
            output_dir=output_dir,
            write_json=output_dir is not None,
            preprocessed=(shots, records),
        )
        manifest["indexed"] = indexed
        manifest["cache"] = indexed.get("cache")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline AIC video processing pipeline.")
    parser.add_argument("video_path", help="Path to the source video file")
    parser.add_argument("--output-dir", default=None, help="Optional folder for JSON manifests")
    parser.add_argument("--index", action="store_true", help="Index generated keyframes into the local FAISS + SQLite stores when configured")
    args = parser.parse_args()

    manifest = run_pipeline(
        args.video_path,
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

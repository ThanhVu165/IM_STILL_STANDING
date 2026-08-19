"""CLI entry point for running the AIC offline video pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.indexing.pipeline import VideoIndexingPipeline
from src.preprocessing.video_processor import AICVideoPipeline


def run_pipeline(
    video_input: str,
    *,
    index_to_stores: bool = False,
    output_dir: str | None = None,
    data_root: str | None = "data",
    use_real_models: bool = False,
    processing_pipeline: AICVideoPipeline | None = None,
    indexing_pipeline: VideoIndexingPipeline | None = None,
) -> dict[str, Any]:
    """Run the artifact-first video processing pipeline for a single input video."""
    path = Path(video_input)
    resolved_input = str(path) if path.exists() else str(video_input)

    pipeline = processing_pipeline or AICVideoPipeline(
        use_real_models=use_real_models,
        organizer_root=data_root,
    )
    shots, records = pipeline.process(resolved_input)
    manifest = pipeline.run(
        resolved_input,
        output_dir=output_dir,
        write_json=output_dir is not None,
        preprocessed=(shots, records),
    )

    if index_to_stores:
        index_pipeline = indexing_pipeline or VideoIndexingPipeline(
            use_real_models=use_real_models,
            preprocessor=pipeline,
        )
        indexed = index_pipeline.run(
            resolved_input,
            output_dir=output_dir,
            write_json=output_dir is not None,
            preprocessed=(shots, records),
        )
        manifest["indexed"] = indexed
        manifest["cache"] = indexed.get("cache")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline AIC video processing pipeline.")
    parser.add_argument("video_input", help="Video path or video identifier used to resolve organizer artifacts")
    parser.add_argument("--data-root", default="data", help="Organizer artifact root (contains processed/, metadata/, etc.)")
    parser.add_argument("--use-real-models", action="store_true", help="Enable real-model preprocessing fallback when organizer artifacts are missing")
    parser.add_argument("--output-dir", default=None, help="Optional folder for JSON manifests")
    parser.add_argument("--index", action="store_true", help="Index generated keyframes into the local FAISS + SQLite stores when configured")
    args = parser.parse_args()

    manifest = run_pipeline(
        args.video_input,
        index_to_stores=args.index,
        output_dir=args.output_dir,
        data_root=args.data_root,
        use_real_models=args.use_real_models,
    )
    print(json.dumps({
        "video_path": manifest["video_path"],
        "summary": manifest["summary"],
        "stages": manifest["stages"],
        "manifest_path": manifest.get("manifest_path"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

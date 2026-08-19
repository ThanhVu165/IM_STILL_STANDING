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
    force: bool = False,
) -> dict[str, Any]:
    """Run the artifact-first video processing pipeline for a single input video or a directory of videos.

    When a directory is provided, process each video file inside. If force is False and a manifest already
    exists for a video in the output_dir, that video will be skipped.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    pipeline = AICVideoPipeline(use_real_models=True)

    # Single file
    if path.is_file():
        manifest = pipeline.run(str(path), output_dir=output_dir, write_json=output_dir is not None)
        if index_to_stores:
            index_pipeline = VideoIndexingPipeline(use_real_models=True)
            indexed = index_pipeline.run(str(path), output_dir=output_dir, write_json=output_dir is not None)
            manifest["indexed"] = indexed
            manifest["cache"] = indexed.get("cache")
        return manifest

    # Directory: iterate videos
    results: dict[str, Any] = {"processed": 0, "skipped": 0, "manifests": []}
    for candidate in sorted(path.iterdir()):
        if not candidate.is_file() or candidate.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            continue
        out_dir = Path(output_dir) if output_dir is not None else Path("data") / "processed" / "manifests"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / f"{candidate.stem}_pipeline.json"
        if manifest_path.exists() and not force:
            results["skipped"] += 1
            continue
        manifest = pipeline.run(str(candidate), output_dir=str(out_dir), write_json=True)
        if index_to_stores:
            index_pipeline = VideoIndexingPipeline(use_real_models=True)
            indexed = index_pipeline.run(str(candidate), output_dir=str(out_dir), write_json=True)
            manifest["indexed"] = indexed
            manifest["cache"] = indexed.get("cache")
        results["processed"] += 1
        results["manifests"].append(manifest.get("manifest_path"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline AIC video processing pipeline.")
    parser.add_argument("video_path", help="Path to the source video file or a directory of videos")
    parser.add_argument("--output-dir", default=None, help="Optional folder for JSON manifests")
    parser.add_argument("--index", action="store_true", help="Index generated keyframes into the local FAISS + SQLite stores when configured")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even when manifests already exist (directory mode)")
    args = parser.parse_args()

    manifest = run_pipeline(
        args.video_path,
        index_to_stores=args.index,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(json.dumps({
        "video_path": manifest["video_path"],
        "summary": manifest["summary"],
        "stages": manifest["stages"],
        "manifest_path": manifest.get("manifest_path"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

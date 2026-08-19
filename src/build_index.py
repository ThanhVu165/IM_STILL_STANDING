"""Build the local FAISS + SQLite indexes for image/video retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.indexing.build_service import build_artifact_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the repo's local retrieval indexes once, then reuse them for many queries.")
    parser.add_argument("--data-root", default="data", help="Root folder that contains processed metadata and keyframes")
    parser.add_argument("--force", action="store_true", help="Rebuild the indexes even when they already exist")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    index_root = data_root / "indexes"
    faiss_path = index_root / "video_keyframes.npy"
    sqlite_path = index_root / "video_keyframes.sqlite"
    should_build = args.force or not (faiss_path.exists() and sqlite_path.exists())

    if not should_build:
        print(json.dumps({
            "status": "already_built",
            "faiss_index": str(faiss_path),
            "sqlite_index": str(sqlite_path),
        }, ensure_ascii=False, indent=2))
        return

    records = build_artifact_index(data_root=data_root)
    print(json.dumps({
        "status": "built",
        "indexed_records": len(records),
        "faiss_index": str(faiss_path),
        "sqlite_index": str(sqlite_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Run TKIS / Q&A / TRAKE tasks using the project retrieval pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.indexing.build_service import build_artifact_index
from src.retrieval.pipeline import VideoRetrievalPipeline
from src.retrieval.tasks import PreliminaryTaskRunner, TaskQuery
from src.submission import write_submission_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a preliminary-round task using the repo's retrieval pipeline.")
    parser.add_argument("--task", choices=["tkis", "qa", "trake"], required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--previous-query", default=None)
    parser.add_argument("--next-query", default=None)
    parser.add_argument("--event", action="append", default=[], help="Add one event query for a TRAKE task")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--build-index", action="store_true", help="Build artifact indexes before running query tasks")
    parser.add_argument("--output", default=None)
    parser.add_argument("--submission-dir", default=None, help="Optional directory to write submission/<query>.csv and submission.zip")
    parser.add_argument("--submission-zip", default=None, help="Optional explicit zip path for the submission bundle")
    parser.add_argument("--query-file-name", default=None, help="Optional query csv file name (must match organizer naming when submitting)")
    parser.add_argument("--expected-event-count", type=int, default=None, help="Required TRAKE event count for strict submission validation")
    parser.add_argument("--internal-zero-based", action="store_true", help="Convert internal zero-based frame ids to external one-based numbering at export")
    args = parser.parse_args()

    if args.build_index:
        build_artifact_index(data_root=args.data_root)
    pipeline = VideoRetrievalPipeline(data_root=args.data_root, load_index_only=True)
    runner = PreliminaryTaskRunner(pipeline=pipeline)
    task = TaskQuery(
        query_id=args.query_id,
        task_type=args.task,
        query=args.query,
        previous_query=args.previous_query,
        next_query=args.next_query,
        event_queries=list(args.event),
        top_k=args.top_k,
    )
    try:
        ranked_items = runner.run_task_ranked(task, top_k=args.top_k)
        payload = {
            "task": args.task,
            "query_id": args.query_id,
            "result_count": len(ranked_items),
            "ranked_answers": [asdict(ranked) for ranked, _ in ranked_items],
            "submissions": [asdict(submission) for _, submission in ranked_items],
        }
        if args.submission_dir:
            records = [submission for _, submission in ranked_items]
            bundle = write_submission_bundle(
                task_type=args.task,
                records=records,
                output_dir=Path(args.submission_dir),
                query_filename=args.query_file_name,
                expected_event_count=args.expected_event_count,
                internal_zero_based=args.internal_zero_based,
                zip_path=args.submission_zip,
            )
            payload["submission_bundle"] = bundle
    except ValueError as exc:
        payload = {"error": str(exc), "task": args.task, "query_id": args.query_id}
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

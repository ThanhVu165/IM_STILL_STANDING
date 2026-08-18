"""Run TKIS / Q&A / TRAKE tasks using the project retrieval pipeline."""

from __future__ import annotations

import argparse
import json

from src.retrieval.pipeline import VideoRetrievalPipeline
from src.retrieval.tasks import PreliminaryTaskRunner, TaskQuery


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a preliminary-round task using the repo's retrieval pipeline.")
    parser.add_argument("--task", choices=["tkis", "qa", "trake"], required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--previous-query", default=None)
    parser.add_argument("--next-query", default=None)
    parser.add_argument("--event", action="append", default=[], help="Add one event query for a TRAKE task")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pipeline = VideoRetrievalPipeline(data_root=args.data_root)
    runner = PreliminaryTaskRunner(pipeline=pipeline)
    task = TaskQuery(
        query_id=args.query_id,
        task_type=args.task,
        query=args.query,
        previous_query=args.previous_query,
        next_query=args.next_query,
        event_queries=list(args.event),
    )
    try:
        ranked, submission = runner.run_task(task)
        payload = {"ranked_answer": ranked.__dict__, "submission": submission.__dict__}
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

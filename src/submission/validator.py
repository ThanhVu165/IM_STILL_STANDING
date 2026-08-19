"""Deterministic validation for competition-facing submission records."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.schemas.common import internal_to_external_frame
from src.schemas.submission import QASubmissionRecord, TKISSubmissionRecord, TRAKESubmissionRecord

SubmissionRecord = TKISSubmissionRecord | QASubmissionRecord | TRAKESubmissionRecord


def normalize_submission_frame(frame_id: int, *, internal_zero_based: bool) -> int:
    """Normalize a frame id to external competition numbering (starts at 1)."""
    if internal_zero_based:
        return internal_to_external_frame(int(frame_id))
    normalized = int(frame_id)
    if normalized < 1:
        raise ValueError("frame_id must be >= 1 in external numbering; set internal_zero_based=True if using zero-based frames.")
    return normalized


def validate_submission_records(
    task_type: str,
    records: Iterable[SubmissionRecord],
    *,
    expected_event_count: int | None = None,
    internal_zero_based: bool = False,
) -> list[SubmissionRecord]:
    normalized_task = (task_type or "").strip().lower()
    if normalized_task not in {"tkis", "qa", "trake"}:
        raise ValueError(f"Unsupported task type for submission: {task_type}")

    validated = list(records)
    if not validated:
        raise ValueError("No submission records to validate.")

    ranks_by_query: dict[str, set[int]] = defaultdict(set)
    frames_per_record: list[int] = []

    for record in validated:
        query_id = str(getattr(record, "query_id", "")).strip()
        if not query_id:
            raise ValueError("Each submission record must contain query_id.")
        rank = int(getattr(record, "rank", 0))
        if rank < 1:
            raise ValueError(f"Rank must be positive for query '{query_id}'.")
        if rank in ranks_by_query[query_id]:
            raise ValueError(f"Duplicate rank {rank} detected for query '{query_id}'.")
        ranks_by_query[query_id].add(rank)

        video_id = str(getattr(record, "video_id", "")).strip()
        if not video_id:
            raise ValueError(f"Missing video_id for query '{query_id}', rank {rank}.")

        if normalized_task == "tkis":
            if not isinstance(record, TKISSubmissionRecord):
                raise ValueError("TKIS submission contains a non-TKIS record type.")
            normalize_submission_frame(record.frame_id, internal_zero_based=internal_zero_based)
            frames_per_record.append(1)
            continue

        if normalized_task == "qa":
            if not isinstance(record, QASubmissionRecord):
                raise ValueError("Q&A submission contains a non-Q&A record type.")
            normalize_submission_frame(record.frame_id, internal_zero_based=internal_zero_based)
            if not str(record.answer or "").strip():
                raise ValueError(f"Q&A answer must be non-empty for query '{query_id}', rank {rank}.")
            frames_per_record.append(1)
            continue

        if not isinstance(record, TRAKESubmissionRecord):
            raise ValueError("TRAKE submission contains a non-TRAKE record type.")
        if not record.frames:
            raise ValueError(f"TRAKE record must contain at least one event frame for query '{query_id}', rank {rank}.")
        for frame in record.frames:
            normalize_submission_frame(frame, internal_zero_based=internal_zero_based)
        frames_per_record.append(len(record.frames))

    if normalized_task == "trake":
        if expected_event_count is not None:
            if expected_event_count < 1:
                raise ValueError("expected_event_count must be positive when provided.")
            for count in frames_per_record:
                if count != expected_event_count:
                    raise ValueError(f"TRAKE record event count mismatch: expected {expected_event_count}, got {count}.")
        else:
            required = frames_per_record[0]
            for count in frames_per_record:
                if count != required:
                    raise ValueError(
                        "TRAKE records have inconsistent event counts. "
                        "Provide expected_event_count for strict validation against organizer query schema."
                    )

    return validated

"""CSV/ZIP serialization for competition-facing submission payloads."""

from __future__ import annotations

import csv
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from src.schemas.submission import QASubmissionRecord, TKISSubmissionRecord, TRAKESubmissionRecord
from src.submission.validator import normalize_submission_frame, validate_submission_records


def _csv_filename(query_filename: str | None, query_id: str) -> str:
    candidate = (query_filename or query_id).strip()
    if not candidate:
        candidate = query_id
    return candidate if candidate.lower().endswith(".csv") else f"{candidate}.csv"


def _rows_for_task(
    task_type: str,
    records: list[TKISSubmissionRecord | QASubmissionRecord | TRAKESubmissionRecord],
    *,
    internal_zero_based: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    normalized_task = task_type.strip().lower()
    if normalized_task == "tkis":
        for record in records:
            assert isinstance(record, TKISSubmissionRecord)
            rows.append(
                {
                    "query_id": record.query_id,
                    "rank": int(record.rank),
                    "video_id": record.video_id,
                    "frame_id": normalize_submission_frame(record.frame_id, internal_zero_based=internal_zero_based),
                }
            )
        return rows

    if normalized_task == "qa":
        for record in records:
            assert isinstance(record, QASubmissionRecord)
            rows.append(
                {
                    "query_id": record.query_id,
                    "rank": int(record.rank),
                    "video_id": record.video_id,
                    "frame_id": normalize_submission_frame(record.frame_id, internal_zero_based=internal_zero_based),
                    "answer": record.answer,
                }
            )
        return rows

    max_events = max(len(record.frames) for record in records if isinstance(record, TRAKESubmissionRecord))
    for record in records:
        assert isinstance(record, TRAKESubmissionRecord)
        row: dict[str, object] = {
            "query_id": record.query_id,
            "rank": int(record.rank),
            "video_id": record.video_id,
        }
        for index, frame in enumerate(record.frames, start=1):
            row[f"frame_{index}"] = normalize_submission_frame(frame, internal_zero_based=internal_zero_based)
        for index in range(len(record.frames) + 1, max_events + 1):
            row[f"frame_{index}"] = ""
        rows.append(row)
    return rows


def write_submission_csv(
    *,
    task_type: str,
    records: list[TKISSubmissionRecord | QASubmissionRecord | TRAKESubmissionRecord],
    output_path: str | Path,
    expected_event_count: int | None = None,
    internal_zero_based: bool = False,
) -> Path:
    validated = validate_submission_records(
        task_type,
        records,
        expected_event_count=expected_event_count,
        internal_zero_based=internal_zero_based,
    )
    rows = _rows_for_task(task_type, validated, internal_zero_based=internal_zero_based)
    if not rows:
        raise ValueError("No rows to write.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_submission_bundle(
    *,
    task_type: str,
    records: list[TKISSubmissionRecord | QASubmissionRecord | TRAKESubmissionRecord],
    output_dir: str | Path,
    query_filename: str | None = None,
    expected_event_count: int | None = None,
    internal_zero_based: bool = False,
    zip_path: str | Path | None = None,
) -> dict[str, str]:
    if not records:
        raise ValueError("No submission records to export.")
    query_id = str(records[0].query_id)
    csv_name = _csv_filename(query_filename, query_id)

    root = Path(output_dir)
    submission_dir = root / "submission"
    csv_path = submission_dir / csv_name
    csv_output = write_submission_csv(
        task_type=task_type,
        records=records,
        output_path=csv_path,
        expected_event_count=expected_event_count,
        internal_zero_based=internal_zero_based,
    )

    archive_path = Path(zip_path) if zip_path is not None else (root / "submission.zip")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.write(csv_output, arcname=f"submission/{csv_name}")

    return {
        "csv_path": str(csv_output),
        "zip_path": str(archive_path),
        "query_filename": csv_name,
    }

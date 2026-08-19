from pathlib import Path
from zipfile import ZipFile

import pytest

from src.schemas.submission import QASubmissionRecord, TKISSubmissionRecord, TRAKESubmissionRecord
from src.submission.serializer import write_submission_bundle, write_submission_csv


def test_submission_bundle_converts_zero_based_frames_at_boundary(tmp_path: Path) -> None:
    records = [TKISSubmissionRecord(query_id="q1", rank=1, video_id="L21_V001", frame_id=0)]

    result = write_submission_bundle(
        task_type="tkis",
        records=records,
        output_dir=tmp_path,
        query_filename="query_01.csv",
        internal_zero_based=True,
    )

    csv_path = Path(result["csv_path"])
    zip_path = Path(result["zip_path"])
    assert csv_path.exists()
    assert zip_path.exists()
    assert "frame_id" in csv_path.read_text(encoding="utf-8")
    assert ",1\n" in csv_path.read_text(encoding="utf-8")
    with ZipFile(zip_path, "r") as archive:
        assert "submission/query_01.csv" in archive.namelist()


def test_submission_rejects_implicit_zero_based_frame_ids(tmp_path: Path) -> None:
    records = [TKISSubmissionRecord(query_id="q1", rank=1, video_id="L21_V001", frame_id=0)]
    with pytest.raises(ValueError, match="internal_zero_based=True"):
        write_submission_csv(
            task_type="tkis",
            records=records,
            output_path=tmp_path / "submission.csv",
            internal_zero_based=False,
        )


def test_trake_submission_enforces_expected_event_count(tmp_path: Path) -> None:
    records = [TRAKESubmissionRecord(query_id="q1", rank=1, video_id="L21_V001", frames=[10, 11])]
    with pytest.raises(ValueError, match="expected 3, got 2"):
        write_submission_csv(
            task_type="trake",
            records=records,
            output_path=tmp_path / "trake.csv",
            expected_event_count=3,
        )


def test_qa_submission_requires_non_empty_answer(tmp_path: Path) -> None:
    records = [QASubmissionRecord(query_id="q2", rank=1, video_id="L21_V005", frame_id=10, answer="")]
    with pytest.raises(ValueError, match="non-empty"):
        write_submission_csv(
            task_type="qa",
            records=records,
            output_path=tmp_path / "qa.csv",
        )

from pathlib import Path
from dataclasses import asdict

from src.run_video_pipeline import run_pipeline
from src.schemas.video import KeyframeRecord, ShotRecord


class _FakeProcessingPipeline:
    def __init__(self) -> None:
        self.process_calls = 0
        self.stages = ["sample", "annotate", "embed"]

    def process(self, video_path: str):
        self.process_calls += 1
        shots = [
            ShotRecord(
                video_id=Path(video_path).stem,
                shot_id="shot_0",
                start_time=0.0,
                end_time=1.0,
                start_frame=0,
                end_frame=8,
            )
        ]
        records = [
            KeyframeRecord(
                video_id=Path(video_path).stem,
                frame_id=0,
                timestamp=0.0,
                image_ref=f"{video_path}#frame=0",
                caption="scene",
                ocr="",
                asr="",
                clip_embedding=[1.0, 0.0],
                siglip2_embedding=[1.0, 0.0],
            )
        ]
        return shots, records

    def run(self, video_path: str, *, output_dir=None, write_json=True, preprocessed=None):
        shots, records = preprocessed if preprocessed is not None else self.process(video_path)
        return {
            "video_path": video_path,
            "stages": list(self.stages),
            "shots": [asdict(shot) for shot in shots],
            "keyframes": [asdict(record) for record in records],
            "summary": {"shot_count": len(shots), "keyframe_count": len(records)},
        }


class _FakeIndexingPipeline:
    def __init__(self) -> None:
        self.run_calls = 0

    def run(self, video_path: str, *, output_dir=None, write_json=True, preprocessed=None):
        self.run_calls += 1
        shots, records = preprocessed
        return {
            "video_path": video_path,
            "summary": {"shot_count": len(shots), "keyframe_count": len(records)},
            "cache": {"video_path": video_path, "keyframe_count": len(records)},
        }


def test_run_pipeline_reuses_preprocessed_records_for_indexing(tmp_path: Path) -> None:
    video_path = tmp_path / "demo_manifest_video.mp4"
    video_path.write_bytes(b"fake-video-content")

    processing = _FakeProcessingPipeline()
    indexing = _FakeIndexingPipeline()
    manifest = run_pipeline(
        str(video_path),
        index_to_stores=True,
        processing_pipeline=processing,
        indexing_pipeline=indexing,
    )

    assert processing.process_calls == 1
    assert indexing.run_calls == 1
    assert manifest["summary"]["keyframe_count"] == 1
    assert manifest["indexed"]["summary"]["keyframe_count"] == 1
    assert manifest["cache"]["keyframe_count"] == 1

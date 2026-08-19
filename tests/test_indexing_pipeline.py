from pathlib import Path

from src.indexing.pipeline import VideoIndexingPipeline
from src.schemas.video import KeyframeRecord, ShotRecord


class _StubPreprocessor:
    def process(self, video_path: str):
        video_id = Path(video_path).stem
        shots = [
            ShotRecord(
                video_id=video_id,
                shot_id="shot_0",
                start_time=0.0,
                end_time=1.0,
                start_frame=0,
                end_frame=8,
            )
        ]
        records = [
            KeyframeRecord(
                video_id=video_id,
                frame_id=0,
                timestamp=0.0,
                image_ref=f"{video_path}#frame=0",
                ocr="red shirt",
                caption="scene with red shirt",
                asr="red shirt appears",
                clip_embedding=[1.0, 0.0, 0.0],
                siglip2_embedding=[1.0, 0.0, 0.0],
                metadata={"source": "stub"},
            )
        ]
        return shots, records


def test_video_indexing_pipeline_runs_with_stubbed_preprocessor() -> None:
    pipeline = VideoIndexingPipeline(preprocessor=_StubPreprocessor())
    shots, records = pipeline.index_video("demo_index_video.mp4")

    assert shots
    assert records
    assert pipeline.cached_video("demo_index_video.mp4") is not None
    assert pipeline.vector_index.search(records[0].clip_embedding or [1.0, 0.0, 0.0], 1, field="clip_embedding")
    assert pipeline.text_index.search("red shirt", 1, fields=("caption", "ocr"))


def test_video_indexing_pipeline_run_accepts_preprocessed_records() -> None:
    pipeline = VideoIndexingPipeline(preprocessor=_StubPreprocessor())
    shots, records = _StubPreprocessor().process("demo_index_video.mp4")

    manifest = pipeline.run("demo_index_video.mp4", write_json=False, preprocessed=(shots, records))

    assert manifest["summary"]["shot_count"] == 1
    assert manifest["summary"]["keyframe_count"] == 1
    assert manifest["cache"]["video_id"] == "demo_index_video"

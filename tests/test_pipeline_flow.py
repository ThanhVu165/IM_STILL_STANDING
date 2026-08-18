from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.video_processor import AICVideoPipeline


def test_aic_pipeline_exposes_standard_stage_order() -> None:
    pipeline = AICVideoPipeline()
    assert pipeline.stages[0].startswith("AutoShot")
    assert "sample every 8 frames" in pipeline.stages[1]
    assert "relative L2" in pipeline.stages[3]
    assert "Whisper ASR" in pipeline.stages[9]
    assert "multimodal keyframe records" in pipeline.stages[-1]


def test_aic_pipeline_run_creates_manifest_for_real_video(tmp_path: Path) -> None:
    video_path = tmp_path / "demo_manifest_video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for _ in range(20):
        frame = np.full((32, 32, 3), 64, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    manifest = AICVideoPipeline().run(str(video_path), output_dir=tmp_path / "artifacts")

    assert manifest["summary"]["keyframe_count"] > 0
    assert manifest["stages"]
    assert manifest["summary"]["shot_count"] > 0
    assert Path(manifest["manifest_path"]).exists()

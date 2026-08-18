from pathlib import Path

import cv2
import numpy as np

from src.indexing.pipeline import VideoIndexingPipeline


def test_video_indexing_pipeline_runs_end_to_end() -> None:
    video_path = Path("demo_index_video.mp4")
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for _ in range(20):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    pipeline = VideoIndexingPipeline()
    shots, records = pipeline.index_video(str(video_path))

    assert shots
    assert records
    assert pipeline.cached_video(str(video_path)) is not None
    assert pipeline.milvus.search(records[0].clip_embedding or [1.0, 1.0, 1.0], 1, field="clip_embedding")
    assert pipeline.elasticsearch.search("scene", 1, fields=("caption", "ocr"))

    video_path.unlink(missing_ok=True)

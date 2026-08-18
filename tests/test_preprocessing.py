from pathlib import Path

import cv2

from src.preprocessing.video_processor import VideoPreprocessor


def test_video_preprocessor_runs_end_to_end() -> None:
    path = Path("demo_video.mp4")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for _ in range(20):
        frame = 255 * (1 - __import__("numpy").ones((32, 32, 3), dtype="uint8"))
        writer.write(frame)
    writer.release()

    preprocessor = VideoPreprocessor()
    shots, records = preprocessor.process(str(path))

    assert shots
    assert records
    assert all(record.clip_embedding for record in records)
    assert all(record.siglip2_embedding for record in records)
    assert all(record.asr for record in records)

    path.unlink(missing_ok=True)

#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import json
from src.schemas.video import KeyframeRecord
from src.retrieval.pipeline import VideoRetrievalPipeline

# Adjust these to a single example video present in your bundle
DATA_ROOT = Path(r"C:\IMLANGLAVANG\IM_STILL_STANDING\data")
VIDEO_ID = "L21_V001"

keyframe_dir = DATA_ROOT / "processed" / "keyframes" / VIDEO_ID
emb_file = DATA_ROOT / "processed" / "embeddings" / "clip" / f"{VIDEO_ID}.npy"

records = []
if keyframe_dir.exists():
    frame_files = sorted([p for p in keyframe_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
    emb = None
    if emb_file.exists():
        try:
            arr = np.load(str(emb_file))
            emb = arr
        except Exception as e:
            print('Failed to load embeddings:', e)
    for idx, frame in enumerate(frame_files[:200]):
        rec = KeyframeRecord(
            video_id=VIDEO_ID,
            frame_id=int(frame.stem) if frame.stem.isdigit() else idx,
            timestamp=float(frame.stem)/30.0 if frame.stem.isdigit() else float(idx)/30.0,
            image_ref=str(frame),
            metadata={"source":"organizer-keyframes"},
        )
        if emb is not None and emb.ndim==2 and idx < len(emb):
            rec.clip_embedding = [float(v) for v in emb[idx]]
        records.append(rec)

if not records:
    raise SystemExit('No sample records found; check DATA_ROOT and VIDEO_ID')

pipeline = VideoRetrievalPipeline(records=records, data_root=DATA_ROOT)
results = pipeline.query_frames('một câu truy vấn', top_k=10)
print(json.dumps(results, ensure_ascii=False, indent=2))

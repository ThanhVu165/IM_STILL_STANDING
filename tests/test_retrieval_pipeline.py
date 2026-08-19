from src.retrieval.pipeline import VideoRetrievalPipeline
from src.schemas.video import KeyframeRecord


def test_video_retrieval_pipeline_uses_lexical_match_for_text_queries() -> None:
    records = [
        KeyframeRecord(
            video_id="L21_V001",
            frame_id=1,
            timestamp=1.0,
            image_ref="/tmp/frame_1.jpg",
            ocr="red shirt",
            caption="person wearing a red shirt in the studio",
            asr="a red shirt appears on camera",
            clip_embedding=[1.0, 0.0, 0.0],
            siglip2_embedding=[1.0, 0.0, 0.0],
            metadata={"scene": "studio"},
        ),
        KeyframeRecord(
            video_id="L21_V001",
            frame_id=2,
            timestamp=2.0,
            image_ref="/tmp/frame_2.jpg",
            ocr="blue car",
            caption="a blue car moves on the road",
            asr="the route continues",
            clip_embedding=[0.0, 1.0, 0.0],
            siglip2_embedding=[0.0, 1.0, 0.0],
            metadata={"scene": "road"},
        ),
    ]

    pipeline = VideoRetrievalPipeline(records=records)
    results = pipeline.search("red shirt", top_k=5)

    assert results
    assert results[0].video_id == "L21_V001"
    assert results[0].frame_id == 1


def test_video_retrieval_pipeline_supports_temporal_reranking() -> None:
    records = [
        KeyframeRecord(
            video_id="L21_V001",
            frame_id=10,
            timestamp=10.0,
            image_ref="/tmp/scene_a.jpg",
            ocr="person enters stage",
            caption="person enters stage",
            asr="person enters stage",
            clip_embedding=[1.0, 0.0, 0.0],
            siglip2_embedding=[1.0, 0.0, 0.0],
        ),
        KeyframeRecord(
            video_id="L21_V001",
            frame_id=11,
            timestamp=11.0,
            image_ref="/tmp/scene_b.jpg",
            ocr="person speaks to the camera",
            caption="person speaks to the camera",
            asr="person speaks to the camera",
            clip_embedding=[0.8, 0.2, 0.0],
            siglip2_embedding=[0.8, 0.2, 0.0],
        ),
        KeyframeRecord(
            video_id="L21_V001",
            frame_id=12,
            timestamp=12.0,
            image_ref="/tmp/scene_c.jpg",
            ocr="person leaves stage",
            caption="person leaves stage",
            asr="person leaves stage",
            clip_embedding=[0.2, 0.8, 0.0],
            siglip2_embedding=[0.2, 0.8, 0.0],
        ),
    ]

    pipeline = VideoRetrievalPipeline(records=records)
    results = pipeline.search("person speaks to the camera", previous_query="person enters stage", next_query="person leaves stage", top_k=3)

    assert results
    assert results[0].video_id == "L21_V001"
    assert results[0].frame_id == 11


def test_video_retrieval_pipeline_can_load_persisted_index(tmp_path) -> None:
    records = [
        KeyframeRecord(
            video_id="L21_V001",
            frame_id=1,
            timestamp=1.0,
            image_ref="/tmp/frame_1.jpg",
            ocr="red shirt",
            caption="person wearing a red shirt",
            asr="red shirt",
            clip_embedding=[1.0, 0.0, 0.0],
            siglip2_embedding=[1.0, 0.0, 0.0],
            metadata={"scene": "studio"},
        ),
    ]

    data_root = tmp_path / "data"
    index_root = data_root / "indexes"
    pipeline = VideoRetrievalPipeline(records=records, data_root=data_root, index_root=index_root)
    assert (index_root / "video_keyframes.npy").exists()
    assert (index_root / "video_keyframes.sqlite").exists()

    reloaded = VideoRetrievalPipeline(data_root=data_root, index_root=index_root, load_index_only=True)
    results = reloaded.search("red shirt", top_k=5)

    assert results
    assert results[0].video_id == "L21_V001"
    assert results[0].frame_id == 1

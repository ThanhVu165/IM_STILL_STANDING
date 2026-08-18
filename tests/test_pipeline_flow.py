from src.preprocessing.video_processor import AICVideoPipeline


def test_aic_pipeline_exposes_standard_stage_order() -> None:
    pipeline = AICVideoPipeline()
    assert pipeline.stages[0].startswith("AutoShot")
    assert "sample every 8 frames" in pipeline.stages[1]
    assert "relative L2" in pipeline.stages[3]
    assert "Whisper ASR" in pipeline.stages[9]
    assert "multimodal keyframe records" in pipeline.stages[-1]

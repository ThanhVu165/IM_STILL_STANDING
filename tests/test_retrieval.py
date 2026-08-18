from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.temporal import apply_temporal_rerank
from src.schemas.retrieval import RetrievalResult


def test_rrf_uses_rank_positions() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "d", "e"]], k=60)
    assert fused[0][0] == "a"
    assert fused[1][0] == "b"
    assert fused[0][1] > fused[1][1]


def test_temporal_rerank_applies_same_video_boost() -> None:
    current = [
        RetrievalResult(video_id="v1", frame_id=10, score=0.5, source="clip"),
        RetrievalResult(video_id="v2", frame_id=7, score=0.7, source="clip"),
    ]
    previous = [RetrievalResult(video_id="v1", frame_id=9, score=0.2, source="clip")]
    next_results = [RetrievalResult(video_id="v1", frame_id=11, score=0.3, source="clip")]

    reranked = apply_temporal_rerank(current, previous, next_results)

    assert reranked[0].video_id == "v1"
    assert reranked[0].final_score == 1.0
    assert reranked[0].previous_score == 0.2
    assert reranked[0].next_score == 0.3

"""Temporal reranking for sequential and same-video retrieval candidates."""

from __future__ import annotations

from collections.abc import Sequence

from src.preprocessing.video_pipeline import same_video_temporal_boost
from src.schemas.retrieval import RetrievalResult, TemporalCandidate


def _best_score_by_video(results: Sequence[RetrievalResult]) -> dict[str, float]:
    best: dict[str, float] = {}
    for result in results:
        best[result.video_id] = max(best.get(result.video_id, float("-inf")), result.score)
    return best


def apply_temporal_rerank(
    current_results: Sequence[RetrievalResult],
    previous_results: Sequence[RetrievalResult] | None = None,
    next_results: Sequence[RetrievalResult] | None = None,
) -> list[TemporalCandidate]:
    """Boost current candidates using best same-video support from previous/next subqueries."""
    previous_by_video = _best_score_by_video(previous_results or [])
    next_by_video = _best_score_by_video(next_results or [])

    reranked: list[TemporalCandidate] = []
    for result in current_results:
        prev_score = previous_by_video.get(result.video_id, 0.0)
        next_score = next_by_video.get(result.video_id, 0.0)
        final_score = same_video_temporal_boost(result.score, prev_score, next_score)
        reranked.append(
            TemporalCandidate(
                video_id=result.video_id,
                frame_id=result.frame_id,
                current_score=result.score,
                previous_score=prev_score,
                next_score=next_score,
                final_score=final_score,
                timestamp=result.timestamp,
            )
        )

    reranked.sort(key=lambda item: item.final_score, reverse=True)
    return reranked

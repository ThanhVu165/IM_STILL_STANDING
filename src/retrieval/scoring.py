"""Preliminary-round scoring helpers for TKIS, Q&A, and TRAKE."""

from __future__ import annotations

import re
from collections.abc import Sequence


def _normalize_answer(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
    return re.sub(r"[^\w\s]", "", cleaned)


def rscore_tkis(*, video_id: str, frame_id: int, gt_video_id: str, start_frame: int, end_frame: int) -> float:
    """Binary match for TKIS: video must match and frame must lie in [start_frame, end_frame]."""
    if str(video_id) != str(gt_video_id):
        return 0.0
    return 1.0 if int(start_frame) <= int(frame_id) <= int(end_frame) else 0.0


def rscore_qa(
    *,
    video_id: str,
    frame_id: int,
    answer: str,
    gt_video_id: str,
    start_frame: int,
    end_frame: int,
    gt_answer: str,
) -> float:
    """Binary match for Q&A: TKIS condition plus normalized semantic answer equality."""
    if rscore_tkis(
        video_id=video_id,
        frame_id=frame_id,
        gt_video_id=gt_video_id,
        start_frame=start_frame,
        end_frame=end_frame,
    ) == 0.0:
        return 0.0
    return 1.0 if _normalize_answer(answer) == _normalize_answer(gt_answer) else 0.0


def rscore_trake(
    *,
    video_id: str,
    frame_ids: Sequence[int],
    gt_video_id: str,
    gt_intervals: Sequence[tuple[int, int]],
) -> float:
    """TRAKE score: 0 for wrong video; otherwise matched-event ratio against event intervals."""
    if str(video_id) != str(gt_video_id):
        return 0.0
    if not gt_intervals:
        return 0.0

    matches = 0
    for (start_frame, end_frame), frame_id in zip(gt_intervals, frame_ids):
        if int(start_frame) <= int(frame_id) <= int(end_frame):
            matches += 1
    return float(matches) / float(len(gt_intervals))


def top_k_rscore(r_scores: Sequence[float], k: int) -> float:
    """Top-k R@k value: max R-score in first k ranked answers."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not r_scores:
        return 0.0
    return max(float(score) for score in r_scores[:k])


def final_score(r_scores: Sequence[float], *, cutoffs: Sequence[int] = (1, 5, 20, 50, 100)) -> float:
    """Final preliminary score: mean R@k over the configured rank cutoffs."""
    if not cutoffs:
        raise ValueError("cutoffs must not be empty")
    values = [top_k_rscore(r_scores, cutoff) for cutoff in cutoffs]
    return sum(values) / float(len(values))

from src.retrieval.scoring import final_score, rscore_qa, rscore_tkis, rscore_trake, top_k_rscore


def test_tkis_rscore_matches_interval_rule() -> None:
    assert rscore_tkis(video_id="L01_V001", frame_id=505, gt_video_id="L01_V001", start_frame=500, end_frame=510) == 1.0
    assert rscore_tkis(video_id="L01_V001", frame_id=600, gt_video_id="L01_V001", start_frame=500, end_frame=510) == 0.0
    assert rscore_tkis(video_id="L02_V003", frame_id=505, gt_video_id="L01_V001", start_frame=500, end_frame=510) == 0.0


def test_qa_rscore_requires_semantic_answer_match() -> None:
    assert (
        rscore_qa(
            video_id="L05_V005",
            frame_id=888,
            answer="mau xanh",
            gt_video_id="L05_V005",
            start_frame=800,
            end_frame=900,
            gt_answer="Mau xanh.",
        )
        == 1.0
    )
    assert (
        rscore_qa(
            video_id="L05_V005",
            frame_id=888,
            answer="mau trang",
            gt_video_id="L05_V005",
            start_frame=800,
            end_frame=900,
            gt_answer="mau xanh",
        )
        == 0.0
    )


def test_trake_rscore_is_ratio_and_zero_on_wrong_video() -> None:
    intervals = [(95, 105), (145, 155), (195, 205), (245, 255)]
    assert rscore_trake(video_id="L10_V010", frame_ids=[101, 156, 203, 251], gt_video_id="L10_V010", gt_intervals=intervals) == 0.75
    assert rscore_trake(video_id="L11_V001", frame_ids=[101, 156, 203, 251], gt_video_id="L10_V010", gt_intervals=intervals) == 0.0


def test_final_score_uses_r_at_cutoffs() -> None:
    r_scores = [0.5, 0.2, 0.8] + [0.1] * 97
    assert top_k_rscore(r_scores, 1) == 0.5
    assert top_k_rscore(r_scores, 5) == 0.8
    assert final_score(r_scores) == 0.74

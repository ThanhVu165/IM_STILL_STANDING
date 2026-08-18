from src.schemas.agent import AgentAction, FeedbackRecord
from src.schemas.answers import RankedAnswer
from src.schemas.common import external_to_internal_frame, internal_to_external_frame
from src.schemas.evidence import EvidenceRecord
from src.schemas.retrieval import RetrievalResult, TemporalCandidate
from src.schemas.submission import QASubmissionRecord, TKISSubmissionRecord, TRAKESubmissionRecord
from src.schemas.video import KeyframeRecord, ShotRecord, VideoRecord


def test_frame_numbering_round_trip() -> None:
    for internal in (0, 1, 10, 100):
        assert external_to_internal_frame(internal_to_external_frame(internal)) == internal


def test_frame_numbering_rejects_invalid_values() -> None:
    import pytest

    with pytest.raises(ValueError):
        internal_to_external_frame(-1)
    with pytest.raises(ValueError):
        external_to_internal_frame(0)


def test_core_schemas_can_be_constructed() -> None:
    video = VideoRecord(video_id="v1")
    shot = ShotRecord(video_id="v1", shot_id="s1", start_time=0.0, end_time=1.0)
    keyframe = KeyframeRecord(video_id="v1", frame_id=1, timestamp=0.0, image_ref="frame.jpg")
    result = RetrievalResult(video_id="v1", frame_id=1, score=0.9, source="clip")
    temporal = TemporalCandidate(
        video_id="v1",
        frame_id=1,
        current_score=0.9,
        previous_score=0.2,
        next_score=0.3,
        final_score=1.4,
    )
    evidence = EvidenceRecord(video_id="v1", frame_id=1, reason="visual match")
    answer = RankedAnswer(query_id="q1", rank=1, video_id="v1", frame_id=1, evidence=[evidence])
    action = AgentAction(step_id=1, tool_name="semantic_search", input={}, status="success")
    feedback = FeedbackRecord(query_id="q1", positive_frame_ids=[1], negative_frame_ids=[], source="user", created_at="2026-01-01T00:00:00Z")
    t = TKISSubmissionRecord(query_id="q1", rank=1, video_id="v1", frame_id=1)
    qa = QASubmissionRecord(query_id="q1", rank=1, video_id="v1", frame_id=1, answer="answer")
    trake = TRAKESubmissionRecord(query_id="q1", rank=1, video_id="v1", frames=[1, 2, 3])

    assert video.video_id == shot.video_id == keyframe.video_id == result.video_id == temporal.video_id
    assert answer.rank == action.step_id == t.rank == qa.rank == trake.rank == 1
    assert feedback.positive_frame_ids == [1]

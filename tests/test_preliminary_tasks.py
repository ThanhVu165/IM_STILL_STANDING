from src.retrieval.pipeline import VideoRetrievalPipeline
from src.retrieval.tasks import PreliminaryTaskRunner, TaskQuery
from src.schemas.video import KeyframeRecord


def test_tkis_task_returns_ranked_answer_and_submission() -> None:
    records = [
        KeyframeRecord(
            video_id="L21_V001",
            frame_id=10,
            timestamp=10.0,
            image_ref="/tmp/frame.jpg",
            ocr="person with red shirt",
            caption="person with red shirt in the studio",
            asr="a red shirt appears",
            clip_embedding=[1.0, 0.0, 0.0],
            siglip2_embedding=[1.0, 0.0, 0.0],
            metadata={"scene": "studio"},
        ),
        KeyframeRecord(
            video_id="L21_V002",
            frame_id=15,
            timestamp=15.0,
            image_ref="/tmp/frame2.jpg",
            ocr="blue car",
            caption="a blue car on the road",
            asr="blue car moves",
            clip_embedding=[0.0, 1.0, 0.0],
            siglip2_embedding=[0.0, 1.0, 0.0],
            metadata={"scene": "road"},
        ),
    ]
    runner = PreliminaryTaskRunner(pipeline=VideoRetrievalPipeline(records=records))
    ranked, submission = runner.run_tkis("q1", "red shirt")

    assert ranked.video_id == "L21_V001"
    assert submission.video_id == "L21_V001"
    assert submission.frame_id == 10


def test_qa_task_returns_answer_and_submission() -> None:
    records = [
        KeyframeRecord(
            video_id="L21_V003",
            frame_id=21,
            timestamp=21.0,
            image_ref="/tmp/frame.jpg",
            ocr="5 people",
            caption="five people are standing in a room",
            asr="there are five people",
            clip_embedding=[1.0, 0.0, 0.0],
            siglip2_embedding=[1.0, 0.0, 0.0],
            metadata={"scene": "room"},
        )
    ]
    runner = PreliminaryTaskRunner(pipeline=VideoRetrievalPipeline(records=records))
    ranked, submission = runner.run_qa("q2", "how many people are in the room")

    assert ranked.answer == "5"
    assert submission.answer == "5"


def test_trake_task_returns_ordered_frames() -> None:
    records = [
        KeyframeRecord(
            video_id="L21_V004",
            frame_id=30,
            timestamp=30.0,
            image_ref="/tmp/a.jpg",
            caption="person enters room",
            asr="person enters room",
            clip_embedding=[1.0, 0.0, 0.0],
            siglip2_embedding=[1.0, 0.0, 0.0],
        ),
        KeyframeRecord(
            video_id="L21_V004",
            frame_id=31,
            timestamp=31.0,
            image_ref="/tmp/b.jpg",
            caption="person speaks",
            asr="person speaks",
            clip_embedding=[0.8, 0.2, 0.0],
            siglip2_embedding=[0.8, 0.2, 0.0],
        ),
        KeyframeRecord(
            video_id="L21_V004",
            frame_id=32,
            timestamp=32.0,
            image_ref="/tmp/c.jpg",
            caption="person exits room",
            asr="person exits room",
            clip_embedding=[0.2, 0.8, 0.0],
            siglip2_embedding=[0.2, 0.8, 0.0],
        ),
    ]
    runner = PreliminaryTaskRunner(pipeline=VideoRetrievalPipeline(records=records))
    ranked, submission = runner.run_trake("q3", ["person enters room", "person speaks", "person exits room"])

    assert ranked.video_id == "L21_V004"
    assert submission.frames == [30, 31, 32]


def test_task_runner_supports_task_query_wrapper() -> None:
    records = [
        KeyframeRecord(
            video_id="L21_V005",
            frame_id=50,
            timestamp=50.0,
            image_ref="/tmp/actual.jpg",
            caption="red shirt on stage",
            asr="red shirt on stage",
            clip_embedding=[1.0, 0.0, 0.0],
            siglip2_embedding=[1.0, 0.0, 0.0],
        )
    ]
    runner = PreliminaryTaskRunner(pipeline=VideoRetrievalPipeline(records=records))
    task = TaskQuery(query_id="q4", task_type="tkis", query="red shirt")
    ranked, submission = runner.run_task(task)

    assert ranked.video_id == "L21_V005"
    assert submission.frame_id == 50

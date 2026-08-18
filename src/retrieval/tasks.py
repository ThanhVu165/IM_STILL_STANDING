"""Task-specific execution layer for the preliminary-round video retrieval tasks."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.retrieval.pipeline import VideoRetrievalPipeline
from src.schemas.answers import RankedAnswer
from src.schemas.evidence import EvidenceRecord
from src.schemas.submission import QASubmissionRecord, TKISSubmissionRecord, TRAKESubmissionRecord


@dataclass(slots=True)
class TaskQuery:
    query_id: str
    task_type: str
    query: str
    previous_query: str | None = None
    next_query: str | None = None
    event_queries: list[str] = field(default_factory=list)
    answer: str | None = None


class PreliminaryTaskRunner:
    """Execute task-specific retrieval aligned with TKIS, Q&A, and TRAKE."""

    def __init__(self, pipeline: VideoRetrievalPipeline | None = None) -> None:
        self.pipeline = pipeline or VideoRetrievalPipeline()

    @staticmethod
    def _summarize_evidence(result: Any, *, query: str) -> EvidenceRecord:
        text = query
        if getattr(result, "metadata", None):
            metadata = result.metadata or {}
            parts = []
            for part in (
                metadata.get("field_hits"),
                metadata.get("secondary_source"),
                str(metadata),
                query,
            ):
                if part is None:
                    continue
                if isinstance(part, (tuple, list, set)):
                    parts.extend(str(item) for item in part)
                else:
                    parts.append(str(part))
            text = " ".join(part for part in parts if part)
        return EvidenceRecord(
            video_id=str(result.video_id),
            frame_id=int(result.frame_id),
            timestamp=result.timestamp,
            reason=f"retrieved for query: {query}",
            signals={"query": query, "source": result.source},
            provenance={"rank": result.rank, "score": result.score},
        )

    @staticmethod
    def _best_candidate(results: Sequence[Any]) -> Any | None:
        if not results:
            return None
        return sorted(results, key=lambda item: item.score, reverse=True)[0]

    @staticmethod
    def _extract_answer(query: str, evidence_text: str) -> str:
        lowered = evidence_text.strip()
        if not lowered:
            return "insufficient evidence"
        cleaned = re.sub(r"\s+", " ", lowered).strip()
        if "what" in query.lower() or "which" in query.lower():
            if len(cleaned) > 160:
                return cleaned[:160].rstrip(" .")
            return cleaned
        if "how many" in query.lower():
            numbers = re.findall(r"\d+", cleaned)
            if numbers:
                return numbers[0]
        if "color" in query.lower() or "colour" in query.lower():
            colors = [token for token in re.findall(r"[A-Za-z]+", cleaned) if token.lower() in {"red", "blue", "green", "black", "white", "yellow", "orange", "purple", "pink", "brown", "gray", "grey"}]
            if colors:
                return colors[0].lower()
        return cleaned[:160].rstrip(" .")

    def _record_for_result(self, result: Any) -> Any | None:
        if hasattr(self.pipeline, "_records_by_video"):
            video_records = self.pipeline._records_by_video.get(str(result.video_id), [])
            for record in video_records:
                if int(record.frame_id) == int(result.frame_id):
                    return record
        for record in getattr(self.pipeline, "_records", []):
            if str(record.video_id) == str(result.video_id) and int(record.frame_id) == int(result.frame_id):
                return record
        return None

    def run_tkis(self, query_id: str, query: str, *, top_k: int = 10) -> tuple[RankedAnswer, TKISSubmissionRecord]:
        candidates = self.pipeline.query(query, top_k=top_k)
        best = self._best_candidate(candidates)
        if best is None:
            raise ValueError(f"No candidate found for TKIS query '{query_id}'")

        evidence = self._summarize_evidence(best, query=query)
        ranked = RankedAnswer(
            query_id=query_id,
            rank=1,
            video_id=str(best.video_id),
            frame_id=int(best.frame_id),
            retrieval_score=best.score,
            evidence=[evidence],
            provenance={"source": best.source, "timestamp": best.timestamp},
        )
        submission = TKISSubmissionRecord(query_id=query_id, rank=1, video_id=str(best.video_id), frame_id=int(best.frame_id))
        return ranked, submission

    def run_qa(self, query_id: str, query: str, *, top_k: int = 10) -> tuple[RankedAnswer, QASubmissionRecord]:
        candidates = self.pipeline.query(query, top_k=top_k)
        best = self._best_candidate(candidates)
        if best is None:
            raise ValueError(f"No candidate found for Q&A query '{query_id}'")

        matched_record = self._record_for_result(best)
        evidence_text = " ".join(
            part for part in (
                matched_record.ocr if matched_record is not None else "",
                matched_record.caption if matched_record is not None else "",
                matched_record.asr if matched_record is not None else "",
                json.dumps(matched_record.metadata or {}, ensure_ascii=False) if matched_record is not None else "",
                str(best.metadata or ""),
            )
            if part
        )
        answer = self._extract_answer(query, evidence_text)
        evidence = self._summarize_evidence(best, query=query)
        ranked = RankedAnswer(
            query_id=query_id,
            rank=1,
            video_id=str(best.video_id),
            frame_id=int(best.frame_id),
            answer=answer,
            retrieval_score=best.score,
            evidence=[evidence],
            provenance={"source": best.source, "timestamp": best.timestamp},
        )
        submission = QASubmissionRecord(query_id=query_id, rank=1, video_id=str(best.video_id), frame_id=int(best.frame_id), answer=answer)
        return ranked, submission

    def run_trake(self, query_id: str, event_queries: Sequence[str], *, top_k: int = 10) -> tuple[RankedAnswer, TRAKESubmissionRecord]:
        if not event_queries:
            raise ValueError(f"TRAKE query '{query_id}' requires at least one event query")

        frames: list[int] = []
        evidence: list[EvidenceRecord] = []
        video_scores: dict[str, float] = defaultdict(float)

        for event_query in event_queries:
            candidates = self.pipeline.query(event_query, top_k=top_k)
            best = self._best_candidate(candidates)
            if best is None:
                continue
            frames.append(int(best.frame_id))
            evidence.append(self._summarize_evidence(best, query=event_query))
            video_scores[str(best.video_id)] = max(video_scores.get(str(best.video_id), 0.0), float(best.score))

        if not frames:
            raise ValueError(f"No valid event results found for TRAKE query '{query_id}'")

        video_id = max(video_scores, key=video_scores.get, default="")
        ranked = RankedAnswer(
            query_id=query_id,
            rank=1,
            video_id=video_id,
            frames=frames,
            retrieval_score=sum(video_scores.values()),
            evidence=evidence,
            provenance={"event_queries": list(event_queries), "video_scores": dict(video_scores)},
        )
        submission = TRAKESubmissionRecord(query_id=query_id, rank=1, video_id=video_id, frames=frames)
        return ranked, submission

    def run_task(self, task: TaskQuery) -> tuple[RankedAnswer, Any]:
        task_type = (task.task_type or "tkis").lower()
        if task_type == "tkis":
            return self.run_tkis(task.query_id, task.query, top_k=10)
        if task_type == "qa":
            return self.run_qa(task.query_id, task.query, top_k=10)
        if task_type == "trake":
            events = task.event_queries or [task.query]
            return self.run_trake(task.query_id, events, top_k=10)
        raise ValueError(f"Unsupported task type: {task.task_type}")

    @staticmethod
    def export_submission(record: Any, *, output_path: str | None = None) -> dict[str, Any] | None:
        payload = {
            "query_id": getattr(record, "query_id", None),
            "rank": getattr(record, "rank", None),
            "video_id": getattr(record, "video_id", None),
        }
        if hasattr(record, "frame_id"):
            payload["frame_id"] = record.frame_id
        if hasattr(record, "frames"):
            payload["frames"] = record.frames
        if hasattr(record, "answer"):
            payload["answer"] = record.answer
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        return payload

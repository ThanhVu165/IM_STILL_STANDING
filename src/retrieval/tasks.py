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
from src.submission.serializer import write_submission_csv


@dataclass(slots=True)
class TaskQuery:
    query_id: str
    task_type: str
    query: str
    previous_query: str | None = None
    next_query: str | None = None
    event_queries: list[str] = field(default_factory=list)
    answer: str | None = None
    top_k: int = 100


class PreliminaryTaskRunner:
    """Execute task-specific retrieval aligned with TKIS, Q&A, and TRAKE."""

    def __init__(self, pipeline: VideoRetrievalPipeline | None = None) -> None:
        self.pipeline = pipeline or VideoRetrievalPipeline()
        self._current_event_maps: list[dict[str, Any]] = []

    @staticmethod
    def _normalize_top_k(top_k: int) -> int:
        return max(1, min(100, int(top_k)))

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

    def _fallback_frame_for_video(self, video_id: str) -> int:
        for event_map in getattr(self, "_current_event_maps", []):
            candidate = event_map.get(video_id)
            if candidate is not None:
                return int(candidate.frame_id)

        video_records = sorted(
            list(getattr(self.pipeline, "_records_by_video", {}).get(video_id, [])),
            key=lambda record: int(record.frame_id),
        )
        if not video_records:
            raise ValueError(f"No indexed keyframe exists for video '{video_id}'")
        return int(video_records[len(video_records) // 2].frame_id)

    def run_tkis_ranked(self, query_id: str, query: str, *, top_k: int = 100) -> list[tuple[RankedAnswer, TKISSubmissionRecord]]:
        limit = self._normalize_top_k(top_k)
        candidates = self.pipeline.query(query, top_k=limit)
        if not candidates:
            raise ValueError(f"No candidate found for TKIS query '{query_id}'")

        ranked_items: list[tuple[RankedAnswer, TKISSubmissionRecord]] = []
        for rank, candidate in enumerate(candidates[:limit], start=1):
            evidence = self._summarize_evidence(candidate, query=query)
            ranked = RankedAnswer(
                query_id=query_id,
                rank=rank,
                video_id=str(candidate.video_id),
                frame_id=int(candidate.frame_id),
                retrieval_score=float(candidate.score),
                evidence=[evidence],
                provenance={"source": candidate.source, "timestamp": candidate.timestamp},
            )
            submission = TKISSubmissionRecord(
                query_id=query_id,
                rank=rank,
                video_id=str(candidate.video_id),
                frame_id=int(candidate.frame_id),
            )
            ranked_items.append((ranked, submission))
        return ranked_items

    def run_tkis(self, query_id: str, query: str, *, top_k: int = 10) -> tuple[RankedAnswer, TKISSubmissionRecord]:
        return self.run_tkis_ranked(query_id, query, top_k=top_k)[0]

    def run_qa_ranked(self, query_id: str, query: str, *, top_k: int = 100) -> list[tuple[RankedAnswer, QASubmissionRecord]]:
        limit = self._normalize_top_k(top_k)
        candidates = self.pipeline.query(query, top_k=limit)
        if not candidates:
            raise ValueError(f"No candidate found for Q&A query '{query_id}'")

        ranked_items: list[tuple[RankedAnswer, QASubmissionRecord]] = []
        for rank, candidate in enumerate(candidates[:limit], start=1):
            matched_record = self._record_for_result(candidate)
            evidence_text = " ".join(
                part
                for part in (
                    matched_record.ocr if matched_record is not None else "",
                    matched_record.caption if matched_record is not None else "",
                    matched_record.asr if matched_record is not None else "",
                    json.dumps(matched_record.metadata or {}, ensure_ascii=False) if matched_record is not None else "",
                    str(candidate.metadata or ""),
                )
                if part
            )
            answer = self._extract_answer(query, evidence_text)
            evidence = self._summarize_evidence(candidate, query=query)
            ranked = RankedAnswer(
                query_id=query_id,
                rank=rank,
                video_id=str(candidate.video_id),
                frame_id=int(candidate.frame_id),
                answer=answer,
                retrieval_score=float(candidate.score),
                evidence=[evidence],
                provenance={"source": candidate.source, "timestamp": candidate.timestamp},
            )
            submission = QASubmissionRecord(
                query_id=query_id,
                rank=rank,
                video_id=str(candidate.video_id),
                frame_id=int(candidate.frame_id),
                answer=answer,
            )
            ranked_items.append((ranked, submission))
        return ranked_items

    def run_qa(self, query_id: str, query: str, *, top_k: int = 10) -> tuple[RankedAnswer, QASubmissionRecord]:
        return self.run_qa_ranked(query_id, query, top_k=top_k)[0]

    def run_trake_ranked(self, query_id: str, event_queries: Sequence[str], *, top_k: int = 100) -> list[tuple[RankedAnswer, TRAKESubmissionRecord]]:
        if not event_queries:
            raise ValueError(f"TRAKE query '{query_id}' requires at least one event query")

        limit = self._normalize_top_k(top_k)
        per_event_by_video: list[dict[str, Any]] = []
        video_support: dict[str, dict[str, float]] = defaultdict(lambda: {"events": 0.0, "score": 0.0})

        for event_query in event_queries:
            candidates = self.pipeline.query(event_query, top_k=limit)
            if not candidates:
                raise ValueError(f"No candidate found for TRAKE event query '{event_query}'")

            event_best_by_video: dict[str, Any] = {}
            for candidate in candidates:
                video_id = str(candidate.video_id)
                current = event_best_by_video.get(video_id)
                if current is None or float(candidate.score) > float(current.score):
                    event_best_by_video[video_id] = candidate
            per_event_by_video.append(event_best_by_video)

            for video_id, candidate in event_best_by_video.items():
                video_support[video_id]["events"] += 1.0
                video_support[video_id]["score"] += float(candidate.score)

        if not video_support:
            raise ValueError(f"No valid event results found for TRAKE query '{query_id}'")

        ranked_videos = sorted(
            video_support.items(),
            key=lambda item: (item[1]["events"], item[1]["score"]),
            reverse=True,
        )[:limit]

        ranked_items: list[tuple[RankedAnswer, TRAKESubmissionRecord]] = []
        self._current_event_maps = per_event_by_video
        for rank, (video_id, stats) in enumerate(ranked_videos, start=1):
            frames: list[int] = []
            evidence: list[EvidenceRecord] = []
            valid_video = True
            for event_index, event_query in enumerate(event_queries):
                matched = per_event_by_video[event_index].get(video_id)
                if matched is None:
                    try:
                        fallback_frame = self._fallback_frame_for_video(video_id)
                    except ValueError:
                        valid_video = False
                        break
                    frames.append(fallback_frame)
                    evidence.append(
                        EvidenceRecord(
                            video_id=video_id,
                            frame_id=fallback_frame,
                            reason=f"fallback frame for event query: {event_query}",
                            signals={"query": event_query, "source": "fallback"},
                            provenance={"event_index": event_index},
                        )
                    )
                    continue

                frames.append(int(matched.frame_id))
                evidence.append(self._summarize_evidence(matched, query=event_query))

            if not valid_video:
                continue

            ranked = RankedAnswer(
                query_id=query_id,
                rank=rank,
                video_id=video_id,
                frames=frames,
                retrieval_score=float(stats["score"]),
                evidence=evidence,
                provenance={"event_queries": list(event_queries), "event_support": int(stats["events"])},
            )
            submission = TRAKESubmissionRecord(
                query_id=query_id,
                rank=rank,
                video_id=video_id,
                frames=frames,
            )
            ranked_items.append((ranked, submission))

        if not ranked_items:
            raise ValueError(f"No valid event results found for TRAKE query '{query_id}'")

        for normalized_rank, (ranked, submission) in enumerate(ranked_items, start=1):
            ranked.rank = normalized_rank
            submission.rank = normalized_rank

        self._current_event_maps = []
        return ranked_items

    def run_trake(self, query_id: str, event_queries: Sequence[str], *, top_k: int = 10) -> tuple[RankedAnswer, TRAKESubmissionRecord]:
        return self.run_trake_ranked(query_id, event_queries, top_k=top_k)[0]

    def run_task_ranked(self, task: TaskQuery, *, top_k: int | None = None) -> list[tuple[RankedAnswer, Any]]:
        task_type = (task.task_type or "tkis").lower()
        limit = self._normalize_top_k(top_k if top_k is not None else task.top_k)
        if task_type == "tkis":
            return self.run_tkis_ranked(task.query_id, task.query, top_k=limit)
        if task_type == "qa":
            return self.run_qa_ranked(task.query_id, task.query, top_k=limit)
        if task_type == "trake":
            events = task.event_queries or [task.query]
            return self.run_trake_ranked(task.query_id, events, top_k=limit)
        raise ValueError(f"Unsupported task type: {task.task_type}")

    def run_task(self, task: TaskQuery) -> tuple[RankedAnswer, Any]:
        ranked_items = self.run_task_ranked(task)
        if not ranked_items:
            raise ValueError(f"No candidate found for task '{task.query_id}'")
        return ranked_items[0]

    @staticmethod
    def export_submission(record: Any, *, output_path: str | None = None) -> dict[str, Any] | None:
        payload = {
            "query_id": getattr(record, "query_id", None),
            "rank": getattr(record, "rank", None),
            "video_id": getattr(record, "video_id", None),
        }
        if isinstance(record, TKISSubmissionRecord):
            payload["frame_id"] = int(record.frame_id)
            if output_path is not None:
                csv_path = write_submission_csv(task_type="tkis", records=[record], output_path=output_path)
                payload["csv_path"] = str(csv_path)
            return payload
        if isinstance(record, QASubmissionRecord):
            payload["frame_id"] = int(record.frame_id)
            payload["answer"] = record.answer
            if output_path is not None:
                csv_path = write_submission_csv(task_type="qa", records=[record], output_path=output_path)
                payload["csv_path"] = str(csv_path)
            return payload
        if isinstance(record, TRAKESubmissionRecord):
            payload["frames"] = list(record.frames)
            if output_path is not None:
                csv_path = write_submission_csv(task_type="trake", records=[record], output_path=output_path, expected_event_count=len(record.frames))
                payload["csv_path"] = str(csv_path)
            return payload
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        return payload

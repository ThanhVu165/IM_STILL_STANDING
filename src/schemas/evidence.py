"""Evidence attached to verified retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceRecord:
    video_id: str
    frame_id: int
    reason: str
    timestamp: float | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

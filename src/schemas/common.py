"""Common schema primitives and frame-numbering helpers."""

from __future__ import annotations


def internal_to_external_frame(frame_index: int) -> int:
    """Convert an explicitly zero-based internal frame index to competition frame numbering."""
    if frame_index < 0:
        raise ValueError("internal frame index must be non-negative")
    return frame_index + 1


def external_to_internal_frame(frame_id: int) -> int:
    """Convert competition frame numbering (starting at 1) to zero-based indexing."""
    if frame_id < 1:
        raise ValueError("external frame_id must be >= 1")
    return frame_id - 1

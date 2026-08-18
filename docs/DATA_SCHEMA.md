# Data Schema

This document defines conceptual schemas. Concrete Python/dataclass/Pydantic implementations may refine field types without changing semantics.

## 1. VideoRecord

```yaml
video_id: string
path: string|null
duration_seconds: float|null
fps: float|null
source: string|null
title: string|null
description: string|null
channel: string|null
date: string|null
metadata: object
```

## 2. ShotRecord

```yaml
video_id: string
shot_id: string
start_time: float
end_time: float
start_frame: integer|null
end_frame: integer|null
```

## 3. KeyframeRecord

```yaml
video_id: string
frame_id: integer
timestamp: float
image_ref: string
shot_id: string|null

clip_embedding: array<float>|null
siglip2_embedding: array<float>|null

ocr: string|null
caption: string|null
asr: string|null

objects: array<object>|null
metadata: object|null
```

## 4. RetrievalResult

```yaml
video_id: string
frame_id: integer
timestamp: float|null
score: float
source: string
rank: integer|null
metadata: object|null
```

Allowed `source` examples:

- `clip`
- `siglip2`
- `ocr`
- `asr`
- `metadata`
- `temporal`
- `fused`
- `reranked`

## 5. TemporalCandidate

```yaml
video_id: string
frame_id: integer
timestamp: float|null
current_score: float
previous_score: float
next_score: float
final_score: float
```

## 6. EvidenceRecord

```yaml
video_id: string
frame_id: integer
timestamp: float|null
reason: string
signals:
  visual: object|null
  ocr: object|null
  asr: object|null
  temporal: object|null
  objects: object|null
provenance: object
```

## 7. RankedAnswer

```yaml
query_id: string
rank: integer
video_id: string
frame_id: integer|null
frames: array<integer>|null
answer: string|null
retrieval_score: float|null
final_score: float|null
evidence: array<EvidenceRecord>|null
provenance: object|null
```

Use either `frame_id` for single-event tasks or `frames` for multi-event temporal tasks. Do not silently populate both unless a schema explicitly requires it.

## 8. AgentAction

```yaml
step_id: integer
tool_name: string
input: object
output: object|null
reason: string|null
latency_ms: float|null
candidate_count_before: integer|null
candidate_count_after: integer|null
status: string
```

## 9. FeedbackRecord

```yaml
query_id: string
positive_frame_ids: array<integer>
negative_frame_ids: array<integer>
source: string
created_at: string
```

## 10. SubmissionRecord

Canonical internal forms:

### TKIS

```yaml
query_id: string
rank: integer
video_id: string
frame_id: integer
```

### Q&A

```yaml
query_id: string
rank: integer
video_id: string
frame_id: integer
answer: string
```

### TRAKE

```yaml
query_id: string
rank: integer
video_id: string
frames: array<integer>
```

## 11. Frame convention

The external competition representation starts at frame 1. Any internal zero-based index must be converted at the serialization boundary and tested.

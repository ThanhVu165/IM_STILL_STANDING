# Task Contracts

## TKIS

### Goal
Find the correct video and one frame inside the valid event interval.

### Internal flow

```text
query
 -> semantic/OCR/ASR/metadata retrieval
 -> candidate video
 -> local temporal search
 -> fine visual verification
 -> final frame
```

### Internal answer

```yaml
query_id: string
rank: integer
video_id: string
frame_id: integer
score: float|null
evidence: object
provenance: object
```

### Notes

- The supplied organizer keyframes are reference samples, not the only valid frames.
- The valid answer is an interval; prefer a safe frame inside the interval.

## Q&A

### Goal
Find the correct video/frame and answer the question semantically.

### Internal flow

```text
query description + question
 -> retrieve video/event
 -> visual/audio/text evidence
 -> reason / count / compare / follow temporal relation
 -> final answer
```

### Internal answer

```yaml
query_id: string
rank: integer
video_id: string
frame_id: integer
answer: string
score: float|null
evidence: object
provenance: object
```

### Notes

- Keep the evidence frame even when answer generation uses an LLM.
- Audio may matter.
- External knowledge may be used during competition when allowed, but the answer still needs a correct video/frame as evidence.

## TRAKE

### Goal
Retrieve the correct video and one semantic keyframe for each ordered event.

### Internal flow

```text
event sequence
 -> split into event queries
 -> independent retrieval
 -> same-video temporal support
 -> temporal reranking
 -> local verification for each event
 -> one frame per event
```

### Internal answer

```yaml
query_id: string
rank: integer
video_id: string
frames:
  - integer
  - integer
  - integer
score: float|null
evidence: object
provenance: object
```

### Notes

- The correct video is a prerequisite for scoring.
- Each event has a separate valid frame interval.
- Event ordering must be preserved.

## Tracking

Tracking is referenced by the organizer training materials as a task type in preliminary rounds. It is treated as an extensible task module rather than one of the three task types defined in the initial AIC 2026 preliminary-round document used for the core pipeline.

Potential internal flow:

```text
query
 -> coarse video retrieval
 -> object detection
 -> object association across frames
 -> trajectory / direction analysis
 -> temporal verification
 -> evidence frame(s)
```

Do not change the official task list or submission schema based solely on this reference; confirm current organizer instructions first.

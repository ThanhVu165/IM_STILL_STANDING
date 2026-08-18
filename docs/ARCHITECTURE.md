# Architecture Contract

## System layers

```text
[Data]
  Raw videos + organizer keyframes + objects + CLIP features + metadata
          |
          v
[Preprocessing]
  AutoShot -> every-8th-frame sampling -> CLIP/L2 keyframe selection
          |
          +--> Qwen2.5-VL OCR
          +--> Qwen2.5-VL caption
          +--> CLIP DFN5B embedding
          +--> SigLIP2 embedding
          +--> Whisper ASR + temporal alignment
          |
          v
[Indexing]
  Milvus / Elasticsearch / Redis
          |
          v
[Query Orchestration]
  query understanding -> routing -> retrieval -> fusion -> reranking
          |
          +--> semantic retrieval
          +--> OCR / metadata retrieval
          +--> ASR retrieval
          +--> image retrieval
          +--> temporal retrieval
          +--> tracking
          |
          v
[Verification]
  nearby frame / QBE / crop / zoom / OCR / object / temporal checks
          |
          v
[Task Output]
  TKIS / Q&A / TRAKE
          |
          v
[Submission]
  ranked answers -> CSV -> validator -> ZIP
```

## Dependency direction

Use this dependency direction:

`tasks -> orchestration -> retrieval/reranking/verification -> indexing -> preprocessing`

Shared schemas/configuration may be depended on by all layers.

Lower layers should not import high-level task orchestration.

## Core interfaces

### KeyframeRecord

Minimum conceptual fields:

```text
video_id
frame_id
timestamp
image_ref
clip_embedding
siglip2_embedding
ocr
caption
asr
shot_id
metadata
objects
```

### RetrievalResult

```text
video_id
frame_id
timestamp
score
source
metadata
```

`source` should identify retrieval provenance such as `clip`, `siglip2`, `ocr`, `asr`, `metadata`, `temporal`, `fused`.

### RankedAnswer

```text
query_id
rank
video_id
frame_id(s)
answer (optional)
evidence
score
provenance
```

Keep `score` and `rank` distinct.

## Reranking contract

Rerankers receive an explicit candidate set and return the same candidate identities with updated scores/order. They should not silently create unrelated candidates unless the interface explicitly allows candidate expansion.

## Temporal contract

Temporal retrieval must preserve event ordering and video identity. For Previous/Current/Next:

1. retrieve each subquery independently;
2. compute best per-video previous/next support;
3. boost current candidates;
4. rerank;
5. optionally inspect neighboring frames to verify actual temporal order.

## Agent contract

The agent may choose tools, but tool execution must be explicit and traceable.

Recommended action record:

```text
step_id
tool_name
input_summary
output_summary
latency
candidate_count_before
candidate_count_after
reason
```

Avoid putting hidden business logic solely in prompts. Deterministic validation, frame semantics, submission serialization, and score calculations belong in code.

## Verification contract

Verification operates on already reduced candidates and should expose why a candidate passed or failed whenever practical.

## Submission contract

Submission serialization is a boundary. Internal objects may be rich; official files should contain exactly the fields required by the current organizer format.

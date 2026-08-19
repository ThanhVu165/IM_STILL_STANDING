# Architecture Contract

## System layers

```text
[Data]
  Organizer-provided keyframes + CLIP .npy + metadata + objects + optional raw videos
          |
          v
[Minimal local artifact layer]
  frames.csv or frame catalog
  -> video_id / frame_id / timestamp / image_ref
  -> CLIP vectors loaded into FAISS
  -> OCR/ASR/caption/metadata loaded into SQLite/FTS
          |
          v
[Query Orchestration]
  query understanding -> routing -> retrieval -> fusion -> reranking
          |
          +--> semantic retrieval (FAISS)
          +--> lexical retrieval (SQLite / FTS)
          +--> temporal retrieval
          +--> metadata/object constraints
          |
          v
[Verification]
  nearby frame / OCR / object / temporal checks
          |
          v
[Task Output]
  TKIS / Q&A / TRAKE
          |
          v
[Submission]
  ranked answers -> CSV -> validator -> ZIP
```

This is the default architecture for the local challenge setup. It intentionally avoids forcing the whole raw organizer archive or heavyweight production backends into the daily workflow.

## Local implementation policy

Use the following policy for this repository:

- default: FAISS for vector search, SQLite/FTS for text search
- optional: Milvus / Elasticsearch only when scale or operational constraints require it
- optional: GPU-backed FAISS only when the local machine has suitable CUDA hardware and profiling shows a real latency benefit
- do not add backend detection logic or pipeline branches that become complexity for the sake of completeness

## GPU vs CPU recommendation

For the organizer-provided artifact format, CPU is enough for the required search quality and speed. The primary cost is not the vector index itself but building and querying a large frame catalog with fusion and reranking.

Use GPU only if:

- the index is large enough that CPU latency becomes the bottleneck
- repeated batch indexing or large-scale nearest-neighbor queries are being run often
- the local machine has CUDA support and enough VRAM to make the speedup worthwhile

Otherwise, `faiss-cpu` is the correct default and keeps the pipeline simpler, cheaper, and easier to reproduce on ordinary machines.

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

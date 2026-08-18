# Project Context — IM_STILL_STANDING

## A. Project goal

Build a practical AIC 2026 multimodal video retrieval system that can go from organizer-provided video/query data to verified, submission-ready answers.

The system is designed around three preliminary-round task types: TKIS, Q&A, and TRAKE. The architecture is intentionally modular and supports human-in-the-loop operation.

## B. Canonical end-to-end pipeline

```text
OFFLINE
Raw Video
 -> AutoShot
 -> shot segmentation
 -> sample every 8 frames
 -> CLIP ViT-L/14-quickgelu candidate embeddings
 -> relative L2 difference filtering (> 0.4)
 -> keyframes
 -> Qwen2.5-VL OCR
 -> Qwen2.5-VL caption / scene description
 -> CLIP DFN5B 1024-d
 -> SigLIP2 1152-d
 -> audio extraction
 -> Whisper timestamped ASR
 -> ASR/keyframe temporal alignment
 -> multimodal keyframe records
 -> Milvus + Elasticsearch + Redis

ONLINE
Query
 -> task/intent understanding
 -> optional query interpretation / expansion
 -> routing
 -> candidate retrieval
 -> fusion / RRF
 -> metadata/OCR/ASR filtering
 -> temporal reranking when relevant
 -> fine-grained reranking
 -> agent reasoning/planning when useful
 -> nearby-frame / query-by-example / OCR / object / crop / zoom verification
 -> optional user feedback / Rocchio refinement
 -> final verification
 -> task-specific output
 -> submission generation
 -> validator
```

## C. Data representation

A searchable keyframe record should conceptually contain:

- `video_id`
- `frame_id`
- `timestamp`
- `image` or image path/reference
- `clip_embedding_1024`
- `siglip2_embedding_1152`
- `ocr`
- `caption`
- `asr`
- `shot_id` or shot metadata where available
- object annotations and organizer metadata when available

## D. Competition-specific semantics

### TKIS

Output shape:

`video_id, frame_id`

A frame is correct when the video is correct and the submitted frame lies inside the allowed ground-truth interval.

### Q&A

Output shape:

`video_id, frame_id, answer`

Correctness requires the correct video, a frame inside the allowed interval, and a semantically matching answer.

### TRAKE

Output shape:

`video_id, frame_id_1, ..., frame_id_n`

The video must be correct. Each event frame is checked against that event's own valid interval; the score is proportional to the number of correctly matched events.

## E. Retrieval architecture

### Semantic retrieval

Encode text/image queries with:

- CLIP DFN5B: 1024 dimensions
- SigLIP2: 1152 dimensions

Search each independently in Milvus and fuse ranked lists using RRF.

### RRF

`RRF(d) = sum_i 1/(k + rank_i(d))`

Use an explicit configurable `k`; the supplied Vortex reference uses values such as 60.

### Lexical / metadata retrieval

Use Elasticsearch for:

- OCR
- caption
- ASR
- metadata

Do not force distinctive OCR/entity queries through semantic embeddings if lexical filtering can directly narrow the candidate set.

### Temporal retrieval

For ordered-event queries, support:

`Qprevious`, `Qcurrent`, `Qnext`

Perform independent retrieval, group by video ID, compute best previous/next scores per video, and boost current candidates.

Vortex reference heuristic:

`S_final(rc) = S(rc) + Smax(rp) + Smax(rn)`

## F. Coarse-to-fine search

The system should reduce the candidate space before expensive models are run.

Typical stages:

```text
large candidate pool
 -> lexical/semantic coarse retrieval
 -> filtering
 -> fusion
 -> reranking
 -> small candidate pool
 -> expensive visual/temporal/LLM verification
```

The exact K values are implementation/configuration choices and must not be hard-coded into the project context as competition rules.

## G. Agent layer

The agent is an orchestrator over existing tools.

Core ideas from Training Session 3:

- Reasoning
- Memory
- Planning
- Action space
- multimodal observation

A ReAct-style loop is an appropriate implementation pattern:

`Reason -> Action -> Observation -> Reason -> ...`

Tool descriptions must explicitly define:

- tool purpose
- inputs
- outputs
- expected behavior
- constraints

Possible tools:

- semantic search
- OCR search
- ASR search
- metadata filtering
- image search
- temporal search
- nearby frames
- query-by-example
- crop/zoom
- object detection
- tracking
- answer generation
- submission builder

## H. Long-video reasoning

The supplied training material references a VideoTool-style strategy for long videos:

- maintain a visible-frame dictionary rather than passing every frame to the multimodal model;
- alternate temporal and spatial tools;
- expand the visible set only when the planner needs additional evidence.

This is a design reference, not a mandatory implementation.

## I. Feedback and refinement

### Query interpretation

The LLM can propose explicit interpretations rather than silently rewriting the query. The user/agent should retain control of which interpretation is used.

### Rocchio

`q_m = alpha*q_0 + beta*mean(C_r) - gamma*mean(C_nr)`

where positive and negative sets are collected from result feedback.

## J. Verification

Final verification should combine only the evidence actually needed for the task:

- temporal neighborhood
- fine-grained visual match
- OCR check
- object check
- trajectory check
- answer evidence

The final evidence must preserve video/frame identity. A generated answer without evidence is not a valid final submission artifact.

## K. Frame semantics

- First displayed frame is frame 1.
- Be explicit about any internal zero-based representation.
- Use presentation-time semantics.
- Approximate mapping may use `frame_index ~= PTS * FPS`, subject to rounding.
- Prefer a frame comfortably inside the valid interval rather than an interval edge.
- Organizer-provided keyframes are reference samples and are not necessarily the only valid frames.

## L. Submission model

Keep a canonical internal ranked-answer representation and only serialize to the organizer's exact CSV/ZIP format at the boundary.

Canonical development records:

### TKIS

`query_id, rank, video_id, frame_id`

### Q&A

`query_id, rank, video_id, frame_id, answer`

### TRAKE

`query_id, rank, video_id, frame_1, ..., frame_n`

The system must retain answer ordering because Top-k ranking affects competition scoring.

## M. Engineering boundaries

### preprocessing/
Responsible for video-to-keyframe and multimodal feature generation.

### indexing/
Responsible for writing/searching Milvus, Elasticsearch, Redis.

### retrieval/
Responsible for individual retrieval branches.

### reranking/
Responsible for fusion and score combination.

### temporal/
Responsible for sequence-aware retrieval/alignment logic.

### verification/
Responsible for local frame exploration and evidence checks.

### agent/
Responsible for routing, planning, tool use, memory, and iterative reasoning.

### tasks/
Responsible for task-specific orchestration for TKIS, Q&A, TRAKE, and future tracking components.

### submission/
Responsible for answer ranking, schema validation, CSV serialization, and ZIP assembly.

## N. Coding-assistant invariants

1. Do not collapse the architecture into a single model call.
2. Do not assume every query uses CLIP + SigLIP2 + RRF.
3. Preserve lexical/OCR/ASR retrieval paths.
4. Preserve temporal retrieval as a separate concern.
5. Keep candidate score and final submission rank separate.
6. Preserve frame numbering semantics.
7. Make external dependencies configurable.
8. Keep expensive verification after candidate reduction.
9. Preserve evidence alongside generated Q&A answers.
10. Keep human-in-the-loop workflows possible.
11. Do not change competition submission semantics without an explicit project decision.
12. Every new module must document its input, output, dependencies, and place in the pipeline.

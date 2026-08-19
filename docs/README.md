# IM_STILL_STANDING

AIC 2026 video retrieval system project.

This repository is being organized around an end-to-end multimodal video retrieval pipeline for the Ho Chi Minh City AI Challenge (AIC) 2026. The project combines the supplied AIC data/baseline, the four AIC 2026 training sessions, and the Vortex system paper as references for system design.

## 1. Competition scope

The AIC 2026 preliminary round specifies three query types:

- **Textual KIS (TKIS)**: retrieve the correct video and a frame inside the valid answer interval.
- **Q&A**: retrieve the relevant video/frame and provide a semantically matching answer.
- **TRAKE**: retrieve the correct video and one semantic keyframe for each event in a temporal event sequence.

A key distinction is that the competition's semantic keyframe is a content-defined moment, not a codec I-frame. For TKIS and Q&A, a valid answer is a frame inside the ground-truth interval. For TRAKE, each event has its own usually short valid interval.

The preliminary-round evaluation allows up to 100 ranked answers per query. Final scoring uses R@1, R@5, R@20, R@50, and R@100, so ranking order matters.

## 2. Design principles

1. Do not force every query through one retrieval method.
2. Use a coarse-to-fine pipeline: cheap filtering/retrieval first, expensive verification later.
3. Combine visual, textual, speech, metadata, and temporal evidence.
4. Treat human-in-the-loop interaction as a valid part of the competition workflow.
5. Use pretrained models and spend engineering effort on retrieval, ranking, verification, and interaction rather than training large models from scratch.
6. The final target is not merely a relevant candidate; it is a submission-ready evidence record with the correct video/frame (and answer for Q&A).

## 3. Offline video-processing pipeline

```text
RAW VIDEO
  -> AutoShot shot detection
  -> sample every 8 frames
  -> CLIP ViT-L/14-quickgelu candidate embeddings
  -> relative L2-difference filtering (retain when rel_diff > 0.4)
  -> KEYFRAMES
  -> Qwen2.5-VL-3B-Instruct OCR
  -> Qwen2.5-VL-3B-Instruct caption / scene description
  -> CLIP DFN5B 1024-d embedding
  -> SigLIP2 1152-d embedding
  -> audio extraction
  -> Whisper timestamped ASR
  -> temporal alignment of ASR to keyframes
  -> multimodal keyframe records
  -> FAISS + SQLite/FTS + Redis
```

Important: the CLIP used for keyframe selection is distinct from the CLIP used for retrieval.

### Keyframe selection

For current embedding `e_current` and the previous retained keyframe embedding `e_prev`:

`rel_diff = ||e_current - e_prev|| / ||e_prev||`

Keep the candidate when `rel_diff > 0.4`; otherwise discard it as too similar.

### Multimodal record

Each keyframe should carry at least:

- video_id
- frame_id
- timestamp
- image/keyframe reference
- CLIP DFN5B 1024-d embedding
- SigLIP2 1152-d embedding
- OCR
- caption / scene description
- timestamp-aligned ASR transcription
- shot information when available

## 4. Indexing

### FAISS (default local backend)

For the current local and competition environment, FAISS is the default vector index. It is lightweight, fast enough for CPU execution, and matches the organizer-provided artifact pattern well: index a CLIP embedding table once, then query top-k candidates with minimal deployment overhead.

Use FAISS when:

- the corpus is already built from organizer-provided keyframes and `.npy` embeddings
- the system must run on a local workstation without dedicated database services
- we want a simpler, deterministic, artifact-first pipeline

### SQLite / FTS (default lexical backend)

Use SQLite (or SQLite FTS) for OCR, captions, ASR, metadata, and object text search. This is sufficient for local search over processed keyframe metadata and is a good replacement for Elasticsearch in the preliminary pipeline.

### Optional production backends

Milvus and Elasticsearch remain valid optional backends, but they are not required for the local project baseline. They are useful only when data scale or operational requirements justify the added complexity.

### Redis

Use Redis as a cache for repeated or expensive retrieval results.

## GPU vs CPU for FAISS

For this project, CPU is the correct default choice unless the local machine has a modern NVIDIA GPU and the indexing/query workload becomes noticeably bottlenecked.

Recommended policy:

- use `faiss-cpu` by default for local development and the challenge setup
- use GPU only when the corpus is large and vector search latency or batch indexing becomes the main bottleneck
- keep the software architecture backend-agnostic so switching to GPU-backed FAISS later is a configuration change, not a redesign

In practice, for organizer-provided keyframes and CLIP features, a CPU FAISS index is usually enough to satisfy the required retrieval speed and keeps the system lightweight enough for everyday team usage.

## 5. Online query pipeline

```text
QUERY
  -> task / intent understanding
  -> optional query interpretation / expansion
  -> query router
  -> candidate retrieval
  -> fusion / RRF
  -> metadata filtering
  -> temporal reranking when needed
  -> fine-grained reranking
  -> agent reasoning / planning when useful
  -> nearby-frame / query-by-example / OCR / object / crop / zoom verification
  -> optional user feedback and Rocchio refinement
  -> final verification
  -> task-specific answer construction
  -> submission generation
  -> validation
```

## 6. Retrieval branches

### Semantic retrieval

For text or image queries, encode with both:

- CLIP DFN5B (1024-d)
- SigLIP2 (1152-d)

Perform independent searches and fuse the ranked lists using Reciprocal Rank Fusion (RRF):

`RRF(d) = sum_i 1 / (k + rank_i(d))`

The Vortex reference uses `k` such as 60 as the RRF constant.

### OCR / metadata retrieval

If a query contains a highly distinctive text/entity cue, use Elasticsearch directly. Example strategy: extract a distinctive term -> OCR search -> candidate frames.

### ASR retrieval

Use timestamp-aligned Whisper transcripts for queries depending on spoken content.

### Temporal retrieval

For sequential queries, decompose into:

- `Qprevious`
- `Qcurrent`
- `Qnext`

Retrieve each independently, then boost current results when the previous/current/next candidates occur in the same video. The Vortex heuristic is:

`S_final(r_c) = S(r_c) + S_max(r_p) + S_max(r_n)`

This is a lightweight heuristic reranking step rather than a full dynamic-programming alignment over the whole database.

## 7. Ranking and coarse-to-fine search

Do not rely on a single vector search.

A practical pattern is:

```text
large candidate set
  -> coarse ranking
  -> metadata / OCR / ASR filtering
  -> smaller candidate set
  -> semantic + visual + temporal reranking
  -> fine verification
```

Possible ranking signals include:

- CLIP similarity
- SigLIP2 similarity
- OCR match
- ASR match
- caption/scene match
- metadata constraints
- temporal consistency
- object/trajectory evidence
- task-specific verification score

Early fusion is more expressive but more expensive; late fusion is faster but may be less precise. The practical design is to retrieve independently, then fuse/rerank on a reduced candidate set.

## 8. Agentic layer

The agent is an orchestration/reasoning layer, not a replacement for the retrieval engine.

### Core capabilities

- **Reasoning**: decide what evidence is still missing.
- **Memory**: retain relevant interaction history and reusable procedure/knowledge.
- **Planning**: choose a sequence of retrieval/verification actions.

### ReAct-style tool use

```text
Reason -> Action/tool -> Observation -> Reason -> ...
```

Tools should expose clear descriptions, input/output contracts, and expected behavior.

Potential tools include:

- semantic_search
- ocr_search
- asr_search
- metadata_filter
- image_search
- temporal_search
- nearby_frames
- query_by_example
- object_detection
- crop_or_zoom
- tracking
- answer_generation
- submission_builder

### Long-video reasoning pattern

Use a spatial-temporal search strategy inspired by the VideoTool case study:

```text
temporal grounding
  -> relevant clip / frames
  -> spatial operation (caption/OCR/object/crop)
  -> frame selection
  -> temporal refinement
  -> final evidence
```

The point is to avoid pushing an entire long video into a single multimodal language-model context.

## 9. Relevance feedback

User feedback serves both:

- expansion of the search space when the current region is wrong;
- narrowing of the search space when the current region is broadly correct.

For Rocchio refinement:

`q_m = alpha*q_0 + beta*mean(C_r) - gamma*mean(C_nr)`

where `C_r` is the positive set and `C_nr` is the negative set.

This is an optional post-query refinement loop.

## 10. Fine-grained verification

After coarse retrieval, inspect local temporal/spatial neighborhoods:

- nearby frames
- temporal navigation
- query-by-example visual search
- OCR re-check
- object detection
- crop/zoom
- tracking/trajectory verification

The objective is to move from a roughly correct candidate to the exact valid frame/event.

## 11. Task-specific finalization

### TKIS

Target:

`video_id + frame_id`

Pipeline:

```text
query -> retrieve candidate video -> localize event -> verify interval -> choose safe frame
```

### Q&A

Target:

`video_id + frame_id + answer`

Pipeline:

```text
query description + question
  -> retrieve video/event
  -> collect visual/audio evidence
  -> reason/verify
  -> generate answer
```

The answer may be Vietnamese or English. Audio may matter.

### TRAKE

Target:

`video_id + frame_id_1 + ... + frame_id_n`

For each event, choose one semantic keyframe inside that event's valid interval. The video must be correct before partial event matching can contribute to the score.

## 12. Frame conventions

- Competition frame numbering starts at frame **1**, not frame 0.
- Use presentation time / displayed-frame semantics rather than decoder-internal ordering.
- Approximate conversion can use `frame_index ~= PTS_time * FPS`, with possible rounding differences.
- Prefer frames comfortably inside the valid interval rather than at the edge.
- The supplied organizer keyframes are reference samples, not necessarily the only valid answer frames.

## 13. Submission workflow

```text
Final ranked answers
  -> per-query CSV
  -> exact query filename matching
  -> submission/ folder
  -> ZIP
  -> organizer Submission Validator
  -> official submission
```

The organizer's preliminary-round format uses one query per file and a CSV/run-list style submission. Video names do not need the `.mp4` suffix.

The repository should keep **ranking order explicit** because the competition evaluates Top-k behavior.

For development, maintain a task-specific canonical representation such as:

```text
TKIS:  query_id, rank, video_id, frame_id
Q&A:   query_id, rank, video_id, frame_id, answer
TRAKE: query_id, rank, video_id, frame_1, ..., frame_n
```

Then convert to the exact organizer-required submission format at export time.

## 14. FiftyOne

FiftyOne is a dataset exploration / visualization baseline, not the complete retrieval architecture.

Useful functions include:

- dataset exploration
- annotation inspection
- visual search/filtering
- embedding similarity / nearest-neighbor exploration
- integration with vector databases such as Milvus, Qdrant, Pinecone, LanceDB
- FAISS-style local similarity workflows

Use it primarily for dataset inspection, debugging, search experimentation, and human verification.

## 15. Engineering rules for AI coding assistants

When changing code in this repository:

1. Preserve the pipeline boundaries above.
2. Keep preprocessing, indexing, retrieval, ranking, verification, agent orchestration, and submission as separate concerns.
3. Do not silently replace one retrieval branch with another.
4. Do not assume every query needs every model.
5. Prefer pretrained models and reusable components.
6. Make frame-numbering conventions explicit in code and tests.
7. Treat the final submission format as a first-class interface, not an afterthought.
8. When adding a module, document its input, output, model/tool, and role in the pipeline.
9. For expensive models, use them on narrowed candidate sets whenever possible.
10. Preserve human-in-the-loop capability even if an agentic mode is added.
11. Distinguish **candidate retrieval score** from **final submission rank**.
12. Never use a model-generated answer as final evidence without retaining the supporting video/frame evidence.

## 16. Recommended repository structure

```text
IM_STILL_STANDING/
├── README.md
├── PROJECT_CONTEXT.md
├── ARCHITECTURE.md
├── TASKS.md
├── AGENT_GUIDE.md
├── DATA_SCHEMA.md
├── SUBMISSION_SPEC.md
├── .gitignore
├── configs/
├── docs/
├── src/
│   ├── preprocessing/
│   ├── indexing/
│   ├── retrieval/
│   ├── reranking/
│   ├── temporal/
│   ├── verification/
│   ├── agent/
│   ├── tasks/
│   └── submission/
├── tests/
├── scripts/
├── notebooks/
└── experiments/
```

## 17. Reference sources used for project context

- AIC 2026 preliminary-round specification supplied to the team.
- AIC 2026 Training Session 1: problem context and video retrieval architecture.
- AIC 2026 Training Session 2: query architecture, technical challenges, ranking/re-ranking, feedback, and interaction.
- AIC 2026 Training Session 3: Agentic AI, reasoning, memory, planning, and multimedia agent case studies.
- AIC 2026 Training Session 4: submission rules, frame conventions, validator, and FiftyOne baseline.
- Supplied `VIDEO-PROCESSING` baseline.
- Supplied `XỬ LÝ CHÍNH CỦA HỆ THỐNG` baseline.
- Vortex: *Multi-Modal Fusion System for Intelligent Video Retrieval*.

This README is the project-level context for coding assistants. Detailed task contracts, schemas, implementation decisions, and development rules should live in the accompanying project-context files.
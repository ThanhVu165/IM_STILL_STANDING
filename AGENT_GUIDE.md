# AGENT GUIDE

This file is the operational context for AI coding assistants working in `IM_STILL_STANDING`.

## 1. Mission

Build and incrementally improve an AIC 2026 multimodal video retrieval system. The system must support offline indexing and online retrieval, verification, task-specific output, and submission generation.

## 2. Source-of-truth hierarchy

When making project decisions, use this order:

1. Current official AIC 2026 organizer rules/data formats.
2. Existing repository code and tests.
3. The project context files in this repository.
4. Supplied AIC training materials and supplied baselines.
5. Vortex paper as an implementation/reference baseline.

Do not silently turn a reference design into a competition rule.

## 3. Required conceptual pipeline

```text
Raw video
 -> keyframes
 -> OCR/caption/CLIP/SigLIP2/ASR
 -> aligned multimodal records
 -> Milvus/Elasticsearch/Redis
 -> query understanding
 -> retrieval routing
 -> candidate retrieval
 -> fusion/reranking
 -> temporal reasoning when needed
 -> fine verification
 -> human/agent refinement when useful
 -> task output
 -> submission validation
```

## 4. Important model separation

There are two CLIP roles:

- `CLIP ViT-L/14-quickgelu` with L2 filtering: keyframe selection.
- `CLIP DFN5B` 1024-d: retrieval embedding.

Do not merge these roles in code or documentation without an explicit reason.

## 5. Retrieval routing

A query does not automatically use every module.

Possible routes:

- semantic text/image retrieval;
- OCR/entity retrieval;
- ASR retrieval;
- metadata filtering;
- temporal retrieval;
- tracking/object retrieval;
- combinations of the above.

A router/agent may select multiple routes and fuse them.

## 6. Ranking rules

A retrieval score is not the same thing as final submission rank.

Candidate ranking may use:

- CLIP similarity;
- SigLIP2 similarity;
- OCR/ASR/caption matching;
- metadata constraints;
- temporal support;
- object/trajectory evidence;
- task-specific verification.

RRF is one supported fusion mechanism. Keep the implementation configurable.

## 7. Temporal logic

For sequential queries:

```text
Qprevious -> Rprevious
Qcurrent  -> Rcurrent
Qnext     -> Rnext
```

Then use same-video support and temporal reranking. Do not assume that co-occurrence in one video alone proves temporal ordering; verification may need nearby frames/timestamps.

## 8. Agent rules

The agent is a planner/router/orchestrator.

It may call tools such as:

- semantic_search
- ocr_search
- asr_search
- metadata_filter
- image_search
- temporal_search
- nearby_frames
- query_by_example
- object_detection
- tracking
- crop_or_zoom
- answer_generation
- submission_builder

Prefer deterministic tools for deterministic constraints. Do not put file-format validation, frame indexing, or official serialization rules only into an LLM prompt.

Use an explicit tool contract:

```text
name
purpose
input schema
output schema
cost/latency notes
failure behavior
```

## 9. Agent reasoning safety

The agent must not invent evidence.

For every final answer, retain:

- selected video;
- selected frame(s);
- supporting evidence;
- retrieval provenance.

For Q&A, the textual answer is not enough without a valid evidence frame.

## 10. Feedback

Rocchio-style feedback is optional and should be implemented as a post-query refinement path.

Positive and negative result sets should be explicit. Do not silently change the original query representation.

## 11. Human-in-the-loop

The competition supports manual interaction. Keep it possible to:

- inspect nearby frames;
- select a better frame;
- reject candidates;
- mark positive/negative results;
- edit/refine a query;
- reorder final answers.

Do not design the system so that an autonomous agent is the only route to the correct output.

## 12. Long-video strategy

Do not pass a full long video into a large multimodal model by default.

Prefer:

```text
retrieve -> temporal narrow -> spatial inspect -> temporal refine -> verify
```

This follows the supplied training material's long-video reasoning direction.

## 13. Frame semantics

- External competition frame numbering starts at 1.
- Internal zero-based indexing is allowed only if conversions are explicit and tested.
- Presentation-time semantics matter.
- Prefer safe frames inside valid intervals rather than interval boundaries.

## 14. Submission

Keep canonical internal ranked answers:

### TKIS

`query_id, rank, video_id, frame_id`

### Q&A

`query_id, rank, video_id, frame_id, answer`

### TRAKE

`query_id, rank, video_id, frame_1, ..., frame_n`

Only the submission adapter should translate these objects into organizer-specific CSV/ZIP formatting.

## 15. When changing code

Before editing:

1. Read the nearest module's README/docs/tests.
2. Identify the layer being changed.
3. Identify its input/output contract.
4. Check whether the change affects frame semantics, ranking, task schema, or submission.
5. Add/update tests for deterministic behavior.

After editing:

1. Run focused tests.
2. Run integration tests relevant to the changed pipeline path.
3. Update documentation when architecture or interfaces changed.
4. Keep changes small and traceable.

## 16. Do not do these without explicit project approval

- replace Milvus with another DB globally;
- remove Elasticsearch/OCR/ASR paths;
- remove temporal reasoning;
- replace the multimodal pipeline with a single embedding;
- silently change frame numbering;
- change official submission fields;
- hard-code competition answers into production retrieval logic;
- make external network/API access a required runtime dependency without documenting it;
- introduce model training when a pretrained solution is adequate.

## 17. Preferred implementation style

- Python for ML/data/backend unless the repository establishes another standard.
- Typed schemas for records crossing module boundaries.
- Config-driven model/database parameters.
- Reproducible scripts for preprocessing and indexing.
- Structured logging for retrieval/reranking/agent steps.
- Unit tests for frame conversions, ranking, RRF, temporal scoring, and submission serialization.

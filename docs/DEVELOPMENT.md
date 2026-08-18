# Development Workflow

## 1. Before implementing a feature

Read:

- `README.md`
- `PROJECT_CONTEXT.md`
- `ARCHITECTURE.md`
- `AGENT_GUIDE.md`
- the nearest module documentation/tests

Identify:

- pipeline stage;
- input schema;
- output schema;
- dependencies;
- whether the change affects competition-facing semantics.

## 2. Implementation order for the current system

Recommended implementation sequence:

1. schemas/configuration
2. preprocessing interfaces
3. indexing adapters
4. retrieval adapters
5. fusion/reranking
6. temporal search
7. verification tools
8. task orchestrators
9. submission layer
10. agent orchestration
11. UI/interaction layer

This order prioritizes a deterministic retrieval core before optional agentic automation.

## 3. Retrieval development

Every retrieval branch should return a common candidate representation. This enables later fusion and reranking without branch-specific output handling.

Do not let a model-specific embedding format leak through the whole application. Put model-specific transformations behind adapters.

## 4. Experiments

Experiments should record:

- model/version;
- preprocessing configuration;
- retrieval branch;
- candidate K;
- reranking configuration;
- temporal settings;
- feedback state;
- latency;
- final ranking;
- task and query ID.

## 5. Testing priorities

At minimum, add deterministic tests for:

- frame 0/1 conversions;
- PTS/FPS conversion;
- L2 keyframe threshold;
- RRF scoring;
- temporal same-video scoring;
- Top-k ranking preservation;
- task-specific output schemas;
- CSV serialization;
- ZIP structure.

## 6. Expensive model usage

Use expensive multimodal reasoning only after candidate reduction whenever possible. Prefer retrieval + verification over full-dataset LLM/VLM processing.

## 7. External services

If a feature depends on a paid API or remote service:

- make it optional/configurable;
- document credentials and rate limits;
- provide a deterministic fallback where feasible;
- never commit secrets.

## 8. Documentation rule

When a change modifies the architecture or an interface:

1. update the relevant `.md` file;
2. record the design decision near the code;
3. add/modify tests;
4. keep the README high-level and project-oriented.

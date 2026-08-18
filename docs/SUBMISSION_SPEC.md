# Submission Specification

## 1. Purpose

This file defines the internal-to-external submission boundary. The exact current organizer format always takes precedence over this document.

## 2. Internal canonical format

### TKIS

```text
query_id | rank | video_id | frame_id
```

### Q&A

```text
query_id | rank | video_id | frame_id | answer
```

### TRAKE

```text
query_id | rank | video_id | frame_1 | frame_2 | ... | frame_n
```

## 3. Ranking

Preserve `rank` explicitly. The preliminary-round scoring uses Top-k cutoffs, so ordering is part of the result semantics.

## 4. Validation rules

Before export, validate:

### Common

- query ID is present;
- rank is positive and unique within query;
- video ID is present;
- frame value(s) are present and valid;
- no unexpected task fields are serialized.

### TKIS

- exactly one frame ID.

### Q&A

- exactly one frame ID;
- non-empty answer.

### TRAKE

- exactly the required number of event frames;
- event order preserved.

## 5. Frame numbering

Competition-facing frame numbering starts at 1. Never change this silently.

If internal indexing is zero-based:

```text
external_frame = internal_frame + 1
```

only when the internal representation is explicitly defined as zero-based. Test this conversion.

## 6. CSV / ZIP boundary

The supplied training material describes a submission containing:

```text
submission/
  <query-file-name>.csv
  ...
```

and a final `.zip` archive.

The exact query filename must match the organizer-provided query filename convention.

Video names do not require the `.mp4` suffix according to the supplied training material.

## 7. Validator

Use the organizer-provided Input/Submission Validator whenever available. A local validator should still check deterministic schema conditions before export.

## 8. Do not mix candidate ranking with export

Candidate retrieval may use many scores and evidence fields. Submission output should be generated from the final ranked answer list only.

IM_STILL_STANDING — quick developer notes

This repo splits indexing from query execution for the local video retrieval pipeline.

Build the indexes once (offline):

  python src\build_index.py --data-root data

Run queries repeatedly without rebuilding:

  python src\run_video_retrieval.py "your query text" --data-root data --top-k 10

If you need to force a rebuild when testing, pass --force to build_index.py or use --build-index with run_video_retrieval.py.

Artifact-first video processing (no raw video required if organizer keyframes exist):

  python src\run_video_pipeline.py L21_V001 --data-root data --index

Preliminary task run with submission bundle export (CSV + ZIP + validation):

  python src\run_preliminary_tasks.py --task tkis --query-id q1 --query "red shirt" --top-k 100 --data-root data --submission-dir data\outputs

If your internal frame IDs are zero-based, export with explicit conversion:

  python src\run_preliminary_tasks.py --task tkis --query-id q1 --query "red shirt" --data-root data --submission-dir data\outputs --internal-zero-based

See docs/README.md for full design and development guidance.

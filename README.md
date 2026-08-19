IM_STILL_STANDING — quick developer notes

This repo splits indexing from query execution for the local video retrieval pipeline.

Build the indexes once (offline):

  python src\build_index.py --data-root data

Run queries repeatedly without rebuilding:

  python src\run_video_retrieval.py "your query text" --data-root data --top-k 10

If you need to force a rebuild when testing, pass --force to build_index.py or use --build-index with run_video_retrieval.py.

See docs/README.md for full design and development guidance.

#!/usr/bin/env python3
"""Verify dataset layout and basic consistency.

Usage:
  python scripts/verify_data.py --data-root "C:\\IMLANGLAVANG\\IM_STILL_STANDING\\data"

This script is read-only and intended to be run locally by the dataset owner. It prints
a short report and exits with code 0 for OK (no critical errors) or 2 for fatal problems
(e.g., no frames or embeddings detected at all).
"""

from pathlib import Path
import argparse
import csv
import json
import sqlite3
import subprocess
import sys

try:
    import numpy as np
except Exception:
    np = None


def probe_ffprobe(path: Path):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ], stderr=subprocess.STDOUT)
        return float(out.decode().strip())
    except Exception:
        return None


def list_sqlite_tables(db_path: Path):
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return None


def check_map_csv(csv_path: Path):
    required = {"frame", "frame_id", "timestamp", "image", "image_ref"}
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            header_l = [h.strip().lower() for h in header]
            ok = bool(set(header_l) & required)
            return {"path": str(csv_path), "headers": header_l, "has_required": ok}
    except Exception as e:
        return {"path": str(csv_path), "error": str(e), "has_required": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=False,
                    default=str(Path("data")), help="data root folder")
    args = ap.parse_args()

    root = Path(args.data_root)
    print(f"Data root: {root}")

    # Known layout
    raw_videos = root / "raw" / "videos"
    keyframes = root / "processed" / "keyframes"
    embeddings_clip = root / "processed" / "embeddings" / "clip"
    objects_dir = root / "processed" / "objects"
    metadata_dir = root / "metadata"
    map_keyframes = metadata_dir / "map-keyframes"
    sqlite_files = list(root.rglob("*.sqlite"))

    issues = []
    summary = {}

    # Existence
    summary['raw_videos_exists'] = raw_videos.exists()
    summary['keyframes_exists'] = keyframes.exists()
    summary['embeddings_exists'] = embeddings_clip.exists()
    summary['objects_exists'] = objects_dir.exists()
    summary['metadata_exists'] = metadata_dir.exists()
    summary['map_keyframes_exists'] = map_keyframes.exists()

    print('\nExistence checks:')
    for k, v in summary.items():
        print(f" - {k}: {v}")

    # Map-keyframes CSV checks
    csvs = list(map_keyframes.glob("*.csv")) if map_keyframes.exists() else []
    print(f"\nFound {len(csvs)} map-keyframes CSVs")
    csv_reports = []
    for c in csvs[:200]:
        r = check_map_csv(c)
        csv_reports.append(r)
        if not r.get('has_required'):
            issues.append(f"CSV missing expected headers: {c}")

    # Embeddings checks
    npy_files = list(embeddings_clip.glob("*.npy")) if embeddings_clip.exists() else []
    print(f"Found {len(npy_files)} embeddings (.npy) in embeddings/clip")
    emb_report = []
    for p in npy_files[:200]:
        info = {"file": str(p)}
        if np is None:
            info['note'] = "numpy not available: skipping load"
        else:
            try:
                arr = np.load(str(p))
                info['shape'] = arr.shape
            except Exception as e:
                info['error'] = str(e)
        emb_report.append(info)

    # Keyframes sampling
    missing_images = []
    csv_count_map = {}
    for c in csvs:
        vid = c.stem
        csv_count = 0
        try:
            with c.open('r', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                for i, row in enumerate(reader):
                    csv_count += 1
                    if i < 3:
                        # try to resolve image path
                        img_ref = None
                        for k in ['image', 'image_ref', 'file', 'filename']:
                            if k in row and row[k]:
                                img_ref = row[k]
                                break
                        if img_ref:
                            img_path = (keyframes / vid / img_ref) if (keyframes / vid / img_ref).exists() else Path(img_ref)
                            if not img_path.exists():
                                missing_images.append(str(keyframes / vid / img_ref))
            csv_count_map[vid] = csv_count
        except Exception as e:
            issues.append(f"Failed parse CSV {c}: {e}")

    print(f"\nCSV -> sample missing images: {len(missing_images)} (showing up to 10)")
    for m in missing_images[:10]:
        print(' -', m)

    # Objects / metadata checks
    obj_videos = []
    if objects_dir.exists():
        for d in sorted(objects_dir.iterdir())[:200]:
            if d.is_dir():
                files = list(d.glob('*.json'))
                if not files:
                    issues.append(f"objects folder empty for {d.name}")
                else:
                    obj_videos.append((d.name, len(files)))
    print(f"\nChecked objects: {len(obj_videos)} video folders with JSONs")

    meta_bad = []
    if metadata_dir.exists():
        for m in metadata_dir.glob('*.json'):
            try:
                j = json.loads(m.read_text(encoding='utf-8'))
            except Exception as e:
                meta_bad.append((str(m), str(e)))
    print(f"Metadata JSON parse failures: {len(meta_bad)}")

    # Raw videos probe
    ffprobe_ok = True
    try:
        subprocess.check_output(['ffprobe', '-version'], stderr=subprocess.STDOUT)
    except Exception:
        ffprobe_ok = False
    print(f"\nffprobe available: {ffprobe_ok}")
    if ffprobe_ok and raw_videos.exists():
        sample_videos = list(raw_videos.glob('*.mp4'))[:3]
        for v in sample_videos:
            dur = probe_ffprobe(v)
            print(f" - {v.name}: duration={dur}")

    # SQLite files
    print(f"\nFound {len(sqlite_files)} sqlite files")
    for s in sqlite_files[:20]:
        tables = list_sqlite_tables(s)
        print(f" - {s.name}: tables={tables}")

    # Cross-check counts between CSV and embeddings per-video
    mismatches = []
    if np is not None:
        emb_map = {p.stem: p for p in npy_files}
        for vid, rows in csv_count_map.items():
            if vid in emb_map:
                try:
                    arr = np.load(str(emb_map[vid]))
                    if arr.shape[0] != rows:
                        mismatches.append((vid, rows, arr.shape))
                except Exception:
                    mismatches.append((vid, rows, 'load-error'))
            else:
                mismatches.append((vid, rows, 'missing-embedding'))

    print('\nSummary:')
    print(f" - csv files: {len(csvs)}")
    print(f" - npy files: {len(npy_files)}")
    print(f" - object video folders: {len(obj_videos)}")
    print(f" - metadata jsons parse failures: {len(meta_bad)}")
    print(f" - csv/embedding mismatches found: {len(mismatches)}")
    if mismatches:
        for m in mismatches[:10]:
            print('  *', m)

    if not csvs and not list(keyframes.iterdir() if keyframes.exists() else []):
        print('\nFATAL: No frames CSVs and no keyframes found under keyframes/.')
        sys.exit(2)

    print('\nDone. Please inspect issues list for warnings/errors.')
    if issues:
        print('\nProblems detected (sample):')
        for i in issues[:20]:
            print(' -', i)

    sys.exit(0)


if __name__ == '__main__':
    main()

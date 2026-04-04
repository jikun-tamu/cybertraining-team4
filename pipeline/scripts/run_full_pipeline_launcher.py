#!/usr/bin/env python3
"""Parallel full-run launcher for all 295 LA Fire cells under the canonical data root.

Splits cells into two equal batches and runs them in parallel on cuda:0 and cuda:1.
Each batch calls run_multidate_experiment.py.

Outputs go to `/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run`.
Optional Stage-1 reuse should also point at canonical LA Fire outputs.

Usage
-----
  python scripts/run_full_pipeline_launcher.py
  python scripts/run_full_pipeline_launcher.py --start_from cell_00100  # resume
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

from la_fire_paths import canonical_manifest_path, canonical_run_root

PKG_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = canonical_manifest_path()
OUT_ROOT = canonical_run_root()
REUSE_FROM = None
PYTHON = sys.executable

CHUNK_SIZE = 30   # cells per subprocess call (limits per-process memory / log size)


def all_cells() -> list[str]:
    seen, cells = set(), []
    with open(MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            cid = f"cell_{int(row['cell_id']):05d}"
            if cid not in seen:
                cells.append(cid)
                seen.add(cid)
    return cells


def already_done(cell_id: str) -> bool:
    """True if cell already has aggregated output OR is a known 0-instance cell."""
    cell_dir = OUT_ROOT / cell_id
    agg = cell_dir / "aggregated_predictions.jsonl"
    marker = cell_dir / "shared_base/zero_instances.marker"
    return agg.exists() or marker.exists()


def run_batch(cells: list[str], device: str, log_path: Path) -> int:
    cmd = [
        PYTHON, str(PKG_ROOT / "scripts/run_multidate_experiment.py"),
        "--cells", *cells,
        "--manifest", str(MANIFEST),
        "--out_root", str(OUT_ROOT),
        "--device", device,
    ]
    if REUSE_FROM is not None:
        cmd.extend(["--reuse_stage1_from", str(REUSE_FROM)])
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    with open(log_path, "a") as log:
        result = subprocess.run(cmd, cwd=str(PKG_ROOT), env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start_from", type=str, default=None,
                   help="Resume from this cell ID (skip cells before it).")
    p.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    p.add_argument("--dry_run", action="store_true", help="Print plan, do not execute.")
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    cells = all_cells()

    # Filter to only pending cells
    if args.start_from:
        try:
            start_idx = cells.index(args.start_from)
            cells = cells[start_idx:]
            print(f"Resuming from {args.start_from} (index {start_idx})")
        except ValueError:
            print(f"WARNING: {args.start_from} not found, starting from beginning")

    pending = [c for c in cells if not already_done(c)]
    skipped = len(cells) - len(pending)
    print(f"Total cells: {len(all_cells())}")
    print(f"Already done: {skipped}")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All cells already processed.")
        return

    # Split into two halves (interleaved so each GPU gets a mix of sparse/dense cells)
    batch_a = pending[0::2]   # even indices → cuda:0
    batch_b = pending[1::2]   # odd  indices → cuda:1

    print(f"Batch A (cuda:0): {len(batch_a)} cells")
    print(f"Batch B (cuda:1): {len(batch_b)} cells")

    if args.dry_run:
        print(f"\n-- dry run --")
        print(f"Batch A first 5: {batch_a[:5]}")
        print(f"Batch B first 5: {batch_b[:5]}")
        return

    # Launch two parallel streams, each processing CHUNK_SIZE cells at a time
    chunks_a = list(chunked(batch_a, args.chunk_size))
    chunks_b = list(chunked(batch_b, args.chunk_size))
    max_chunks = max(len(chunks_a), len(chunks_b))

    print(f"\nStarting parallel run: {max_chunks} chunk rounds × up to 2 GPUs")
    print(f"Log dir: {log_dir}")
    print("Each chunk log: logs/batch_a_chunk_NNN.log, logs/batch_b_chunk_NNN.log")

    t0 = time.time()
    for chunk_idx in range(max_chunks):
        procs = []

        for batch_label, chunks, gpu_idx in [
            ("a", chunks_a, "0"),
            ("b", chunks_b, "1"),
        ]:
            if chunk_idx >= len(chunks):
                continue
            chunk = chunks[chunk_idx]
            # Skip cells already done (in case of partial prior run)
            chunk = [c for c in chunk if not already_done(c)]
            if not chunk:
                continue

            log_path = log_dir / f"batch_{batch_label}_chunk_{chunk_idx:03d}.log"
            print(f"\n[chunk {chunk_idx}] batch_{batch_label} (CUDA_VISIBLE_DEVICES={gpu_idx}): "
                  f"{len(chunk)} cells → {log_path.name}")

            cmd = [
                PYTHON, str(PKG_ROOT / "scripts/run_multidate_experiment.py"),
                "--cells", *chunk,
                "--manifest", str(MANIFEST),
                "--out_root", str(OUT_ROOT),
                "--device", "cuda:0",   # always cuda:0 within the isolated env
            ]
            if REUSE_FROM is not None:
                cmd.extend(["--reuse_stage1_from", str(REUSE_FROM)])
            # Use CUDA_VISIBLE_DEVICES to isolate GPUs — prevents cross-device tensor errors
            env = {**os.environ, "HF_HUB_OFFLINE": "1",
                   "CUDA_VISIBLE_DEVICES": gpu_idx}
            log_file = open(log_path, "a")
            proc = subprocess.Popen(cmd, cwd=str(PKG_ROOT), env=env,
                                    stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((proc, log_file, batch_label, chunk_idx))

        # Wait for both GPU processes to finish before starting the next chunk round
        for proc, log_file, batch_label, cidx in procs:
            rc = proc.wait()
            log_file.close()
            elapsed = time.time() - t0
            status = "OK" if rc == 0 else f"EXIT={rc}"
            print(f"  [chunk {cidx}] batch_{batch_label} finished: {status}  "
                  f"elapsed={elapsed/60:.1f}min")

    total_elapsed = time.time() - t0
    print(f"\n[done] Full run complete. Total elapsed: {total_elapsed/3600:.1f}h")

    # Final count
    done_now = [c for c in all_cells() if already_done(c)]
    print(f"Cells with outputs: {len(done_now)}/{len(all_cells())}")


if __name__ == "__main__":
    main()

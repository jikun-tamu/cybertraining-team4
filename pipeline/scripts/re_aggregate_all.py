#!/usr/bin/env python3
"""Re-run aggregate_multidate_predictions.py on all completed cells.

Run after any change to aggregate_multidate_predictions.py that needs
to be propagated across all cells without re-running Stage 1 / Stage 2b.

Usage:
    python scripts/re_aggregate_all.py
    python scripts/re_aggregate_all.py --cells cell_00080 cell_00356  # specific cells
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
# Default: authoritative LA fire data at /media/data/building_instance_tamu/la_fire_2025/
DEFAULT_RUN_ROOT = Path("/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run")
SCRIPT   = PKG_ROOT / "scripts/aggregate_multidate_predictions.py"
PYTHON   = sys.executable


def cells_to_reaggregate(run_root: Path, only: list[str] | None) -> list[Path]:
    cells = []
    for cell_dir in sorted(run_root.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("cell_"):
            continue
        if (cell_dir / "shared_base/zero_instances.marker").exists():
            continue
        if not (cell_dir / "dates").exists():
            continue
        if only and cell_dir.name not in only:
            continue
        cells.append(cell_dir)
    return cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT,
                   help="Root directory containing cell_XXXXX dirs (default: LA fire data)")
    p.add_argument("--cells", nargs="*", default=None,
                   help="Re-aggregate only these cells (default: all)")
    args = p.parse_args()

    cells = cells_to_reaggregate(args.run_root, args.cells)
    print(f"Re-aggregating {len(cells)} cells under {args.run_root}...")

    n_ok = n_fail = 0
    not_id_m1b = 0
    not_id_m2b = 0

    for cell_dir in cells:
        out_jsonl = cell_dir / "aggregated_predictions.jsonl"
        out_csv   = cell_dir / "aggregated_predictions.csv"
        cmd = [
            PYTHON, str(SCRIPT),
            "--cell_run_dir", str(cell_dir),
            "--out_jsonl", str(out_jsonl),
            "--out_csv",   str(out_csv),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [FAIL] {cell_dir.name}: {result.stderr[-200:]}")
            n_fail += 1
        else:
            for line in result.stdout.splitlines():
                if "not_identifiable_m1b=" in line:
                    try:
                        not_id_m1b += int(line.split("not_identifiable_m1b=")[1].split("/")[0])
                    except Exception:
                        pass
                if "not_identifiable_m2b=" in line:
                    try:
                        not_id_m2b += int(line.split("not_identifiable_m2b=")[1].split("/")[0])
                    except Exception:
                        pass
            n_ok += 1
            print(f"  [ok] {cell_dir.name}")

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed")
    print(f"Total 'not identifiable' buildings (m1b): {not_id_m1b}")
    print(f"Total 'not identifiable' buildings (m2b): {not_id_m2b}")


if __name__ == "__main__":
    main()

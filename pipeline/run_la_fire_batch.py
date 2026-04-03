#!/usr/bin/env python3
"""Batch runner: apply II_package pipeline over LA fire chips_600m cells.

Selects earliest post-disaster date per cell.
Fixes stale base path in manifest CSV.
Calls run_pipeline.sh for each pre/post pair.
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[1]
PKG_ROOT   = Path(__file__).resolve().parent
LA_FIRE_ROOT = Path("/media/data/building_instance_tamu/la_fire_2025")
MANIFEST   = LA_FIRE_ROOT / "grids/chips_600m_manifest.csv"
CHIPS_ROOT = LA_FIRE_ROOT / "chips"

# Stale base paths in manifest CSV → correct to new location
OLD_PREFIXES = [
    "/media/gisense/xihan/250812_CyberTraining_Team4/data/chips_600m",
    "/media/gisense/xihan/250812_CyberTraining_Team4/data/interim/chips_600m",
    str(REPO_ROOT / "data/interim/chips_600m"),
]
NEW_PREFIX = str(CHIPS_ROOT)


def fix_path(p: str) -> Path:
    for old in OLD_PREFIXES:
        if old in p:
            return Path(p.replace(old, NEW_PREFIX))
    return Path(p)


def load_pairs(manifest: Path) -> list[dict]:
    """Return list of {cell_id, pre, post_date, post_path} sorted by cell."""
    cells = defaultdict(lambda: {"pre": None, "posts": []})
    with open(manifest) as f:
        for row in csv.DictReader(f):
            cid = f"cell_{int(row['cell_id']):05d}"
            ptype = row["type"]
            path = fix_path(row["path"])
            if ptype == "pre":
                cells[cid]["pre"] = path
            elif ptype == "post":
                date = row.get("date", "").strip()
                cells[cid]["posts"].append((date, path))

    pairs = []
    for cid, data in sorted(cells.items()):
        pre = data["pre"]
        posts = sorted(set(data["posts"]))  # deduplicate, sort by date
        if not pre or not posts:
            continue
        # Pick earliest post date
        post_date, post_path = posts[0]
        pairs.append({"cell_id": cid, "pre": pre, "post_date": post_date, "post": post_path})
    return pairs


def run_cell(pair: dict, out_root: Path, device: str, verbose: bool, overwrite: bool) -> bool:
    run_id = f"la_fire_{pair['cell_id']}"
    cmd = [
        "bash", str(PKG_ROOT / "run_pipeline.sh"),
        "--pre_image", str(pair["pre"]),
        "--post_image", str(pair["post"]),
        "--run_id", run_id,
        "--out_root", str(out_root),
        "--stage1_output_style", "notebook",
        "--stage1_tile_size", "512",
        "--stage1_overlap", "64",
        "--stage1_min_size", "30",
        "--stage1_backend", "meta",  # use cached model, avoids HF 401 for transformers backend
        "--stage1_device", device,
        "--stage2a_device", "cuda",
        "--stage2b_device", "cuda",
    ]
    if overwrite:
        cmd.append("--overwrite_run_dir")
    if verbose:
        cmd.append("--verbose")

    import os
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    env_cmd = ["conda", "run", "-n", "geoai_sam", "--no-capture-output"] + cmd
    print(f"\n[{pair['cell_id']}] post_date={pair['post_date']}")
    print(f"  pre:  {pair['pre']}")
    print(f"  post: {pair['post']}")
    result = subprocess.run(env_cmd, cwd=str(PKG_ROOT), env=env)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False
    print(f"  OK → {out_root / run_id}")
    return True


def main():
    p = argparse.ArgumentParser(description="Batch LA-fire II_package runner")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--out_root", type=Path, default=PKG_ROOT / "outputs/driver_runs")
    p.add_argument("--max_cells", type=int, default=None, help="Limit number of cells (for smoke test)")
    p.add_argument("--cells", nargs="+", default=None, help="Specific cell IDs to process (e.g. cell_00065 cell_00074)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    pairs = load_pairs(args.manifest)
    print(f"Found {len(pairs)} cells with pre+post pairs")

    if args.cells:
        pairs = [p for p in pairs if p["cell_id"] in args.cells]
        print(f"  → running {len(pairs)} selected cells: {args.cells}")
    elif args.max_cells:
        pairs = pairs[:args.max_cells]
        print(f"  → running first {len(pairs)} cells (smoke test)")

    # Validate paths before running
    missing = []
    for pair in pairs:
        if not pair["pre"].exists():
            missing.append(str(pair["pre"]))
        if not pair["post"].exists():
            missing.append(str(pair["post"]))
    if missing:
        print(f"\nERROR: {len(missing)} missing paths:")
        for m in missing[:10]:
            print(f"  {m}")
        sys.exit(1)

    args.out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for pair in pairs:
        ok = run_cell(pair, args.out_root, args.device, args.verbose, args.overwrite)
        results.append((pair["cell_id"], ok))

    print("\n=== Batch Summary ===")
    ok_count = sum(1 for _, ok in results if ok)
    print(f"  Passed: {ok_count}/{len(results)}")
    failed = [cid for cid, ok in results if not ok]
    if failed:
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-date inference experiment driver.

Runs Stage-1 (once per cell), Stage-2a (once per cell), then Stage-2b for
EVERY post date, and aggregates predictions across dates using three methods.

Outputs go to --out_root (default: outputs/multidate_experiment/).
Does NOT touch outputs/driver_runs/ (the single-date smoke test outputs).

Pipeline per cell
-----------------
  1. Stage 1 — SAM3 on pre image (once; reuse from --reuse_stage1_from if given)
  2. Shared base artifacts — pre crops, masks (once, using earliest valid post)
  3. Stage 2a — building type / population (once, uses pre crops)
  4. For each post date:
       a. Tile-level quality filter
       b. generate_post_crops_for_date.py → per-date post crops + shared_for_date.csv
       c. infer_stage2_ensemble.py → stage2b_{date}.jsonl
  5. aggregate_multidate_predictions.py → aggregated_predictions.{jsonl,csv}

Usage
-----
  python scripts/run_multidate_experiment.py \\
      --cells cell_00064 cell_00065 cell_00072 cell_00073 cell_00074 \\
              cell_00045 cell_00046 cell_00080 \\
      --manifest data/processed/chips_600m_manifest.csv \\
      --out_root outputs/multidate_experiment \\
      --device cuda:0 \\
      --reuse_stage1_from outputs/driver_runs   # optional
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
PKG_ROOT = Path(__file__).resolve().parents[1]

OLD_CHIPS_PREFIX = "/media/gisense/xihan/250812_CyberTraining_Team4/data/chips_600m"
NEW_CHIPS_PREFIX = str(PKG_ROOT.parent / "data/interim/chips_600m")

PYTHON = sys.executable  # same interpreter that runs this script


# ── Helpers ───────────────────────────────────────────────────────────────────

def fix_path(p: str) -> Path:
    return Path(p.replace(OLD_CHIPS_PREFIX, NEW_CHIPS_PREFIX))


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Return {cell_id: {pre: Path, posts: [(date, Path)]}}."""
    cells: dict[str, dict] = defaultdict(lambda: {"pre": None, "posts": []})
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            cid = f"cell_{int(row['cell_id']):05d}"
            path = fix_path(row["path"])
            if row["type"] == "pre":
                cells[cid]["pre"] = path
            else:
                date = row.get("date", "").strip()
                cells[cid]["posts"].append((date, path))
    # Deduplicate and sort posts
    for cid in cells:
        cells[cid]["posts"] = sorted(set(cells[cid]["posts"]))
    return dict(cells)


def run(cmd: list[str], cwd: Path, env: dict | None = None, label: str = "") -> int:
    """Run a command, stream output to stdout, return exit code."""
    display = label or " ".join(str(c) for c in cmd[:4])
    print(f"\n  [run] {display}")
    cmd_s = [str(c) for c in cmd]
    result = subprocess.run(cmd_s, cwd=str(cwd), env=env)
    if result.returncode != 0:
        print(f"  [FAILED] exit={result.returncode}")
    return result.returncode


def symlink_force(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def infer_tile_id(pre_image: Path) -> str:
    """Mirror the driver's tile_id inference: strip _pre_disaster suffix if present."""
    stem = pre_image.stem
    if stem.endswith("_pre_disaster"):
        return stem[: -len("_pre_disaster")]
    return stem


def resolve_stage1_labels_dir(stage1_dir: Path) -> Path:
    """Return the labels/predictions directory, preferring 'predictions/' (stage1 package)."""
    preds = stage1_dir / "predictions"
    if preds.exists():
        return preds
    return stage1_dir / "labels"


def count_stage1_instances(label_files: list[Path]) -> int:
    """Count instances across label files, handling both SAM3_Final and stage1 package formats."""
    total = 0
    for lf in label_files:
        try:
            doc = json.loads(lf.read_text())
            if "instances" in doc and isinstance(doc["instances"], list):
                total += len(doc["instances"])
            else:
                total += len((doc.get("features", {}) or {}).get("xy", []))
        except Exception:
            pass
    return total


def run_stage1(
    cell_id: str,
    pre_image: Path,
    cell_out: Path,
    device: str,
    reuse_from: Path | None,
    use_stage1_package: bool = False,
) -> tuple[Path, str] | None:
    """Run (or reuse) Stage 1. Returns (stage1_dir, tile_id) or None on failure."""
    stage1_dir = cell_out / "stage1"
    pair_dir = cell_out / "pair_inputs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Tile id and canonical pre-image name
    tile_id = infer_tile_id(pre_image)
    pre_link_name = f"{tile_id}_pre_disaster.png"
    pre_link = pair_dir / pre_link_name
    symlink_force(pre_image, pre_link)

    # Try to reuse existing Stage-1 output
    if reuse_from is not None:
        # Look for outputs/driver_runs/la_fire_{cell_id}/stage1/labels/*.json
        candidate = reuse_from / f"la_fire_{cell_id}" / "stage1"
        label_files = list((candidate / "labels").glob("*_prediction.json")) if candidate.exists() else []
        if label_files:
            print(f"  [stage1] reusing {candidate}")
            stage1_dir.mkdir(parents=True, exist_ok=True)
            # Symlink/copy the labels dir
            labels_dst = stage1_dir / "labels"
            if not labels_dst.exists():
                shutil.copytree(candidate / "labels", labels_dst)
            return stage1_dir, tile_id

    # Run Stage 1 fresh
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if use_stage1_package:
        rc = run(
            [PYTHON, "-m", "sam3_building_identifier",
             "--input-dir", pair_dir,
             "--output-dir", stage1_dir,
             "--max-images", "1",
             "--prompt", "building",
             "--min-size", "30",
             "--batch-size", "1",
             "--device", device,
             "--disaster-type", "pre"],
            cwd=PKG_ROOT, env=env,
            label=f"stage1 {cell_id}",
        )
    else:
        stage1_script = PKG_ROOT / "stage1/SAM3_Final_20260226/scripts/run_sam3_building_infer.py"
        rc = run(
            [PYTHON, stage1_script,
             "--input", pair_dir,
             "--output", stage1_dir,
             "--pattern", pre_link_name,
             "--max-images", "1",
             "--prompt", "building",
             "--min-size", "30",
             "--output-style", "notebook",
             "--batch-size", "1",
             "--device", device,
             "--backend", "meta",
             "--tile-size", "512",
             "--overlap", "64"],
            cwd=PKG_ROOT, env=env,
            label=f"stage1 {cell_id}",
        )
    if rc != 0:
        return None
    return stage1_dir, tile_id


# ── Shared base artifacts (pre crops + masks) ─────────────────────────────────

def run_shared_base(
    cell_id: str,
    tile_id: str,
    stage1_dir: Path,
    pair_dir: Path,
    earliest_post: Path,
    cell_out: Path,
) -> Path | None:
    """Generate shared base artifacts (pre crops, masks, baseline post crops).

    Returns path to shared_instance_samples.csv or None on failure.
    """
    shared_dir = cell_out / "shared_base"
    out_csv = shared_dir / "shared_instance_samples.csv"
    if out_csv.exists():
        print(f"  [shared_base] already exists, skipping ({out_csv})")
        return out_csv

    # Check how many instances Stage 1 found — skip shared_base if 0
    labels_dir = resolve_stage1_labels_dir(stage1_dir)
    label_files = list(labels_dir.glob("*_prediction.json"))
    n_instances = count_stage1_instances(label_files)
    if n_instances == 0:
        print(f"  [shared_base] 0 Stage-1 instances — writing empty marker, skipping")
        shared_dir.mkdir(parents=True, exist_ok=True)
        (shared_dir / "zero_instances.marker").write_text("Stage-1 found 0 building instances.")
        return "EMPTY"

    # Symlink the baseline post image into pair_inputs/
    post_link_name = f"{tile_id}_post_disaster.png"
    post_link = pair_dir / post_link_name
    symlink_force(earliest_post, post_link)

    script = PKG_ROOT / "scripts/generate_shared_instance_subimages.py"
    rc = run(
        [PYTHON, script,
         "--stage1_labels_dir", labels_dir,
         "--pre_images_dir", pair_dir,
         "--post_images_dir", pair_dir,
         "--out_root", shared_dir,
         "--crop_size", "256",
         "--ring_radius_px", "48",
         "--strict_images",
         "--num_workers", "4",
         "--chunk_size", "100",
         "--log_every", "50"],
        cwd=PKG_ROOT,
        label=f"shared_base {cell_id}",
    )
    if rc != 0:
        return None
    return out_csv if out_csv.exists() else None


# ── Stage 2a ──────────────────────────────────────────────────────────────────

def run_stage2a(cell_id: str, shared_csv: Path, cell_out: Path, device: str) -> Path | None:
    stage2a_csv = cell_out / "stage2a_predictions.csv"
    if stage2a_csv.exists():
        print(f"  [stage2a] already exists, skipping ({stage2a_csv})")
        return stage2a_csv

    infer_csv = cell_out / "stage2a_infer_input.csv"
    build_script = PKG_ROOT / "scripts/build_stage2a_infer_csv.py"
    rc = run(
        [PYTHON, build_script,
         "--shared_csv", shared_csv,
         "--out_csv", infer_csv,
         "--crop_col", "pre_crop",
         "--mask_col", "mask_M"],
        cwd=PKG_ROOT, label=f"build_stage2a_csv {cell_id}",
    )
    if rc != 0:
        return None

    infer_script = PKG_ROOT / "scripts/infer_stage2a.py"
    rc = run(
        [PYTHON, infer_script,
         "--input_csv", infer_csv,
         "--ckpt", PKG_ROOT / "models/stage2a/stage2a_best_model.pt",
         "--out_csv", stage2a_csv,
         "--batch_size", "64",
         "--num_workers", "4",
         "--device", device],
        cwd=PKG_ROOT, label=f"stage2a {cell_id}",
    )
    return stage2a_csv if stage2a_csv.exists() else None


# ── Per-date: post crop generation + Stage 2b ─────────────────────────────────

def run_date(
    cell_id: str,
    date: str,
    post_image: Path,
    shared_csv: Path,
    cell_out: Path,
    device: str,
) -> bool:
    date_dir = cell_out / "dates" / date
    date_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = date_dir / f"stage2b_{date}.jsonl"
    if out_jsonl.exists():
        print(f"  [date {date}] already done, skipping")
        return True

    # ── Generate post crops ──────────────────────────────────────────────────
    crop_script = PKG_ROOT / "scripts/generate_post_crops_for_date.py"
    rc = run(
        [PYTHON, crop_script,
         "--base_shared_csv", shared_csv,
         "--post_image", post_image,
         "--date", date,
         "--out_dir", date_dir,
         "--png_compress_level", "1"],
        cwd=PKG_ROOT, label=f"post_crops {cell_id} {date}",
    )
    if rc != 0:
        return False

    shared_for_date = date_dir / "shared_for_date.csv"
    if not shared_for_date.exists():
        print(f"  [date {date}] shared_for_date.csv missing after crop gen")
        return False

    # Check if the tile was rejected by quality filter
    quality_json = date_dir / "quality_metrics.json"
    if quality_json.exists():
        qm = json.loads(quality_json.read_text())
        if not qm.get("tile_quality_ok", True):
            print(f"  [date {date}] TILE REJECTED by quality filter: {qm.get('reason','')}")
            print(f"  [date {date}] Stage-2b will still run (for comparison), but flagged in CSV.")

    # Check if any rows survived (non-empty CSV)
    try:
        with open(shared_for_date) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"  [date {date}] no rows in shared_for_date.csv — skipping Stage-2b")
            return True  # not a failure, just no instances
    except Exception as e:
        print(f"  [date {date}] cannot read shared_for_date.csv: {e}")
        return False

    # ── Stage 2b ensemble inference ──────────────────────────────────────────
    infer_script = PKG_ROOT / "scripts/infer_stage2_ensemble.py"
    rc = run(
        [PYTHON, infer_script,
         "--csv", shared_for_date,
         "--ckpts", ",".join([
             str(PKG_ROOT / "models/stage2b/inference0.7273.pt"),
             str(PKG_ROOT / "models/stage2b/inference0.7066_seed9999.pt"),
             str(PKG_ROOT / "models/stage2b/inference0.7034_seed7777.pt"),
         ]),
         "--weights", "4,3,2",
         "--configs", ",".join([
             str(PKG_ROOT / "configs/stage2b/run019_seed2025_train_config.json"),
             str(PKG_ROOT / "configs/stage2b/seed9999_train_config.json"),
             str(PKG_ROOT / "configs/stage2b/seed7777_train_config.json"),
         ]),
         "--calibration_method", "temperature",
         "--calibration_dirs", ",".join([
             str(PKG_ROOT / "calibration/calibration_run019_r48"),
             str(PKG_ROOT / "calibration/calibration_seed9999_r48"),
             str(PKG_ROOT / "calibration/calibration_seed7777_r48"),
         ]),
         "--out_jsonl", out_jsonl,
         "--batch_size", "64",
         "--num_workers", "4",
         "--device", device,
         "--print_examples", "0",
         "--log_every_steps", "20"],
        cwd=PKG_ROOT, label=f"stage2b {cell_id} {date}",
    )
    return rc == 0


# ── Aggregation ───────────────────────────────────────────────────────────────

def run_aggregation(cell_id: str, cell_out: Path) -> bool:
    agg_script = PKG_ROOT / "scripts/aggregate_multidate_predictions.py"
    out_jsonl = cell_out / "aggregated_predictions.jsonl"
    out_csv = cell_out / "aggregated_predictions.csv"
    rc = run(
        [PYTHON, agg_script,
         "--cell_run_dir", cell_out,
         "--out_jsonl", out_jsonl,
         "--out_csv", out_csv],
        cwd=PKG_ROOT, label=f"aggregate {cell_id}",
    )
    return rc == 0


# ── Cell summary ──────────────────────────────────────────────────────────────

def summarize_cell(cell_id: str, cell_out: Path, all_dates: list[str]) -> dict:
    agg_csv = cell_out / "aggregated_predictions.csv"
    stage2a_csv = cell_out / "stage2a_predictions.csv"
    stage1_summary = cell_out / "stage1/run_summary.json"

    n_instances = 0
    if stage1_summary.exists():
        try:
            s = json.loads(stage1_summary.read_text())
            n_instances = s.get("instances", 0)
        except Exception:
            pass

    n_instances_stage2 = 0
    type_counts: dict = {}
    if stage2a_csv.exists():
        with open(stage2a_csv, newline="") as f:
            rows = list(csv.DictReader(f))
            n_instances_stage2 = len(rows)
            for r in rows:
                t = r.get("pred_type", r.get("type", ""))
                type_counts[t] = type_counts.get(t, 0) + 1

    unstable = 0
    m1_dist: dict = {}
    m2_dist: dict = {}
    m3_dist: dict = {}
    if agg_csv.exists():
        with open(agg_csv, newline="") as f:
            rows = list(csv.DictReader(f))
            for r in rows:
                if r.get("is_unstable", "").lower() == "true":
                    unstable += 1
                for col, dist in [("m1_prob_avg_class", m1_dist),
                                  ("m2_majority_class", m2_dist),
                                  ("m3_quality_filtered_class", m3_dist)]:
                    v = r.get(col, "")
                    if v:
                        dist[v] = dist.get(v, 0) + 1

    # Per-date quality
    date_quality: dict = {}
    for date in all_dates:
        qj = cell_out / "dates" / date / "quality_metrics.json"
        if qj.exists():
            m = json.loads(qj.read_text())
            date_quality[date] = m.get("tile_quality_ok", "?")

    return {
        "cell_id": cell_id,
        "n_instances_stage1": n_instances,
        "n_instances_stage2": n_instances_stage2,
        "stage2a_type_dist": type_counts,
        "n_dates_total": len(all_dates),
        "date_quality": date_quality,
        "n_dates_passed_quality": sum(1 for v in date_quality.values() if v is True),
        "n_unstable_instances": unstable,
        "m1_damage_dist": m1_dist,
        "m2_damage_dist": m2_dist,
        "m3_damage_dist": m3_dist,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Multi-date inference experiment.")
    p.add_argument("--cells", nargs="+", required=True, help="Cell IDs to process.")
    p.add_argument("--manifest", type=Path,
                   default=PKG_ROOT.parent / "data/processed/chips_600m_manifest.csv")
    p.add_argument("--out_root", type=Path,
                   default=PKG_ROOT / "outputs/multidate_experiment")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--reuse_stage1_from", type=Path, default=None,
                   help="Look here for existing la_fire_{cell_id}/stage1/ outputs to reuse.")
    p.add_argument("--use_stage1_package", action="store_true",
                   help="Use stage1/sam3_building_identifier package instead of SAM3_Final script.")
    p.add_argument(
        "--workflow",
        choices=["training", "realworld"],
        default="realworld",
        help=(
            "Workflow mode. "
            "'training': single pre/post pair per cell, no multidate aggregation "
            "(use run_instance_impact_driver.py instead for this). "
            "'realworld': multiple post-disaster dates per cell, quality filtering, "
            "identifiability-first logic, aggregation across valid dates (includes M2b). "
            "Default: realworld."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.workflow == "training":
        print("[workflow] TRAINING mode — single post date, no multidate aggregation")
        print("[workflow] For full single-pair workflow, consider run_instance_impact_driver.py")
    else:
        print("[workflow] REALWORLD mode — multiple post dates, quality filtering, M2b aggregation")

    manifest = load_manifest(args.manifest)
    print(f"Loaded manifest: {len(manifest)} cells total")

    cell_summaries = []
    failed_cells = []

    for cell_id in args.cells:
        if cell_id not in manifest:
            print(f"\n[SKIP] {cell_id} not in manifest")
            continue

        cell_data = manifest[cell_id]
        pre_image = cell_data["pre"]
        posts = cell_data["posts"]  # [(date, path), ...]

        if not pre_image or not pre_image.exists():
            print(f"\n[SKIP] {cell_id}: pre image missing: {pre_image}")
            failed_cells.append(cell_id)
            continue
        if not posts:
            print(f"\n[SKIP] {cell_id}: no post images")
            failed_cells.append(cell_id)
            continue

        print(f"\n{'='*60}")
        print(f"[cell] {cell_id}  pre={pre_image.name}  n_post_dates={len(posts)}")
        print(f"{'='*60}")

        cell_out = args.out_root / cell_id
        cell_out.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Stage 1 ──────────────────────────────────────────────────
        result = run_stage1(
            cell_id, pre_image, cell_out, args.device, args.reuse_stage1_from,
            use_stage1_package=args.use_stage1_package,
        )
        if result is None:
            print(f"[FAIL] {cell_id}: Stage 1 failed")
            failed_cells.append(cell_id)
            continue
        stage1_dir, tile_id = result

        # Check if Stage 1 produced any instances
        s1_labels_dir = resolve_stage1_labels_dir(stage1_dir)
        label_files = list(s1_labels_dir.glob("*_prediction.json"))
        if label_files:
            n_inst = count_stage1_instances(label_files)
            print(f"  [stage1] instances={n_inst}")
            if n_inst == 0:
                print(f"  [stage1] WARNING: 0 instances detected — continuing anyway")

        pair_dir = cell_out / "pair_inputs"

        # ── Step 2: Shared base artifacts ────────────────────────────────────
        # Use earliest post date that actually exists
        earliest_post = None
        for date, path in posts:
            if path.exists():
                earliest_post = path
                break
        if earliest_post is None:
            print(f"[FAIL] {cell_id}: no existing post images")
            failed_cells.append(cell_id)
            continue

        shared_csv = run_shared_base(cell_id, tile_id, stage1_dir, pair_dir, earliest_post, cell_out)
        if shared_csv is None:
            print(f"[FAIL] {cell_id}: shared base artifact generation failed")
            failed_cells.append(cell_id)
            continue

        if shared_csv == "EMPTY":
            print(f"  [skip] {cell_id}: 0 instances — no Stage 2 to run")
            cell_summaries.append({
                "cell_id": cell_id, "n_instances_stage1": 0, "n_instances_stage2": 0,
                "stage2a_type_dist": {}, "n_dates_total": len(posts),
                "date_quality": {}, "n_dates_passed_quality": 0,
                "n_unstable_instances": 0,
                "m1_damage_dist": {}, "m2_damage_dist": {}, "m3_damage_dist": {},
            })
            continue

        # ── Step 3: Stage 2a ─────────────────────────────────────────────────
        stage2a_csv = run_stage2a(cell_id, shared_csv, cell_out, args.device)
        if stage2a_csv is None:
            print(f"  [WARN] {cell_id}: Stage 2a failed — continuing with dates")

        # ── Step 4: Per-date Stage 2b ─────────────────────────────────────────
        if args.workflow == "training":
            # Training mode: use only the first available post date
            use_posts = [(date, path) for date, path in posts if path.exists()][:1]
            if not use_posts:
                print(f"[FAIL] {cell_id}: no existing post images for training mode")
                failed_cells.append(cell_id)
                continue
            print(f"  [training] using single post date: {use_posts[0][0]}")
        else:
            use_posts = posts

        all_dates = [date for date, _ in use_posts]
        date_results = {}
        for date, post_path in use_posts:
            if not post_path.exists():
                print(f"  [date {date}] post image missing: {post_path}")
                date_results[date] = False
                continue
            ok = run_date(cell_id, date, post_path, shared_csv, cell_out, args.device)
            date_results[date] = ok

        n_ok = sum(1 for ok in date_results.values() if ok)
        print(f"\n  [dates] {n_ok}/{len(date_results)} dates succeeded")

        # ── Step 5: Aggregation ───────────────────────────────────────────────
        if args.workflow == "training":
            print(f"  [training] skipping aggregation (single date)")
        else:
            agg_ok = run_aggregation(cell_id, cell_out)
            if not agg_ok:
                print(f"  [WARN] {cell_id}: aggregation failed")

        # ── Summarize ─────────────────────────────────────────────────────────
        summary = summarize_cell(cell_id, cell_out, all_dates)
        cell_summaries.append(summary)
        print(f"\n  [summary] {cell_id}: instances={summary['n_instances_stage1']}  "
              f"unstable={summary['n_unstable_instances']}  "
              f"dates_ok={summary['n_dates_passed_quality']}/{summary['n_dates_total']}")

    # ── Write experiment summary JSON ─────────────────────────────────────────
    exp_summary = {
        "cells_attempted": args.cells,
        "cells_failed": failed_cells,
        "cell_summaries": cell_summaries,
    }
    summary_path = args.out_root / "experiment_summary.json"
    summary_path.write_text(json.dumps(exp_summary, indent=2))
    print(f"\n[done] experiment_summary.json → {summary_path}")
    print(f"[done] outputs → {args.out_root}")

    if failed_cells:
        print(f"\nFailed cells: {failed_cells}")
        sys.exit(1)


if __name__ == "__main__":
    main()

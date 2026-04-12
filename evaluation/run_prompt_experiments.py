#!/usr/bin/env python3
"""
Run SAM3 building detection with multiple prompt strategies on the xView2 test set.
Evaluates each strategy and generates a comparison figure.

Usage:
    conda run -n geoai_sam python evaluation/run_prompt_experiments.py [--device cuda:0]

Prompts tested:
    building          (baseline — reuses existing predictions, no rerun)
    house
    rooftop
    building rooftop
    structure

Outputs:
    /media/data/building_instance_tamu/sam3_prompt_experiments/<slug>/   ← predictions
    evaluation/results/prompt_experiments/                                ← metrics + figures
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT  = Path("/media/data/building_instance_tamu")
LABEL_DIR  = DATA_ROOT / "test/labels"
IMAGE_DIR  = DATA_ROOT / "test/images"
EXPT_ROOT  = DATA_ROOT / "sam3_prompt_experiments"
OUT_DIR    = REPO_ROOT / "evaluation/results/prompt_experiments"

# Baseline reuses existing full-test-set predictions
BASELINE_PRED_DIR = DATA_ROOT / "xview2_sam3_outputs/test/predictions"

# ─── experiment definitions ────────────────────────────────────────────────────
EXPERIMENTS = [
    {"slug": "building",         "prompt": "building",         "baseline": True},
    {"slug": "house",            "prompt": "house",            "baseline": False},
    {"slug": "rooftop",          "prompt": "rooftop",          "baseline": False},
    {"slug": "building_rooftop", "prompt": "building rooftop", "baseline": False},
    {"slug": "structure",        "prompt": "structure",        "baseline": False},
]

# ─── evaluation helpers (inline — avoids import path issues) ───────────────────

def _iou(a, b):
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _poly_from_pred(inst):
    from shapely.geometry import Polygon
    pts = inst.get("polygon", [])
    if len(pts) < 3:
        return None
    try:
        p = Polygon(pts)
        if not p.is_valid:
            p = p.buffer(0)
        return p if p.area > 0 else None
    except Exception:
        return None


def _polys_from_gt(label):
    from shapely.wkt import loads as wkt_loads
    from shapely.geometry import Polygon
    out = []
    for feat in label.get("features", {}).get("xy", []):
        wkt = feat.get("wkt", "")
        if not wkt:
            continue
        try:
            p = wkt_loads(wkt)
            if not p.is_valid:
                p = p.buffer(0)
            if p.area > 0:
                out.append(p)
        except Exception:
            pass
    return out


def evaluate_predictions(pred_dir: Path, label_dir: Path, iou_thresh: float = 0.5) -> dict:
    """Return aggregate + per-disaster metrics for a prediction directory."""
    from collections import defaultdict

    pred_files = sorted(pred_dir.glob("*_prediction.json"))
    if not pred_files:
        print(f"  [WARN] No prediction files in {pred_dir}")
        return {}

    all_ious = []
    disaster_stats = defaultdict(lambda: dict(tp=0, fp=0, fn=0, ious=[], images=0,
                                               pred_total=0, gt_total=0))
    totals = dict(tp=0, fp=0, fn=0, images=0, pred_total=0, gt_total=0,
                  images_no_pred=0)

    for pred_path in pred_files:
        stem = pred_path.stem.replace("_prediction", "")
        label_path = label_dir / f"{stem}.json"
        if not label_path.exists():
            continue

        with open(pred_path) as f:
            pred_data = json.load(f)
        with open(label_path) as f:
            gt_data = json.load(f)

        pred_polys = [p for inst in pred_data.get("instances", [])
                      if (p := _poly_from_pred(inst)) is not None]
        gt_polys   = _polys_from_gt(gt_data)

        n_pred, n_gt = len(pred_polys), len(gt_polys)

        # greedy IoU matching
        tp = fp = fn = 0
        matched_ious = []
        if n_pred == 0 and n_gt == 0:
            pass
        elif n_pred == 0:
            fn = n_gt
        elif n_gt == 0:
            fp = n_pred
        else:
            iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
            for i, pp in enumerate(pred_polys):
                for j, gp in enumerate(gt_polys):
                    iou_mat[i, j] = _iou(pp, gp)
            matched_pred, matched_gt = set(), set()
            pairs = sorted(
                [(iou_mat[i, j], i, j) for i in range(n_pred) for j in range(n_gt)],
                key=lambda x: -x[0]
            )
            for score, i, j in pairs:
                if score < iou_thresh:
                    break
                if i in matched_pred or j in matched_gt:
                    continue
                matched_pred.add(i)
                matched_gt.add(j)
                matched_ious.append(float(score))
            tp = len(matched_pred)
            fp = n_pred - tp
            fn = n_gt - len(matched_gt)

        disaster = stem.split("_")[0]
        totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn
        totals["images"] += 1
        totals["pred_total"] += n_pred; totals["gt_total"] += n_gt
        if n_pred == 0: totals["images_no_pred"] += 1
        all_ious.extend(matched_ious)

        d = disaster_stats[disaster]
        d["tp"] += tp; d["fp"] += fp; d["fn"] += fn
        d["images"] += 1; d["pred_total"] += n_pred; d["gt_total"] += n_gt
        d["ious"].extend(matched_ious)

    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    prec, rec, f1 = prf(totals["tp"], totals["fp"], totals["fn"])
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    per_disaster = {}
    for dis, d in sorted(disaster_stats.items()):
        dp, dr, df = prf(d["tp"], d["fp"], d["fn"])
        per_disaster[dis] = {
            "images": d["images"], "pred_total": d["pred_total"], "gt_total": d["gt_total"],
            "tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
            "precision": round(dp, 4), "recall": round(dr, 4),
            "f1": round(df, 4), "mean_iou_matched": round(float(np.mean(d["ious"])) if d["ious"] else 0.0, 4),
        }

    return {
        "images": totals["images"], "images_no_pred": totals["images_no_pred"],
        "pred_total": totals["pred_total"], "gt_total": totals["gt_total"],
        "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "mean_iou_matched": round(mean_iou, 4),
        "matched_pairs": len(all_ious),
        "per_disaster": per_disaster,
    }


# ─── figures ───────────────────────────────────────────────────────────────────

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def plot_overall_comparison(all_metrics: list[dict], out_path: Path):
    """4-metric bar chart comparing all prompt strategies."""
    metrics = ["precision", "recall", "f1", "mean_iou_matched"]
    labels  = ["Precision", "Recall", "F1", "Mean IoU"]
    names   = [m["name"] for m in all_metrics]
    n = len(names)
    x = np.arange(len(metrics))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(all_metrics):
        vals = [m["metrics"].get(k, 0) for k in metrics]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=m["name"],
                      color=PALETTE[i % len(PALETTE)], alpha=0.87, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("SAM3 Prompt Strategies — Overall Comparison (xView2 Test Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_f1_by_disaster(all_metrics: list[dict], out_path: Path):
    """Per-disaster F1 bars for all prompt strategies (grouped by disaster)."""
    all_disasters = set()
    for m in all_metrics:
        all_disasters.update(m["metrics"].get("per_disaster", {}).keys())
    disasters = sorted(all_disasters)

    names = [m["name"] for m in all_metrics]
    n = len(names)
    x = np.arange(len(disasters))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, m in enumerate(all_metrics):
        vals = [m["metrics"].get("per_disaster", {}).get(d, {}).get("f1", 0) for d in disasters]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=m["name"],
               color=PALETTE[i % len(PALETTE)], alpha=0.87, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(disasters, rotation=38, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("SAM3 Prompt Strategies — F1 per Disaster Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_recall_by_disaster(all_metrics: list[dict], out_path: Path):
    """Per-disaster Recall bars — recall is the main bottleneck."""
    all_disasters = set()
    for m in all_metrics:
        all_disasters.update(m["metrics"].get("per_disaster", {}).keys())
    disasters = sorted(all_disasters)

    names = [m["name"] for m in all_metrics]
    n = len(names)
    x = np.arange(len(disasters))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, m in enumerate(all_metrics):
        vals = [m["metrics"].get("per_disaster", {}).get(d, {}).get("recall", 0) for d in disasters]
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=m["name"],
               color=PALETTE[i % len(PALETTE)], alpha=0.87, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(disasters, rotation=38, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("SAM3 Prompt Strategies — Recall per Disaster Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_comparison_table(all_metrics: list[dict], out_path: Path):
    """Write a markdown summary table."""
    lines = ["# SAM3 Prompt Strategy Comparison\n",
             "**Dataset**: xView2 test set (933 images, 10 disaster types)  \n"
             "**IoU threshold**: 0.5\n",
             "---\n",
             "## Overall Metrics\n",
             "| Prompt | Precision | Recall | F1 | Mean IoU | Images w/o pred |",
             "|--------|----------:|-------:|---:|---------:|----------------:|"]
    for m in all_metrics:
        r = m["metrics"]
        lines.append(
            f"| `{m['prompt']}` | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {r['mean_iou_matched']:.4f} | {r['images_no_pred']} |"
        )
    lines.append("")
    lines.append("## F1 per Disaster\n")
    all_disasters = sorted({d for m in all_metrics
                             for d in m["metrics"].get("per_disaster", {})})
    header = "| Disaster |" + "".join(f" {m['name']} |" for m in all_metrics)
    lines.append(header)
    lines.append("|----------|" + "".join(" -----:|" for _ in all_metrics))
    for dis in all_disasters:
        row = f"| {dis} |"
        for m in all_metrics:
            f1 = m["metrics"].get("per_disaster", {}).get(dis, {}).get("f1", 0)
            row += f" {f1:.3f} |"
        lines.append(row)
    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"  Saved: {out_path}")


# ─── runner ────────────────────────────────────────────────────────────────────

def run_inference(expt: dict, device: str, overwrite: bool):
    """Run sam3_building_identifier for a non-baseline experiment."""
    slug = expt["slug"]
    pred_dir = EXPT_ROOT / slug / "predictions"
    if pred_dir.exists() and any(pred_dir.glob("*_prediction.json")) and not overwrite:
        n = len(list(pred_dir.glob("*_prediction.json")))
        print(f"  [SKIP] {slug}: {n} predictions already exist")
        return EXPT_ROOT / slug

    out_dir = EXPT_ROOT / slug
    cmd = [
        "conda", "run", "-n", "geoai_sam", "--no-capture-output",
        "python", "-m", "sam3_building_identifier",
        "--input-dir",  str(IMAGE_DIR),
        "--output-dir", str(out_dir),
        "--prompt",     expt["prompt"],
        "--disaster-type", "pre",      # pre-disaster images only (same as baseline)
        "--device",     device,
        "--min-size",   "100",
        "--epsilon",    "2.0",
        "--batch-size", "1",
    ]
    import os
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    print(f"\n  Running: prompt='{expt['prompt']}' → {out_dir}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        print(f"  [ERROR] Inference failed for {slug} (exit {result.returncode})")
        return None
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Run SAM3 prompt experiments and evaluate")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run inference even if predictions exist")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, only run evaluation on existing predictions")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Limit to specific prompt slugs (e.g. house rooftop)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPT_ROOT.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.prompts:
        experiments = [e for e in experiments if e["slug"] in args.prompts]
        print(f"Running subset: {[e['slug'] for e in experiments]}")

    # ── Step 1: inference ──────────────────────────────────────────────────────
    if not args.eval_only:
        for expt in experiments:
            if expt["baseline"]:
                print(f"\n[building] Using existing baseline: {BASELINE_PRED_DIR}")
                continue
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: {expt['slug']}  (prompt: '{expt['prompt']}')")
            print(f"{'='*60}")
            run_inference(expt, args.device, args.overwrite)

    # ── Step 2: evaluation ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    all_metrics = []
    for expt in experiments:
        slug = expt["slug"]
        if expt["baseline"]:
            pred_dir = BASELINE_PRED_DIR
        else:
            pred_dir = EXPT_ROOT / slug / "predictions"

        if not pred_dir.exists():
            print(f"  [SKIP] {slug}: prediction dir missing ({pred_dir})")
            continue

        n_preds = len(list(pred_dir.glob("*_prediction.json")))
        print(f"\n  Evaluating: {slug}  ({n_preds} files) ...")
        metrics = evaluate_predictions(pred_dir, LABEL_DIR)
        if not metrics:
            continue

        print(f"    P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  mIoU={metrics['mean_iou_matched']:.3f}  "
              f"zero-pred={metrics['images_no_pred']}")

        all_metrics.append({
            "slug":    slug,
            "name":    expt["prompt"],
            "prompt":  expt["prompt"],
            "metrics": metrics,
        })

        # Save per-experiment JSON
        (OUT_DIR / f"eval_{slug}.json").write_text(json.dumps(metrics, indent=2))

    if not all_metrics:
        print("\nNo results to plot — no predictions found.")
        sys.exit(1)

    # ── Step 3: figures ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FIGURES")
    print(f"{'='*60}")

    plot_overall_comparison(all_metrics, OUT_DIR / "prompt_comparison_overall.png")
    plot_f1_by_disaster(all_metrics,     OUT_DIR / "prompt_comparison_f1_by_disaster.png")
    plot_recall_by_disaster(all_metrics, OUT_DIR / "prompt_comparison_recall_by_disaster.png")
    write_comparison_table(all_metrics,  OUT_DIR / "prompt_comparison_report.md")

    print(f"\nAll outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()

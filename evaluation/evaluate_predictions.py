"""
Evaluate SAM3 building segmentation predictions against xView2 ground truth labels.

Usage:
    python evaluate_predictions.py [--split test|train|both] [--iou-thresh 0.5]
                                   [--output-dir results/sam3_eval]
"""

import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from shapely.geometry import Polygon
    from shapely.wkt import loads as wkt_loads
except ImportError:
    raise ImportError("pip install shapely")

# ─── paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/media/data/building_instance_tamu")

SPLITS = {
    "test": {
        "pred_dir":  DATA_ROOT / "xview2_sam3_outputs/test/predictions",
        "label_dir": DATA_ROOT / "test/labels",
    },
    "train": {
        "pred_dir":  DATA_ROOT / "xview2_sam3_outputs/train/predictions",
        "label_dir": DATA_ROOT / "train/labels",
    },
}


# ─── polygon helpers ──────────────────────────────────────────────────────────

def poly_from_prediction(instance: dict) -> Polygon | None:
    """Build Shapely polygon from a prediction instance."""
    pts = instance.get("polygon", [])
    if len(pts) < 3:
        return None
    try:
        p = Polygon(pts)
        if not p.is_valid:
            p = p.buffer(0)
        return p if p.area > 0 else None
    except Exception:
        return None


def polys_from_gt(label: dict) -> list[Polygon]:
    """Extract Shapely polygons from an xView2 label JSON (pixel coords via features.xy)."""
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


# ─── IoU ─────────────────────────────────────────────────────────────────────

def iou(a: Polygon, b: Polygon) -> float:
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


# ─── per-image evaluation ────────────────────────────────────────────────────

def evaluate_image(pred_polys: list[Polygon], gt_polys: list[Polygon],
                   iou_thresh: float = 0.5) -> dict:
    """
    Greedy IoU matching: sort candidate pairs by IoU desc, match greedily.

    Returns dict with tp, fp, fn, matched_ious, pred_count, gt_count.
    """
    n_pred = len(pred_polys)
    n_gt   = len(gt_polys)

    if n_pred == 0 and n_gt == 0:
        return dict(tp=0, fp=0, fn=0, matched_ious=[], pred_count=0, gt_count=0)
    if n_pred == 0:
        return dict(tp=0, fp=0, fn=n_gt, matched_ious=[], pred_count=0, gt_count=n_gt)
    if n_gt == 0:
        return dict(tp=0, fp=n_pred, fn=0, matched_ious=[], pred_count=n_pred, gt_count=0)

    # build IoU matrix
    iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, pp in enumerate(pred_polys):
        for j, gp in enumerate(gt_polys):
            iou_mat[i, j] = iou(pp, gp)

    matched_pred = set()
    matched_gt   = set()
    matched_ious = []

    # greedy matching by descending IoU
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

    return dict(tp=tp, fp=fp, fn=fn, matched_ious=matched_ious,
                pred_count=n_pred, gt_count=n_gt)


# ─── split evaluation ────────────────────────────────────────────────────────

def evaluate_split(split_name: str, pred_dir: Path, label_dir: Path,
                   iou_thresh: float = 0.5, verbose: bool = True) -> dict:
    pred_files = sorted(pred_dir.glob("*_prediction.json"))
    print(f"\n{'='*60}")
    print(f"Split: {split_name.upper()} — {len(pred_files)} prediction files")
    print(f"{'='*60}")

    all_ious       = []
    disaster_stats = defaultdict(lambda: dict(tp=0, fp=0, fn=0, ious=[], images=0,
                                               pred_total=0, gt_total=0))
    totals = dict(tp=0, fp=0, fn=0, images=0, pred_total=0, gt_total=0,
                  images_no_gt=0, images_no_pred=0)

    for pred_path in pred_files:
        # derive GT label path from stem
        stem = pred_path.stem.replace("_prediction", "")
        label_path = label_dir / f"{stem}.json"

        if not label_path.exists():
            if verbose:
                print(f"  [WARN] No GT for {stem}")
            continue

        with open(pred_path) as f:
            pred_data = json.load(f)
        with open(label_path) as f:
            gt_data = json.load(f)

        pred_polys = [p for inst in pred_data.get("instances", [])
                      if (p := poly_from_prediction(inst)) is not None]
        gt_polys   = polys_from_gt(gt_data)

        res = evaluate_image(pred_polys, gt_polys, iou_thresh)

        # disaster type from stem (e.g. "hurricane-florence_...")
        disaster = stem.split("_")[0]  # "hurricane-florence"

        totals["tp"]         += res["tp"]
        totals["fp"]         += res["fp"]
        totals["fn"]         += res["fn"]
        totals["images"]     += 1
        totals["pred_total"] += res["pred_count"]
        totals["gt_total"]   += res["gt_count"]
        if res["gt_count"]   == 0: totals["images_no_gt"]   += 1
        if res["pred_count"] == 0: totals["images_no_pred"] += 1
        all_ious.extend(res["matched_ious"])

        d = disaster_stats[disaster]
        d["tp"]         += res["tp"]
        d["fp"]         += res["fp"]
        d["fn"]         += res["fn"]
        d["images"]     += 1
        d["pred_total"] += res["pred_count"]
        d["gt_total"]   += res["gt_count"]
        d["ious"].extend(res["matched_ious"])

    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    prec, rec, f1 = prf(totals["tp"], totals["fp"], totals["fn"])
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    result = {
        "split": split_name,
        "iou_thresh": iou_thresh,
        "images": totals["images"],
        "images_no_gt":   totals["images_no_gt"],
        "images_no_pred": totals["images_no_pred"],
        "pred_total":  totals["pred_total"],
        "gt_total":    totals["gt_total"],
        "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "mean_iou_matched": round(mean_iou, 4),
        "matched_pair_count": len(all_ious),
        "all_matched_ious": all_ious,
        "per_disaster": {},
    }

    for dis, d in sorted(disaster_stats.items()):
        dp, dr, df = prf(d["tp"], d["fp"], d["fn"])
        miou = float(np.mean(d["ious"])) if d["ious"] else 0.0
        result["per_disaster"][dis] = {
            "images":    d["images"],
            "pred_total": d["pred_total"],
            "gt_total":   d["gt_total"],
            "tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
            "precision": round(dp, 4),
            "recall":    round(dr, 4),
            "f1":        round(df, 4),
            "mean_iou_matched": round(miou, 4),
        }

    return result


# ─── reporting ────────────────────────────────────────────────────────────────

def print_summary(r: dict):
    print(f"\n{'─'*60}")
    print(f"  SPLIT: {r['split'].upper()}  (IoU threshold = {r['iou_thresh']})")
    print(f"{'─'*60}")
    print(f"  Images evaluated : {r['images']:>6}")
    print(f"  Images w/o GT    : {r['images_no_gt']:>6}")
    print(f"  Images w/o pred  : {r['images_no_pred']:>6}")
    print(f"  GT buildings     : {r['gt_total']:>6}")
    print(f"  Predicted bldgs  : {r['pred_total']:>6}")
    print(f"  Avg pred/image   : {r['pred_total']/r['images']:.1f}" if r['images'] else "")
    print(f"  Avg GT/image     : {r['gt_total']/r['images']:.1f}"   if r['images'] else "")
    print()
    print(f"  {'Metric':<22} {'Value':>8}")
    print(f"  {'─'*31}")
    print(f"  {'TP':<22} {r['tp']:>8}")
    print(f"  {'FP':<22} {r['fp']:>8}")
    print(f"  {'FN':<22} {r['fn']:>8}")
    print(f"  {'Precision':<22} {r['precision']:>8.4f}")
    print(f"  {'Recall':<22} {r['recall']:>8.4f}")
    print(f"  {'F1':<22} {r['f1']:>8.4f}")
    print(f"  {'Mean IoU (matched)':<22} {r['mean_iou_matched']:>8.4f}")
    print(f"  {'Matched pairs':<22} {r['matched_pair_count']:>8}")
    print()
    print(f"  Per-disaster breakdown:")
    header = f"  {'Disaster':<35} {'Imgs':>5} {'GT':>6} {'Pred':>6} {'P':>6} {'R':>6} {'F1':>6} {'mIoU':>6}"
    print(header)
    print(f"  {'─'*(len(header)-2)}")
    for dis, d in r["per_disaster"].items():
        print(f"  {dis:<35} {d['images']:>5} {d['gt_total']:>6} {d['pred_total']:>6} "
              f"{d['precision']:>6.3f} {d['recall']:>6.3f} {d['f1']:>6.3f} {d['mean_iou_matched']:>6.3f}")


def plot_iou_hist(all_ious: list[float], split: str, out_path: Path):
    if not all_ious:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_ious, bins=50, range=(0, 1), color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(all_ious), color="red", linestyle="--",
               label=f"Mean = {np.mean(all_ious):.3f}")
    ax.axvline(np.median(all_ious), color="orange", linestyle="--",
               label=f"Median = {np.median(all_ious):.3f}")
    ax.set_xlabel("IoU score (matched pairs)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"IoU Distribution of Matched Pairs — {split.upper()} set", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pr_by_disaster(results: list[dict], out_path: Path):
    """Bar chart comparing F1 across disasters and splits."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["precision", "recall", "f1"]
    titles  = ["Precision", "Recall", "F1 Score"]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    all_disasters = set()
    for r in results:
        all_disasters.update(r["per_disaster"].keys())
    disasters = sorted(all_disasters)

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        x = np.arange(len(disasters))
        width = 0.35 / max(len(results), 1)
        for ri, r in enumerate(results):
            vals = [r["per_disaster"].get(d, {}).get(metric, 0) for d in disasters]
            offset = (ri - (len(results)-1)/2) * width * 2
            ax.bar(x + offset, vals, width*1.8, label=r["split"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(disasters, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SAM3 Building Detection — Per-Disaster Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_markdown_report(results: list[dict], out_path: Path):
    lines = ["# SAM3 Building Segmentation — Evaluation Report\n"]
    lines.append(f"**IoU threshold**: {results[0]['iou_thresh']} (detection match criterion)\n")
    lines.append(
        "**Metrics definition**: TP = predicted building matched to a GT building (IoU ≥ threshold); "
        "FP = unmatched prediction; FN = unmatched GT building.\n"
    )
    lines.append("---\n")

    for r in results:
        lines.append(f"## Split: {r['split'].upper()}\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Images evaluated | {r['images']} |")
        lines.append(f"| GT buildings | {r['gt_total']} |")
        lines.append(f"| Predicted buildings | {r['pred_total']} |")
        lines.append(f"| Avg pred / image | {r['pred_total']/r['images']:.1f} |" if r['images'] else "")
        lines.append(f"| Avg GT / image | {r['gt_total']/r['images']:.1f} |"   if r['images'] else "")
        lines.append(f"| True Positives (TP) | {r['tp']} |")
        lines.append(f"| False Positives (FP) | {r['fp']} |")
        lines.append(f"| False Negatives (FN) | {r['fn']} |")
        lines.append(f"| **Precision** | **{r['precision']:.4f}** |")
        lines.append(f"| **Recall** | **{r['recall']:.4f}** |")
        lines.append(f"| **F1** | **{r['f1']:.4f}** |")
        lines.append(f"| **Mean IoU (matched pairs)** | **{r['mean_iou_matched']:.4f}** |")
        lines.append(f"| Matched pairs | {r['matched_pair_count']} |")
        lines.append("")

        lines.append("### Per-Disaster Breakdown\n")
        lines.append("| Disaster | Images | GT | Pred | Precision | Recall | F1 | Mean IoU |")
        lines.append("|----------|-------:|---:|-----:|----------:|-------:|---:|---------:|")
        for dis, d in sorted(r["per_disaster"].items()):
            lines.append(
                f"| {dis} | {d['images']} | {d['gt_total']} | {d['pred_total']} | "
                f"{d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {d['mean_iou_matched']:.3f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"  Saved: {out_path}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM3 building predictions vs xView2 GT")
    parser.add_argument("--split", choices=["test", "train", "both"], default="both")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results/sam3_eval")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_to_run = (["test", "train"] if args.split == "both"
                     else [args.split])

    results = []
    for split in splits_to_run:
        cfg = SPLITS[split]
        if not cfg["pred_dir"].exists():
            print(f"[SKIP] Prediction dir not found: {cfg['pred_dir']}")
            continue
        if not cfg["label_dir"].exists():
            print(f"[SKIP] Label dir not found: {cfg['label_dir']}")
            continue
        r = evaluate_split(split, cfg["pred_dir"], cfg["label_dir"], args.iou_thresh)
        results.append(r)
        print_summary(r)

        # per-split IoU histogram
        plot_iou_hist(r["all_matched_ious"], split,
                      out_dir / f"iou_hist_{split}.png")

        # save per-split JSON (without the big ious list to keep it small)
        r_save = {k: v for k, v in r.items() if k != "all_matched_ious"}
        (out_dir / f"eval_{split}.json").write_text(json.dumps(r_save, indent=2))

    if results:
        # combined bar chart
        plot_pr_by_disaster(results, out_dir / "pr_by_disaster.png")
        # markdown report
        write_markdown_report(results, out_dir / "report.md")
        print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()

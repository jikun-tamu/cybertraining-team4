#!/usr/bin/env python3
"""
Compare SAM3 text prompts on the xView2 test set (pre-disaster, 933 images).

Every prompt, including the "building" baseline, is run with the same stage1
code and the same parameters (STAGE1_ARGS plus any overrides), so results are
comparable. Only prediction JSONs are kept (no mask TIFs or annotation PNGs).

Usage:
    conda run -n geoai_sam python evaluation/run_prompt_experiments.py [--device cuda:0]
    conda run -n geoai_sam python evaluation/run_prompt_experiments.py --eval-only

Outputs:
    /media/data/building_instance_tamu/sam3_prompt_experiments_v2/<slug>/   predictions
    evaluation/results/prompt_experiments/                                metrics + figures
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_predictions import evaluate_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/data/building_instance_tamu")
LABEL_DIR = DATA_ROOT / "test/labels"
IMAGE_DIR = DATA_ROOT / "test/images"
EXPT_ROOT = DATA_ROOT / "sam3_prompt_experiments_v2"
OUT_DIR = REPO_ROOT / "evaluation/results/prompt_experiments"

PROMPTS = ["building", "house", "rooftop", "building rooftop", "structure"]

# Stage-1 parameters shared by every prompt; everything not listed here uses
# the stage1 package defaults (tiling, merge IoU, confidence threshold, model).
STAGE1_ARGS = [
    "--disaster-type", "pre",
    "--no-masks", "--no-annotations",
]

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def slug(prompt: str) -> str:
    return prompt.replace(" ", "_")


def run_inference(prompt: str, args) -> bool:
    out_dir = EXPT_ROOT / slug(prompt)
    if any((out_dir / "predictions").glob("*_prediction.json")) and not args.overwrite:
        print(f"  [SKIP] {prompt}: predictions exist (use --overwrite)")
        return True
    cmd = [sys.executable, "-m", "sam3_building_identifier",
           "--input-dir", str(IMAGE_DIR), "--output-dir", str(out_dir),
           "--prompt", prompt, "--device", args.device, "--no-skip", *STAGE1_ARGS,
           *args.stage1_extra]
    env = {**os.environ, "HF_HUB_OFFLINE": "1",
           "PYTHONPATH": str(REPO_ROOT / "stage1") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    print(f"\n  Running prompt '{prompt}' -> {out_dir}")
    return subprocess.run(cmd, env=env).returncode == 0


def plot_overall(results: list[dict], out_path: Path) -> None:
    metrics = ["precision", "recall", "f1", "mean_iou_matched"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(results):
        vals = [r[k] for k in metrics]
        bars = ax.bar(x + (i - (len(results) - 1) / 2) * width, vals, width,
                      label=r["split"], color=PALETTE[i % len(PALETTE)], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x, ["Precision", "Recall", "F1", "Mean IoU"], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("SAM3 prompts: overall (xView2 test, IoU 0.5)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_by_disaster(results: list[dict], metric: str, out_path: Path) -> None:
    disasters = sorted({d for r in results for d in r["per_disaster"]})
    x = np.arange(len(disasters))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, r in enumerate(results):
        vals = [r["per_disaster"].get(d, {}).get(metric, 0) for d in disasters]
        ax.bar(x + (i - (len(results) - 1) / 2) * width, vals, width,
               label=r["split"], color=PALETTE[i % len(PALETTE)], edgecolor="white")
    ax.set_xticks(x, disasters, rotation=38, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"SAM3 prompts: {metric} per disaster", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(results: list[dict], run_config: dict, out_path: Path) -> None:
    lines = [
        "# SAM3 Prompt Comparison\n",
        f"**Dataset**: xView2 test set, pre-disaster ({results[0]['images']} images)  ",
        "**Match criterion**: IoU >= 0.5  ",
        f"**Stage-1 config**: `{json.dumps(run_config)}`\n",
        "| Prompt | Precision | Recall | F1 | Mean IoU | Predicted | Images w/o pred |",
        "|--------|----------:|-------:|---:|---------:|----------:|----------------:|",
    ]
    for r in results:
        lines.append(f"| `{r['split']}` | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                     f"{r['mean_iou_matched']:.4f} | {r['pred_total']} | {r['images_no_pred']} |")
    disasters = sorted({d for r in results for d in r["per_disaster"]})
    lines += ["", "## F1 per disaster\n",
              "| Disaster |" + "".join(f" {r['split']} |" for r in results),
              "|----------|" + " -----:|" * len(results)]
    for d in disasters:
        lines.append(f"| {d} |" + "".join(
            f" {r['per_disaster'].get(d, {}).get('f1', 0):.3f} |" for r in results))
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run and evaluate SAM3 prompt experiments")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompts", nargs="+", default=PROMPTS)
    parser.add_argument("--overwrite", action="store_true", help="Re-run existing predictions.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing predictions.")
    args, args.stage1_extra = parser.parse_known_args()  # extra args go to stage1, e.g. --tile-size 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        for prompt in args.prompts:
            if not run_inference(prompt, args):
                print(f"  [ERROR] inference failed for '{prompt}'")

    results, run_config = [], {}
    for prompt in args.prompts:
        out_dir = EXPT_ROOT / slug(prompt)
        if not (out_dir / "predictions").is_dir():
            print(f"  [SKIP] {prompt}: no predictions")
            continue
        summary = json.loads((out_dir / "run_summary.json").read_text())
        cfg = summary["config"]
        run_config = {k: cfg.get(k) for k in ("model_id", "confidence_threshold", "tile_size",
                                              "tile_overlap", "merge_iou", "min_size",
                                              "min_polygon_area", "polygon_epsilon")}
        run_config["code_version"] = summary.get("code_version")
        r = evaluate_split(prompt, out_dir / "predictions", LABEL_DIR, verbose=False)
        r.pop("all_matched_ious")
        (OUT_DIR / f"eval_{slug(prompt)}.json").write_text(json.dumps(r, indent=2))
        print(f"  {prompt:18s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
        results.append(r)

    if not results:
        sys.exit("No predictions to evaluate.")
    plot_overall(results, OUT_DIR / "prompt_comparison_overall.png")
    plot_by_disaster(results, "f1", OUT_DIR / "prompt_comparison_f1_by_disaster.png")
    plot_by_disaster(results, "recall", OUT_DIR / "prompt_comparison_recall_by_disaster.png")
    write_report(results, run_config, OUT_DIR / "prompt_comparison_report.md")
    print(f"\nOutputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()

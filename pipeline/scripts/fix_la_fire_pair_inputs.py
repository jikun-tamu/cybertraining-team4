#!/usr/bin/env python3
"""Repair LA fire multidate pair_inputs symlinks after dataset relocation.

This script relinks:
- pair_inputs/<cell>_pre_pre_disaster.png  -> chips/<cell>/pre/<cell>_pre.tif
- pair_inputs/<cell>_pre_post_disaster.png -> earliest chips/<cell>/post/<cell>_post_YYYYMMDD.tif

The original multidate outputs often contain broken links into the old project
interim directory. Those links block shared-base regeneration and any stale-cell
refresh after moving the dataset root.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_LA_ROOT = Path("/media/data/building_instance_tamu/la_fire_2025")
DEFAULT_RUN_ROOT = DEFAULT_LA_ROOT / "stage2_damage/multidate_full_run"
DEFAULT_CHIPS_ROOT = DEFAULT_LA_ROOT / "chips"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair pair_inputs symlinks for LA fire multidate outputs.")
    p.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--chips_root", type=Path, default=DEFAULT_CHIPS_ROOT)
    p.add_argument("--cells", nargs="*", default=None, help="Optional subset of cells to repair.")
    p.add_argument("--dry_run", action="store_true", help="Report changes without modifying links.")
    return p.parse_args()


def choose_post_image(post_dir: Path, cell_id: str) -> Path | None:
    tif_paths = sorted(post_dir.glob(f"{cell_id}_post_*.tif"))
    return tif_paths[0] if tif_paths else None


def relink(dst: Path, src: Path, dry_run: bool) -> bool:
    changed = False
    if dst.is_symlink():
        current = dst.resolve(strict=False)
        if current == src.resolve():
            return False
        changed = True
        if not dry_run:
            dst.unlink()
    elif dst.exists():
        raise RuntimeError(f"Refusing to replace non-symlink path: {dst}")
    else:
        changed = True

    if changed and not dry_run:
        dst.symlink_to(src.resolve())
    return changed


def iter_cells(run_root: Path, only: list[str] | None):
    for cell_dir in sorted(run_root.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("cell_"):
            continue
        if only and cell_dir.name not in only:
            continue
        yield cell_dir


def main() -> int:
    args = parse_args()

    repaired_links = 0
    skipped_cells = 0

    for cell_dir in iter_cells(args.run_root, args.cells):
        cell_id = cell_dir.name
        pair_dir = cell_dir / "pair_inputs"
        if not pair_dir.exists():
            skipped_cells += 1
            print(f"[skip] {cell_id}: missing pair_inputs/")
            continue

        pre_src = args.chips_root / cell_id / "pre" / f"{cell_id}_pre.tif"
        post_src = choose_post_image(args.chips_root / cell_id / "post", cell_id)
        if not pre_src.exists() or post_src is None or not post_src.exists():
            skipped_cells += 1
            print(
                f"[skip] {cell_id}: missing chip path(s) "
                f"pre_exists={pre_src.exists()} post_exists={post_src is not None and post_src.exists()}"
            )
            continue

        pre_dst = pair_dir / f"{cell_id}_pre_pre_disaster.png"
        post_dst = pair_dir / f"{cell_id}_pre_post_disaster.png"

        changed_pre = relink(pre_dst, pre_src, args.dry_run)
        changed_post = relink(post_dst, post_src, args.dry_run)
        repaired_links += int(changed_pre) + int(changed_post)

        status = "would_relink" if args.dry_run else "relinked"
        if changed_pre or changed_post:
            print(f"[{status}] {cell_id}: pre->{pre_src.name} post->{post_src.name}")

    print(f"[done] repaired_links={repaired_links} skipped_cells={skipped_cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Utility helpers: file discovery, disaster-type filtering, image metadata,
and a lightweight timing context manager.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# Image discovery and filtering
# ---------------------------------------------------------------------------

def _detect_disaster_naming(files: List[Path]) -> Tuple[bool, bool]:
    """Return (has_pre, has_post): whether the file list uses xView2 naming."""
    has_pre = any("_pre_disaster" in p.stem for p in files)
    has_post = any("_post_disaster" in p.stem for p in files)
    return has_pre, has_post


def discover_images(
    input_dir: str,
    disaster_type: str = "auto",
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    max_images: Optional[int] = None,
) -> List[Path]:
    """Return sorted list of image paths matching the disaster-type filter.

    Parameters
    ----------
    input_dir:
        Directory to search (non-recursive).
    disaster_type:
        'auto' (default) — inspect the folder:
                           • If any *_pre_disaster.* files exist → use pre only.
                           • Otherwise → use all image files.
        'pre'  → filenames containing '_pre_disaster' (raises if none found).
        'post' → filenames containing '_post_disaster' (raises if none found).
        'all'  → all files with accepted extensions.
    extensions:
        Accepted file extensions (lower-case, with leading dot).
    max_images:
        Optional cap on results (useful for smoke tests).
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    all_files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not all_files:
        log(f"WARNING: No image files found in {root}")
        return []

    if disaster_type == "auto":
        has_pre, has_post = _detect_disaster_naming(all_files)
        if has_pre:
            filtered = [p for p in all_files if "_pre_disaster" in p.stem]
            log(
                f"Auto-detected xView2 naming — using pre-disaster images "
                f"({len(filtered)} of {len(all_files)} total)"
            )
        else:
            filtered = all_files
            if has_post:
                log(
                    f"Auto-detected: only post-disaster naming found — "
                    f"processing all {len(filtered)} images"
                )
            else:
                log(
                    f"Auto-detected: no pre/post disaster naming — "
                    f"processing all {len(filtered)} images"
                )
    elif disaster_type == "pre":
        filtered = [p for p in all_files if "_pre_disaster" in p.stem]
        if not filtered:
            log(
                f"WARNING: disaster_type='pre' but no '_pre_disaster' files found "
                f"in {root}. Found {len(all_files)} total image(s). "
                f"Use --disaster-type auto or --disaster-type all to process them."
            )
        else:
            log(f"Filtering to pre-disaster images: {len(filtered)} of {len(all_files)}")
    elif disaster_type == "post":
        filtered = [p for p in all_files if "_post_disaster" in p.stem]
        if not filtered:
            log(
                f"WARNING: disaster_type='post' but no '_post_disaster' files found "
                f"in {root}. Found {len(all_files)} total image(s). "
                f"Use --disaster-type auto or --disaster-type all to process them."
            )
        else:
            log(f"Filtering to post-disaster images: {len(filtered)} of {len(all_files)}")
    elif disaster_type == "all":
        filtered = all_files
        log(f"Processing all {len(filtered)} images (no disaster-type filter)")
    else:
        raise ValueError(
            f"disaster_type must be 'auto', 'pre', 'post', or 'all'; "
            f"got {disaster_type!r}"
        )

    if max_images is not None:
        filtered = filtered[:max_images]

    return filtered


def get_image_size(path: Path) -> Tuple[int, int]:
    """Return (width, height) of an image without loading full pixel data."""
    with Image.open(path) as img:
        return img.size  # (width, height)


def infer_disaster_type(stem: str) -> str:
    """Extract 'pre' or 'post' from an xView2-style filename stem."""
    if "_pre_disaster" in stem:
        return "pre"
    if "_post_disaster" in stem:
        return "post"
    return "unknown"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer() -> Generator[dict, None, None]:
    """Context manager that measures elapsed wall-clock time in seconds.

    Usage::

        with timer() as t:
            do_work()
        print(t['elapsed'])
    """
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = round(time.perf_counter() - start, 3)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Print a timestamped log line."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_section(title: str) -> None:
    """Print a section header."""
    bar = "─" * (len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)

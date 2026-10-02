"""
Unit tests for window layout and cross-window merging (no GPU needed).

    conda run -n geoai_sam python stage1/tests/test_tiling.py   (or with pytest)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam3_building_identifier.tiling import (
    instance_from_mask, merge_instances, paint_labels, tile_windows,
)


def test_windows_cover_image_without_slivers():
    wins = tile_windows(1024, 1024, 512, 64)
    assert len(wins) == 9
    assert all(w == 512 and h == 512 for _, _, w, h in wins)
    assert sorted({x for x, _, _, _ in wins}) == [0, 256, 512]
    assert tile_windows(300, 300, 512, 64) == [(0, 0, 300, 300)]


def _window_view(full_mask, win):
    x, y, w, h = win
    return full_mask[y:y + h, x:x + w]


def test_building_across_seam_is_merged():
    # A 200x60 building crossing the seam between two side-by-side windows.
    full = np.zeros((512, 900), dtype=bool)
    full[100:160, 350:550] = True
    left, right = (0, 0, 512, 512), (388, 0, 512, 512)
    a = instance_from_mask(_window_view(full, left), left, 0.7)
    b = instance_from_mask(_window_view(full, right), right, 0.9)
    merged = merge_instances([a, b])
    assert len(merged) == 1
    m = merged[0]
    assert (m.x0, m.y0, m.x1, m.y1) == (350, 100, 550, 160)
    assert m.area == 200 * 60
    assert m.score == 0.9


def test_neighbouring_buildings_stay_separate():
    full = np.zeros((512, 900), dtype=bool)
    full[100:160, 400:440] = True  # building 1, inside both windows
    left, right = (0, 0, 512, 512), (388, 0, 512, 512)
    a = instance_from_mask(_window_view(full, left), left, 0.8)
    other = np.zeros_like(full)
    other[100:160, 450:490] = True  # building 2, only reported by the right window
    b = instance_from_mask(_window_view(other, right), right, 0.8)
    assert len(merge_instances([a, b])) == 2


def test_paint_labels_higher_score_wins():
    win = (0, 0, 50, 50)
    m1 = np.zeros((50, 50), dtype=bool); m1[10:30, 10:30] = True
    m2 = np.zeros((50, 50), dtype=bool); m2[20:40, 20:40] = True
    low, high = instance_from_mask(m1, win, 0.6), instance_from_mask(m2, win, 0.9)
    labels, scores, label_scores = paint_labels([high, low], 50, 50)
    assert labels.dtype == np.int32
    assert scores[25, 25] == np.float32(0.9)
    assert label_scores[labels[25, 25]] == 0.9
    assert len(label_scores) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")

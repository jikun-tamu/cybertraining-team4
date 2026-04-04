#!/usr/bin/env python3
"""Canonical path helpers for the LA Fire 2025 validation pipeline."""

from __future__ import annotations

from pathlib import Path


CANONICAL_LA_FIRE_ROOT = Path("/media/data/building_instance_tamu/la_fire_2025")

CANONICAL_CHIPS_ROOT = CANONICAL_LA_FIRE_ROOT / "chips"
CANONICAL_MANIFEST = CANONICAL_LA_FIRE_ROOT / "grids/chips_600m_manifest.csv"
CANONICAL_RUN_ROOT = CANONICAL_LA_FIRE_ROOT / "stage2_damage/multidate_full_run"
CANONICAL_QC_ROOT = CANONICAL_LA_FIRE_ROOT / "qc_overlays_m2b"

OLD_CHIPS_PREFIXES = [
    "/media/data/la_fire_2025/chips",
    "/media/gisense/xihan/250812_CyberTraining_Team4/data/chips_600m",
    "/media/gisense/xihan/250812_CyberTraining_Team4/data/interim/chips_600m",
    "/media/gisense/xihan/250812_tamu_cybertraining_team4/data/interim/chips_600m",
]


def canonical_la_fire_root() -> Path:
    return CANONICAL_LA_FIRE_ROOT


def canonical_manifest_path() -> Path:
    return CANONICAL_MANIFEST


def canonical_chips_root() -> Path:
    return CANONICAL_CHIPS_ROOT


def canonical_run_root() -> Path:
    return CANONICAL_RUN_ROOT


def canonical_qc_root() -> Path:
    return CANONICAL_QC_ROOT


def rewrite_la_fire_path(path_like: str | Path) -> Path:
    """Rewrite legacy chip paths to the canonical LA root."""
    text = str(path_like)
    for old in OLD_CHIPS_PREFIXES:
        if old in text:
            text = text.replace(old, str(CANONICAL_CHIPS_ROOT))
    return Path(text)


def canonical_cell_dir(cell_id: str) -> Path:
    return CANONICAL_RUN_ROOT / cell_id

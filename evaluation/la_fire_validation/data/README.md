# LA ground truth (LARIAC outlines + CAL FIRE DINS)

Copied from `/media/data/building_instance_tamu/la_fire_2025/validation/` so the LA metrics can be
recomputed without the lab server. Built by `../validate_la.py`; see `reports/sam3_audit_2026-09.md` §6.

| File | Contents |
|---|---|
| `lariac_dins_in_region.gpkg` | 14,369 LA County LARIAC building outlines inside the evaluation region, with CAL FIRE DINS max/min damage (27 Feb 2025), `AIN` (assessor parcel number), `HEIGHT`, fire name. Source: LA County feature service `WildFire_BUILDINGS_100M_Plus_DINS_MaxMinDAMAGE` |
| `evaluation_region.gpkg` | Imaged area ∩ (Eaton + Palisades perimeters + 100 m), 83.3 km², EPSG:32611 |
| `perimeters_2025.geojson` | WFIGS final perimeters, Eaton and Palisades |
| `matched_v2.csv` | Ground-truth building ↔ Stage 1 footprint matches (`gt_index`, `lariac_bld_id`, `bldg_uid`, `dins`, `m2b`, per-cell shift `shift_dx_m`/`shift_dy_m` used for co-registration) |

DINS positive class for "destroyed": `Destroyed (>50%)`. Buildings with an empty DINS field were
not inspected (outside the perimeter buffer of DINS points).

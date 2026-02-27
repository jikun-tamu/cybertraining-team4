import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union

# -----------------------------
# PolyWorld Model Import
# -----------------------------
from models.matching import OptimalMatching
from models.backbone import R2U_Net, DetectionBranch, NonMaxSuppression


# ============================================================
# 1. LOAD POLYWORLD MODEL
# ============================================================
def load_polyworld(weights_dir: Path, device: torch.device):
    backbone = R2U_Net().to(device).train()
    head = DetectionBranch().to(device).train()
    nms = NonMaxSuppression().to(device)
    matching = OptimalMatching().to(device).train()

    backbone.load_state_dict(torch.load(weights_dir / "polyworld_backbone", map_location=device))
    head.load_state_dict(torch.load(weights_dir / "polyworld_seg_head", map_location=device))
    matching.load_state_dict(torch.load(weights_dir / "polyworld_matching", map_location=device))

    return backbone, head, nms, matching


# ============================================================
# 2. PATCH PREPROCESSING
# ============================================================
def preprocess_patch(img_patch: Image.Image):
    arr = np.array(img_patch).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


# ============================================================
# 3. RUN POLYWORLD ON ONE PATCH
# ============================================================
def run_on_patch(poly_model, img_patch, device):
    backbone, head, nms, matching = poly_model

    inp = preprocess_patch(img_patch).to(device).float()
    with torch.no_grad():
        feat = backbone(inp)
        occ = head(feat)
        _, pts = nms(occ)
        polys = matching.predict(inp, feat, pts)

    return polys[0] if len(polys) else []


# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================
def polygon_from_seg(seg):
    return Polygon(np.array(seg).reshape(-1, 2))


def iou(poly1, poly2):
    try:
        p1 = poly1.buffer(0)
        p2 = poly2.buffer(0)

        if p1.is_empty or p2.is_empty:
            return 0

        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        if union == 0:
            return 0
        return inter / union
    except Exception as e:
        # Fallback: treat invalid intersection as IoU = 0
        return 0

def make_ann(image_id, seg):
    xs = seg[0::2]
    ys = seg[1::2]
    return {
        "image_id": image_id,
        "category_id": 100,
        "score": 1,
        "segmentation": [seg],
        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
    }


# ============================================================
# 5. TILE POSITION GENERATOR
# ============================================================
def compute_positions(length, tile):
    # No overlap, but ensure coverage of right/bottom edges
    positions = list(range(0, length, tile))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return sorted(set(positions))


# ============================================================
# 6. FULL TILED INFERENCE + MERGE + CLEAN PIPELINE
# ============================================================
def polyworld_inference_and_clean(
    image_path,
    weights_dir,
    output_json,
    iou_thresh=0.20,
    small_thresh=50,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir = Path(weights_dir)

    # ---- LOAD MODEL ----
    poly_model = load_polyworld(weights_dir, device)

    # ---- LOAD IMAGE ----
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    print(f"[INFO] Image size: {W} × {H}")

    TILE = 320
    MODEL_IN = 320

    xs = compute_positions(W, TILE)
    ys = compute_positions(H, TILE)

    print("[INFO] Tile X positions:", xs)
    print("[INFO] Tile Y positions:", ys)

    raw_polygons = []
    image_id = 1

    # ==================================================
    #     STEP A — TILED INFERENCE
    # ==================================================
    for ty in ys:
        for tx in xs:
            x2 = min(tx + TILE, W)
            y2 = min(ty + TILE, H)
            x1 = max(x2 - TILE, 0)
            y1 = max(y2 - TILE, 0)

            patch = img.crop((x1, y1, x2, y2))
            pw, ph = patch.size

            patch_resized = patch.resize((MODEL_IN, MODEL_IN), Image.BILINEAR)
            polys = run_on_patch(poly_model, patch_resized, device)

            for poly in polys:
                coords = np.array(poly).reshape(-1, 2)

                # scale from model-space (320) → patch space
                coords[:, 0] = x1 + (coords[:, 0] / MODEL_IN) * pw
                coords[:, 1] = y1 + (coords[:, 1] / MODEL_IN) * ph

                poly = Polygon(coords).buffer(0)   # fix shape
                if not poly.is_empty:
                    raw_polygons.append(poly)


    print(f"[INFO] Raw polygons detected: {len(raw_polygons)}")

    # ==================================================
    #     STEP B — IoU MERGING
    # ==================================================
    merged = []
    consumed = set()

    for i in range(len(raw_polygons)):
        if i in consumed:
            continue

        base = raw_polygons[i]
        group = [base]

        for j in range(i + 1, len(raw_polygons)):
            if j in consumed:
                continue
            if iou(base, raw_polygons[j]) > iou_thresh:
                group.append(raw_polygons[j])
                consumed.add(j)

        merged_poly = unary_union(group)
        merged_poly = merged_poly.buffer(0)
        merged.append(merged_poly)

    print(f"[INFO] After IoU merging: {len(merged)} polygons")

    # ==================================================
    #     STEP C — REMOVE SMALL ATTACHED FRAGMENTS
    # ==================================================
    final = []
    for p in merged:
        if p.area < small_thresh:
            # Remove if touching a significantly larger polygon
            if any((p != big) and (big.area > small_thresh * 3) and (p.buffer(1).intersects(big))
                   for big in merged):
                continue
        final.append(p)

    print(f"[INFO] After small-fragment cleaning: {len(final)} polygons")

    # ==================================================
    #     STEP D — WRITE FINAL JSON  (FIXED MultiPolygon)
    # ==================================================
    output_list = []
    for poly in final:
        if poly.is_empty:
            continue

        # Case A — MultiPolygon → split into individual polygons
        if poly.geom_type == "MultiPolygon":
            for sub in poly.geoms:
                if sub.is_empty:
                    continue
                x, y = sub.exterior.coords.xy
                seg = []
                for xx, yy in zip(x, y):
                    seg += [float(xx), float(yy)]
                output_list.append(make_ann(image_id, seg))

        # Case B — simple Polygon
        elif poly.geom_type == "Polygon":
            x, y = poly.exterior.coords.xy
            seg = []
            for xx, yy in zip(x, y):
                seg += [float(xx), float(yy)]
            output_list.append(make_ann(image_id, seg))

        # Case C — something unexpected (rare)
        else:
            print("[WARN] Unknown geom type:", poly.geom_type)
            continue

    Path(output_json).write_text(json.dumps(output_list))
    print(f"[INFO] Saved cleaned polygons → {output_json}")



# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    polyworld_inference_and_clean(
        image_path="/media/data/building_instance_tamu/test/images/hurricane-harvey_00000023_pre_disaster.png",
        weights_dir="/media/gisense/xihan/250812_tamu_cybertraining_team4/PolyWorld/trained_weights",
        output_json="/media/gisense/xihan/250812_tamu_cybertraining_team4/PolyWorld/sample_test/tiled_predictions_v8.json",
        iou_thresh=0.20,
        small_thresh=50,
    )

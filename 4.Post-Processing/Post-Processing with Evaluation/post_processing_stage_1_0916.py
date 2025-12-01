# Sep.20 : Updated to handle nested subfolders in input JSON dir.

"""
======================
Coordinate System Notes
======================
Upstream reality TODAY:
- Tile-level prediction JSONs currently store points as [y, x] (row, col).

This script's policy:
- On load, convert JSON coords from [y, x] -> (x, y) exactly once.
- Internally we always reason with:
    JSON-normalized points: (x, y)
    NumPy indexing:        arr[y, x]
- Unified JSONs we write use (x, y) for interoperability (COCO/GIS/plotting).
- CSV exports vertices as [row, col] = [y, x].

Summary:
- INPUT (tile JSON):         [y, x]    (legacy from exporter)
- Normalized in this script: (x, y)
- NumPy access:              arr[y, x]
- Unified JSON output:       (x, y)
- CSV vertices:              [y, x]
"""

import os
import re
import json
import csv
import hashlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
from shapely.geometry import Polygon
from tqdm import tqdm

# =======================
# CONFIG
# =======================
json_folder = "/shared/data/climateplus2025/Prediction_for_poster_3_images_Mask2Former_1024_Nov20/prediction_outputs_Mask2Former_newdata_swinS"

# Where to write outputs
out_dir = "/shared/data/climateplus2025/Postprocessing_for_poster_3_images_1024_Nov20/output_stage_1"
os.makedirs(out_dir, exist_ok=True)

# Unified JSONs (global coords, background omitted)
SAVE_UNIFIED_JSON = True
unified_json_dir = os.path.join(out_dir, "unified_json_global")
if SAVE_UNIFIED_JSON:
    os.makedirs(unified_json_dir, exist_ok=True)

# Polygonization switch
RUN_POLYGONIZATION = True

# Optional: parallelize per base
PARALLEL_BASES = False  # set True to use multiprocessing Pool for per-base processing

# Canvas / tiles
CANVAS_H = 12500
CANVAS_W = 12500
TILE_SIZE = 1024  # from your tiler

# Pixel-to-meters (8cm GSD)
PIXEL_AREA_M2 = 0.08 ** 2

# Labels and priority (higher wins in overlaps)
CLASS_ORDER = ["background", "PV_normal", "PV_heater", "PV_pool"]
CLASS_PRIORITY = {"background": 0, "PV_normal": 1, "PV_heater": 2, "PV_pool": 3}

# Normalize incoming keys (e.g., "class_1" -> "PV_normal", etc.)
CLASS_TO_LABEL = {
    "class_1": "PV_normal",
    "class_2": "PV_heater",
    "class_3": "PV_pool",
}

# If/when your exporter changes to emit (x, y), set this to False.
LEGACY_JSON_IS_YX = True  # current exporter writes [y, x]

# I/O performance
try:
    import orjson
    def _load_json_file(path):
        with open(path, "rb") as f:
            return orjson.loads(f.read())
except Exception:
    def _load_json_file(path):
        with open(path, "r") as f:
            return json.load(f)

# =======================
# FILENAME PARSER
# =======================
# A) pixel-offset filenames: ..._tile_<x>_<y>[ _pred_* ].json
TILE_PIXEL_RE = re.compile(r"_tile_(\d+)_(\d+)", re.I)

# B) index filenames: i_<base>_<ix>_<iy>[ _pred_* ].json
INDEX_JSON_RE = re.compile(
    r"^(?P<prefix>i_)?(?P<base>.+?)_(?P<ix>\d+)_(?P<iy>\d+)(?:_pred_[^.]*)?\.json$",
    re.I
)

def parse_base_and_offsets(json_name: str, tile_size=TILE_SIZE):
    """
    Return (base, tile_x_px, tile_y_px) for a JSON filename.
    Supports:
      A) ..._tile_x_y...
      B) i_<base>_<ix>_<iy>[ _pred_* ].json
    """
    bn = os.path.basename(json_name)
    stem, _ = os.path.splitext(bn)

    # A) pixel offsets directly in name
    m = TILE_PIXEL_RE.search(stem)
    if m:
        x_px, y_px = int(m.group(1)), int(m.group(2))
        base = stem.split("_tile_")[0]  # keep i_ prefix if present
        return base, x_px, y_px

    # B) index style (ix, iy)
    m = INDEX_JSON_RE.match(bn)
    if m:
        prefix = m.group("prefix") or ""
        base = prefix + m.group("base")
        ix, iy = int(m.group("ix")), int(m.group("iy"))
        x_px = ix * tile_size
        y_px = iy * tile_size
        return base, x_px, y_px

    raise ValueError(f"Cannot parse tile offsets from: {bn}")

# =======================
# PRED COORDS NORMALIZATION
# =======================
def _norm_key_to_label(k: str):
    """ Map 'class_1/2/3' or numeric '1/2/3' to PV_*; pass through PV_* """
    ks = str(k).strip().lower()
    if ks in CLASS_TO_LABEL:
        return CLASS_TO_LABEL[ks]
    if ks in {"1", "2", "3"}:
        return CLASS_TO_LABEL[f"class_{ks}"]
    # already PV_* or untracked
    return k

def normalize_pred_coords(pred_coords: dict, clip_h=CANVAS_H, clip_w=CANVAS_W) -> dict:
    """
    Normalize incoming JSON coords to (x, y).
    Current upstream format is [y, x] (row, col) if LEGACY_JSON_IS_YX=True.
    Returns: {label: np.ndarray(N,2) of (x,y)}
    """
    out = {}
    for k, coords in (pred_coords or {}).items():
        label = _norm_key_to_label(k)
        if label not in CLASS_PRIORITY or not coords:
            continue

        arr = np.asarray(coords, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            continue

        # Convert [y, x] -> (x, y) once if legacy
        if LEGACY_JSON_IS_YX:
            arr = arr[:, [1, 0]]

        # Clip to canvas bounds: x in [0, W), y in [0, H)
        m = (arr[:, 0] >= 0) & (arr[:, 0] < clip_w) & (arr[:, 1] >= 0) & (arr[:, 1] < clip_h)
        if not np.any(m):
            continue
        arr = arr[m]

        out[label] = arr

    out.pop("background", None)
    return out

# =======================
# POLYGON HELPERS
# =======================
def canonicalize_ring_rc(verts_rc):
    pts = [(int(round(r)), int(round(c))) for (r, c) in verts_rc]
    if not pts:
        return pts
    def rotate_to_min(seq):
        k = min(range(len(seq)), key=lambda i: seq[i])
        return seq[k:] + seq[:k]
    fwd = rotate_to_min(pts)
    rev = rotate_to_min(list(reversed(pts)))
    return fwd if fwd <= rev else rev

def polygon_hash_id(base_name: str, label: str, verts_rc):
    canon = canonicalize_ring_rc(verts_rc)
    payload = {"label": label, "verts": canon}
    h8 = hashlib.sha1(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:8]
    return f"{base_name}_pred_{h8}"

# =======================
# STITCHING
# =======================
def stitch_one_base(base: str, items, json_dir: str):
    """
    Build a single global class-index canvas for the base image:
      0 background, 1 PV_normal, 2 PV_heater, 3 PV_pool
    Conflict-resolve by CLASS_PRIORITY (higher wins).
    """
    class_map = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    class_map_flat = class_map.ravel()  # 1D view
    W64 = np.int64(CANVAS_W)
    H64 = np.int64(CANVAS_H)

    for fn, tile_x, tile_y in items:
        data = _load_json_file(os.path.join(json_dir, fn))
        pred_coords = normalize_pred_coords(data.get("predicted_coords", {}))

        for label, arr in pred_coords.items():
            if arr is None or arr.size == 0:
                continue
            cls_idx = CLASS_PRIORITY[label]

            # arr is (N, 2) = (x, y) after normalization
            gx = arr[:, 0].astype(np.int64) + np.int64(tile_x)
            gy = arr[:, 1].astype(np.int64) + np.int64(tile_y)

            # Quick dev-time guard (comment out in prod if noisy)
            # Asserts 99% of points are in-bounds; helps catch accidental swaps.
            if gx.size:
                inb = (gx >= 0) & (gx < W64) & (gy >= 0) & (gy < H64)
                if inb.mean() < 0.99:
                    print(f"[WARN] Many points out-of-bounds in {fn}/{label}. "
                          f"Check LEGACY_JSON_IS_YX or tile offsets.")

            # In-bounds mask
            m = (gx >= 0) & (gx < W64) & (gy >= 0) & (gy < H64)
            if not np.any(m):
                continue
            gx = gx[m]; gy = gy[m]

            flat = gy * W64 + gx
            cur = class_map_flat[flat]
            upd = cls_idx > cur
            if np.any(upd):
                class_map_flat[flat[upd]] = cls_idx

    return class_map

# =======================
# OUTPUTS
# =======================
def save_unified_json(base: str, class_map: np.ndarray, out_dir_json: str):
    output_coords = {}
    for label in ["PV_normal", "PV_heater", "PV_pool"]:
        idx = CLASS_PRIORITY[label]
        ys, xs = np.where(class_map == idx)  # ys = rows, xs = cols
        if ys.size == 0:
            continue
        # Write as (x, y)
        coords = [[int(x), int(y)] for (y, x) in zip(ys.tolist(), xs.tolist())]
        output_coords[label] = coords

    out_json = {"predicted_coords": output_coords}
    out_path = os.path.join(out_dir_json, f"{base}.json")
    with open(out_path, "w") as f:
        json.dump(out_json, f)
    return out_path

def polygonize_to_csv(base: str, class_map: np.ndarray, out_dir_csv: str):
    """
    For each PV_* class, extract external contours, compute area (m^2),
    and write one per-base CSV with columns:
      prediction_id, image_id, label, area_m2, vertice1_pixel, vertice2_pixel, ...
    Vertices are global pixel coords in [row, col] = [y, x] order.
    """
    rows = []
    max_verts = 0

    # Reused mask buffer
    mask = np.empty_like(class_map, dtype=np.uint8)

    RET = cv2.RETR_EXTERNAL
    CHAIN = cv2.CHAIN_APPROX_NONE  # preserves all vertices

    for label in ("PV_normal", "PV_heater", "PV_pool"):
        idx = CLASS_PRIORITY[label]

        # Fill mask in-place: (class_map == idx) -> {0,255}
        np.equal(class_map, idx, out=mask)  # boolean 0/1
        mask *= 255                         # -> 0/255

        if mask.max() == 0:
            continue

        contours, _ = cv2.findContours(mask, RET, CHAIN)
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            # ring in (x, y)
            ring_xy = cnt[:, 0, :]  # (N,2)
            # area in pixels (OpenCV is fast and stable)
            area_px = cv2.contourArea(cnt)
            if area_px <= 0:
                continue
            area_m2 = area_px * PIXEL_AREA_M2

            # CSV requires [row, col] = [y, x]
            verts_rc = [(float(y), float(x)) for (x, y) in ring_xy.tolist()]
            max_verts = max(max_verts, len(verts_rc))
            pred_id = polygon_hash_id(base, label, verts_rc)
            rows.append((pred_id, base, label, f"{area_m2:.6f}", verts_rc))

    # Write CSV (even empty -> header only)
    csv_path = os.path.join(out_dir_csv, f"{base}_instances.csv")
    if not rows:
        header = ["prediction_id", "image_id", "label", "area_m2"]
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return csv_path, 0

    header = ["prediction_id", "image_id", "label", "area_m2"] + \
             [f"vertice{i}_pixel" for i in range(1, max_verts + 1)]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pid, image_id, label, area_m2, verts_rc in rows:
            cells = [f"[{int(round(r))},{int(round(c))}]" for (r, c) in verts_rc]
            if len(cells) < max_verts:
                cells += [""] * (max_verts - len(cells))
            w.writerow([pid, image_id, label, area_m2] + cells)

    return csv_path, len(rows)

# =======================
# DISCOVERY
# =======================
from pathlib import Path

def discover_by_base(folder: str):
    """
    Recursively find all *.json files under `folder` and group by base image,
    attaching tile pixel offsets (x,y). Works with nested subfolders.
    """
    grouped = defaultdict(list)
    skipped = 0
    total = 0

    root = Path(folder)
    for p in root.rglob("*.json"):
        total += 1
        relpath = str(p.relative_to(root))   # e.g., "subdir1/subdir2/file.json"
        try:
            base, tile_x, tile_y = parse_base_and_offsets(p.name, tile_size=TILE_SIZE)
            if tile_x % TILE_SIZE != 0 or tile_y % TILE_SIZE != 0:
                raise ValueError("tile offsets not aligned to TILE_SIZE")
            grouped[base].append((relpath, tile_x, tile_y))
        except Exception as e:
            print(f"[WARN] {e} :: {relpath}")
            skipped += 1

    return grouped, skipped, total

# =======================
# PER-BASE WORKER
# =======================
def process_one_base(args_pack):
    base, items = args_pack
    # deterministic order helps debugging
    items_sorted = sorted(items, key=lambda t: (t[1], t[2]))
    class_map = stitch_one_base(base, items_sorted, json_folder)

    unified_path = ""
    if SAVE_UNIFIED_JSON:
        unified_path = save_unified_json(base, class_map, unified_json_dir)

    polys_csv = ""
    n_polys = 0
    if RUN_POLYGONIZATION:
        polys_csv, n_polys = polygonize_to_csv(base, class_map, out_dir)

    # quick per-base stats
    area_total_m2 = 0.0
    for label in ("PV_normal", "PV_heater", "PV_pool"):
        idx = CLASS_PRIORITY[label]
        area_total_m2 += float((class_map == idx).sum()) * PIXEL_AREA_M2

    # free memory per base
    del class_map
    return [base, n_polys, f"{area_total_m2:.2f}", unified_path, polys_csv]

# =======================
# MAIN
# =======================
def main():
    # 1) discover
    grouped, skipped, total = discover_by_base(json_folder)
    print(f"Discovered {sum(len(v) for v in grouped.values())} tile JSONs "
          f"across {len(grouped)} base images (skipped {skipped}/{total}).")

    bases = list(grouped.items())

    # 2) process per base (optionally in parallel)
    if PARALLEL_BASES and len(bases) > 1:
        nproc = max(1, min(cpu_count(), len(bases)))
        print(f"[INFO] Parallelizing per-base processing with {nproc} workers.")
        with Pool(processes=nproc) as pool:
            summary_rows = list(tqdm(pool.imap_unordered(process_one_base, bases),
                                     total=len(bases), desc="Processing bases"))
    else:
        summary_rows = []
        for pack in tqdm(bases, desc="Processing bases"):
            summary_rows.append(process_one_base(pack))

    # 3) write small summary
    summary_csv = os.path.join(out_dir, "SUMMARY_per_base.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base_image", "num_polygons", "pixel_area_sum_m2",
                    "unified_json_path", "polygons_csv_path"])
        w.writerows(summary_rows)

    print(f"\n✅ Done. Summary: {summary_csv}")
    if SAVE_UNIFIED_JSON:
        print(f"   Unified global JSONs in: {unified_json_dir}")
    if RUN_POLYGONIZATION:
        print(f"   Per-base polygon CSVs in: {out_dir}")

if __name__ == "__main__":
    main()


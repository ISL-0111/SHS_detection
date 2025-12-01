"""
(Nov29)
The old code used tile JSON filenames to get base image and tile offsets.
The new code reads JSONL metadata (image_name, tile_x, tile_y) and no longer depends on filenames.
This corresponds to changing from pe-tiles JSON files -> merged JSONL files, reducing I/O overhead in full-dataset inference.

======================
Coordinate System Notes
======================
Upstream::
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
from pathlib import Path

import numpy as np
import cv2
from shapely.geometry import Polygon
from tqdm import tqdm

# =======================
# CONFIG
# =======================
json_folder = "/shared/data/climateplus2025/Prediction_EntireDataset_Mask2Former_1024/2023/prediction_outputs_Mask2Former_probe_strict/"

# Output root
out_dir = "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2023/output_stage_1"
os.makedirs(out_dir, exist_ok=True)

# Unified JSONs
SAVE_UNIFIED_JSON = True
unified_json_dir = os.path.join(out_dir, "unified_json_global")
if SAVE_UNIFIED_JSON:
    os.makedirs(unified_json_dir, exist_ok=True)

# Polygonization
RUN_POLYGONIZATION = True

# Multi-processing
PARALLEL_BASES = True

# Canvas size
CANVAS_H = 12500
CANVAS_W = 12500
TILE_SIZE = 1024  # tiling size

# Pixel area (8cm GSD)
PIXEL_AREA_M2 = 0.08 ** 2

# Class priorities
CLASS_ORDER = ["background", "PV_normal", "PV_heater", "PV_pool"]
CLASS_PRIORITY = {"background": 0, "PV_normal": 1, "PV_heater": 2, "PV_pool": 3}

# Label normalization
CLASS_TO_LABEL = {
    "class_1": "PV_normal",
    "class_2": "PV_heater",
    "class_3": "PV_pool",
}

# Upstream JSON format is [y, x]
LEGACY_JSON_IS_YX = True

# Optional fast JSON loader (orjson)
try:
    import orjson
    def _load_json_file(path):
        with open(path, "rb") as f:
            return orjson.loads(f.read())
except Exception:
    def _load_json_file(path):
        with open(path, "r") as f:
            return json.load(f)

# Filename parsers (legacy support)
TILE_PIXEL_RE = re.compile(r"_tile_(\d+)_(\d+)", re.I)
INDEX_JSON_RE = re.compile(
    r"^(?P<prefix>i_)?(?P<base>.+?)_(?P<ix>\d+)_(?P<iy>\d+)(?:_pred_[^.]*)?\.json$",
    re.I
)

def parse_base_and_offsets(json_name: str, tile_size=TILE_SIZE):
    bn = os.path.basename(json_name)
    stem, _ = os.path.splitext(bn)

    m = TILE_PIXEL_RE.search(stem)
    if m:
        x_px, y_px = int(m.group(1)), int(m.group(2))
        base = stem.split("_tile_")[0]
        return base, x_px, y_px

    m = INDEX_JSON_RE.match(bn)
    if m:
        prefix = m.group("prefix") or ""
        base = prefix + m.group("base")
        ix, iy = int(m.group("ix")), int(m.group("iy"))
        return base, ix * tile_size, iy * tile_size

    raise ValueError(f"Cannot parse offsets from: {bn}")

# =======================
# NORMALIZE COORDS
# =======================
def _norm_key_to_label(k: str):
    ks = str(k).strip().lower()
    if ks in CLASS_TO_LABEL:
        return CLASS_TO_LABEL[ks]
    if ks in {"1", "2", "3"}:
        return CLASS_TO_LABEL[f"class_{ks}"]
    return k

def normalize_pred_coords(pred_coords: dict, clip_h=CANVAS_H, clip_w=CANVAS_W):
    out = {}
    for k, coords in (pred_coords or {}).items():
        label = _norm_key_to_label(k)
        if label not in CLASS_PRIORITY:
            continue

        arr = np.asarray(coords, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
            continue

        if LEGACY_JSON_IS_YX:
            arr = arr[:, [1, 0]]  # [y,x] -> (x,y)

        m = (arr[:, 0] >= 0) & (arr[:, 0] < clip_w) & (arr[:, 1] >= 0) & (arr[:, 1] < clip_h)
        if not np.any(m):
            continue

        out[label] = arr[m]

    out.pop("background", None)
    return out

# =======================
# POLYGON HELPERS
# =======================
def canonicalize_ring_rc(verts_rc):
    pts = [(int(round(r)), int(round(c))) for (r, c) in verts_rc]
    if not pts:
        return pts
    def rotate_to_min(seq): return seq[min(range(len(seq)), key=lambda i: seq[i]):] + seq[:min(range(len(seq)), key=lambda i: seq[i])]
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
def stitch_one_base(base: str, items):
    class_map = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    flat = class_map.ravel()
    W64, H64 = np.int64(CANVAS_W), np.int64(CANVAS_H)

    for image_name, tile_x, tile_y, pred_raw in items:
        pred = normalize_pred_coords(pred_raw)
        for label, arr in pred.items():
            cls = CLASS_PRIORITY[label]
            gx = arr[:, 0].astype(np.int64) + tile_x
            gy = arr[:, 1].astype(np.int64) + tile_y

            m = (gx >= 0) & (gx < W64) & (gy >= 0) & (gy < H64)
            gx, gy = gx[m], gy[m]
            idx = gy * W64 + gx

            cur = flat[idx]
            upd = cls > cur
            flat[idx[upd]] = cls

    return class_map

# =======================
# OUTPUT WRITERS
# =======================
def save_unified_json(base, class_map, out_dir_json):
    output = {}
    for label in ["PV_normal", "PV_heater", "PV_pool"]:
        idx = CLASS_PRIORITY[label]
        ys, xs = np.where(class_map == idx)
        if len(xs) == 0:
            continue
        output[label] = [[int(x), int(y)] for (y, x) in zip(ys, xs)]

    out_path = os.path.join(out_dir_json, f"{base}.json")
    with open(out_path, "w") as f:
        json.dump({"predicted_coords": output}, f)
    return out_path

def polygonize_to_csv(base, class_map, out_dir_csv):
    rows = []
    max_verts = 0
    mask = np.empty_like(class_map, dtype=np.uint8)

    for label in ("PV_normal", "PV_heater", "PV_pool"):
        idx = CLASS_PRIORITY[label]
        np.equal(class_map, idx, out=mask)
        mask *= 255
        if mask.max() == 0:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            ring_xy = cnt[:, 0, :]
            area_px = cv2.contourArea(cnt)
            if area_px <= 0:
                continue

            verts_rc = [(float(y), float(x)) for (x, y) in ring_xy.tolist()]
            max_verts = max(max_verts, len(verts_rc))

            pred_id = polygon_hash_id(base, label, verts_rc)
            rows.append((pred_id, base, label, f"{area_px * PIXEL_AREA_M2:.6f}", verts_rc))

    csv_path = os.path.join(out_dir_csv, f"{base}_instances.csv")
    if not rows:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["prediction_id", "image_id", "label", "area_m2"])
        return csv_path, 0

    header = ["prediction_id", "image_id", "label", "area_m2"] + \
             [f"vertice{i}_pixel" for i in range(1, max_verts + 1)]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pid, image_id, label, area, verts in rows:
            cells = [f"[{int(r)},{int(c)}]" for (r, c) in verts]
            if len(cells) < max_verts:
                cells += [""] * (max_verts - len(cells))
            w.writerow([pid, image_id, label, area] + cells)

    return csv_path, len(rows)

# =======================
# DISCOVERY (JSONL)
# =======================
def discover_by_base_jsonl(folder: str):
    grouped = defaultdict(list)
    skipped = 0
    total = 0

    root = Path(folder)
    files = list(root.rglob("*.jsonl"))
    print(f"[INFO] Found {len(files)} JSONL files.")

    for jsonl_path in files:
        with open(jsonl_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                total += 1
                try:
                    rec = json.loads(s)
                    image_name = rec["image_name"]
                    pred = rec.get("predicted_coords", {})

                    stem = os.path.splitext(os.path.basename(image_name))[0]
                    if "_tile_" not in stem:
                        raise ValueError("Missing _tile_ in image_name")

                    base = stem.split("_tile_")[0]
                    off = stem.split("_tile_")[1]
                    tile_x, tile_y = map(int, off.split("_"))

                    grouped[base].append((image_name, tile_x, tile_y, pred))

                except Exception:
                    skipped += 1

    return grouped, skipped, total

# =======================
# PER-BASE PROCESSOR
# =======================
def process_one_base(args_pack):
    base, items = args_pack
    items = sorted(items, key=lambda x: (x[1], x[2]))  # sort by tile_x, tile_y
    cmap = stitch_one_base(base, items)

    unified_path = save_unified_json(base, cmap, unified_json_dir) if SAVE_UNIFIED_JSON else ""
    polys_path, n_poly = polygonize_to_csv(base, cmap, out_dir) if RUN_POLYGONIZATION else ("", 0)

    area_total_m2 = float((cmap > 0).sum()) * PIXEL_AREA_M2
    del cmap
    return [base, n_poly, f"{area_total_m2:.2f}", unified_path, polys_path]

# =======================
# MAIN
# =======================
def main():
    grouped, skipped, total = discover_by_base_jsonl(json_folder)
    print(f"Discovered {sum(len(v) for v in grouped.values())} tile records "
          f"across {len(grouped)} base images (skipped {skipped}/{total}).")

    bases = list(grouped.items())

    if PARALLEL_BASES and len(bases) > 1:
        nproc = min(cpu_count(), len(bases))
        print(f"[INFO] Using {nproc} workers.")
        with Pool(nproc) as pool:
            summary = list(tqdm(pool.imap_unordered(process_one_base, bases),
                                total=len(bases), desc="Processing bases"))
    else:
        summary = []
        for pack in tqdm(bases, desc="Processing bases"):
            summary.append(process_one_base(pack))

    out_csv = os.path.join(out_dir, "SUMMARY_per_base.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base_image", "num_polygons", "pixel_area_sum_m2",
                    "unified_json_path", "polygons_csv_path"])
        w.writerows(summary)

    print(f"\nDone. Summary: {out_csv}")
    if SAVE_UNIFIED_JSON:
        print(f"Unified global JSONs in: {unified_json_dir}")
    if RUN_POLYGONIZATION:
        print(f"Per-base polygon CSVs in: {out_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm  # NEW: progress bars

# ===== Config (keep simple) =====
INPUT_PATH = Path("/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2023/output_stage_2")  # dir or single .csv
FILE_GLOB  = "*2023_*.csv"  # used only if INPUT_PATH is a dir
OUTPUT_CSV = Path("/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2023/output_stage_4/prediction_merged_2023.csv")
CHUNK_ROWS = 100_000
KEEP_DEFAULT_NA = False       # False => keep empty strings as ""

# Your actual pinned column names (appear first if present)
PINNED = [
    "image_id", "prediction_id", "label", "area_m2",
    "polygon_centroid_pixel[y,x]",
    "polygon_centroid_CRS[X,Y]",
    "polygon_centroid_GPS[lat,lon]",
]

# Match both "vertice1_pixel[y,x]" and "vertices_1_CRS[X,Y]" (ignore bracket suffix)
VERTEX_RE = re.compile(r"^vertices?_?(\d+)_(pixel|CRS|GPS)(?:\[.*\])?$")
TYPE_ORDER = {"pixel": 0, "CRS": 1, "GPS": 2}

def list_inputs(p: Path):
    if p.is_dir(): return sorted(p.glob(FILE_GLOB))
    if p.is_file() and p.suffix.lower()==".csv": return [p]
    return []

def choose_output(p: Path):
    if OUTPUT_CSV: return Path(OUTPUT_CSV)
    if p.is_dir(): return p / "Merged_2023_instances_with_centroid_crs_gps.csv"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return p.parent / f"Merged_{p.stem}_{ts}.csv"

def order_columns(all_cols):
    pinned = [c for c in PINNED if c in all_cols]
    rest   = [c for c in all_cols if c not in pinned]
    verts, others = [], []
    for c in rest:
        m = VERTEX_RE.match(c)
        (verts if m else others).append((int(m.group(1)), TYPE_ORDER[m.group(2)], c) if m else c)
    verts_sorted = [c for _,_,c in sorted(verts)]
    return pinned + verts_sorted + [c for c in others if isinstance(c, str)]

def main():
    files = list_inputs(INPUT_PATH)
    if not files:
        print("No inputs."); return

    out_csv = choose_output(INPUT_PATH)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: union of columns (preserve discovery order) — with progress
    all_cols, seen = [], set()
    for f in tqdm(files, desc="Scan headers", unit="file"):
        try:
            for c in pd.read_csv(f, nrows=0).columns:
                if c not in seen:
                    seen.add(c); all_cols.append(c)
        except Exception as e:
            print(f"[WARN] header: {f.name}: {e}")

    if not all_cols:
        print("No columns found."); return

    ordered = order_columns(all_cols)
    pd.DataFrame(columns=ordered).to_csv(out_csv, index=False)

    # Pass 2: append rows in chunks — with progress
    total = 0
    for f in tqdm(files, desc="Process files", unit="file"):
        try:
            file_rows = 0
            with tqdm(desc=f.name, unit="rows", leave=False) as pbar:
                for chunk in pd.read_csv(f, dtype=str, keep_default_na=KEEP_DEFAULT_NA, chunksize=CHUNK_ROWS):
                    chunk = chunk.reindex(columns=ordered)
                    # If you prefer no NaN at all: chunk = chunk.fillna("")
                    chunk.to_csv(out_csv, mode="a", header=False, index=False)
                    n = len(chunk)
                    total += n
                    file_rows += n
                    pbar.update(n)
            print(f"[INFO] {f.name}: {file_rows} rows")
        except Exception as e:
            print(f"[WARN] {f.name}: {e}")

    print(f"Done. Wrote {total} rows -> {out_csv}")

if __name__ == "__main__":
    main()

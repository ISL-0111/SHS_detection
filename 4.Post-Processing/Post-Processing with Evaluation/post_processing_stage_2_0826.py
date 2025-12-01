# (Sep.20) Automatic resume/skip logic is added

'''
===============================
Column & Coordinate Conventions
===============================
This script enforces explicit, order-annotated column names:

1) Pixel coordinates (row, col)  → **[y,x]**
   - **polygon_centroid_pixel[y,x]**
   - **verticeN_pixel[y,x]**  (N=1,2,...)
   - Rationale: images/arrays are indexed arr[row, col] = arr[y, x].

2) Projected CRS coordinates (Easting, Northing) → **[X,Y]**
   - **polygon_centroid_CRS[X,Y]**
   - **vertices_N_CRS[X,Y]**  (N=1,2,...)
   - Computed from pixel coords using rasterio.xy(row=y, col=x).

3) GPS coordinates (WGS84) → **[lat,lon]**
   - **polygon_centroid_GPS[lat,lon]**
   - Computed by transforming (X,Y) to WGS84. We output (lat,lon)
     for human-facing tools (QGIS, Google Maps search, etc.).

Backward compatibility:
- If input CSV uses legacy names:
    * "polygon_centroid" (no bracket hint)  → renamed to "polygon_centroid_pixel[y,x]"
    * "verticeN_pixel"                      → renamed to "verticeN_pixel[y,x]"
- Pixel strings are always parsed as "[y,x]".
- CRS strings are written as "[X,Y]".
- GPS strings are written as "[lat,lon]".

Summary:
- Pixel  : [y,x]
- CRS    : [X,Y]
- GPS    : [lat,lon]
'''

import os
import re
import ast
import math
import atexit
from pathlib import Path
from contextlib import contextmanager
from typing import Any, List, Tuple

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from rasterio import open as rio_open
from rasterio.transform import xy
from rasterio.env import Env
from pyproj import CRS, Transformer
from tqdm.auto import tqdm

# ------------------------
# I/O CONFIG
# ------------------------
INPUT_DIR = Path("/shared/data/climateplus2025/Postprocessing_for_poster_3_images_1024_Nov20/output_stage_1")
FILE_GLOB = "i_2023_*.csv"      # e.g. "i_2021_*.csv" or "*.csv"

OUTPUT_DIR = Path("/shared/data/climateplus2025/Postprocessing_for_poster_3_images_1024_Nov20/output_stage_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Folder containing FULL-SIZE base GeoTIFFs (not tiles)
BASE_TIF_DIR = Path("/data/data/capetown_bc_2025/Data/CapeTown_Image_2023_original")

# ------------------------
# CRS CONFIG
# ------------------------
CUSTOM_CRS_WKT = """
PROJCS["Hartebeesthoek94_Lo19_(E-N)",
    GEOGCS["Hartebeesthoek94",
        DATUM["Hartebeesthoek94",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6148"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4148"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",19],
    PARAMETER["scale_factor",1],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["ESRI","102562"]]
"""
FALLBACK_SOURCE_CRS = CRS.from_wkt(CUSTOM_CRS_WKT)
TARGET_CRS = CRS.from_epsg(4326)  # WGS84

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
    PROJ_NETWORK="OFF",
    CPL_DEBUG="OFF",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=""
)

# ------------------------
# Column name constants
# ------------------------
PIXEL_CENTROID_COL_NEW = "polygon_centroid_pixel[y,x]"
PIXEL_CENTROID_COL_OLD = "polygon_centroid"  # legacy accepted

CRS_CENTROID_COL = "polygon_centroid_CRS[X,Y]"
GPS_CENTROID_COL = "polygon_centroid_GPS[lat,lon]"

# vertex columns: accept both "verticeN_pixel" (legacy) and "verticeN_pixel[y,x]" (new)
VERTEX_COL_RE = re.compile(r"^vertice(\d+)_pixel(?:\[y,x\])?$")  # matches both styles

# ------------------------
# Helpers
# ------------------------
def parse_pixel_rc(cell):
    """
    Parse a pixel string like "[y,x]" → (y, x) as floats. Returns None if invalid.
    """
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            y, x = float(val[0]), float(val[1])
            if math.isfinite(y) and math.isfinite(x):
                return (y, x)
    except Exception:
        pass
    return None


def detect_vertex_columns(columns) -> List[Tuple[int, str]]:
    """
    Return all vertex columns that look like:
      - 'verticeN_pixel' (legacy)
      - 'verticeN_pixel[y,x]' (new)
    sorted by N ascending. Returns [(N, name), ...]
    """
    items: List[Tuple[int, str]] = []
    for c in columns:
        m = VERTEX_COL_RE.match(c)
        if m:
            items.append((int(m.group(1)), c))
    items.sort(key=lambda t: t[0])
    return items


def rename_vertex_columns_inplace(df: pd.DataFrame) -> None:
    """
    Rename any legacy 'verticeN_pixel' columns → 'verticeN_pixel[y,x]'.
    """
    renames = {}
    for n, name in detect_vertex_columns(df.columns):
        if name == f"vertice{n}_pixel":
            renames[name] = f"vertice{n}_pixel[y,x]"
    if renames:
        df.rename(columns=renames, inplace=True)


def get_vertex_column_names(df: pd.DataFrame) -> list[str]:
    """
    After potential renaming, return the canonical vertex column names.
    """
    return [name for _, name in detect_vertex_columns(df.columns)]


def polygon_centroid_from_row(row, vertex_cols):
    """
    Build a Shapely polygon from verticeN_pixel[y,x] columns to compute centroid.
    Steps:
      - Parse all present vertices to (y,x)
      - Keep order by N
      - Swap to (x,y) for Shapely
      - Return centroid as (x,y) floats or None
    """
    pts = []
    for name in vertex_cols:
        rc = parse_pixel_rc(row.get(name, None))
        if rc is not None:
            n = int(VERTEX_COL_RE.match(name).group(1))  # extract N
            pts.append((n, rc))

    if len(pts) < 3:
        return None

    pts_sorted_rc = [rc for _, rc in sorted(pts, key=lambda t: t[0])]
    xy = [(x, y) for (y, x) in pts_sorted_rc]

    try:
        poly = Polygon(xy)
        if not poly.is_empty and poly.area > 0:
            c = poly.centroid
            return (float(c.x), float(c.y))  # (x,y)
    except Exception:
        pass
    return None


def ensure_pixel_centroid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a **polygon_centroid_pixel[y,x]** column (pixel, [y,x]).
    - If legacy "polygon_centroid" exists, it will be **renamed** to the new name.
    - If missing or empty, compute from verticeN_pixel[y,x] columns (if available).
    """
    # 0) Rename legacy centroid if present
    if PIXEL_CENTROID_COL_OLD in df.columns and PIXEL_CENTROID_COL_NEW not in df.columns:
        df.rename(columns={PIXEL_CENTROID_COL_OLD: PIXEL_CENTROID_COL_NEW}, inplace=True)

    # 1) Ensure vertex columns are in canonical form
    rename_vertex_columns_inplace(df)
    vertex_cols = get_vertex_column_names(df)

    # 2) Check centroid presence/emptiness
    has_col = PIXEL_CENTROID_COL_NEW in df.columns
    if has_col:
        empty_mask = df[PIXEL_CENTROID_COL_NEW].astype(str).str.strip().eq("")
    else:
        empty_mask = pd.Series([True] * len(df), index=df.index)

    if empty_mask.any():
        centroids = [] if not has_col else df[PIXEL_CENTROID_COL_NEW].tolist()
        if not centroids:
            centroids = ["" for _ in range(len(df))]

        for i in tqdm(empty_mask[empty_mask].index, desc="Computing missing pixel centroids", unit="row"):
            c_xy = polygon_centroid_from_row(df.loc[i], vertex_cols)  # (x,y)
            if c_xy is None:
                centroids[i] = ""
            else:
                y, x = c_xy[1], c_xy[0]  # store as [y,x]
                y_int, x_int = int(round(y)), int(round(x))
                centroids[i] = f"[{y_int},{x_int}]"

        if has_col:
            df[PIXEL_CENTROID_COL_NEW] = centroids
        else:
            # Insert after "area_m2" if present, else append
            insert_at = df.columns.get_loc("area_m2") + 1 if "area_m2" in df.columns else len(df.columns)
            df.insert(insert_at, PIXEL_CENTROID_COL_NEW, centroids)

    return df


def resolve_base_tif(image_id_value: str) -> str | None:
    """
    Resolve a per-base GeoTIFF path for a given image_id (e.g., "i_2023_RGB_8cm_W24A_17").
    Tries:
      BASE_TIF_DIR / "<image_id>.tif"
      BASE_TIF_DIR / "<image_id-without-leading-i_>.tif"
    """
    if not image_id_value:
        return None
    stem = Path(image_id_value).stem
    cand1 = BASE_TIF_DIR / f"{stem}.tif"
    if cand1.exists():
        return str(cand1)
    if stem.startswith("i_"):
        cand2 = BASE_TIF_DIR / f"{stem[2:]}.tif"
        if cand2.exists():
            return str(cand2)
    return None


# ---------- Dataset/Transformer caches ----------
_DS_CACHE: dict[str, Any] = {}
_TF_CACHE: dict[tuple[str, str], Transformer] = {}


@contextmanager
def open_cached(path: str):
    ds = _DS_CACHE.get(path)
    if ds is None:
        ds = rio_open(path)
        _DS_CACHE[path] = ds
    try:
        yield ds
    finally:
        pass  # keep open; closed at exit


def get_src_crs(ds) -> CRS:
    """
    Robustly extract a pyproj.CRS from a rasterio dataset.
    Fallback to CUSTOM_CRS_WKT if missing or ambiguous.
    """
    if ds.crs is not None:
        # Try to keep exactly what the file says first
        try:
            return CRS.from_wkt(ds.crs.to_wkt())
        except Exception:
            # Fall back to EPSG if resolvable; else our project fallback
            try:
                epsg = ds.crs.to_epsg()
                return CRS.from_epsg(epsg) if epsg is not None else FALLBACK_SOURCE_CRS
            except Exception:
                return FALLBACK_SOURCE_CRS
    return FALLBACK_SOURCE_CRS


def get_transformer(src_crs: CRS, tgt_crs: CRS = TARGET_CRS) -> Transformer:
    """
    Cache & return a pyproj Transformer. always_xy=True to avoid axis-order surprises.
    """
    key = (src_crs.to_wkt(), tgt_crs.to_wkt())
    tf = _TF_CACHE.get(key)
    if tf is None:
        tf = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
        _TF_CACHE[key] = tf
    return tf


@atexit.register
def _close_all_ds():
    for ds in list(_DS_CACHE.values()):
        try:
            ds.close()
        except Exception:
            pass
    _DS_CACHE.clear()


def build_output_path(input_csv: Path) -> Path:
    """Write as '<stem>_with_centroid_crs_gps.csv' to OUTPUT_DIR."""
    return OUTPUT_DIR / f"{input_csv.stem}_with_centroid_crs_gps.csv"


# ------------------------
# NEW: Per-vertex CRS columns
# ------------------------
def vertex_crs_col_name(n: int) -> str:
    """
    Column name for per-vertex CRS coordinates.
    Uses the user's requested pattern: 'vertices_{N}_CRS[X,Y]'
    """
    return f"vertices_{n}_CRS[X,Y]"


def ensure_vertex_crs_columns(df: pd.DataFrame) -> List[Tuple[int, str, str]]:
    """
    Ensure CRS columns exist for each detected vertex.
    Returns a list of (n, pixel_col_name, crs_col_name).
    Inserts each CRS column immediately after its pixel column for readability.
    """
    rename_vertex_columns_inplace(df)  # normalize legacy names
    pairs: List[Tuple[int, str, str]] = []
    for n, pixel_col in detect_vertex_columns(df.columns):
        crs_col = vertex_crs_col_name(n)
        if crs_col not in df.columns:
            insert_at = df.columns.get_loc(pixel_col) + 1
            df.insert(insert_at, crs_col, [""] * len(df))
        pairs.append((n, pixel_col, crs_col))
    return pairs


# =========================
# ====== MAIN LOGIC =======
# =========================

def process_csv(csv_path: Path) -> Path | None:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])

    # 1) Ensure pixel centroid exists, column name normalized
    df = ensure_pixel_centroid(df)
    if PIXEL_CENTROID_COL_NEW not in df.columns:
        print(f"[WARN] {csv_path.name}: cannot compute '{PIXEL_CENTROID_COL_NEW}' — skipping.")
        return None

    # 2) Resolve base TIF per row (batch by image_id for fewer opens)
    image_ids = df.get("image_id", pd.Series([""] * len(df))).astype(str)
    unique_ids = image_ids.unique().tolist()

    # 3) Prepare centroid output columns (insert after pixel centroid)
    insert_at = df.columns.get_loc(PIXEL_CENTROID_COL_NEW) + 1
    if CRS_CENTROID_COL not in df.columns:
        df.insert(insert_at, CRS_CENTROID_COL, [""] * len(df))
        insert_at += 1
    if GPS_CENTROID_COL not in df.columns:
        df.insert(insert_at, GPS_CENTROID_COL, [""] * len(df))

    # 3.5) NEW: Prepare per-vertex CRS output columns (next to their pixel columns)
    vertex_triplets = ensure_vertex_crs_columns(df)  # [(n, pixel_col, crs_col), ...]

    # 4) For each base image, open TIF once and convert centroids & vertices
    for img_id in tqdm(unique_ids, desc="Per-base conversion", unit="image"):
        tif_path = resolve_base_tif(img_id)
        if not tif_path:
            # No GeoTIFF found for this base — leave entries blank
            continue

        with open_cached(tif_path) as src:
            src_crs = get_src_crs(src)
            tf = get_transformer(src_crs, TARGET_CRS)
            tform = src.transform  # rasterio Affine

            # rows for this base
            idxs = np.where(image_ids.values == img_id)[0]
            for i in idxs:
                # ----- centroid as before -----
                c = df.at[i, PIXEL_CENTROID_COL_NEW]
                rc = parse_pixel_rc(c)  # [y,x] → (y,x)
                if rc:
                    y_px, x_px = rc[0], rc[1]  # row, col
                    X, Y = xy(tform, y_px, x_px, offset="center")  # map CRS
                    lon, lat = tf.transform(X, Y)
                    df.at[i, CRS_CENTROID_COL] = f"[{round(X, 3)},{round(Y, 3)}]"           # [X,Y]
                    df.at[i, GPS_CENTROID_COL] = f"[{round(lat, 8)},{round(lon, 8)}]"       # [lat,lon]

                # ----- NEW: per-vertex conversion -----
                # vertex_triplets = [(n, pixel_col, crs_col), ...]
                for _, px_col, crs_col in vertex_triplets:
                    v = df.at[i, px_col] if px_col in df.columns else ""
                    v_rc = parse_pixel_rc(v)
                    if not v_rc:
                        continue
                    vy_px, vx_px = v_rc[0], v_rc[1]  # row, col
                    VX, VY = xy(tform, vy_px, vx_px, offset="center")  # map CRS
                    df.at[i, crs_col] = f"[{round(VX, 3)},{round(VY, 3)}]"  # [X,Y]

    # 5) Save atomically
    out_path = build_output_path(csv_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    return out_path


def main():
    csv_files = sorted(INPUT_DIR.glob(FILE_GLOB))
    if not csv_files:
        print(f"No matching CSV files ({FILE_GLOB}) found in: {INPUT_DIR}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in: {INPUT_DIR}")
    ok = 0
    for f in tqdm(csv_files, desc="Processing CSVs", unit="file"):
        out_path = build_output_path(f)
        part_path = out_path.with_suffix(out_path.suffix + ".part")

        # --- resume/skip logic ---
        # Case 1: completed output exists and no partial -> skip
        if out_path.exists() and not part_path.exists():
            # print(f"Skip (exists): {out_path}")  # optional logging
            continue

        # Case 2: stale partial file -> remove and regenerate
        if part_path.exists():
            try:
                part_path.unlink()
            except Exception as e:
                print(f"[WARN] Could not remove stale partial {part_path}: {e}")

        try:
            outp = process_csv(f)
            if outp is not None:
                ok += 1
                print(f"Wrote: {outp}")
        except Exception as e:
            print(f"[WARN] {f.name}: {e}")

    print(f"Done. Wrote {ok}/{len(csv_files)} file(s) to: {OUTPUT_DIR}")


if __name__ == "__main__":
    with Env(**GDAL_ENV):
        main()


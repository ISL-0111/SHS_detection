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
INPUT_DIR = Path("/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2023/output_stage_1")
FILE_GLOB = "*2023_*.csv"      # e.g. "i_2021_*.csv" or "*.csv"

OUTPUT_DIR = Path("/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2023/output_stage_2")
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
# Column Name Convention
# ------------------------
PIXEL_CENTROID_COL_NEW = "polygon_centroid_pixel[y,x]"
PIXEL_CENTROID_COL_OLD = "polygon_centroid"

CRS_CENTROID_COL = "polygon_centroid_CRS[X,Y]"
GPS_CENTROID_COL = "polygon_centroid_GPS[lat,lon]"

# vertex pixel column pattern
VERTEX_COL_RE = re.compile(r"^vertice(\d+)_pixel(?:\[y,x\])?$")


# ------------------------
# Helpers
# ------------------------
def parse_pixel_rc(cell):
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
    except:
        pass
    return None


def detect_vertex_columns(columns):
    items = []
    for c in columns:
        m = VERTEX_COL_RE.match(c)
        if m:
            items.append((int(m.group(1)), c))
    return sorted(items, key=lambda t: t[0])


def canonical_vertex_name(n):
    return f"vertice{n}_pixel[y,x]"


def rename_vertex_columns(df):
    renames = {}
    for n, name in detect_vertex_columns(df.columns):
        if name == f"vertice{n}_pixel":
            renames[name] = canonical_vertex_name(n)
    if renames:
        df.rename(columns=renames, inplace=True)


def polygon_centroid_from_row(row, vertex_cols):
    pts = []
    for n, col in vertex_cols:
        rc = parse_pixel_rc(row.get(col, None))
        if rc is not None:
            pts.append((n, rc))
    if len(pts) < 3:
        return None
    pts_sorted = [rc for _, rc in sorted(pts, key=lambda t: t[0])]
    xy = [(x, y) for (y, x) in pts_sorted]

    try:
        poly = Polygon(xy)
        if not poly.is_empty and poly.area > 0:
            c = poly.centroid
            return (float(c.x), float(c.y))
    except:
        return None
    return None


def ensure_pixel_centroid(df):
    # rename legacy centroid
    if PIXEL_CENTROID_COL_OLD in df.columns and PIXEL_CENTROID_COL_NEW not in df.columns:
        df.rename(columns={PIXEL_CENTROID_COL_OLD: PIXEL_CENTROID_COL_NEW}, inplace=True)

    rename_vertex_columns(df)
    vertex_cols = detect_vertex_columns(df.columns)
    has_col = PIXEL_CENTROID_COL_NEW in df.columns

    if has_col:
        empty_mask = df[PIXEL_CENTROID_COL_NEW].astype(str).str.strip().eq("")
    else:
        empty_mask = pd.Series([True] * len(df), index=df.index)

    if not empty_mask.any():
        return df

    centroids = df[PIXEL_CENTROID_COL_NEW].tolist() if has_col else [""] * len(df)

    for i in tqdm(empty_mask[empty_mask].index, desc="Computing missing pixel centroids"):
        c_xy = polygon_centroid_from_row(df.loc[i], vertex_cols)
        if c_xy is None:
            centroids[i] = ""
        else:
            y, x = c_xy[1], c_xy[0]
            centroids[i] = f"[{int(round(y))},{int(round(x))}]"

    df[PIXEL_CENTROID_COL_NEW] = centroids
    return df


def resolve_base_tif(image_id):
    if not image_id:
        return None
    stem = Path(image_id).stem
    cand1 = BASE_TIF_DIR / f"{stem}.tif"
    if cand1.exists():
        return str(cand1)
    if stem.startswith("i_"):
        cand2 = BASE_TIF_DIR / f"{stem[2:]}.tif"
        if cand2.exists():
            return str(cand2)
    return None


_DS_CACHE = {}
_TF_CACHE = {}


@contextmanager
def open_cached(path):
    ds = _DS_CACHE.get(path)
    if ds is None:
        ds = rio_open(path)
        _DS_CACHE[path] = ds
    try:
        yield ds
    finally:
        pass


def get_src_crs(ds):
    if ds.crs is not None:
        try:
            return CRS.from_wkt(ds.crs.to_wkt())
        except:
            try:
                epsg = ds.crs.to_epsg()
                return CRS.from_epsg(epsg) if epsg else FALLBACK_SOURCE_CRS
            except:
                return FALLBACK_SOURCE_CRS
    return FALLBACK_SOURCE_CRS


def get_transformer(src_crs, tgt_crs=TARGET_CRS):
    key = (src_crs.to_wkt(), tgt_crs.to_wkt())
    tf = _TF_CACHE.get(key)
    if tf is None:
        tf = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
        _TF_CACHE[key] = tf
    return tf


@atexit.register
def close_all():
    for ds in _DS_CACHE.values():
        try:
            ds.close()
        except:
            pass
    _DS_CACHE.clear()


def build_output(csv_path):
    return OUTPUT_DIR / f"{csv_path.stem}_with_centroid_crs_gps.csv"


# ------------------------
# Main optimized function
# ------------------------
def process_csv(csv_path):

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    df = ensure_pixel_centroid(df)
    rename_vertex_columns(df)

    vertex_cols = detect_vertex_columns(df.columns)

    # === STEP 1: Prepare all new columns in dicts (no insert!) ===
    new_cols = {}

    # centroid CRS/GPS columns (empty, will fill later)
    if CRS_CENTROID_COL not in df.columns:
        new_cols[CRS_CENTROID_COL] = [""] * len(df)
    if GPS_CENTROID_COL not in df.columns:
        new_cols[GPS_CENTROID_COL] = [""] * len(df)

    # per-vertex CRS columns
    for n, pixel_col in vertex_cols:
        crs_col = f"vertices_{n}_CRS[X,Y]"
        if crs_col not in df.columns:
            new_cols[crs_col] = [""] * len(df)

    # === STEP 2: Add all new columns at once ===
    if len(new_cols) > 0:
        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
        df = df.copy()  # defragment → huge speed gain

    # === STEP 3: CRS/GPS coordinate computation ===
    image_ids = df.get("image_id", pd.Series([""] * len(df))).astype(str)
    unique_ids = image_ids.unique().tolist()

    for img_id in tqdm(unique_ids, desc="Computing CRS/GPS"):
        tif_path = resolve_base_tif(img_id)
        if not tif_path:
            continue

        with open_cached(tif_path) as src:
            src_crs = get_src_crs(src)
            tf = get_transformer(src_crs)
            tform = src.transform

            idxs = np.where(image_ids.values == img_id)[0]

            for i in idxs:
                # centroid
                rc = parse_pixel_rc(df.at[i, PIXEL_CENTROID_COL_NEW])
                if rc:
                    y_px, x_px = rc
                    X, Y = xy(tform, y_px, x_px, offset="center")
                    lon, lat = tf.transform(X, Y)
                    df.at[i, CRS_CENTROID_COL] = f"[{round(X,3)},{round(Y,3)}]"
                    df.at[i, GPS_CENTROID_COL] = f"[{round(lat,8)},{round(lon,8)}]"

                # vertices
                for n, pixel_col in vertex_cols:
                    crs_col = f"vertices_{n}_CRS[X,Y]"
                    v_rc = parse_pixel_rc(df.at[i, pixel_col])
                    if v_rc:
                        vy, vx = v_rc
                        VX, VY = xy(tform, vy, vx, offset="center")
                        df.at[i, crs_col] = f"[{round(VX,3)},{round(VY,3)}]"

    # === STEP 4: atomic save ===
    out_path = build_output(csv_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    return out_path


def main():
    csv_files = sorted(INPUT_DIR.glob(FILE_GLOB))
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} files.")
    ok = 0

    for f in tqdm(csv_files, desc="Processing files"):
        out_path = build_output(f)
        part = out_path.with_suffix(out_path.suffix + ".part")

        if out_path.exists() and not part.exists():
            continue

        if part.exists():
            try:
                part.unlink()
            except:
                pass

        try:
            if process_csv(f) is not None:
                ok += 1
        except Exception as e:
            print(f"[WARN] {f.name}: {e}")

    print(f"Completed {ok}/{len(csv_files)} files.")


if __name__ == "__main__":
    with Env(**GDAL_ENV):
        main()

#(Sep.21) Update attribute fields name error = drop columns starting with vertices_ / vertice

"""
CSV -> GeoPackage (GPKG) converter for instance polygons.

Sep.21 fast + column-safe:
- Uses itertuples(name=None) + a fast '[x,y]' parser (no AST)
- Calls make_valid() only when a polygon is invalid
- OUTPUT KEEPS ONLY: geometry, prediction_id, image_id, label, area_m2, centroid fields (if present), vertex_count
- EXCLUDES all vertices_* / vertice* columns from the written GPKG
- Sanitizes ONLY the remaining attribute column names (geometry untouched)
"""

import re
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid

# ----------------------------
# === CONFIG YOU WILL CHANGE ===
# ----------------------------
INPUT_DIR = "/shared/data/climateplus2025/Postprocessing_for_poster_3_images_1024_Nov20/output_stage_2"
OUTPUT_DIR = "/shared/data/climateplus2025/Postprocessing_for_poster_3_images_1024_Nov20/output_stage_3"
COMBINE_INTO_SINGLE_GPKG = False
SINGLE_GPKG_NAME = "pv_instances.gpkg"
LAYER_NAME_SUFFIX = ""
CRS_FOR_OUTPUT = """
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
# ----------------------------

# ---------- Sanitization helpers (attribute fields only) ----------
SAFE_COL_CHARS = re.compile(r'[^A-Za-z0-9_]')

def _dedupe(name: str, used: set, limit: int = 63) -> str:
    base = name[:limit] if name else "f_"
    if base not in used:
        return base
    i = 1
    while True:
        suf = f"_{i}"
        cand = (base[: limit - len(suf)]) + suf
        if cand not in used:
            return cand
        i += 1

def sanitize_field_names(columns: List[str]) -> Dict[str, str]:
    """Return mapping original -> sanitized (A-Za-z0-9_, starts with [A-Za-z_], max 63 chars, deduped)."""
    mapping: Dict[str, str] = {}
    used = set()
    for col in columns:
        new = SAFE_COL_CHARS.sub('_', col)
        if not new or not re.match(r'[A-Za-z_]', new[0]):
            new = 'f_' + new
        new = _dedupe(new, used, limit=63)
        used.add(new)
        mapping[col] = new
    return mapping

def apply_sanitized_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sanitize ONLY non-geometry columns; no JSON sidecar."""
    geom_col = gdf.geometry.name  # usually "geometry"
    cols = list(gdf.columns)
    to_sanitize = [c for c in cols if c != geom_col]
    mapping = sanitize_field_names(to_sanitize)
    return gdf.rename(columns=mapping)
# ------------------------------------------------------------------

# Patterns for vertex columns (used ONLY to build geometry; not written to output)
PAIR_COL_PATTERNS = [
    r"^vertices?_(\d+)_CRS\[X,Y\]$",
    r"^vertice?(\d+)_CRS\[X,Y\]$",
]

# Attribute columns to KEEP in output (centroids included if present)
BASE_ATTR_COLS = [
    "prediction_id", "image_id", "label", "area_m2",
    "polygon_centroid_pixel[y,x]",
    "polygon_centroid_CRS[X,Y]",
    "polygon_centroid_GPS[lat,lon]",
]

# --------- Fast parser & geometry builder ----------
def parse_pair_fast(s: object) -> Optional[Tuple[float, float]]:
    """Fast parse for strings like '[x,y]' without AST overhead."""
    if s is None:
        return None
    t = str(s).strip()
    if len(t) < 5 or t[0] != '[' or t[-1] != ']':
        return None
    try:
        x_str, y_str = t[1:-1].split(',', 1)
        return float(x_str), float(y_str)
    except Exception:
        return None

def build_geometry(coords: List[Tuple[float, float]]):
    """Create polygon; call make_valid only when actually invalid; normalize basic collections."""
    try:
        poly = Polygon(coords)
        if poly.is_valid:
            return poly
        poly = make_valid(poly)  # expensive; do only if invalid
        if poly is None or poly.is_empty:
            return None
        if isinstance(poly, GeometryCollection):
            polys = [g for g in poly.geoms if isinstance(g, (Polygon, MultiPolygon))]
            if not polys:
                return None
            parts = []
            for g in polys:
                if isinstance(g, Polygon):
                    parts.append(g)
                elif isinstance(g, MultiPolygon):
                    parts.extend(list(g.geoms))
            if not parts:
                return None
            return MultiPolygon(parts) if len(parts) > 1 else parts[0]
        return poly
    except Exception:
        return None
# ---------------------------------------------------

def discover_vertex_cols(df: pd.DataFrame) -> Dict[str, int]:
    """Find all vertex CRS columns and map to their integer index."""
    col_to_idx: Dict[str, int] = {}
    for col in df.columns:
        for pat in PAIR_COL_PATTERNS:
            m = re.match(pat, col)
            if m:
                try:
                    col_to_idx[col] = int(m.group(1))
                except Exception:
                    pass
    return col_to_idx

def coords_from_row_tuple(row_tuple: tuple, col_to_idx: Dict[str, int], col_index: Dict[str, int]) -> Optional[List[Tuple[float, float]]]:
    """Extract ordered coords for a row (tuple) using discovered vertex columns."""
    pairs: List[Tuple[int, Tuple[float, float]]] = []
    for col, idx in col_to_idx.items():
        j = col_index.get(col)
        if j is None:
            continue
        xy = parse_pair_fast(row_tuple[j])
        if xy is not None:
            pairs.append((idx, xy))
    if len(pairs) < 3:
        return None
    pairs.sort(key=lambda t: t[0])
    coords = [xy for _, xy in pairs]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords

def csv_to_gdf(csv_path: Path, crs_for_output) -> gpd.GeoDataFrame:
    """Convert a CSV to a GeoDataFrame with geometry and selected attributes. Fast path using itertuples."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Discover vertex columns (CRS coordinates only)
    col_to_idx = discover_vertex_cols(df)
    if not col_to_idx:
        raise RuntimeError(f"No CRS vertex columns found in {csv_path.name}")

    # Build records
    records = []
    keep_cols = [c for c in BASE_ATTR_COLS if c in df.columns]  # DO NOT include vertex columns

    df_cols = list(df.columns)
    col_index = {c: i for i, c in enumerate(df_cols)}

    for row in df.itertuples(index=False, name=None):  # fast tuples
        coords = coords_from_row_tuple(row, col_to_idx, col_index)
        if coords is None:
            continue
        geom = build_geometry(coords)
        if geom is None or geom.is_empty:
            continue

        rec = {c: row[col_index[c]] for c in keep_cols if c in col_index}
        rec["geometry"] = geom
        rec["vertex_count"] = len(coords) - 1  # optional QA
        records.append(rec)

    if not records:
        raise RuntimeError(f"No valid polygons constructed from {csv_path.name}")

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=None)
    if crs_for_output is not None:
        # define (do NOT reproject) the CRS of the input coords
        gdf = gdf.set_crs(crs_for_output)

    return gdf

def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(glob.glob(str(in_dir / "*.csv")))
    if not csv_files:
        print(f"No CSV files found under {in_dir.resolve()}")
        sys.exit(1)

    if COMBINE_INTO_SINGLE_GPKG:
        gpkg_path = out_dir / SINGLE_GPKG_NAME
        if gpkg_path.exists():
            gpkg_path.unlink()

    for csvf in csv_files:
        csv_path = Path(csvf)
        print(f"Processing: {csv_path.name}")
        gdf = csv_to_gdf(csv_path, CRS_FOR_OUTPUT)

        # Sanitize only the small set of kept attributes (geometry untouched)
        gdf_out = apply_sanitized_columns(gdf)

        if COMBINE_INTO_SINGLE_GPKG:
            layer_name = csv_path.stem + (f"_{LAYER_NAME_SUFFIX}" if LAYER_NAME_SUFFIX else "")
            gdf_out.to_file(gpkg_path, layer=layer_name, driver="GPKG")
            print(f"  -> wrote layer '{layer_name}' to {gpkg_path.name} ({len(gdf_out)} features)")
        else:
            layer_name = "instances" + (f"_{LAYER_NAME_SUFFIX}" if LAYER_NAME_SUFFIX else "")
            gpkg_name = csv_path.stem + ".gpkg"
            gpkg_path = out_dir / gpkg_name
            if gpkg_path.exists():
                gpkg_path.unlink()
            gdf_out.to_file(gpkg_path, layer=layer_name, driver="GPKG")
            print(f"  -> wrote {gpkg_name} (layer '{layer_name}', {len(gdf_out)} features)")

    print("Done.")

if __name__ == "__main__":
    main()
# Run this Streamlit app in GAIA CLI
'''
This code is designed to run within the GAIA environment using Streamlit.
To execute the application, you must launch it through the GAIA CLI, specifying the port settings and the file path as shown below. 
After running the command, access the locally assigned port through a Firefox browser (Safari is not supported).

/home/il72/.local/bin/streamlit run /shared/data/climateplus2025/Installation_year_detection_POC_Dec1/Visualization/app.py --server.port 8502 --server.address 0.0.0.0
'''

import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from PIL import Image, ImageDraw
import numpy as np
import os

# ---------------------------------------------------
# 1. File Paths
# ---------------------------------------------------
GPKG_PATH = "/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023_Mask2Former_1024_Nov29/2023/output_post_processing_polygonization_grouping_drop_small_objects/prediction_merged_2023_final_visualization.gpkg"

YEAR_DIRS = {
    2023: "/data/data/capetown_bc_2025/Data/CapeTown_Image_2023_original",
    2022: "/data/data/capetown_bc_2025/Data/CapeTown_Image_2022_original",
    2021: "/data/data/capetown_bc_2025/Data/CapeTown_Image_2021_original",
}

# Colors (30% transparent fill)
CLASS_COLORS = {
    "PV_normal": (0, 255, 0, 76),
    "PV_pool":   (255, 0, 0, 76),
    "PV_heater": (0, 0, 255, 76),
}

OUTLINE_COLORS = {
    "PV_normal": (0, 255, 0, 255),
    "PV_pool":   (255, 0, 0, 255),
    "PV_heater": (0, 0, 255, 255),
}

# ---------------------------------------------------
# 2. Helper: Clean image ID by removing "i_"
# ---------------------------------------------------
def clean_id(image_id):
    """Remove leading 'i_' prefix from TIFF name."""
    if image_id.startswith("i_"):
        return image_id[2:]
    return image_id


# ---------------------------------------------------
# 3. Load GPKG
# ---------------------------------------------------
st.title("Temporal Visualization Viewer (2021–2023, 1024 Crop Window)")

@st.cache_resource
def load_gpkg():
    gdf = gpd.read_file(GPKG_PATH)
    grouped = gdf.groupby("image_id")
    return gdf, grouped

gdf, grouped = load_gpkg()
image_ids = sorted(list(grouped.groups.keys()))


# ---------------------------------------------------
# 4. Navigation
# ---------------------------------------------------
if "idx" not in st.session_state:
    st.session_state.idx = 0

selected_from_dropdown = st.selectbox("Select image_id", image_ids, index=st.session_state.idx)
st.session_state.idx = image_ids.index(selected_from_dropdown)

col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("◀ Previous"):
        st.session_state.idx = max(0, st.session_state.idx - 1)
with col_next:
    if st.button("Next ▶"):
        st.session_state.idx = min(len(image_ids) - 1, st.session_state.idx + 1)

selected_id_raw = image_ids[st.session_state.idx]
rows = grouped.get_group(selected_id_raw)

# Cleaned ID for filenames
selected_id = clean_id(selected_id_raw)


# ---------------------------------------------------
# 5. Compute crop window based on 2023 polygon centroid
# ---------------------------------------------------
reference_row = rows.iloc[0]
geom = reference_row.geometry
centroid = geom.centroid
cx, cy = centroid.x, centroid.y

def compute_window(src, center_x, center_y, size=1024):
    transform = src.transform
    px, py = ~transform * (center_x, center_y)

    px, py = int(px), int(py)
    half = size // 2
    return Window(px - half, py - half, size, size)


# ---------------------------------------------------
# 6. Polygon overlay
# ---------------------------------------------------
def overlay_polygons_on_img(img, rows, win_transform):
    draw = ImageDraw.Draw(img, "RGBA")

    for _, row in rows.iterrows():
        geom = row.geometry
        label = row.label
        pred = row.prediction_id
        area = row.area_m2

        fill_color = CLASS_COLORS[label]
        outline_color = OUTLINE_COLORS[label]

        # Correct MultiPolygon handling
        if geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        elif geom.geom_type == "Polygon":
            polygons = [geom]
        else:
            continue  # skip unexpected geometry types

        for poly in polygons:
            coords = poly.exterior.coords

            px_list = []
            for x, y in coords:
                px, py = ~win_transform * (x, y)
                px_list.append((int(px), int(py)))

            draw.polygon(px_list, fill=fill_color, outline=outline_color)

            min_x = min(p[0] for p in px_list)
            min_y = min(p[1] for p in px_list)

            text = f"{pred}\n{label}\n{area:.1f} m²"
            draw.rectangle((min_x, min_y - 12, min_x + 150, min_y + 40), fill=(0, 0, 0, 120))
            draw.text((min_x + 3, min_y - 10), text, fill=(255, 255, 255, 255))

    return img


# ---------------------------------------------------
# 7. Load crop for each year
# ---------------------------------------------------
def load_crop(base_id_clean, year, center_x, center_y, size=1024):
    # Replace the year numerically in the cleaned filename
    year_id = base_id_clean.replace("2023", str(year))

    path = os.path.join(YEAR_DIRS[year], year_id + ".tif")

    if not os.path.exists(path):
        return None, None

    with rasterio.open(path) as src:
        window = compute_window(src, center_x, center_y, size)
        data = src.read([1, 2, 3], window=window)
        win_transform = src.window_transform(window)

    rgb = np.transpose(data, (1, 2, 0))
    rgb = np.nan_to_num(rgb)
    rgb = ((rgb - rgb.min()) / (rgb.ptp() + 1e-6) * 255).astype(np.uint8)

    return Image.fromarray(rgb).convert("RGBA"), win_transform


# ---------------------------------------------------
# 8. Display images
# ---------------------------------------------------
cols = st.columns(3)
years = [2021, 2022, 2023]

for col, year in zip(cols, years):
    img_crop, win_transform = load_crop(selected_id, year, cx, cy)

    if img_crop is None:
        col.write(f"No image for {year}")
        continue

    img_overlay = overlay_polygons_on_img(img_crop.copy(), rows, win_transform)
    col.image(img_overlay, caption=f"{year}")

st.success("1024 Crop Visualization Complete!")

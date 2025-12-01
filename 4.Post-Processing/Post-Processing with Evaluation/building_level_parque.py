import pandas as pd
import geopandas as gpd
from shapely import wkb

# 1. Load Parquet file
file_path = '/shared/data/climateplus2025/Post_Processing_with_Evaluation_1024_Nov20/1+2+3+4.Drop_any_predicts_no_buildings/capetown_buildings2.parquet'
df = pd.read_parquet(file_path)
df.head(3)

# 2. Convert WKB to geometry
df['geometry'] = df['geometry'].apply(wkb.loads)

# 3. Create GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# 4. ✅ Assign the correct *original* CRS (OGC:CRS84 = WGS84 = GPS degrees)
gdf.set_crs("OGC:CRS84", inplace=True)

# 5. ✅ Reproject to match your imagery (Hartebeesthoek94 / Lo19)
gdf = gdf.to_crs("ESRI:102562")

# 6. Save to GeoPackage
output_path = '/shared/data/climateplus2025/Post_Processing_with_Evaluation_1024_Nov20/1+2+3+4.Drop_any_predicts_no_buildings/output_buildings_sameCRS.gpkg'
gdf.to_file(output_path, driver='GPKG')

print(f"File correctly reprojected and saved with ESRI:102562 CRS -> {output_path}")

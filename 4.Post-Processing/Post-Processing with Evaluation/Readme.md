# Post Processing
This converts pixel-level predictions into instance-level outputs and applies additional refinement steps to improve prediction quality, mainly by reducing false positives.

The pipeline consists of four stages:

**1. Polygonization (Connect Adjacent Pixels)**
**2. Grouping Neighboring Polygons**
**3. Dropping small objects**
**4. Building footprint filtering**

* The combination of **steps 1+2+3** provides the most balanced results, and performance is evaluated at each cumulative stage (1 or 1+2 or 1+2+3 or 1+2+3+4)

* The `.gpkg` files generated at each stage, enabling visualization in QGIS

    <img width="1497" height="676" alt="Image" src="https://github.com/user-attachments/assets/177f9bc8-fa76-406a-b16c-6e23d4e60f80" />

* The econometric pipeline receives a single merged prediction `CSV` from `post_processing_stage_4_0827.py` and independently applies both Drop small objects (Stage3) and Building footprint filtering (Stage4).
**Path:** `/shared/data/climateplus2025/Postprocessing_EntireDataset_CapeTown_Image_2018_2023/post_processing_stage_4_0827.py`

#### ✅[1] Polygonization(Connect Adjacent Pixels)

In this Stage 1, processing and evaluation were implemented in separate files, whereas all subsequent stages were executed within a single file.

**1.1 Processing**
**Path:** `/shared/data/climateplus2025/Postprocessing_for_poster_3_images_1024_Nov20/`

This consist of 3 files in order:
- `post_processing_stage_1_0916.py`: Reconstructs full-resolution (12500*12500) predictions from tile-level JSONs. Uses `findContours()` to extract boundaries—each contour becomes one PV related instance.

- `post_processing_stage_2_0916.py`: Converts Stage-1 polygon CSVs into georeferenced data: Pixel → CRS (meters) → GPS (lat/lon), and computes missing centroids.

- `post_processing_stage_3_0916.py`: Builds clean GIS-ready GeoPackages: Reconstructs polygons using CRS vertices, removes raw vertex columns, sanitizes fields, and writes valid `GPKG` layers for QGIS.

**1.2 Evaluation** 
**File:** `evaluation_instance_v0_1_including_FP.ipynb` 
**Path:** `/shared/data/climateplus2025/Post_Processing_with_Evaluation_1024_Nov20/1.Connect_adjacent_pixels/evaluation_instance_v0_1_including_FP.ipynb`

* This evaluation matches each predicted polygon to ground truth using COCO-style one-to-one IoU-based matching, without using confidence scores. Any unmatched or duplicated predictions become false positives, and any GT without a matched prediction becomes a false negative, making the evaluation strict and instance-oriented.




#### [1]Polygonization + ✅[2]Groupping neighboring polygons

Fragments split at map boundaries are typically within 2 pixels. so, polygons of the same class that are within 2 pixels of each other are merged.

**File:** `evaluation_instance_v1_neighboring.ipynb`
**path:** `/shared/data/climateplus2025/Post_Processing_with_Evaluation_1024_Nov20/1+2.Group_neighboring_polygons/evaluation_instance_v1_neighboring.ipynb`

#### [1]Polygonization + [2]Groupping neighboring polygons + ✅[3]Dropping small object → Most Balanced combination

Objects smaller than 1.7 m² (the smallest true target size) are treated as false positives.
- *Note: Real objects may appear smaller, so a lower threshold such as 0.816 m² is also feasible based on GT distribution.*

**File:** `evaluation_instance_v1_drop_small_polygons.ipynb`
**Path:** `/shared/data/climateplus2025/Post_Processing_with_Evaluation_1024_Nov20/1+2+3.Drop_small_polygons/evaluation_instance_v1_drop_small_polygons.ipynb`

#### [1]Polygonization + [2]Groupping neighboring polygons + [3]Dropping small objects + ✅[4]Building footpring filtering

Predicted objects overlapping less than 80% with the Cape Town building footprint dataset are dropped as false positives.

**4.1 Building footprint data preprocessing**
**File1:** `building_level_parque.py`
**Path:** `/shared/data/climateplus2025/Post_Processing_with_Evaluation_1024_Nov20/1+2+3+4.Drop_any_predicts_no_buildings/building_level_parque.py`
- *Note:* Input file (Building_footprint_parque) `/shared/data/climateplus2025/Post_Processing_with_Evaluation_1024_Nov20/1+2+3+4.Drop_any_predicts_no_buildings/capetown_buildings2.parquet`

**4.2 Instance Filtering**
**File:** `evaluation_instance_v1_building_level.ipynb`
**Path:** `/shared/data/climateplus2025/Post_Processing_with_Evaluation_1024_Nov20/1+2+3+4.Drop_any_predicts_no_buildings/evaluation_instance_v1_building_level.ipynb`


#### Evaluation summary by Stage

**[1] Polygonization** 

Overall Metrics: 'Precision': 0.6384065372829418, 'Recall': 0.7275902211874272, 'TP': 625, 'FP': 354, 'FN': 234

| Class      | TP  | FP  | FN  | *Precision* | *Recall*  |
|------------|-----|-----|-----|-----------|---------|
| PV_normal  | 243 | 134 | 89  | 0.6445    | 0.7319  |
| PV_heater  | 221 | 110 | 82 | 0.6676    | 0.7293  |
| PV_pool    | 161 | 110 | 63  | 0.5940    | 0.7187  |

**[1] + [2] Groupping_neighboring_polygons (Threshold : 0.16m = 2pixels)**

Overall Metrics: 'Precision': 0.6921373200442967, 'Recall': 0.7275902211874272, 'TP': 625, 'FP': 278, 'FN': 234

| Class      | TP  | FP  | FN  | *Precision* | *Recall*  |
|------------|-----|-----|-----|-----------|---------|
| PV_normal  | 243 | 86 | 89  | 0.7386    | 0.7319  |
| PV_heater  | 221 | 105 | 82 | 0.6779    | 0.7293  |
| PV_pool    | 161 | 87 | 63  | 0.6491    | 0.7187  |

**[1] + [2] + [3] Dropping small objects (=<1.7m^2)**

==Overall Metrics: 'Precision': 0.7751552795031056, 'Recall': 0.7264260768335273, 'TP': 624, 'FP': 181, 'FN': 235==

| Class      | TP  | FP  | FN  | *Precision* | *Recall*  |
|------------|-----|-----|-----|-----------|---------|
| PV_normal  | 243 | 68 | 89  | 0.7813    | ==0.7319==  |
| PV_heater  | 220 | 62 | 83 | 0.7801    | ==0.7260==  |
| PV_pool    | 161 | 51 | 63  | 0.7594    | ==0.7187==  |

**[1] + [2] + [3] + [4] Building Footpring Filtering**

Overall Metrics: 'Precision': 0.8219584569732937, 'Recall': 0.642691415313225, 'TP': 554, 'FP': 120, 'FN': 308

| Class      | TP  | FP  | FN  | *Precision* | *Recall*  |
|------------|-----|-----|-----|-----------|---------|
| PV_normal  | 216 | 51 | 116  | 0.8089    | 0.6506  |
| PV_heater  | 189 |  32 | 115 | 0.8552    | 0.6217  |
| PV_pool    | 149 |  37 | 77  | 0.8010    | 0.6592  |

------

#### Models' Overall Performance Evaluation 
* *Note: S1 =  [1]Polygonization, S2 =  [1]+[2], S3 =  [1]+[2]+[3], S4 = [1]+[2]+[3]+[4]*

* S1–S3 reduce false positives, but S4 raises false negatives due to true positives being removed by the building filter.



<img width="614" height="504" alt="Image" src="https://github.com/user-attachments/assets/516afdd5-dea5-49f9-99c9-f44d699c512d" />

#### Models' Performance Evaluation by class (S3 Only)

<img width="1022" height="612" alt="Image" src="https://github.com/user-attachments/assets/8872f7ff-c95b-41ff-87b3-a0ab023a95fd" />

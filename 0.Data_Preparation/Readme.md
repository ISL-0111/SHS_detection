# Data processing

`data_processing.ipynb`

This code integrates and refines aerial imagery and annotations for training. It improves the data preparation pipeline developed by the Bass Connections team in 2024, available at https://github.com/slaicha/cape_town_segmentation. The final output of this process is a GPKG (GeoPackage) file, containing about 19,000 annotations corresponding to 268 annotation files and approximately 19,000 raw annotations. The quality of these annotations is subsequently assessed in the next stage, titled Annotation Checking.

*Note: For prediction, complete Step 1 (Download data) and then proceed to Pre-Processing for image cropping.
----
## Step 1: Download data

To download files from `dukebox`, using BOX API is recommended
or refer to `/Users/ilseoplee/cape_town_annotation_checker/1.db_pipeline/download`

* `CapeTown_ImageIDs.xlxs`: The list of Aerial Images ID and Annotators
* `Aerial Imagery`: Cape Town, 12500 * 12500 size, 8cm/pixel (around 55GB for 2023)

-----
CapeTown_Image_2023_Training_1024_Oct.28
├── ...
├── ...
├── ...
├── output5k_stratified/
│   └── test/
│       ├── images/
│       └── masks/


-----

## Step 2: Data Loading

Load files and clean the data, unify format

* `CRS`: Coordinate Reference System = ESRI:102562
* `Annotator Lists`: Contains a list of annotated images
* `Shapefile`: List of anotations


## Step 3: Bounding Box extraction

* Extract vertices of each aerial image directly from files
* Match annoatations with aerial imagery
* Calculate the area of each polygon using its geometry

## Step 4: Annotation Post-processing

* Drop duplicated annotations correponsing to 'geometry' 
* Merge PV_Pool into PV_pool
* Ensure binary values '1' or 'Nan' for PV related columns
* Create a 'PV_normal' column if all other PV related colums are NaN
* Reorder columns for clarity
* Drop unnecessary columns (layer, path = '2020 layers')
* Reindex 'id' for consistency

## Step 5: Save Dataframe as GPKG

* NOTE: ESRI is not directly readbable in python. Therefore it's safe to use WKT format instead.

# (Optional) Annotation checking and confusion matrix

<img src="https://github.com/user-attachments/assets/223e0ddd-6bde-4741-a392-79cc586298c2" alt="설명" width="700"/>

-----
This code is a GUI-based program designed for quality checking of pre-generated annotations. It is intended to run locally and is not compatible with Pizer environments.
The program loads an annotation file in GPKG format, and for each reviewed annotation, it records the result by creating a new column name

**Note:**
1) When clicking either `resizing` or `uncertain`, the user must click one additional button (e.g., PV_normal, PV_pool, or PV_heater)
2) When `uncertain` is selected, the corresponding image is saved. So it can be reviewed seperately later.
3) The heater mat was decided to be classified as a type of water heater. However, this code still includes PV_heater_mat class, which can be removed
   
-----

## Step 1 Set the input data 

Before launching the tool, ensure the following:

1. GeoPackage (.gpkg) file containing annotation geometries and metadata (with columns like image_name, PV_normal, PV_heater, etc.)
2. A folder containing the corresponding .tif images.

##  Step 2 Launch the QC Checker GUI

**Run `checker_final_exe.ipynb`**

if using the notebook version:
  1. Open Annotation_QC_Checker_GUI.ipynb
  2. Run all cells from top to bottom

The GUI will launch and allow you to:
* View original and annotated image tiles
* Apply QC labels via buttons
* Zoom in/out on image tiles
* Automatically save labels to the `.gpkg` file
> * You can inspect the `.gpkg` file using DB Browser for SQLite.
> * If any mislabeling is found, open the file in QGIS and manually correct the label based on its unique ID.

## Label Name and its discription 
* PV_normal_qc → Solar panel (Glossy, uniform, and neatly arranged in rows)
* PV_heater_qc → Solar water heater (Small square-like and has a White rectangular tank attached to it)
* PV_pool_qc → Solar pool heater (Typically located next to a pool and in a darker shade, Sometimes there is visible piping nearby)
* uncertflag_qc → Not confident about the annotation
* delete_qc → Mark for deletion
* resizing_qc → Annotation needs resizing
* PV_heater_mat → a type of tankless water heater mat **(This type should merged into PV_heater)**

##  Step 3 Label name cleaning
`post_annotation_check.ipynb`

This imports the `.gpkg` file after the annotation check and cleans the label column names. In this step, `PV_heater_mat` is merged with `PV_heater`.

Final output is cleaned `.gpkg`file based on 5K annotations.
Please check `final_annotations_PV_all_types_5K_cleaned.gpkg`, or refer to 
(/shared/data/climateplus2025/CapeTown_Image_2023_Training_1024_Oct.28/final_annotations_PV_all_types_5K_cleaned.gpkg)


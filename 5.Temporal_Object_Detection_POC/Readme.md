#### Temporal-Object Detection

*Updated as of Dec.05.2025*

**Time-series analysis**

**File:** `Installation_year_detection_Dec2.ipynb` <br>
**Path:** `/shared/data/climateplus2025/Installation_year_detection_POC_Dec1/temporal_matching_results/Installation_year_detection_Dec2.ipynb`

- This proof-of-concept code performs temporal object detection using .gpkg files generated from the notebook located at:
`/Users/ilseoplee/SHS_detection/4.Post-Processing/Post-Processing_Scaleup/post_processing_polygonization_grouping_drop_small_objects.ipynb`.
<br>
- The method takes all objects detected in 2023 as the reference layer. It then iterates through all objects from previous years and determines whether each historical object corresponds to the same real-world object as the 2023 reference.

- The determination is based on three criteria:
    * **Centroid Distance**: The distance between centroids must be less than 1 meter.
    * **Polygon IoU**: The Intersection over Union between polygons must meet or exceed the 50% threshold.
    * **Class Consistency**: The object class must match across years. The applicable classes are(Solar Panel, Water Heater, Pool Heater)

- If a object fails to be matched, the code records which of the three criteria caused the mismatch. A sample output can be found in `Temporal_Object_Detection_POC.csv`.

    ```
    # Sample output
    2023,2022,2021,prediction_id,2022_Distance_Reason,2022_IOU_Reason,2022_Class_Reason,2021_Distance_Reason,2021_IOU_Reason,2021_Class_Reason
    Y,N,N,i_2023_RGB_8cm_W13A_6_pred_63b21368;;i_2023_RGB_8cm_W13A_6_pred_2fb34ff6,Mismatch,,,Mismatch,,
    Y,N,N,i_2023_RGB_8cm_W16C_20_pred_2e4ddf1a;;i_2023_RGB_8cm_W16C_20_pred_f1aced30,,Mismatch,,,Mismatch,
    ```

*Note: Because IoU and centroid-distance thresholds influence results, it is recommended to perform a **grid search** to identify optimal threshold values and to **visualize** results for cross-validation.*

**(Sample) Grid search output**
<img width="985" height="414" alt="Image" src="https://github.com/user-attachments/assets/bb75e2c8-fe7c-49d4-bb96-1c5dbbf0cce6" />


**Visualization**

**File:** `app.py`
**Path:** `/shared/data/climateplus2025/Installation_year_detection_POC_Dec1/Visualization/app.py`

- This application runs in the GAIA environment using Streamlit and is currently used to display and overlay large-scale temporal object detection images (three years, each at 12,500 × 12,500 resolution). Additional features will need to be implemented for full analysis and interaction.
<br>
- To launch the application, run it through the GAIA CLI with the command below. After execution, open the assigned local port in Firefox (Safari is not supported).


    /home/il72/.local/bin/streamlit run /shared/data/climateplus2025/Installation_year_detection_POC_Dec1/Visualization/app.py --server.port 8502 --server.address 0.0.0.0
    

![Image](https://github.com/user-attachments/assets/9b9e7228-6c73-4303-a498-a6abe50979c9)
 

# Small-Scale prediction

This code is for post-processing and evaluating **three unseen but annotated images**.

**Path:** `/shared/data/climateplus2025/Prediction_for_poster_3_images_Mask2Former_1024_Nov20`

**File Components:**
* `callbacks.csv.py` : No functional role (only needed to avoid import errors during prediction)
* `data_gen_M.py` : Provides preprocessing, normalization, and class definitions → required
* `model_M_patched.py` : Defines the model architecture
* `Prediction_M_Sep.16_ddp.py` : Standard prediction script for normal-size datasets.
  - {project_root}/prediction_outputs_{tag}/
  - {project_root}/predicted_masks_{tag}/
  * To run this script, enter the following command in the CLI
    ```
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=WARN
    export OMP_NUM_THREADS=1
    python3 /shared/data/climateplus2025/Prediction_for_poster_3_images_Mask2Former_1024_Nov20/Prediction_M_Sep.16_ddp.py \
      --arch Mask2Former \
      --mode eval \
      --devices 6 \
      --strategy ddp_find_unused_parameters_false \
      --precision 16-mixed \
      --img_size 1024 \
      --batch_size 8 \
      --workers 3 \
      --project_root /shared/data/climateplus2025/Prediction_for_poster_3_images_Mask2Former_1024_Nov20 \
      --test_image_dir /shared/data/climateplus2025/CapeTown_Image_2023_3samples_poster_1024size_Nov20/original_cropped_images\
      --save_preds true --pred_out_version newdata_swinS \
      --ckpt_path /shared/data/climateplus2025/Shawn_Nov.20_New_Mask2Former_Nov.20_1024size/logs/Mask2Former_run/checkpoints/epoch=031-valid_macro_fg_mIoU=0.6620.ckpt \
    ```

# Large-Scale Prediction

This code is for performing inference on the entire dataset

**Path:** `/shared/data/climateplus2025/Prediction_EntireDataset_Mask2Former_1024`
**File Components:**
* `callbacks.csv.py` : No functional role (only needed to avoid import errors during prediction)
* `data_gen_M.py` : Provides preprocessing, normalization, and class definitions → required
* `model_M_patched.py` : Defines the model architecture
* `Prediction_M_Sep.16_for_entire_data.py` : Adjusted script for large-scale inference, generating n `.jsonl` files (one per GPU).
  - `--skip_empty_json` : true → If an image has no predicted objects, then does not create JSON file for that image (saves storage and I/O)
  - `{project_root}/prediction_outputs_{tag}/rank_{rank_id}/`
    *Note: `rank_id` corresponds to the GPU rank used during distributed prediction*
  - To run `Prediction_M_Sep.19_for_entire_data.py`, enter the following command in the CLI
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 /shared/data/climateplus2025/Prediction_EntireDataset_Mask2Former_1024/Prediction_M_Sep.19_for_entire_data.py \
      --devices 8 \
      --strategy ddp_find_unused_parameters_false \
      --precision 16-mixed \
      --img_size 1024 \
      --batch_size 8 \
      --workers 8 \
      --project_root /shared/data/climateplus2025/Prediction_EntireDataset_Mask2Former_1024/2022 \
      --test_image_dir /data/data/capetown_bc_2025/Data/CapeTown_Image_2022_cropped_1024 \
      --pred_out_version probe_strict \
      --ckpt_path /shared/data/climateplus2025/Shawn_Nov.20_New_Mask2Former_Nov.20_1024size/logs/Mask2Former_run/checkpoints/epoch=031-valid_macro_fg_mIoU=0.6620.ckpt \
      --norm_mode hf255 \
      --skip_empty_json true \
      --positives_index positives_list.txt \
      --stream_save true
      > log.txt 2>&1
    ```

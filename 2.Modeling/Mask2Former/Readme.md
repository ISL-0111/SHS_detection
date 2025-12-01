# Modeling (Mask2Former)

The model performs pixel-level segmentation. It uses a *Swin-Small ADE-semantic encoder* pretrained on the *ADE20K* dataset (an urban scene dataset that includes many rooftop and building types). Due to this, the encoder already understands common rooftop features such as textures, edges, shadows, and building materials. The Cape Town dataset is then used to fine-tune the model so it can identify the specific types of rooftop energy systems found in the city.

Reference: https://huggingface.co/facebook/mask2former-swin-large-ade-semantic

The model predicts four classes: Background, Solar Panel, Solar Water Heater, and Solar Pool Heater. Training uses a learning rate of 0.0003, weight decay of 0.0001, 100 epochs, mixed precision, and cross-entropy loss. The encoder is fine-tuned, while the decoder and segmentation head learn from scratch.


#### Input image size: 1024*1024 (Best)
**Path:** `/shared/data/climateplus2025/Shawn_Nov.20_New_Mask2Former_Nov.20_1024size`

* The following files should be stored in the same folder:
  * `callbacks_csv.py`: A logger callback that automatically saves epoch-level train/validation metrics and per-class detailed metrics to CSV files during segmentation training.
(train_epoch.csv, valid_epoch.csv, metrics_epoch_per_class.csv)
<br>
  * `data_gen_M.py`: Integrated data pipeline that includes Data loading (4 classes including background), mask matching, augmentation (e.g., Affine and Elastic transformations), normalization based on Albumentations (e.g., hf255, imagenet01), and DataLoader construction
<br>
  * `eval_from_png.py`: *Pixel-level* evaluation tool that reads PNG masks, computes IoU-based metrics, and exports results(Per-class IoU, Macro IoU and others).
<br>
  * `model_M_pathced.py`: Defines a patched LightningModule for Mask2Former that ensures stable pixel-wise logits, custom losses, proper metrics, and clean training/validation behavior across different HF versions.
<br>
  * `train_M_Sep.04_ddp.py`: Provides a CLI to configure architecture, GPUs, Distributed training strategy, image size, data paths, learning rate, and prediction options for training or evaluation.
<br>
* The best performing trained parameter checkpoint file 
**File:**`epoch=031-valid_macro_fg_mIoU=0.6620.ckpt`
**Path:** `/shared/data/climateplus2025/Shawn_Nov.20_New_Mask2Former_Nov.20_1024size/logs/Mask2Former_run/checkpoints/epoch=031-valid_macro_fg_mIoU=0.6620.ckpt`
<br>
* To run the training code in Gaia, type the commands in terminal
  ```
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_DEBUG=WARN
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  export OMP_NUM_THREADS=1

  python3 /shared/data/climateplus2025/Shawn_Nov.20_New_Mask2Former_Nov.20_1024size/train_M_Sep.04_ddp.py \
    --arch Mask2Former \
    --mode train \
    --devices 8 \
    --strategy ddp_find_unused_parameters_false \
    --precision 16-mixed \
    --img_size 1024 \
    --batch_size 1 \
    --epochs 100 \
    --workers 3 \
    --project_root /shared/data/climateplus2025/Shawn_Nov.20_New_Mask2Former_Nov.20_1024size \
    --train_image_dir /shared/data/climateplus2025/CapeTown_Image_2023_Training_1024_Oct.28/output5k_stratified/train/images \
    --train_mask_dir  /shared/data/climateplus2025/CapeTown_Image_2023_Training_1024_Oct.28/output5k_stratified/train/masks \
    --valid_image_dir /shared/data/climateplus2025/CapeTown_Image_2023_Training_1024_Oct.28/output5k_stratified/val/images \
    --valid_mask_dir  /shared/data/climateplus2025/CapeTown_Image_2023_Training_1024_Oct.28/output5k_stratified/val/masks \
    --test_image_dir  /shared/data/climateplus2025/CapeTown_Image_2023_Training_1024_Oct.28/output5k_stratified/test/images \
    --test_mask_dir   /shared/data/climateplus2025/CapeTown_Image_2023_Training_1024_Oct.28/output5k_stratified/test/masks \
    --save_preds true --pred_out_version m2f_swinS \
    --lr 3e-4
  ```

* Output(sample)
    - `/logs`: checkpoints(`.ckpt`), hyperparameters(`hparams.yaml`), `masks(RGB images)`, `outputs(Json)`
    - JSON output(Sample)
      - {"image_name":"i_2023_RGB_8cm_W16C_20_4_4.png","predicted_coords":{"class_1":[[900,653],[900,654],[900,655],[900,656],[900,657],[900,658],[900,659],[901,650],[901,651],[901,652],[901,653],[901,654],[901,655],[901,656],[901,657],[901,658],[901,659],[901,660],[901,661],[902,649],[902,650],[902,651],[902,652],[902,653],[902,654],[902,655],[902,656],[902,657],[902,658],[902,659],[902,660],[902,661],[902,662],[903,649],[903,650],[903,651],[903,652],[903,653],[903,654],[903,655],[903,656].......}}
          *Note: class_0 = background. class_1 = PV_normal(Solar Panel), class_2 = PV_heater(Water heater), class_3 = PV_pool(Pool heater)


#### Input image size: 320*320 (Reference)
- Best trained parameter checkpoint : lr3.5e-4 epoch 60 out of 80 training process
**File:** `epoch=060-valid_macro_fg_mIoU=0.7303.ckpt`
**Path:** `/shared/data/climateplus2025/Shawn_Sep.04_New_Mask2Former_Sep.08/model_3_summary/logs_model_3/Mask2Former_run/checkpoints/epoch=060-valid_macro_fg_mIoU=0.7303.ckpt`

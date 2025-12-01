# NNU-Net

**Path:** `/shared/data/climateplus2025/Shawn_Nov18_NNUnet_1024`

This directory contains a fully automated nnU-Net v2 pipeline script that performs:

- Data preprocessing and planning
- 5-fold training with real-time log streaming
- Model ensembling across all folds
- Prediction on the test set (with optional TTA)
- Evaluation using Dice and IoU metrics
- Automatic selection of the best configuration

The script serves as a complete automation layer for the standard nnU-Net v2 2D training, prediction, and evaluation workflow.

**Note:** 
To run NNUnet, strictly follow folder structure (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

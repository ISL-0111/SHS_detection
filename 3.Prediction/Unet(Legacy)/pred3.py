#!/usr/bin/env python
# inference_no_masks.py

import os
import argparse
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model_M import PVModel
from data_gen_pred import get_validation_augmentation
from inference_dataset import InferenceDataset  # NEW


# BGR (OpenCV) colors
COLORS = {
    0: (0, 0, 0),        # background
    1: (0, 255, 0),      # PV_normal
    2: (0, 0, 255),      # PV_heater
    3: (255, 0, 0),      # PV_pool
}

def decode_mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, bgr in COLORS.items():
        color[mask == idx] = bgr
    return color


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        default="/shared/data/climateplus2025/Prediction_for_poster_3_images_U-Net/U-Net_prediction_code/pv-model-epoch=38-valid_avg_PV_iou=0.9572.ckpt"
    )
    parser.add_argument(
        "--images",
        default="/shared/data/climateplus2025/CapeTown_Image_2023_3samples_poster/original_cropped_images",
        help="Folder with RGB tiles (.tif, .png, .jpg…) to predict on",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--outdir",
        default="/shared/data/climateplus2025/Prediction_for_poster_3_images_U-Net/prediction_outputs_u_net_only"
    )
    parser.add_argument(
        "--maskdir",
        default="/shared/data/climateplus2025/Prediction_for_poster_3_images_U-Net/prediction_masks_u_net_only"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.maskdir, exist_ok=True)

    CLASSES = ["background", "PV_normal", "PV_heater", "PV_pool"]
    n_classes = len(CLASSES)

    # Dataset
    dataset = InferenceDataset(
        image_dir=args.images,
        augmentation=get_validation_augmentation()
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PVModel.load_from_checkpoint(
        args.ckpt,
        arch="FPN",
        encoder_name="resnext50_32x4d",
        in_channels=3,
        out_classes=n_classes,
    ).to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        for i, imgs in tqdm(enumerate(loader), total=len(loader)):
            imgs = imgs.float().to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            for b in range(preds.size(0)):
                idx = i * loader.batch_size + b
                img_path = dataset.image_paths[idx]
                base = os.path.splitext(os.path.basename(img_path))[0]
                pred_np = preds[b].cpu().numpy()

                # Save PNG mask
                color = decode_mask_to_color(pred_np)
                cv2.imwrite(os.path.join(args.maskdir, f"{base}_pred.png"), color)

                # Save Mask2Former-style JSON
                coords = {
                    "image_name": f"{base}.png",
                    "predicted_coords": {}
                }

                for cls in range(1, n_classes):  # exclude background (0)
                    ys, xs = np.where(pred_np == cls)
                    if ys.size > 0:
                        coords["predicted_coords"][f"class_{cls}"] = (
                            np.stack([ys, xs], axis=1).tolist()
                        )

                with open(os.path.join(args.outdir, f"{base}.json"), "w") as f:
                    json.dump(coords, f, separators=(",", ":"))

    print(f"Predictions saved to {args.maskdir}")
    print(f"Coordinate data saved to {args.outdir}")
    print("All done – predictions are ready!")


if __name__ == "__main__":
    main()

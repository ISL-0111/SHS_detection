# train_M_Sep.04_ddp.py
# Unified training script for Mask2Former + Lightning + DDP + CSV callback.
# - Uses our patched model_M_patched.Mask2FormerLit
# - Data loaders from data_gen_M (mask prefix/dir handled internally)
# - ModelCheckpoint monitors "valid_avg_PV_iou"
# - CSV callback writes per-class metrics at RUN_DIR/metrics_epoch_per_class.csv

# !!Refer to training_log.txt for training logs(learnig rate variations etc).!!
'''
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

python3 /shared/data/climateplus2025/Shawn_Sep.04_New_Mask2Former_Sep.08/train_M_Sep.04_ddp.py \
  --arch Mask2Former \
  --mode train \
  --devices 6 \
  --strategy ddp_find_unused_parameters_false \
  --precision 16-mixed \
  --img_size 320 \
  --batch_size 8 \
  --epochs 10 \
  --workers 3 \
  --project_root /shared/data/climateplus2025/Shawn_Sep.04_New_Mask2Former_Sep.08 \
  --train_image_dir /shared/data/climateplus2025/Shawn_Sep.04_Attention_U-Net_Sep.07/output5k_stratified_training_db/train/images \
  --train_mask_dir  /shared/data/climateplus2025/Shawn_Sep.04_Attention_U-Net_Sep.07/output5k_stratified_training_db/train/masks \
  --valid_image_dir /shared/data/climateplus2025/Shawn_Sep.04_Attention_U-Net_Sep.07/output5k_stratified_training_db/val/images \
  --valid_mask_dir  /shared/data/climateplus2025/Shawn_Sep.04_Attention_U-Net_Sep.07/output5k_stratified_training_db/val/masks \
  --test_image_dir  /shared/data/climateplus2025/Shawn_Sep.04_Attention_U-Net_Sep.07/output5k_stratified_training_db/test/images \
  --test_mask_dir   /shared/data/climateplus2025/Shawn_Sep.04_Attention_U-Net_Sep.07/output5k_stratified_training_db/test/masks \
  --save_preds true --pred_out_version m2f_swinS \
  --lr 3e-4
'''
# train_M_Sep.04_ddp.py
# Unified training script for Mask2Former + Lightning + DDP + CSV callback + TEST/INFERENCE.

import os
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from model_M_patched import PVModel as Mask2FormerLit
from data_gen_M import (
    make_dataloaders,
    PVDatasetFoldersCT,
    get_validation_augmentation,
    CLASSES,
    CLASS_LABELS,
)
from callbacks_csv import SegmentationCSVLogger

# ------------------------------
# Utils
# ------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y", "t")

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Core
    p.add_argument("--arch", default="Mask2Former", type=str)
    p.add_argument("--mode", default="train", type=str, choices=["train", "eval"])  # <- eval 추가
    p.add_argument("--devices", default="auto", type=str)  # "auto" or number or "0,1,2,3"
    p.add_argument("--strategy", default="ddp_find_unused_parameters_false", type=str)
    p.add_argument("--precision", default="16-mixed", type=str)
    p.add_argument("--img_size", default=320, type=int)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--workers", default=4, type=int)

    # Data
    p.add_argument("--train_image_dir", type=str, required=False, default=os.environ.get("TRAIN_IMAGE_DIR", ""))
    p.add_argument("--train_mask_dir",  type=str, required=False, default=os.environ.get("TRAIN_MASK_DIR", ""))
    p.add_argument("--valid_image_dir", type=str, required=False, default=os.environ.get("VALID_IMAGE_DIR", ""))
    p.add_argument("--valid_mask_dir",  type=str, required=False, default=os.environ.get("VALID_MASK_DIR", ""))
    # Optional test set
    p.add_argument("--test_image_dir",  type=str, required=False, default=os.environ.get("TEST_IMAGE_DIR", ""))
    p.add_argument("--test_mask_dir",   type=str, required=False, default=os.environ.get("TEST_MASK_DIR", ""))

    # Project/logging roots
    p.add_argument("--project_root", type=str, default=os.getcwd())

    # Opt
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ignore_index", type=int, default=255)

    # Inference/save
    p.add_argument("--save_preds", type=str2bool, default=True, help="Save color PNGs and coords JSON after training.")
    p.add_argument("--pred_out_version", type=str, default="", help="Suffix for prediction output folders.")

    # Eval-only
    p.add_argument("--ckpt_path", type=str, default="", help="(eval mode) Path to checkpoint to load for test/inference.")

    # Seed
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()

# ------------------------------
# Helpers: build test loader
# ------------------------------
def build_test_loader(args):
    if not args.test_image_dir:
        return None
    test_img_dir = os.path.abspath(args.test_image_dir)
    test_msk_dir = os.path.abspath(args.test_mask_dir) if args.test_mask_dir else None

    root_guess = os.path.abspath(os.path.join(test_img_dir, "..", ".."))
    split_guess = "test"
    img_dirname = os.path.basename(test_img_dir) if os.path.basename(test_img_dir) != "images" else "images"
    msk_dirname = os.path.basename(test_msk_dir) if test_msk_dir else "masks"

    test_ds = PVDatasetFoldersCT(
        root=root_guess,
        split=split_guess,
        img_dirname=img_dirname,
        msk_dirname=msk_dirname,
        augmentation=get_validation_augmentation(img_size=args.img_size, norm_mode="hf255"),
        img_size=args.img_size,
        norm_mode="hf255",
        strict_mask_size=False,
        debug=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    print(f"[INFO] Test dataset size: {len(test_ds)}")
    return test_loader

# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)

    project_root = os.path.abspath(args.project_root)
    run_dir = os.path.join(project_root, "logs", f"{args.arch}_run")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n================= BOOT =================")
    print(f"arch={args.arch}, mode={args.mode}")
    print(f"devices={args.devices}, strategy={args.strategy}, precision={args.precision}")
    print(f"img_size={args.img_size}, batch_size={args.batch_size}, epochs={args.epochs}")
    print(f"project_root={project_root}")
    print(f"run_dir={run_dir}")
    print(f"ckpt_dir={ckpt_dir}")
    print("========================================\n")

    # ---------------- Eval-only path ----------------
    if args.mode == "eval":
        if not args.test_image_dir:
            raise RuntimeError("--mode eval requires --test_image_dir (and optional --test_mask_dir).")
        if not args.ckpt_path or not os.path.exists(args.ckpt_path):
            raise RuntimeError("--mode eval requires a valid --ckpt_path to a .ckpt file.")

        # Model
        model = Mask2FormerLit.load_from_checkpoint(
            args.ckpt_path,
            out_classes=len(CLASSES),
            lr=args.lr,
            weight_decay=args.weight_decay,
            encoder_name="facebook/mask2former-swin-small-ade-semantic",
            ignore_index=args.ignore_index,
        )

        # Test loader
        test_loader = build_test_loader(args)
        have_test = test_loader is not None

        # TEST: single device, no DDP, no sanity
        print("\nRunning TEST evaluation (eval mode)...")
        test_trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            logger=False,
            enable_progress_bar=True,
            num_sanity_val_steps=0, ## Once you finish sanity check with a single GPU, set 0 for DDP otherwise the process get frozen
            strategy="auto",
        )
        try:
            test_trainer.test(model, dataloaders=test_loader)
        except Exception as e:
            print(f"[WARN] trainer.test skipped ({e}). Will still run manual inference.")

        # Inference (optional save)
        if args.save_preds and have_test:
            _do_inference_and_save(args, model, test_loader, project_root)
        return

    # ---------------- Train path ----------------
    if not args.train_image_dir or not args.valid_image_dir:
        raise RuntimeError("Please provide --train_image_dir and --valid_image_dir (and mask dirs if not under split/masks).")

    train_loader, valid_loader = make_dataloaders(
        train_img_dir=args.train_image_dir,
        train_mask_dir=args.train_mask_dir or None,
        valid_img_dir=args.valid_image_dir,
        valid_mask_dir=args.valid_mask_dir or None,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    test_loader = build_test_loader(args)
    have_test = test_loader is not None

    if args.arch.lower() != "mask2former":
        raise ValueError("This script currently supports --arch Mask2Former only.")
    model = Mask2FormerLit(
        out_classes=len(CLASSES),
        lr=args.lr,
        weight_decay=args.weight_decay,
        encoder_name="facebook/mask2former-swin-small-ade-semantic",
        ignore_index=args.ignore_index,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:03d}-{valid_macro_fg_mIoU:.4f}",
        save_top_k=3,
        save_last=True,
        monitor='valid_macro_fg_mIoU',
        mode="max",
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    perclass_csv_cb = SegmentationCSVLogger(out_dir=run_dir)
    csv_logger = CSVLogger(save_dir=run_dir, name="", version="")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=args.strategy,
        precision=args.precision,
        default_root_dir=run_dir,
        callbacks=[ckpt_cb, lr_cb, perclass_csv_cb],
        logger=csv_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,   # sanity OFF
    )

    print("\nStarting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print("Training completed.")

    # ---- DDP barrier & rank-guard (중요) ----
    if trainer.strategy is not None:
        try:
            trainer.strategy.barrier()
        except Exception:
            pass
    if not trainer.is_global_zero:
        return

    # ========== Post: TEST (optional) ==========
    best_path = getattr(ckpt_cb, "best_model_path", "")
    if best_path and os.path.exists(best_path):
        print(f"[INFO] Best checkpoint: {best_path}")
        model = Mask2FormerLit.load_from_checkpoint(
            best_path,
            out_classes=len(CLASSES),
            lr=args.lr,
            weight_decay=args.weight_decay,
            encoder_name="facebook/mask2former-swin-small-ade-semantic",
            ignore_index=args.ignore_index,
        )
    else:
        print("[WARN] No best checkpoint path found; will use current model state for test/inference.")

    if have_test:
        print("\nRunning TEST evaluation...")
        test_trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",  # single device
            logger=False,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            strategy="auto",   # <- DDP 금지
        )
        try:
            test_trainer.test(model, dataloaders=test_loader)
        except Exception as e:
            print(f"[WARN] trainer.test skipped ({e}). Will still run manual inference.")

    # ========== Post: INFERENCE (optional) ==========
    if args.save_preds and have_test:
        _do_inference_and_save(args, model, test_loader, project_root)

# ------------------------------
# Inference helper
# ------------------------------
def _do_inference_and_save(args, model, test_loader, project_root):
    import json, cv2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    suffix = args.pred_out_version.strip()
    tag = f"Mask2Former{('_' + suffix) if suffix else ''}"
    OUT_DIR = os.path.join(project_root, f"prediction_outputs_{tag}")
    COLOR_DIR = os.path.join(project_root, f"predicted_masks_{tag}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(COLOR_DIR, exist_ok=True)
    print(f"[INFO] Saving predictions to:\n - JSON:  {OUT_DIR}\n - PNGs:  {COLOR_DIR}")

    # Stable RGB palette (class 0 kept as black, but we won't export it to JSON)
    colors = {0: (0, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0)}

    def decode_mask_to_color(mask_np: np.ndarray) -> np.ndarray:
        assert mask_np.ndim == 2, f"mask must be HxW, got {mask_np.shape}"
        h, w = mask_np.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for k, rgb in colors.items():
            color[mask_np == k] = rgb
        return color

    test_ds = getattr(test_loader, "dataset", None)

    with torch.inference_mode():
        idx_global = 0
        for i, (images, gt_masks) in enumerate(test_loader):
            images = images.float().to(device, non_blocking=True)

            logits = model(images)  # [B, C, H, W]
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError(f"Model forward must return Tensor, got {type(logits)}")
            C = int(logits.shape[1])
            preds = torch.argmax(logits, dim=1)  # [B, H, W]

            # Debug: value range before moving to CPU
            minv = int(preds.min().item())
            maxv = int(preds.max().item())
            print(f"[DBG] batch={i} preds range (pre-cpu): [{minv}, {maxv}] expected 0..{C-1}")

            preds = preds.cpu().numpy()  # int64 by default

            B = preds.shape[0]
            for b in range(B):
                pred_mask = preds[b]
                assert pred_mask.ndim == 2, f"pred_mask must be HxW, got {pred_mask.shape}"

                # Prefer original filename if available
                base_name = f"sample_{idx_global:06d}"
                if hasattr(test_ds, "images_fps") and idx_global < len(test_ds.images_fps):
                    img_path = test_ds.images_fps[idx_global]
                    base_name = os.path.splitext(os.path.basename(img_path))[0]

                # Debug: class histogram per image
                uniq, cnt = np.unique(pred_mask, return_counts=True)
                print(f"[DBG] {base_name}: unique classes -> {dict(zip(uniq.tolist(), cnt.tolist()))}")

                # -------------------------
                # JSON: exclude background(0), no image_index
                # -------------------------
                coords = {"image_name": f"{base_name}.png", "predicted_coords": {}}
                # Only classes 1..C-1
                for cls in range(1, C):
                    ys, xs = np.where(pred_mask == cls)
                    if ys.size == 0:
                        continue
                    # Keys like "class_1", "class_2", ...
                    coords["predicted_coords"][f"class_{cls}"] = np.stack([ys, xs], axis=1).tolist()

                json_path = os.path.join(OUT_DIR, f"{base_name}.json")
                with open(json_path, "w") as f:
                    json.dump(coords, f, ensure_ascii=False, separators=(",", ":"))

                # -------------------------
                # Color PNG: RGB → (convert once) → BGR for OpenCV
                # -------------------------
                color_mask_rgb = decode_mask_to_color(pred_mask)
                assert color_mask_rgb.dtype == np.uint8 and color_mask_rgb.shape[-1] == 3
                png_path = os.path.join(COLOR_DIR, f"{base_name}_pred.png")
                ok = cv2.imwrite(png_path, cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    print(f"[WARN] Failed to write PNG: {png_path}")

                idx_global += 1

    print("✅ Inference finished: PNGs and JSONs saved.")
# ------------------------------

if __name__ == "__main__":
    main()
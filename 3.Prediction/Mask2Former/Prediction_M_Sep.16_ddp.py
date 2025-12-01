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
'''

# train_M_Sep.04_ddp.py
# Unified training script for Mask2Former + Lightning + DDP + CSV callback + TEST/INFERENCE.
# - Uses our patched model_M_patched.Mask2FormerLit
# - Data loaders from data_gen_M
# - ModelCheckpoint monitors "valid_macro_fg_mIoU"
# - CSV callback writes per-class metrics at RUN_DIR/metrics_epoch_per_class.csv
# - Prediction JSONs EXCLUDE background (class 0) and DO NOT include image_index.

import os
import argparse
import random
import json
import cv2
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
    p.add_argument("--mode", default="train", type=str, choices=["train", "eval"])
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

    # Free-form eval dataset switch (kept for compatibility)
    p.add_argument("--force_freeform", type=str2bool, default=True,
                   help="Always use free-form eval dataset (no mask requirement).")

    return p.parse_args()


# ------------------------------
# Free-form eval dataset (images anywhere; masks optional)
# ------------------------------
class _FreeFormImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, msk_dir=None, img_size=320, norm_mode="hf255", ignore_index=255):
        import glob
        self.ignore_index = ignore_index

        exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.PNG","*.JPG","*.JPEG","*.TIF","*.TIFF")
        fps = []
        for ext in exts:
            fps.extend(glob.glob(os.path.join(img_dir, ext)))
        self.images_fps = sorted(fps)
        if len(self.images_fps) == 0:
            raise FileNotFoundError(f"No images found under: {img_dir}")

        self.masks_fps = None
        if msk_dir:
            stems = [os.path.splitext(os.path.basename(p))[0] for p in self.images_fps]
            cand = {}
            for ext in ("*.png","*.tif","*.tiff","*.PNG","*.TIF","*.TIFF"):
                for p in glob.glob(os.path.join(msk_dir, ext)):
                    cand[os.path.splitext(os.path.basename(p))[0]] = p
            self.masks_fps = [cand.get(stem, None) for stem in stems]

        self.aug = get_validation_augmentation(img_size=img_size, norm_mode=norm_mode)
        self.img_size = img_size
        self.norm_mode = norm_mode

    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, idx):
        img_path = self.images_fps[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"[READ_FAIL] image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = None
        if self.masks_fps is not None:
            mpath = self.masks_fps[idx]
            if mpath and os.path.isfile(mpath):
                m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
                if m is not None:
                    mask = m if m.ndim == 2 else m[...,0]
        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        augmented = self.aug(image=img, mask=mask)
        img_t = torch.from_numpy(augmented["image"]).permute(2,0,1).float()
        msk_t = torch.from_numpy(augmented["mask"]).long()
        return img_t, msk_t, img_path


# ------------------------------
# Helpers: build test loader
# ------------------------------
def build_test_loader(args):
    if not args.test_image_dir:
        return None
    test_img_dir = os.path.abspath(args.test_image_dir)
    test_msk_dir = os.path.abspath(args.test_mask_dir) if args.test_mask_dir else None

    print(f"[INFO] Using FREE-FORM eval dataset:")
    print(f"      images: {test_img_dir}")
    print(f"      masks : {test_msk_dir if test_msk_dir else '<none>'}")

    test_ds = _FreeFormImageFolderDataset(
        img_dir=test_img_dir,
        msk_dir=test_msk_dir,
        img_size=args.img_size,
        norm_mode="hf255",
        ignore_index=args.ignore_index,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )
    print(f"[INFO] Test dataset size: {len(test_ds)}")
    return test_loader


# ------------------------------
# JSON builder (EXCLUDES class 0, no image_index)
# ------------------------------
def _coords_from_mask(base_name: str, pred_mask: np.ndarray, num_classes: int) -> dict:
    """Return {"image_name":..., "predicted_coords":{class_1:[[y,x],...], ...}}
       Excludes class 0 (background). No image_index field."""
    coords = {"image_name": f"{base_name}.png", "predicted_coords": {}}
    for cls in range(1, int(num_classes)):  # 1..C-1 only
        ys, xs = np.where(pred_mask == cls)
        if ys.size:
            coords["predicted_coords"][f"class_{cls}"] = np.stack([ys, xs], axis=1).tolist()
    return coords


# ------------------------------
# DDP-safe prediction saver (rank-0 aggregates & saves)
# ------------------------------
def _predict_and_save_ddp(args, model, test_loader, project_root, trainer):
    """
    DDP-safe prediction gather/save. Writes color PNGs and JSON,
    with class 0 excluded and no image_index.
    """
    import torch.distributed as dist

    # 1) run predict on each rank
    local_batches = trainer.predict(model, dataloaders=test_loader, return_predictions=True)

    # 2) normalize outputs to list[(path, clsmap_uint8)]
    local_pairs = []

    def _to_pairs(t, paths):
        if isinstance(t, (list, tuple)):
            t = torch.stack(t)
        if t.ndim == 4:
            clsmap = torch.argmax(t, dim=1).detach().cpu().numpy().astype(np.uint8)
        elif t.ndim == 3:
            clsmap = t.detach().cpu().numpy().astype(np.uint8)
        else:
            raise RuntimeError(f"Unexpected predict output shape: {tuple(t.shape)}")

        if paths is None:
            paths_iter = [None] * clsmap.shape[0]
        elif isinstance(paths, (list, tuple)):
            paths_iter = list(paths)
        else:
            paths_iter = [paths] * clsmap.shape[0]

        for i in range(clsmap.shape[0]):
            local_pairs.append((paths_iter[i], clsmap[i]))

    for out in local_batches:
        if isinstance(out, dict):
            t = out.get("logits", None); paths = out.get("paths", None)
            if t is not None: _to_pairs(t, paths)
        elif isinstance(out, (list, tuple)) and out and isinstance(out[0], dict):
            for d in out:
                t = d.get("logits", None); paths = d.get("paths", None)
                if t is not None: _to_pairs(t, paths)
        else:
            _to_pairs(out, None)

    # 3) gather to rank-0
    world_size = getattr(trainer, "world_size", 1)
    rank = getattr(trainer, "global_rank", 0)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_pairs)
        all_pairs = []
        if rank == 0:
            for part in gathered:
                all_pairs.extend(part or [])
    else:
        all_pairs = local_pairs

    if rank != 0:
        return  # only rank-0 saves

    # 4) prepare outputs
    suffix = args.pred_out_version.strip()
    tag = f"Mask2Former{('_' + suffix) if suffix else ''}"
    OUT_DIR = os.path.join(project_root, f"prediction_outputs_{tag}")
    COLOR_DIR = os.path.join(project_root, f"predicted_masks_{tag}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(COLOR_DIR, exist_ok=True)
    print(f"[INFO] Saving predictions to:\n - JSON:  {OUT_DIR}\n - PNGs:  {COLOR_DIR}")

    # 5) sort by filename (None paths go last)
    def _key(p):
        path = p[0]
        return (1, "") if path is None else (0, os.path.basename(path))
    all_pairs.sort(key=_key)

    # 6) colorizer
    colors = {0:(0,0,0), 1:(0,255,0), 2:(0,0,255), 3:(255,0,0)}
    def decode_mask_to_color(mask_np: np.ndarray) -> np.ndarray:
        h, w = mask_np.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for k, rgb in colors.items():
            color[mask_np == k] = rgb
        return color

    # 7) save loop
    saved_png = saved_json = 0
    for idx, (path, clsmap) in enumerate(all_pairs):
        base_name = f"sample_{idx:06d}" if path is None else os.path.splitext(os.path.basename(path))[0]

        # PNG (RGB->BGR once)
        color_mask = decode_mask_to_color(clsmap)
        png_path = os.path.join(COLOR_DIR, f"{base_name}_pred.png")
        cv2.imwrite(png_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        saved_png += 1

        # JSON (exclude background; no image_index)
        coords = _coords_from_mask(base_name, clsmap, num_classes=len(CLASSES))
        json_path = os.path.join(OUT_DIR, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(coords, f, separators=(",", ":"), ensure_ascii=False)
        saved_json += 1

    print("✅ Inference finished (rank-0, DDP-stable).")
    print(f"   predicted pairs: {len(all_pairs)}")
    print(f"   saved PNGs     : {saved_png}")
    print(f"   saved JSONs    : {saved_json}")


# ------------------------------
# Legacy single-device inference helper
# ------------------------------
def _do_inference_and_save(args, model, test_loader, project_root):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    suffix = args.pred_out_version.strip()
    tag = f"Mask2Former{('_' + suffix) if suffix else ''}"
    OUT_DIR = os.path.join(project_root, f"prediction_outputs_{tag}")
    COLOR_DIR = os.path.join(project_root, f"predicted_masks_{tag}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(COLOR_DIR, exist_ok=True)
    print(f"[INFO] Saving predictions to:\n - JSON:  {OUT_DIR}\n - PNGs:  {COLOR_DIR}")

    colors = {0:(0,0,0), 1:(0,255,0), 2:(0,0,255), 3:(255,0,0)}
    def decode_mask_to_color(mask_np: np.ndarray) -> np.ndarray:
        h, w = mask_np.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for k, rgb in colors.items():
            color[mask_np == k] = rgb
        return color

    test_ds = getattr(test_loader, "dataset", None)

    with torch.inference_mode():
        idx_global = 0
        for i, (images, _gt_masks, *rest) in enumerate(test_loader):
            images = images.float().to(device, non_blocking=True)
            logits = model(images)                     # [B,C,H,W]
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            B = preds.shape[0]
            for b in range(B):
                pred_mask = preds[b]
                base_name = f"sample_{idx_global:06d}"
                # try to use original filename if available
                img_fps = getattr(test_ds, "images_fps", None)
                if img_fps is not None and idx_global < len(img_fps):
                    img_path = img_fps[idx_global]
                    base_name = os.path.splitext(os.path.basename(img_path))[0]

                # JSON (exclude background; no image_index)
                coords = _coords_from_mask(base_name, pred_mask, num_classes=len(CLASSES))
                json_path = os.path.join(OUT_DIR, f"{base_name}.json")
                with open(json_path, "w") as f:
                    json.dump(coords, f, separators=(",", ":"), ensure_ascii=False)

                # PNG
                color_mask = decode_mask_to_color(pred_mask)
                png_path = os.path.join(COLOR_DIR, f"{base_name}_pred.png")
                cv2.imwrite(png_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

                idx_global += 1

    print("✅ Inference finished: PNGs and JSONs saved.")


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
            lr=args.lr,  # constructor-only; no training in eval
            weight_decay=args.weight_decay,
            encoder_name="facebook/mask2former-swin-small-ade-semantic",
            ignore_index=args.ignore_index,
        )

        # Test loader
        test_loader = build_test_loader(args)
        have_test = test_loader is not None

        print("\nRunning TEST evaluation (eval mode)...")
        test_trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=args.devices,        # allow multi-GPU from CLI
            strategy=args.strategy,      # DDP strategy from CLI
            precision=args.precision,
            logger=False,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
        )
        # Optional metrics (only if masks available)
        try:
            test_trainer.test(model, dataloaders=test_loader)
        except Exception as e:
            print(f"[WARN] trainer.test skipped ({e}).")

        # Save predictions
        if args.save_preds and have_test:
            _predict_and_save_ddp(args, model, test_loader, project_root, test_trainer)
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
        num_sanity_val_steps=0,
    )

    print("\nStarting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print("Training completed.")

    # ---- DDP barrier & rank-guard ----
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
            strategy="auto",
        )
        try:
            test_trainer.test(model, dataloaders=test_loader)
        except Exception as e:
            print(f"[WARN] trainer.test skipped ({e}). Will still run manual inference.")

    # ========== Post: INFERENCE (optional) ==========
    if args.save_preds and have_test:
        _do_inference_and_save(args, model, test_loader, project_root)


if __name__ == "__main__":
    main()

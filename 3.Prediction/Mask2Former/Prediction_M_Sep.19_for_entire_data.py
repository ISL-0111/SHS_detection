# Output is merged json file per rank in DDP

"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 /shared/data/climateplus2025/Prediction_EntireDataset_Mask2Former_1024/Prediction_M_Sep.19_for_entire_data.py \
  --devices 8 \
  --strategy ddp_find_unused_parameters_false \
  --precision 16-mixed \
  --img_size 1024 \
  --batch_size 8 \
  --workers 8 \
  --project_root /shared/data/climateplus2025/Prediction_EntireDataset_Mask2Former_1024/2023 \
  --test_image_dir /shared/data/climateplus2025/CapeTown_Image_2023_cropped_1024 \
  --pred_out_version probe_strict \
  --ckpt_path /shared/data/climateplus2025/Shawn_Nov.20_New_Mask2Former_Nov.20_1024size/logs/Mask2Former_run/checkpoints/epoch=031-valid_macro_fg_mIoU=0.6620.ckpt \
  --norm_mode hf255 \
  --skip_empty_json true \
  --positives_index positives_list.txt \
  --stream_save true
  > log.txt 2>&1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, random, json
import cv2, numpy as np, torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# --- ensure local imports work ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from model_M_patched import PVModel as Mask2FormerLit
from data_gen_M import get_validation_augmentation, CLASSES


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


def _is_ddp_active():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return True
    return os.environ.get("WORLD_SIZE", "1") not in ("1", "", None)


# ------------------------------
# Robust reader: enforce RGB uint8
# ------------------------------
def _read_as_uint8_rgb(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"[READ_FAIL] {img_path}")

    # dtype conversion
    if img.dtype == np.uint16:
        img = cv2.convertScaleAbs(img, alpha=255.0 / 65535.0)
    elif img.dtype in (np.float32, np.float64):
        finite = np.isfinite(img)
        if finite.any():
            vmin = float(np.nanmin(img[finite]))
            vmax = float(np.nanmax(img[finite]))
            if vmax > vmin:
                img = ((img - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # enforce RGB channels
    try:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    return img


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser("DDP Mask2Former Predict-Only (Rank-level JSONL output)")

    p.add_argument("--devices", default="auto", type=str)
    p.add_argument("--strategy", default="ddp_find_unused_parameters_false")
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--img_size", default=320, type=int)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--workers", default=4, type=int)

    p.add_argument("--test_image_dir", required=True)
    p.add_argument("--project_root", default=os.getcwd())
    p.add_argument("--pred_out_version", default="")
    p.add_argument("--stream_save", type=str2bool, default=True)

    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ignore_index", type=int, default=255)

    p.add_argument("--norm_mode", default="hf255", choices=["hf255", "imagenet", "none"])

    p.add_argument("--skip_empty_json", type=str2bool, default=True)
    p.add_argument("--positives_index", default="positives_list.txt")

    return p.parse_args()


# ------------------------------
# Albumentations Val Aug (fixed)
# ------------------------------
def get_val_aug_compat(img_size: int, norm_mode: str):
    import albumentations as A
    H = W = int(img_size)
    try:
        pad = A.PadIfNeeded(min_height=H, min_width=W,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=(0, 0, 0), mask_value=0)
    except TypeError:
        pad = A.PadIfNeeded(min_height=H, min_width=W,
                            border_mode=cv2.BORDER_CONSTANT,
                            border_value=(0, 0, 0), mask_value=0)

    if norm_mode == "hf255":
        norm = get_validation_augmentation(img_size=img_size, norm_mode="hf255").transforms[-1]
    elif norm_mode == "imagenet":
        import albumentations as A
        norm = A.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225),
                           max_pixel_value=255.0)
    else:
        import albumentations as A
        norm = A.Normalize(mean=(0, 0, 0),
                           std=(1, 1, 1),
                           max_pixel_value=255.0)

    return A.Compose([pad, A.CenterCrop(H, W), norm], is_check_shapes=False)


# ------------------------------
# Dataset
# ------------------------------
class _FreeFormImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_size=320, norm_mode="hf255", ignore_index=255):
        import glob
        self.ignore_index = ignore_index
        fps = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff",
                    "*.PNG", "*.JPG", "*.JPEG", "*.TIF", "*.TIFF"):
            fps.extend(glob.glob(os.path.join(img_dir, ext)))
        self.images_fps = sorted(fps)
        if not self.images_fps:
            raise FileNotFoundError(f"No images found under: {img_dir}")

        self.aug = get_val_aug_compat(img_size=img_size, norm_mode=norm_mode)

    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, idx):
        img_path = self.images_fps[idx]
        img = _read_as_uint8_rgb(img_path)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        aug = self.aug(image=img, mask=mask)
        img_t = torch.from_numpy(aug["image"]).permute(2, 0, 1).float()
        msk_t = torch.from_numpy(aug["mask"]).long()
        return img_t, msk_t, img_path


def build_test_loader(args):
    test_ds = _FreeFormImageFolderDataset(
        img_dir=args.test_image_dir,
        img_size=args.img_size,
        norm_mode=args.norm_mode,
        ignore_index=args.ignore_index,
    )

    sampler = None  # Let PL handle distributed sampler

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[INFO] Test dataset size: {len(test_ds)} (DDP sampler: {sampler is not None})")
    return loader


# ------------------------------
# JSON builder
# ------------------------------
def _coords_from_mask(img_filename: str, pred_mask: np.ndarray, num_classes: int):
    coords = {"image_name": os.path.basename(img_filename),
              "predicted_coords": {}}
    for cls in range(1, num_classes):
        ys, xs = np.where(pred_mask == cls)
        if ys.size:
            coords["predicted_coords"][f"class_{cls}"] = np.stack([ys, xs], axis=1).tolist()
    return coords


# ------------------------------
# Checkpoint loader
# ------------------------------
def load_model_strict(ckpt_path: str):
    try:
        m = Mask2FormerLit.load_from_checkpoint(ckpt_path, strict=True)
        print("[load] strict=True load succeeded")
        return m
    except Exception as e:
        print("[load][fallback] strict load failed:", e)

    m = Mask2FormerLit(
        out_classes=len(CLASSES),
        lr=3e-4,
        weight_decay=1e-4,
        encoder_name="facebook/mask2former-swin-small-ade-semantic",
        ignore_index=255,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    m.load_state_dict(state, strict=False)
    return m


# ------------------------------
# Rank-level JSONL Writer
# ------------------------------
class StreamingJSONWriter(pl.Callback):

    def __init__(self, out_root, tag, num_classes,
                 skip_empty_json=True, positives_index_base=""):
        super().__init__()
        self.out_root = out_root
        self.tag = tag
        self.num_classes = num_classes
        self.skip_empty = skip_empty_json
        self.pos_base = positives_index_base
        # NEW: buffer for rank-level output
        self.buffer = []

    def setup(self, trainer, pl_module, stage):
        self.rank = trainer.global_rank
        self.world = trainer.world_size

        out_dir = os.path.join(self.out_root, f"prediction_outputs_{self.tag}")
        self.out_dir_rank = os.path.join(out_dir, f"rank_{self.rank}") if self.world > 1 else out_dir
        os.makedirs(self.out_dir_rank, exist_ok=True)

        # positives index (ranked)
        self.pos_file = None
        if self.pos_base:
            base, ext = os.path.splitext(self.pos_base)
            fname = f"{base}.rank{self.rank}{ext or '.txt'}" if self.world > 1 else self.pos_base
            self.pos_file = os.path.join(self.out_root, fname)
            open(self.pos_file, "a").close()

        if self.rank == 0:
            print(f"[StreamingJSONWriter] writing rank-level JSONL under: {out_dir}")

    def _extract_logits_and_paths(self, outputs, batch_paths):

        # Case 1 — pure tensor
        if torch.is_tensor(outputs):
            return outputs, batch_paths

        # Case 2 — tuple (logits, paths)
        if isinstance(outputs, tuple) and len(outputs) == 2 and torch.is_tensor(outputs[0]):
            return outputs[0], outputs[1]

        # Case 3 — dict
        if isinstance(outputs, dict):
            logits = None
            for key in ("logits", "preds", "y_hat"):
                if key in outputs:
                    if outputs[key] is not None:
                        logits = outputs[key]
                        break
            if logits is None:
                raise RuntimeError("predict_step dict missing logits/preds/y_hat")
            paths = outputs.get("paths", batch_paths)
            return logits, paths

        # Case 4 — list/tuple of dicts
        if isinstance(outputs, (list, tuple)):
            first = outputs[0]

            # list of dict outputs
            if isinstance(first, dict):
                logits_list, paths_list = [], []
                for d in outputs:
                    lg = None
                    for key in ("logits", "preds", "y_hat"):
                        if key in d and d[key] is not None:
                            lg = d[key]
                            break
                    if lg is None:
                        raise RuntimeError("predict_step element missing logits/preds/y_hat")
                    logits_list.append(lg)
                    paths_list.append(d.get("paths"))
                logits = torch.stack(logits_list, dim=0)
                return logits, paths_list

            # list of tensors
            if torch.is_tensor(first):
                logits = torch.stack(outputs, dim=0)
                return logits, batch_paths

        raise RuntimeError(f"Unsupported predict output type: {type(outputs)}")


    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        _, _, paths_from_batch = batch
        logits, paths = self._extract_logits_and_paths(outputs, paths_from_batch)

        if logits.ndim == 3:
            pred_cls = logits.detach().cpu().numpy().astype(np.uint8)
        elif logits.ndim == 4:
            pred_cls = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)
        else:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        B = pred_cls.shape[0]
        if paths is None:
            paths_iter = [None] * B
        elif isinstance(paths, (list, tuple)):
            paths_iter = list(paths)
            if len(paths_iter) != B:
                paths_iter = [None] * B
        else:
            paths_iter = [paths] * B

        for i in range(B):
            path = paths_iter[i]
            base_name = os.path.splitext(os.path.basename(path))[0] if isinstance(path, str) else f"sample_{self.rank}_{batch_idx}_{i}"
            coords = _coords_from_mask(path if isinstance(path, str) else base_name,
                                       pred_cls[i],
                                       self.num_classes)

            if self.skip_empty and not coords["predicted_coords"]:
                continue

            if self.pos_file and coords["predicted_coords"]:
                with open(self.pos_file, "a") as f:
                    f.write(base_name + "\n")

            # NEW → append only
            self.buffer.append(coords)

        del pred_cls

    def on_predict_end(self, trainer, pl_module):
        out_file = os.path.join(self.out_dir_rank, f"rank_{self.rank}.jsonl")
        with open(out_file, "w") as f:
            for item in self.buffer:
                f.write(json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n")

        if self.rank == 0:
            print(f"[DONE] Rank-level JSONL written → {self.out_root}")


# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()
    seed_everything(42)

    project_root = os.path.abspath(args.project_root)
    os.makedirs(project_root, exist_ok=True)

    print("\n================= PREDICT-ONLY (DDP) =================")
    print(f"devices={args.devices}, strategy={args.strategy}, precision={args.precision}")
    print(f"img_size={args.img_size}, batch={args.batch_size}, workers={args.workers}")
    print(f"test_dir={args.test_image_dir}")
    print(f"ckpt={args.ckpt_path}")
    print("=======================================================\n")

    model = load_model_strict(args.ckpt_path)

    test_loader = build_test_loader(args)

    devices_arg = args.devices
    if isinstance(devices_arg, str) and devices_arg not in ("auto",):
        if devices_arg.isdigit():
            devices_arg = int(devices_arg)
        else:
            devices_arg = [int(x) for x in devices_arg.split(",") if x.strip()]

    suffix = args.pred_out_version.strip()
    tag = f"Mask2Former{('_' + suffix) if suffix else ''}"

    callbacks = [TQDMProgressBar(refresh_rate=50)]
    if args.stream_save:
        callbacks.append(
            StreamingJSONWriter(
                out_root=project_root,
                tag=tag,
                num_classes=len(CLASSES),
                skip_empty_json=args.skip_empty_json,
                positives_index_base=args.positives_index,
            )
        )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices_arg,
        strategy=args.strategy,
        precision=args.precision,
        logger=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        use_distributed_sampler=True,
    )

    trainer.predict(model, dataloaders=test_loader, return_predictions=False)

    if not _is_ddp_active() or trainer.global_rank == 0:
        out_dir = os.path.join(project_root, f"prediction_outputs_{tag}")
        print(f"[FINAL] JSONL saved under {out_dir}\n")


if __name__ == "__main__":
    main()

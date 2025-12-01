# data_gen_M.py
# Robust folder-based dataset for 4-class PV segmentation with Albumentations.
# Classes: 0=background, 1=PV_normal, 2=PV_heater, 3=PV_pool
# Default mask naming rule: masks/m_{image_id[2:]}.png   (e.g., img: abXXXX.png -> mask: m_XXXX.png)
# If that fails, we fall back to a set of reasonable filename candidates.

from __future__ import annotations
import os
import glob
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable

import torch
from torch.utils.data import Dataset as BaseDataset, DataLoader, WeightedRandomSampler
import albumentations as A

# -----------------------------
# Public constants
# -----------------------------
CLASSES = ["background", "PV_normal", "PV_heater", "PV_pool"]
CLASS_LABELS = {
    0: "Background",
    1: "PV_normal",
    2: "PV_heater",
    3: "PV_pool",
}
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255

# -----------------------------
# Albumentations helpers (version-compatible)
# -----------------------------
def _gaussnoise_compat(scale_tuple=(10, 50), p=0.3):
    """Albumentations changed GaussNoise args: newer uses `scale`, older uses `var_limit`."""
    try:
        return A.GaussNoise(scale=scale_tuple, p=p)
    except TypeError:
        return A.GaussNoise(var_limit=tuple(float(x) for x in scale_tuple), p=p)

def _affine_compat(
    scale=(0.9, 1.1),
    translate_percent=(0.1, 0.1),
    rotate=(-15, 15),
    p=0.5,
):
    """Prefer A.Affine; fallback to A.ShiftScaleRotate on older versions."""
    if hasattr(A, "Affine"):
        # Some versions accept interpolation/mask_interpolation, some do not.
        try:
            return A.Affine(
                scale=scale,
                translate_percent=translate_percent,
                rotate=rotate,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                cval=0,
                cval_mask=0,
                mode=cv2.BORDER_CONSTANT,
                p=p,
            )
        except TypeError:
            return A.Affine(
                scale=scale,
                translate_percent=translate_percent,
                rotate=rotate,
                mode=cv2.BORDER_CONSTANT,
                p=p,
            )
    # Fallback: ShiftScaleRotate (older Albumentations)
    rotate_limit = int(max(abs(rotate[0]), abs(rotate[1])))
    scale_limit = (scale[0] - 1.0, scale[1] - 1.0)
    return A.ShiftScaleRotate(
        shift_limit=translate_percent,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        interpolation=cv2.INTER_LINEAR,
        p=p,
    )

def _randomshadow_compat(p=0.3):
    return A.RandomShadow(p=p) if hasattr(A, "RandomShadow") else A.NoOp()

def _elastic_compat(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03):
    """A.ElasticTransform changed arg list in some versions."""
    try:
        return A.ElasticTransform(
            p=p, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine,
            interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
        )
    except TypeError:
        # Older versions without alpha_affine/mask_interpolation
        return A.ElasticTransform(
            p=p, alpha=alpha, sigma=sigma,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT, value=0,
        )

# -----------------------------
# Normalization factory
# -----------------------------
def get_normalize_transform(mode: str = "hf255") -> A.BasicTransform:
    """
    mode:
      - "hf255": Detectron/HF-ish normalization; means/stds in 0..1 because Albumentations expects that scale
      - "imagenet01": Standard ImageNet (0..1) normalization
      - "none": no normalization
    """
    m = mode.lower().strip() if isinstance(mode, str) else "hf255"
    if m == "hf255":
        mean = (123.675/255.0, 116.28/255.0, 103.53/255.0)
        std  = (58.395/255.0, 57.12/255.0, 57.375/255.0)
        return A.Normalize(mean=mean, std=std)
    if m == "imagenet01":
        return A.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225))
    return A.NoOp()

# -----------------------------
# Filename mapping helpers
# -----------------------------
def _candidate_mask_paths(masks_dir: Path, image_id: str) -> List[str]:
    """
    Create a list of candidate mask filepaths based on the image filename.
    Primary rule: 'm_' + image_id without first 2 chars + '.png'
    Fallbacks include:
      - m_<stem>.png
      - <stem>.png
      - m_<stem>.* for common extensions
    """
    stem = os.path.splitext(image_id)[0]
    candidates = []

    # Primary rule (abXXXX.png -> m_XXXX.png)
    if len(stem) >= 2:
        candidates.append(masks_dir / f"m_{stem[2:]}.png")

    # Simple variants
    candidates.append(masks_dir / f"m_{stem}.png")
    candidates.append(masks_dir / f"{stem}.png")

    # Case-insensitive / extension fallbacks
    exts = ["png", "PNG", "jpg", "jpeg", "tif", "tiff"]
    for ext in exts:
        candidates.append(masks_dir / f"m_{stem[2:]}.{ext}") if len(stem) >= 2 else None
        candidates.append(masks_dir / f"m_{stem}.{ext}")
        candidates.append(masks_dir / f"{stem}.{ext}")

    # Glob by patterns (last resort)
    patterns = [
        f"m_{stem[2:]}.*" if len(stem) >= 2 else None,
        f"m_{stem}.*",
        f"{stem}.*",
    ]
    for pat in patterns:
        if not pat:
            continue
        for p in glob.glob(str(masks_dir / pat)):
            candidates.append(Path(p))

    # Deduplicate while preserving order
    uniq: List[str] = []
    for p in candidates:
        if p is None:
            continue
        ps = str(p)
        if ps not in uniq:
            uniq.append(ps)
    return uniq

def find_mask_path(masks_dir: Path, image_id: str) -> Optional[str]:
    """Return the first existing mask file path from a set of candidates, else None."""
    for cand in _candidate_mask_paths(masks_dir, image_id):
        if os.path.isfile(cand):
            return cand
    return None

# -----------------------------
# Mask sanitization
# -----------------------------
def sanitize_mask(mask: np.ndarray, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """
    Ensure mask contains only {0..num_classes-1, ignore_index}.
    Any negative or > num_classes-1 values are set to ignore_index.
    """
    out = mask.astype(np.int64, copy=False)
    invalid = (out < 0) | (out > (num_classes - 1))
    out[invalid] = ignore_index
    return out

# -----------------------------
# Dataset
# -----------------------------
class PVDatasetFoldersCT(BaseDataset):
    """
    Folder layout:
      root/
        train/images/*.png
        train/masks/m_*.png
        val/images/*.png
        val/masks/m_*.png
        (test/images, test/masks) optional

    __getitem__ returns: (image_tensor [3,H,W] float32, mask_tensor [H,W] int64)
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_dirname: str = "images",
        msk_dirname: str = "masks",
        classes: Optional[List[str]] = None,
        augmentation: Optional[Callable] = None,
        img_size: int = 320,
        norm_mode: str = "hf255",
        strict_mask_size: bool = False,
        debug: bool = False,
        return_path: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / split / img_dirname
        self.masks_dir  = self.root / split / msk_dirname
        self.augmentation = augmentation
        self.img_size = int(img_size)
        self.norm_mode = norm_mode
        self.strict_mask_size = strict_mask_size
        self.debug = debug
        self.return_path = bool(return_path)

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Missing masks dir: {self.masks_dir}")

        # List images (skip hidden)
        self.ids = sorted([f for f in os.listdir(self.images_dir) if not f.startswith(".")])
        self.images_fps = [str(self.images_dir / image_id) for image_id in self.ids]

        # Resolve masks with robust mapping
        resolved_masks: List[str] = []
        bad = 0
        for image_id in self.ids:
            mpath = find_mask_path(self.masks_dir, image_id)
            if mpath is None:
                bad += 1
                if self.debug:
                    print(f"[WARN] No mask found for: {image_id}")
                resolved_masks.append("")  # placeholder, will be handled in __getitem__
            else:
                resolved_masks.append(mpath)
        if bad and self.debug:
            print(f"[INFO] {bad} / {len(self.ids)} images missing masks (will be skipped on access).")

        self.masks_fps = resolved_masks

        # Optional class subset (keeps order; remaps labels)
        if classes:
            self.class_values = [CLASSES.index(cls) for cls in classes]
        else:
            self.class_values = list(range(len(CLASSES)))
        # identity mapping by default
        self.class_map = {i: i for i in self.class_values}

        # Augmentations: append normalization if not already included
        # (We keep normalization in the pipeline, not manual per-sample)
        if self.augmentation is None:
            self.augmentation = get_validation_augmentation(self.img_size, self.norm_mode)

    def __len__(self):
        return len(self.ids)

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError(f"Unreadable image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return img

    def _read_mask(self, path: str) -> np.ndarray:
        m = np.array(Image.open(path).convert("L"))  # keep label values exactly
        if m is None or m.size == 0:
            raise ValueError(f"Unreadable mask: {path}")
        if m.shape != (self.img_size, self.img_size):
            if self.strict_mask_size:
                raise ValueError(f"Invalid mask shape: {m.shape}, expected {(self.img_size, self.img_size)}")
            m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return m

def __getitem__(self, i: int):
    """
    Returns:
        if self.return_path is False (default):
            image: float32 [3,H,W] (already normalized if Normalize used)
            mask : int64  [H,W] in {0,1,2,3} or 255 (ignore)
        if self.return_path is True:
            (image, mask, image_path)

    If a sample is broken (missing/corrupt), we skip to the next index.
    """
    image_path = self.images_fps[i]
    mask_path  = self.masks_fps[i]

    # Require a mask for this dataset layout (training/val style).
    if not mask_path or not os.path.isfile(mask_path):
        if self.debug:
            print(f"[WARN] Skipping (no mask): {image_path}")
        j = (i + 1) % len(self)
        if j == i:
            raise RuntimeError("Dataset has no valid samples.")
        return self.__getitem__(j)

    try:
        # --- read ---
        image = self._read_image(image_path)
        mask  = self._read_mask(mask_path)

        # --- optional class subset/remap ---
        if len(self.class_values) < len(CLASSES):
            remap = np.zeros_like(mask, dtype=np.uint8)
            for new_idx, old_val in enumerate(self.class_values):
                remap[mask == old_val] = new_idx
            mask = remap

        # --- augment / normalize (Albumentations expects HWC image, 2D mask) ---
        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # --- sanitize mask values (keep {0..C-1, 255}) ---
        mask = sanitize_mask(mask, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

        # --- to tensors ---
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))              # HWC -> CHW
        image = torch.from_numpy(image)

        if mask.dtype != np.int64:
            mask = mask.astype(np.int64)
        mask = torch.from_numpy(mask)

        # --- RETURN: optionally include path for DDP-stable saving ---
        if getattr(self, "return_path", False):
            return image, mask, image_path
        else:
            return image, mask

    except Exception as e:
        if self.debug:
            print(f"[WARN] Skipping index {i} ({self.ids[i]}): {e}")
        j = (i + 1) % len(self)
        if j == i:
            raise
        return self.__getitem__(j)


# -----------------------------
# Augmentation pipelines
# -----------------------------
def get_training_augmentation(img_size: int = 320, norm_mode: str = "hf255") -> A.Compose:
    H = W = int(img_size)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            _affine_compat(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5),

            # Keep size HxW
            A.PadIfNeeded(min_height=H, min_width=W, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.RandomCrop(height=H, width=W),

            # Noise & blur
            _gaussnoise_compat((10, 50), p=0.3),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.3,
            ),

            # Color
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                    A.RandomGamma(gamma_limit=(80, 120), p=1),
                    A.CLAHE(clip_limit=2.0, p=1),
                ],
                p=0.8,
            ),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),

            # Shadows (compat)
            _randomshadow_compat(p=0.3),

            # Elastic/geometric
            _elastic_compat(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),

            # Normalization at the end
            get_normalize_transform(mode=norm_mode),
        ],
        is_check_shapes=False,
    )

def get_validation_augmentation(img_size: int = 320, norm_mode: str = "hf255") -> A.Compose:
    H = W = int(img_size)
    return A.Compose(
        [
            A.PadIfNeeded(min_height=H, min_width=W, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.CenterCrop(height=H, width=W),
            get_normalize_transform(mode=norm_mode),
        ],
        is_check_shapes=False,
    )

# -----------------------------
# Sampler helpers (optional)
# -----------------------------
def compute_sample_weights(dataset: PVDatasetFoldersCT, num_classes: int = NUM_CLASSES) -> torch.DoubleTensor:
    """
    Very simple sample weighting: images that contain more foreground classes get higher weight.
    Background-only gets lower weight. This increases FG exposure per batch.
    """
    weights: List[float] = []
    for i in range(len(dataset)):
        try:
            _, m = dataset[i]
            m = m.numpy()
            u = np.unique(m[m != IGNORE_INDEX])
            # (num distinct classes among {0..C-1}). Background-only => len({0}) == 1
            # Give more weight if any FG is present (class >=1).
            has_fg = np.any((u >= 1) & (u < num_classes))
            if not u.size:
                w = 0.5
            elif has_fg:
                # boost if multiple fg classes
                k = max(1, len([x for x in u if x >= 1]))
                w = 0.5 + 0.5 * min(1.0, k / (num_classes - 1))
            else:
                # only background
                w = 0.6
            weights.append(float(w))
        except Exception:
            # if sample is broken, give small weight
            weights.append(0.1)
    return torch.DoubleTensor(weights)

# -----------------------------
# Public factory: dataloaders
# -----------------------------
def make_dataloaders(
    train_img_dir: str,
    valid_img_dir: str,
    train_mask_dir: Optional[str] = None,
    valid_mask_dir: Optional[str] = None,
    img_size: int = 320,
    batch_size: int = 8,
    workers: int = 4,
    norm_mode: str = "hf255",
    use_weighted_sampler: bool = False,
    persistent_workers: bool = False,
    debug: bool = False,
):
    """
    Build train/valid DataLoaders.

    Args:
        train_img_dir, valid_img_dir: image folder paths
        train_mask_dir, valid_mask_dir: if provided, the parent split folder is inferred from these;
            if None, we assume masks live at <split>/masks next to <split>/images
        norm_mode: "hf255" | "imagenet01" | "none"
        use_weighted_sampler: enable foreground-friendly sampling
    """
    # Infer common roots if mask dirs not provided
    def _root_of(img_dir: str) -> str:
        # expects .../<split>/images
        p = Path(img_dir)
        if p.name.lower() != "images":
            # Assume parent contains images/, masks/
            return str(p)
        return str(p.parent.parent) if p.parent.name.lower() in ("train", "val", "valid", "test") else str(p.parent)

    train_root = _root_of(train_img_dir) if train_mask_dir is None else str(Path(train_img_dir).parents[1])
    valid_root = _root_of(valid_img_dir) if valid_mask_dir is None else str(Path(valid_img_dir).parents[1])

    # Build datasets
    train_ds = PVDatasetFoldersCT(
        root=train_root,
        split="train",
        img_dirname=Path(train_img_dir).name if Path(train_img_dir).name != "images" else "images",
        msk_dirname=Path(train_mask_dir).name if train_mask_dir else "masks",
        augmentation=get_training_augmentation(img_size=img_size, norm_mode=norm_mode),
        img_size=img_size,
        norm_mode=norm_mode,
        strict_mask_size=False,
        debug=debug,
    )
    valid_ds = PVDatasetFoldersCT(
        root=valid_root,
        split="val" if "val" in Path(valid_img_dir).parts or "valid" in Path(valid_img_dir).parts else "valid",
        img_dirname=Path(valid_img_dir).name if Path(valid_img_dir).name != "images" else "images",
        msk_dirname=Path(valid_mask_dir).name if valid_mask_dir else "masks",
        augmentation=get_validation_augmentation(img_size=img_size, norm_mode=norm_mode),
        img_size=img_size,
        norm_mode=norm_mode,
        strict_mask_size=False,
        debug=debug,
    )

    # DataLoaders
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=persistent_workers)

    if use_weighted_sampler:
        weights = compute_sample_weights(train_ds, num_classes=NUM_CLASSES)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)

    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, valid_loader

# -----------------------------
# Debug helpers (optional)
# -----------------------------
def debug_unique_labels(ds: PVDatasetFoldersCT, k: int = 10):
    """Print unique mask values for first k samples (ignores IO errors by skipping)."""
    print(f"Dataset size: {len(ds)}")
    cnt = 0
    idx = 0
    while cnt < k and idx < len(ds):
        try:
            _, m = ds[idx]
            u = torch.unique(m).tolist()
            print(f"[DEBUG] mask {idx} uniques:", u)
            cnt += 1
        except Exception as e:
            print(f"[DEBUG] skip idx {idx}: {e}")
        idx += 1

def dump_samples(ds: PVDatasetFoldersCT, out_dir: str, k: int = 8, norm_mode: str = "imagenet01"):
    """
    Save first k image/mask pairs for sanity-check.
    For visualization, we inverse-normalize only for 'imagenet01' mode here (example).
    """
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(k, len(ds))):
        x, m = ds[i]
        # x: CHW float32 normalized -> convert to HWC for saving
        img = x.numpy().transpose(1, 2, 0)
        if norm_mode == "imagenet01":
            img = np.clip((img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406]), 0, 1)
            img = (img * 255).astype(np.uint8)
        else:
            # Just rescale to 0..255 heuristically
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"img_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Colorize mask
        mask = m.numpy().astype(np.int32)
        color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colors = {0:(0,0,0),1:(0,255,0),2:(0,0,255),3:(255,0,0),255:(128,128,128)}
        for c, rgb in colors.items():
            color[mask == c] = rgb
        cv2.imwrite(os.path.join(out_dir, f"mask_{i}.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

# # -----------------------------
# # Quick debug utility (optional)
# # -----------------------------
# def debug_unique_labels(images_dir: str, masks_dir: str, img_size: int = 320, k: int = 10):
#     """
#     Print unique label values for the first k samples (after all transforms).
#     Usage:
#         python -c "from data_gen_M import debug_unique_labels; debug_unique_labels('/path/train/images','/path/train/masks')"
#     """
#     ds = PVSegDataset(
#         images_dir=images_dir,
#         masks_dir=masks_dir,
#         img_size=img_size,
#         augmentation=get_validation_augmentation(img_size),
#         debug=True,
#     )
#     print(f"Train dataset size: {len(ds)}")
#     for i in range(min(k, len(ds))):
#         _, m = ds[i]
#         uniques = torch.unique(torch.as_tensor(m)).tolist()
#         print(f"[DEBUG] mask {i} uniques:", uniques)

# """
# data_gen_M.py
# Folder-based dataset for 4-class PV segmentation with Albumentations.
# Classes: 0=background, 1=PV_normal, 2=PV_heater, 3=PV_pool
# Mask naming rule: masks/m_{image_id[2:]}.png   (e.g., img: abXXXX.png -> mask: m_XXXX.png)

# Directory layout:
#   root/
#     train/images/*.png
#     train/masks/m_*.png
#     val/images/*.png
#     val/masks/m_*.png
#     (test/images, test/masks) optional

# Outputs:
#   __getitem__ -> dict(
#       pixel_values: float32 tensor [3, 320, 320] in [0,1],
#       labels:      int64 tensor  [320, 320] in {0,1,2,3} (255 allowed for ignore),
#       path:        str (image path)
#   )
# """

# import os
# import cv2
# import numpy as np
# from PIL import Image
# from pathlib import Path
# from typing import Optional, List, Dict, Tuple, Callable
# import torch
# from torch.utils.data import Dataset as BaseDataset
# import albumentations as A
# import inspect

# CLASSES = ["background", "PV_normal", "PV_heater", "PV_pool"]


# # -----------------------------
# # Albumentations compatibility helpers
# # -----------------------------
# def _gaussnoise_compat(scale_tuple=(10, 50), p=0.3):
#     """
#     Albumentations changed GaussNoise args across versions:
#       - Newer: scale=(lo, hi)
#       - Older: var_limit=(lo, hi) floats
#     Detect by signature and use appropriate argument.
#     """
#     sig = inspect.signature(A.GaussNoise.__init__)
#     if "scale" in sig.parameters:
#         return A.GaussNoise(scale=scale_tuple, p=p)
#     else:
#         return A.GaussNoise(var_limit=tuple(float(x) for x in scale_tuple), p=p)


# def _affine_compat(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5):
#     if hasattr(A, "Affine"):
#         return A.Affine(
#             scale=scale,
#             translate_percent=translate_percent,
#             rotate=rotate,
#             interpolation=cv2.INTER_LINEAR,
#             mask_interpolation=cv2.INTER_NEAREST,
#             cval=0, cval_mask=0,
#             mode=cv2.BORDER_CONSTANT,
#             p=p,
#         )
#     # Fallback: ShiftScaleRotate
#     rotate_limit = int(max(abs(rotate[0]), abs(rotate[1])))
#     scale_limit = (scale[0] - 1.0, scale[1] - 1.0)
#     return A.ShiftScaleRotate(
#         shift_limit=translate_percent,
#         scale_limit=scale_limit,
#         rotate_limit=rotate_limit,
#         border_mode=cv2.BORDER_CONSTANT,
#         value=0, mask_value=0,
#         interpolation=cv2.INTER_LINEAR,
#         p=p,
#     )


# def _randomshadow_compat(p=0.3):
#     """
#     Use A.RandomShadow if exists; else NoOp.
#     """
#     return A.RandomShadow(p=p) if hasattr(A, "RandomShadow") else A.NoOp()


# def _elastic_compat(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03):
#     try:
#         return A.ElasticTransform(
#             p=p, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine,
#             interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
#             border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
#         )
#     except TypeError:
#         return A.ElasticTransform(
#             p=p, alpha=alpha, sigma=sigma,
#             interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
#             border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
#         )



# # -----------------------------
# # Dataset
# # -----------------------------
# class PVDatasetFoldersCT(BaseDataset):
#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         img_dirname: str = "images",
#         msk_dirname: str = "masks",
#         classes: Optional[List[str]] = None,
#         augmentation: Optional[Callable] = None,
#         enforce_size: Tuple[int, int] = (320, 320),
#     ):
#         self.root = Path(root)
#         self.split = split
#         self.images_dir = self.root / split / img_dirname
#         self.masks_dir = self.root / split / msk_dirname
#         self.augmentation = augmentation
#         self.enforce_size = enforce_size

#         if not self.images_dir.is_dir() or not self.masks_dir.is_dir():
#             raise FileNotFoundError(f"Missing images/masks under: {self.root}/{split}")

#         # List images (skip hidden)
#         self.ids = sorted([f for f in os.listdir(self.images_dir) if not f.startswith(".")])
#         self.images_fps = [str(self.images_dir / image_id) for image_id in self.ids]

#         # Mask naming rule: m_{image_id[2:]}.png
#         self.masks_fps = [
#             str(self.masks_dir / f"m_{os.path.splitext(image_id)[0][2:]}.png")
#             for image_id in self.ids
#         ]

#         # Class selection/remap (optional)
#         if classes:
#             self.class_values = [CLASSES.index(cls) for cls in classes]
#         else:
#             self.class_values = list(range(len(CLASSES)))
#         self.class_map = {i: i for i in self.class_values}

#     def __len__(self):
#         return len(self.ids)

#     def _read_image(self, path: str) -> np.ndarray:
#         img = cv2.imread(path)
#         if img is None or img.size == 0:
#             raise ValueError(f"Unreadable image: {path}")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img

#     def _read_mask(self, path: str) -> np.ndarray:
#         m = np.array(Image.open(path).convert("L"))
#         if m is None or m.size == 0:
#             raise ValueError(f"Unreadable mask: {path}")
#         return m

#     def __getitem__(self, i: int) -> Dict:
#         img_path = self.images_fps[i]
#         msk_path = self.masks_fps[i]

#         image = self._read_image(img_path)
#         mask = self._read_mask(msk_path)

#         H, W = self.enforce_size
#         if image.shape[:2] != (H, W):
#             image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
#         if mask.shape != (H, W):
#             mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

#         # Optional class filtering/remap
#         if len(self.class_values) < len(CLASSES):
#             remap = np.zeros_like(mask)
#             for idx, class_value in enumerate(self.class_values):
#                 remap[mask == class_value] = idx
#             mask = remap

#         # Albumentations expects HWC image & 2D mask
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample["image"], sample["mask"]

#         # Tensorize
#         image = image.astype(np.float32)
#         image = np.transpose(image, (2, 0, 1))  # CHW
#         image = torch.from_numpy(image)
#         mask = torch.from_numpy(mask.astype(np.int64))

#         return {"pixel_values": image, "labels": mask, "path": img_path}


# # -----------------------------
# # Albumentations pipelines (train/val) @320x320
# # -----------------------------
# def get_training_augmentation():
#     return A.Compose(
#         [
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.3),
#             _affine_compat(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5),

#             # Keep size 320x320
#             A.PadIfNeeded(min_height=320, min_width=320, border_mode=cv2.BORDER_CONSTANT),
#             A.RandomCrop(height=320, width=320),

#             # Noise & blur
#             _gaussnoise_compat((10, 50), p=0.3),
#             A.OneOf([A.Blur(3, p=1), A.MotionBlur(3, p=1)], p=0.3),

#             # Color
#             A.OneOf(
#                 [
#                     A.RandomBrightnessContrast(0.2, 0.2, p=1),
#                     A.RandomGamma(gamma_limit=(80, 120), p=1),
#                     A.CLAHE(clip_limit=2.0, p=1),
#                 ],
#                 p=0.8,
#             ),
#             A.HueSaturationValue(10, 20, 10, p=0.5),

#             # Shadows (compat)
#             _randomshadow_compat(p=0.3),

#             # Elastic/geometric
#             _elastic_compat(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            
#             # Imagenet normalization
#             A.Normalize(mean=(0.485, 0.456, 0.406),
#                         std=(0.229, 0.224, 0.225)),
#         ]
#     )


# def get_validation_augmentation():
#     return A.Compose(
#         [
#             A.PadIfNeeded(min_height=320, min_width=320, border_mode=cv2.BORDER_CONSTANT),
#             A.CenterCrop(height=320, width=320),
            
#             A.Normalize(mean=(0.485, 0.456, 0.406),
#                         std=(0.229, 0.224, 0.225)),
#         ]
#     )

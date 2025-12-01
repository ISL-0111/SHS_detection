# model_M_patched.py
from __future__ import annotations

# ---------- Standard library ----------
import os
import math
from typing import Optional, List, Dict, Any, Tuple

# ---------- PyTorch / Lightning ----------
import torch
import torch.nn as nn
import torch.distributed as dist
import pytorch_lightning as pl

# ---------- Optimizer & LR schedulers ----------
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# ---------- Torchmetrics ----------
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

# ---------- Hugging Face Transformers ----------
try:
    from transformers import (
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )
except Exception as e:
    raise RuntimeError("transformers is required (pip install transformers).") from e

# To be set by caller:
# self.stream_save_png: bool = True
# self.stream_save_json: bool = True


# ============================
# Utilities
# ============================
def _logit_bias_from_prior(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    mask = (target != ignore_index)
    if not mask.any():
        return logits.new_zeros(())

    t = torch.where(mask, target, torch.zeros_like(target))
    onehot = torch.nn.functional.one_hot(t, num_classes=num_classes).permute(0, 3, 1, 2).float()
    onehot = onehot * mask.unsqueeze(1).float()
    probs = torch.softmax(logits, dim=1) * mask.unsqueeze(1).float()

    dims = (0, 2, 3)
    inter = torch.sum(probs * onehot, dims)
    denom = torch.sum(probs + onehot, dims)
    dice_per_class = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice_per_class.mean()


# ============================
# Lightning Module
# ============================
class PVModel(pl.LightningModule):
    """
    LightningModule wrapping HF Mask2Former for semantic segmentation.
    - 안전한 forward(배치가 list/tuple로 와도 동작)
    - predict_step 스트리밍 저장 옵션(대규모 이미지에서도 메모리 폭주 방지)
    """

    def __init__(
        self,
        arch: str = "Mask2Former",
        encoder_name: str = "facebook/mask2former-swin-small-ade-semantic",
        in_channels: int = 3,
        out_classes: int = 4,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 0.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        max_epochs: int = 50,
        ce_class_weights: Optional[List[float]] = None,
        head_priors: Optional[List[float]] = None,
        enable_diag_print: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ce_class_weights", "head_priors"])

        self.out_classes = int(out_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = class_names or ["background", "PV_normal", "PV_heater", "PV_pool"]

        # ---------- HF model ----------
        config = Mask2FormerConfig.from_pretrained(
            encoder_name,
            num_labels=self.out_classes,
            ignore_mismatched_sizes=True,
            id2label={i: n for i, n in enumerate(self.class_names)},
            label2id={n: i for i, n in enumerate(self.class_names)},
        )
        self.hf = Mask2FormerForUniversalSegmentation.from_pretrained(
            encoder_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
        try:
            self.processor = Mask2FormerImageProcessor(
                do_resize=False, do_rescale=False, do_normalize=False
            )
        except TypeError:
            self.processor = Mask2FormerImageProcessor()

        self.hf.train()

        # Optional head bias init
        try:
            cls_head = getattr(self.hf, "class_predictor", None)
            if isinstance(cls_head, nn.Linear) and head_priors and len(head_priors) == self.out_classes:
                with torch.no_grad():
                    new_bias = torch.tensor([_logit_bias_from_prior(p) for p in head_priors], dtype=cls_head.bias.dtype)
                    cls_head.bias.copy_(new_bias)
        except Exception:
            pass

        # ---------- Losses ----------
        ce_w = torch.tensor(ce_class_weights, dtype=torch.float32) if ce_class_weights is not None else None
        self._ce = nn.CrossEntropyLoss(weight=ce_w, ignore_index=self.ignore_index)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

        # ---------- Metrics ----------
        self.val_iou_none  = MulticlassJaccardIndex(num_classes=self.out_classes, ignore_index=None, average=None)
        self.val_prec_none = MulticlassPrecision  (num_classes=self.out_classes, ignore_index=None, average=None)
        self.val_rec_none  = MulticlassRecall     (num_classes=self.out_classes, ignore_index=None, average=None)
        self.val_f1_none   = MulticlassF1Score    (num_classes=self.out_classes, ignore_index=None, average=None)
        self.val_iou_micro = MulticlassJaccardIndex(num_classes=self.out_classes, average="micro")

        self.register_buffer(
            "val_confmat",
            torch.zeros(self.out_classes, self.out_classes, dtype=torch.long),
            persistent=False,
        )
        self.last_class_stats: Optional[List[Dict[str, Any]]] = None

        # ---------- Scheduler config ----------
        self.base_lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.max_epochs_cfg = int(max_epochs)

        # ---------- Diagnostics ----------
        self.enable_diag_print = bool(enable_diag_print)
        self._diag_seen_val_batches = 0

        # ---------- Streaming saver flags ----------
        self.stream_save: bool = False
        self._save_out_dir: Optional[str] = None
        self._save_color_dir: Optional[str] = None
        # RGB
        self._save_colors: Dict[int, Tuple[int, int, int]] = {0:(0,0,0), 1:(0,255,0), 2:(0,0,255), 3:(255,0,0)}
        # JSON, PNG
        self.stream_save_png: bool = True
        self.stream_save_json: bool = True

    # ============================
    # Streaming saver control
    # ============================
    def enable_stream_saver(
        self,
        out_dir: str,
        color_dir: str,
        colors: Optional[Dict[int, Tuple[int,int,int]]] = None,
        save_png: bool = True,
        save_json: bool = True,
    ):
        """각 rank가 predict_step에서 즉시 PNG/JSON 저장하도록 활성화 (선택 저장 가능)."""
        self.stream_save = True
        self._save_out_dir = out_dir
        self._save_color_dir = color_dir
        if colors:
            self._save_colors = colors
        self.stream_save_png = bool(save_png)
        self.stream_save_json = bool(save_json)
        os.makedirs(self._save_out_dir, exist_ok=True)
        os.makedirs(self._save_color_dir, exist_ok=True)

    # ============================
    # Helpers
    # ============================
    @staticmethod
    def _ensure_batched_tensor(x: Any) -> torch.Tensor:
        """Accept Tensor or list/tuple of Tensors and return a stacked float Tensor [B,3,H,W]."""
        if isinstance(x, (list, tuple)):
            if len(x) and isinstance(x[0], torch.Tensor):
                return torch.stack(list(x), dim=0)
            raise TypeError(f"Expected list/tuple of tensors, got: {type(x[0]) if x else 'empty'}")
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected Tensor, got: {type(x)}")
        return x

    def _extract_semantic_logits(self, out: Any) -> Optional[torch.Tensor]:
        if out is None:
            return None
        logits = getattr(out, "logits", None)
        if logits is None:
            logits = getattr(out, "semantic_logits", None)
        if logits is None and isinstance(out, dict):
            logits = out.get("logits", None) or out.get("semantic_logits", None)
        return logits

    def _semantic_logits_from_queries(
        self,
        out: Any,
        img_hw: Optional[Tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        cls = getattr(out, "class_queries_logits", None)
        if cls is None and isinstance(out, dict):
            cls = out.get("class_queries_logits", None)
        if cls is None:
            cls = getattr(out, "class_logits", None) or (out.get("class_logits", None) if isinstance(out, dict) else None)

        msk = getattr(out, "masks_queries_logits", None)
        if msk is None and isinstance(out, dict):
            msk = out.get("masks_queries_logits", None)
        if msk is None:
            msk = getattr(out, "masks_logits", None) or (out.get("masks_logits", None) if isinstance(out, dict) else None)

        if cls is None or msk is None:
            return None
        if cls.ndim != 3 or msk.ndim != 4:
            return None

        B, Q, C = cls.shape
        Bh, Qh, h, w = msk.shape
        if B != Bh or Q != Qh:
            return None

        cls_b = cls.unsqueeze(-1).unsqueeze(-1)  # (B,Q,C,1,1)
        msk_b = msk.unsqueeze(2)                 # (B,Q,1,h,w)
        combined = cls_b + msk_b                 # (B,Q,C,h,w)
        sem_logits = combined.max(dim=1).values  # (B,C,h,w)

        if img_hw is not None:
            H, W = img_hw
            if sem_logits.shape[-2:] != (H, W):
                sem_logits = torch.nn.functional.interpolate(
                    sem_logits, size=(H, W), mode="bilinear", align_corners=False
                )
        return sem_logits

    def _align_logits_channels(self, logits: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        if C == self.out_classes:
            return logits
        if C > self.out_classes:
            return logits[:, : self.out_classes, :, :]
        pad_ch = self.out_classes - C
        pad = logits.new_full((B, pad_ch, H, W), fill_value=-1e4)
        return torch.cat([logits, pad], dim=1)

    # ============================
    # Forward / Predict
    # ============================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (안전) list/tuple of tensors -> stack
        if isinstance(x, (list, tuple)):
            if len(x) and isinstance(x[0], torch.Tensor):
                x = torch.stack(list(x), dim=0)
            else:
                raise TypeError(f"Forward expected Tensor or list of Tensors; got {type(x)}")

        out = self.hf(pixel_values=x)

        logits = self._extract_semantic_logits(out)
        if logits is None:
            was_training = self.hf.training
            with torch.set_grad_enabled(self.training):
                self.hf.eval()
                out_eval = self.hf(pixel_values=x)
                if was_training:
                    self.hf.train()
            logits = self._extract_semantic_logits(out_eval)
            if logits is None:
                out = out_eval

        if logits is None:
            B, _, H, W = x.shape
            logits = self._semantic_logits_from_queries(out, img_hw=(H, W))

        if logits is None:
            raise RuntimeError(
                "Failed to build semantic logits from HF outputs. "
                "Neither `logits`/`semantic_logits` nor query-level fields were available."
            )

        logits = self._align_logits_channels(logits)
        return logits

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        Return:
          - streaming 저장 활성화 시: {"saved": B}
          - 비활성화 시: {"logits": logits, "paths": paths}
        Accept batch:
          - (images, masks) / (images, masks, paths)
          - images
          - dict: {"pixel_values"/"images"/"x": images, "path"/"paths": ...}
        """
        paths = None

        if isinstance(batch, dict):
            images = batch.get("pixel_values", None)
            if images is None:
                images = batch.get("images", None) or batch.get("x", None)
            paths = batch.get("path", batch.get("paths", None))
        elif isinstance(batch, (list, tuple)):
            images = batch[0] if len(batch) >= 1 else None
            if len(batch) >= 3:
                paths = batch[2]
        elif torch.is_tensor(batch):
            images = batch
        else:
            raise RuntimeError(f"Unexpected batch type in predict_step: {type(batch)}")

        if images is None:
            raise RuntimeError("predict_step could not extract images from batch.")

        images = self._ensure_batched_tensor(images).to(self.device).float()

        with torch.inference_mode():
            logits = self(images)          # [B,C,H,W]
            classmap = torch.argmax(logits, dim=1)  # [B,H,W]

        # ---- Streaming save: 각 rank가 바로 저장 ----
        if self.stream_save and self._save_out_dir and self._save_color_dir:
            import numpy as np
            import cv2, json

            cm = classmap.detach().cpu().numpy()

            # paths 정규화
            if paths is None:
                paths_iter = [None] * cm.shape[0]
            elif isinstance(paths, (list, tuple)):
                paths_iter = list(paths)
            else:
                paths_iter = [paths] * cm.shape[0]

            for i in range(cm.shape[0]):
                base = f"sample_{self.global_rank}_{batch_idx:06d}_{i:02d}"
                p = paths_iter[i]
                if p is not None:
                    base = os.path.splitext(os.path.basename(p))[0]

                mask_np = cm[i]

                # (A) PNG 저장 (옵션)
                if self.stream_save_png:
                    h, w = mask_np.shape
                    color = np.zeros((h, w, 3), dtype=np.uint8)
                    for k, rgb in self._save_colors.items():
                        color[mask_np == k] = rgb
                    png_path = os.path.join(self._save_color_dir, f"{base}_pred.png")
                    cv2.imwrite(png_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

                # (B) JSON 저장 (옵션)
                if self.stream_save_json:
                    coords = {"image_name": f"{base}.png", "predicted_coords": {}}
                    for cls_id in range(self.out_classes):
                        ys, xs = np.where(mask_np == cls_id)
                        coords["predicted_coords"][f"class_{cls_id}"] = (
                            [] if ys.size == 0 else np.stack([ys, xs], axis=1).tolist()
                        )
                    json_path = os.path.join(self._save_out_dir, f"{base}.json")
                    with open(json_path, "w") as f:
                        json.dump(coords, f, separators=(",", ":"))

            return {"saved": cm.shape[0]}

        # ---- Non-streaming: 후단에서 모아 저장하려면 logits/paths 반환 ----
        return {"logits": logits, "paths": paths}

    # ============================
    # Optim
    # ============================
    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)

        def lr_lambda(step: int):
            if self.trainer is None or self.trainer.num_training_batches == 0:
                return 1.0
            if step < self.warmup_steps:
                return float(step + 1) / float(max(1, self.warmup_steps))
            total_steps = max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
            prog = (step - self.warmup_steps) / float(total_steps)
            prog = min(max(prog, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        warmup = LambdaLR(opt, lr_lambda=lr_lambda)
        cosine = CosineAnnealingLR(opt, T_max=max(1, self.max_epochs_cfg))

        return [opt], [
            {"scheduler": warmup, "interval": "step",  "name": "warmup"},
            {"scheduler": cosine, "interval": "epoch", "name": "cosine"},
        ]

    # ============================
    # Train / Valid
    # ============================
    def training_step(self, batch, batch_idx):
        x, y = batch
        if not self.hf.training:
            if self.enable_diag_print and batch_idx == 0:
                print("[DIAG] hf was eval at training_step; switching to train().")
            self.hf.train()

        logits = self.forward(x)

        if getattr(self._ce, "weight", None) is not None:
            if self._ce.weight is not None and self._ce.weight.device != logits.device:
                self._ce.weight = self._ce.weight.to(logits.device)

        loss_ce = self._ce(logits, y)
        loss = self.ce_weight * loss_ce

        if self.dice_weight > 0.0:
            loss_dice = _dice_loss(logits, y, ignore_index=self.ignore_index)
            loss = loss + self.dice_weight * loss_dice
        else:
            loss_dice = logits.new_zeros(())

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss, "loss_ce": loss_ce.detach(), "loss_dice": loss_dice.detach()}

    def on_validation_epoch_start(self) -> None:
        self.val_confmat = torch.zeros(
            self.out_classes, self.out_classes,
            dtype=torch.long, device=self.device
        )
        self.last_class_stats = None
        self._diag_seen_val_batches = 0

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        if self.enable_diag_print and self._diag_seen_val_batches < 2:
            preds_dbg = torch.argmax(logits, dim=1)
            up = torch.unique(preds_dbg).tolist()
            uy = torch.unique(y).tolist()
            print(f"[DIAG][valid] logits shape={tuple(logits.shape)} unique(preds)={up} unique(gts)={uy}")
            self._diag_seen_val_batches += 1

        if getattr(self._ce, "weight", None) is not None:
            if self._ce.weight is not None and self._ce.weight.device != logits.device:
                self._ce.weight = self._ce.weight.to(logits.device)
        loss_ce = self._ce(logits, y)
        loss = self.ce_weight * loss_ce
        if self.dice_weight > 0.0:
            loss_dice = _dice_loss(logits, y, ignore_index=self.ignore_index)
            loss = loss + self.dice_weight * loss_dice

        preds = torch.argmax(logits, dim=1)
        m = (y != self.ignore_index)
        if m.any():
            self.val_iou_micro.update(preds[m], y[m])
            self.val_iou_none.update(preds[m], y[m])
            self.val_prec_none.update(preds[m], y[m])
            self.val_rec_none.update(preds[m], y[m])
            self.val_f1_none.update(preds[m], y[m])

            yv = y[m].view(-1)
            pv = preds[m].view(-1)
            valid = (yv >= 0) & (yv < self.out_classes) & (pv >= 0) & (pv < self.out_classes)
            if valid.any():
                yv = yv[valid]
                pv = pv[valid]
                cm = torch.bincount(
                    (yv * self.out_classes + pv),
                    minlength=self.out_classes * self.out_classes,
                ).reshape(self.out_classes, self.out_classes)
                if cm.device != self.val_confmat.device:
                    cm = cm.to(self.val_confmat.device)
                self.val_confmat += cm

        self.log("valid_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"valid_loss": loss}

    def on_validation_epoch_end(self) -> None:
        if dist.is_available() and dist.is_initialized():
            if not self.val_confmat.is_cuda:
                self.val_confmat = self.val_confmat.to(self.device)
            dist.all_reduce(self.val_confmat, op=dist.ReduceOp.SUM)

        cm = self.val_confmat.to(torch.float32)

        try:
            iou_micro = self.val_iou_micro.compute().item()
        except Exception:
            iou_micro = 0.0

        try:
            iou_none  = self.val_iou_none.compute()
            prec_none = self.val_prec_none.compute()
            rec_none  = self.val_rec_none.compute()
            f1_none   = self.val_f1_none.compute()
        except Exception:
            device = self.device
            iou_none  = torch.zeros(self.out_classes, device=device)
            prec_none = torch.zeros(self.out_classes, device=device)
            rec_none  = torch.zeros(self.out_classes, device=device)
            f1_none   = torch.zeros(self.out_classes, device=device)

        if self.out_classes > 1:
            fg_idx = torch.arange(1, self.out_classes, device=iou_none.device)
            macro_fg_iou  = iou_none[fg_idx].mean().item() if fg_idx.numel() > 0 else 0.0
            macro_fg_prec = prec_none[fg_idx].mean().item() if fg_idx.numel() > 0 else 0.0
            macro_fg_rec  = rec_none[fg_idx].mean().item() if fg_idx.numel() > 0 else 0.0
            macro_fg_f1   = f1_none[fg_idx].mean().item() if fg_idx.numel() > 0 else 0.0
        else:
            macro_fg_iou = macro_fg_prec = macro_fg_rec = macro_fg_f1 = 0.0

        macro_all_iou = iou_none.mean().item() if iou_none.numel() > 0 else 0.0

        self.log("valid_micro_mIoU", iou_micro,       prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_fg_mIoU", macro_fg_iou,  prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_fg_precision", macro_fg_prec, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_fg_recall",    macro_fg_rec,  prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_fg_f1",        macro_fg_f1,   prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_all_mIoU",     macro_all_iou, prog_bar=False, on_epoch=True, sync_dist=True)

        cm = self.val_confmat.to(torch.float32)
        eps = 1e-9
        support = cm.sum(dim=1)
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        iou       = tp / (tp + fp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)

        stats: List[Dict[str, Any]] = []
        for c in range(self.out_classes):
            stats.append({
                "class_id":  int(c),
                "class_name": str(self.class_names[c] if c < len(self.class_names) else f"class_{c}"),
                "precision": float(precision[c].item()),
                "recall":    float(recall[c].item()),
                "iou":       float(iou[c].item()),
                "f1":        float(f1[c].item()),
                "support":   int(support[c].item()),
            })
        self.last_class_stats = stats

        for m in [self.val_iou_none, self.val_prec_none, self.val_rec_none, self.val_f1_none, self.val_iou_micro]:
            try:
                m.reset()
            except Exception:
                pass

# model_M_patched.py
from __future__ import annotations

# ---------- Standard library ----------
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


# ============================
# Utilities
# ============================
def _logit_bias_from_prior(p: float, eps: float = 1e-6) -> float:
    """
    memo: Convert prior probability p in (0,1) to logit bias = log(p/(1-p)).
    For uniform prior (=1/C) this is ~0; rare classes get negative bias.
    """
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    memo: Multi-class soft dice loss.
    logits: (B,C,H,W), target: (B,H,W) with {0..C-1, ignore}
    """
    num_classes = logits.shape[1]
    mask = (target != ignore_index)
    if not mask.any():
        # No valid pixel: return 0 to avoid NaNs
        return logits.new_zeros(())

    # one-hot target (ignore positions zeroed)
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
    memo: LightningModule wrapping HF Mask2Former for semantic segmentation.
    Design goals:
      - Always return per-pixel semantic logits (B,C,H,W) regardless of HF version.
      - If HF does not expose pixel logits, synthesize them from query-level outputs.
      - Train with CE (+ optional Dice). Validation uses torchmetrics and a confusion matrix.
      - Expose per-class stats via self.last_class_stats for CSV callbacks.
    """

    def __init__(
        self,
        arch: str = "Mask2Former",
        encoder_name: str = "facebook/mask2former-swin-small-ade-semantic",
        in_channels: int = 3,      # kept for signature compatibility
        out_classes: int = 4,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,

        # loss weights
        ce_weight: float = 1.0,
        dice_weight: float = 0.0,

        # optimizer / scheduler
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,   # per-step warmup
        max_epochs: int = 50,

        # CE class weights (optional)
        ce_class_weights: Optional[List[float]] = None,

        # optional class head bias priors (sum~1, len=C)
        head_priors: Optional[List[float]] = None,

        # diagnostics
        enable_diag_print: bool = True,
    ):
        super().__init__()
        # Avoid saving large tensors in hparams
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
        # We already handle resize/normalize in our data pipeline
        # (and for old processors, avoid unsupported init kwargs)
        try:
            self.processor = Mask2FormerImageProcessor(
                do_resize=False, do_rescale=False, do_normalize=False
            )
        except TypeError:
            # very old versions may not accept these flags; fall back to defaults
            self.processor = Mask2FormerImageProcessor()

        # Make sure init state is train() (some checkpoints default to eval)
        self.hf.train()

        # Optional: initialize class head bias using priors
        try:
            cls_head = getattr(self.hf, "class_predictor", None)
            if isinstance(cls_head, nn.Linear) and head_priors and len(head_priors) == self.out_classes:
                with torch.no_grad():
                    new_bias = torch.tensor([_logit_bias_from_prior(p) for p in head_priors], dtype=cls_head.bias.dtype)
                    cls_head.bias.copy_(new_bias)
        except Exception:
            # head naming/layout can vary between versions; ignore failures
            pass

        # ---------- Losses ----------
        if ce_class_weights is not None:
            ce_w = torch.tensor(ce_class_weights, dtype=torch.float32)
        else:
            ce_w = None
        self._ce = nn.CrossEntropyLoss(weight=ce_w, ignore_index=self.ignore_index)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

        # ---------- Metrics ----------
        # (We mask ignore_index manually)
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

    # ============================
    # Low-level helpers
    # ============================
    def _extract_semantic_logits(self, out: Any) -> Optional[torch.Tensor]:
        """
        Try to extract per-pixel semantic logits from HF outputs across versions:
          - out.logits
          - out.semantic_logits
          - dict["logits"] / dict["semantic_logits"]
        Returns None if not found.
        """
        if out is None:
            return None
        # attribute access
        logits = getattr(out, "logits", None)
        if logits is None:
            logits = getattr(out, "semantic_logits", None)
        # dict-style access
        if logits is None and isinstance(out, dict):
            logits = out.get("logits", None) or out.get("semantic_logits", None)
        return logits

    def _semantic_logits_from_queries(
        self,
        out: Any,
        img_hw: Optional[Tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Fallback builder: use query-level outputs to synthesize per-pixel semantic logits.
        Expected (old HF API names may vary):
          - class_queries_logits or class_logits: (B, Q, C)
          - masks_queries_logits or masks_logits: (B, Q, h, w)
        Returns logits (B, C, H, W) if possible; otherwise None.
        """
        # Get class query logits
        cls = getattr(out, "class_queries_logits", None)
        if cls is None and isinstance(out, dict):
            cls = out.get("class_queries_logits", None)
        if cls is None:
            cls = getattr(out, "class_logits", None) or (out.get("class_logits", None) if isinstance(out, dict) else None)

        # Get mask query logits
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

        # Broadcast sum: (B,Q,C,1,1) + (B,Q,1,h,w) -> (B,Q,C,h,w)
        cls_b = cls.unsqueeze(-1).unsqueeze(-1)      # (B,Q,C,1,1)
        msk_b = msk.unsqueeze(2)                     # (B,Q,1,h,w)
        combined = cls_b + msk_b                     # (B,Q,C,h,w)

        # Per-class max over queries -> (B,C,h,w)
        sem_logits = combined.max(dim=1).values

        # Upsample to requested size if needed
        if img_hw is not None:
            H, W = img_hw
            if sem_logits.shape[-2:] != (H, W):
                sem_logits = torch.nn.functional.interpolate(
                    sem_logits, size=(H, W), mode="bilinear", align_corners=False
                )
        return sem_logits
    
    # ============================
    # Utility: force logits to have exactly self.out_classes channels
    # ============================
    def _align_logits_channels(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Ensure logits channel dimension == self.out_classes.
        - If C > out_classes: drop extras
        - If C < out_classes: pad with large negative so argmax never picks
        """
        B, C, H, W = logits.shape
        if C == self.out_classes:
            return logits
        if C > self.out_classes:
            return logits[:, : self.out_classes, :, :]
        pad_ch = self.out_classes - C
        pad = logits.new_full((B, pad_ch, H, W), fill_value=-1e4)
        return torch.cat([logits, pad], dim=1)
    

    # ============================
    # Forward
    # ============================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return per-pixel semantic logits (B,C,H,W), version-agnostic.
        Strategy:
          1) Try to get logits/semantic_logits directly.
          2) If missing, try again in eval() (some older checkpoints behave differently).
          3) If still missing, synthesize logits from query-level class/mask outputs.
        """
        out = self.hf(pixel_values=x)

        # 1) Newer or compatible versions
        logits = self._extract_semantic_logits(out)

        # 2) Try eval() path once (some older versions only expose in eval)
        if logits is None:
            was_training = self.hf.training
            # grad 유지 여부를 학습/평가 모드에 맞춰 자동으로 설정
            with torch.set_grad_enabled(self.training):
                self.hf.eval()
                out_eval = self.hf(pixel_values=x)  # no torch.no_grad()!
                if was_training:
                    self.hf.train()
            logits = self._extract_semantic_logits(out_eval)
            if logits is None:
                out = out_eval  # fallback에서 사용할 출력 (학습 중이면 grad 유지됨)

        # 3) Fallback: synthesize from queries
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
    
    # ============================
    # Optim
    # ============================
    def configure_optimizers(self):
        """
        AdamW + per-step warmup (LambdaLR) + per-epoch cosine (CosineAnnealingLR).
        """
        opt = AdamW(self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)

        def lr_lambda(step: int):
            # Before Trainer has correct counters, keep LR scale=1.0
            if self.trainer is None or self.trainer.num_training_batches == 0:
                return 1.0
            # Warmup (per-step)
            if step < self.warmup_steps:
                return float(step + 1) / float(max(1, self.warmup_steps))
            # Cosine (after warmup) based on estimated total steps
            total_steps = max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
            prog = (step - self.warmup_steps) / float(total_steps)
            prog = min(max(prog, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        warmup = LambdaLR(opt, lr_lambda=lr_lambda)                        # interval="step"
        cosine = CosineAnnealingLR(opt, T_max=max(1, self.max_epochs_cfg)) # interval="epoch"

        return [opt], [
            {"scheduler": warmup, "interval": "step",  "name": "warmup"},
            {"scheduler": cosine, "interval": "epoch", "name": "cosine"},
        ]

    # ============================
    # Train
    # ============================
    def training_step(self, batch, batch_idx):
        """
        We always compute our own CE (+optional Dice) on synthesized logits.
        This avoids depending on HF loss APIs that change across versions.
        """
        x, y = batch  # y in {0..C-1, ignore_index}
        # Ensure HF is in train mode
        if not self.hf.training:
            if self.enable_diag_print and batch_idx == 0:
                print("[DIAG] hf was eval at training_step; switching to train().")
            self.hf.train()

        logits = self.forward(x)  # (B,C,H,W)

        # Ensure CE weight tensor is on the same device
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

    # ============================
    # Validation
    # ============================
    def on_validation_epoch_start(self) -> None:
        # 기존 zero_() 대신, 현재 디바이스에 맞춰 새로 생성
        self.val_confmat = torch.zeros(
            self.out_classes, self.out_classes,
            dtype=torch.long, device=self.device
        )
        self.last_class_stats = None
        self._diag_seen_val_batches = 0

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch  # x=(B,3,H,W), y=(B,H,W)
        logits = self.forward(x)  # (B,C,H,W)

        # Diagnostics on the first few batches
        if self.enable_diag_print and self._diag_seen_val_batches < 2:
            preds_dbg = torch.argmax(logits, dim=1)
            up = torch.unique(preds_dbg).tolist()
            uy = torch.unique(y).tolist()
            print(f"[DIAG][valid] logits shape={tuple(logits.shape)} unique(preds)={up} unique(gts)={uy}")
            self._diag_seen_val_batches += 1

        # Loss (for logging)
        if getattr(self._ce, "weight", None) is not None:
            if self._ce.weight is not None and self._ce.weight.device != logits.device:
                self._ce.weight = self._ce.weight.to(logits.device)
        loss_ce = self._ce(logits, y)
        loss = self.ce_weight * loss_ce
        if self.dice_weight > 0.0:
            loss_dice = _dice_loss(logits, y, ignore_index=self.ignore_index)
            loss = loss + self.dice_weight * loss_dice

        # Metrics update (mask ignore)
        preds = torch.argmax(logits, dim=1)  # (B,H,W)
        m = (y != self.ignore_index)
        if m.any():
            self.val_iou_micro.update(preds[m], y[m])
            self.val_iou_none.update(preds[m], y[m])
            self.val_prec_none.update(preds[m], y[m])
            self.val_rec_none.update(preds[m], y[m])
            self.val_f1_none.update(preds[m], y[m])

            # Confusion matrix accumulation [C x C]
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

                # val_confmat 디바이스에 맞춰 cm만 이동시킨 다음 누적
                if cm.device != self.val_confmat.device:
                    cm = cm.to(self.val_confmat.device)
                self.val_confmat += cm

        self.log("valid_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"valid_loss": loss}

    def on_validation_epoch_end(self) -> None:
        # DDP: reduce confusion matrix across ranks
        if dist.is_available() and dist.is_initialized():
            # NCCL은 GPU 텐서만 지원 → 안전하게 보장
            if not self.val_confmat.is_cuda:
                self.val_confmat = self.val_confmat.to(self.device)
            dist.all_reduce(self.val_confmat, op=dist.ReduceOp.SUM)

        # 이후 통계 계산은 float 복사본으로
        cm = self.val_confmat.to(torch.float32)

        # Gather metrics (torchmetrics already sync when sync_dist=True was used)
        try:
            iou_micro = self.val_iou_micro.compute().item()
        except Exception:
            iou_micro = 0.0

        try:
            iou_none  = self.val_iou_none.compute()   # tensor [C]
            prec_none = self.val_prec_none.compute()
            rec_none  = self.val_rec_none.compute()
            f1_none   = self.val_f1_none.compute()
        except Exception:
            device = self.device
            iou_none  = torch.zeros(self.out_classes, device=device)
            prec_none = torch.zeros(self.out_classes, device=device)
            rec_none  = torch.zeros(self.out_classes, device=device)
            f1_none   = torch.zeros(self.out_classes, device=device)

        # Foreground macro from classes 1..C-1
        if self.out_classes > 1:
            fg_idx = torch.arange(1, self.out_classes, device=iou_none.device)
            macro_fg_iou  = iou_none[fg_idx].mean().item() if fg_idx.numel() > 0 else 0.0
            macro_fg_prec = prec_none[fg_idx].mean().item() if fg_idx.numel() > 0 else 0.0
            macro_fg_rec  = rec_none[fg_idx].mean().item() if fg_idx.numel() > 0 else 0.0
            macro_fg_f1   = f1_none[fg_idx].mean().item() if fg_idx.numel() > 0 else 0.0
        else:
            macro_fg_iou = macro_fg_prec = macro_fg_rec = macro_fg_f1 = 0.0

        macro_all_iou = iou_none.mean().item() if iou_none.numel() > 0 else 0.0

        # Log epoch metrics (align with your CSV logger)
        self.log("valid_micro_mIoU", iou_micro,       prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_fg_mIoU", macro_fg_iou,  prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_fg_precision", macro_fg_prec, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_fg_recall",    macro_fg_rec,  prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_fg_f1",        macro_fg_f1,   prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("valid_macro_all_mIoU",     macro_all_iou, prog_bar=False, on_epoch=True, sync_dist=True)

        # Convert confusion matrix → per-class stats
        cm = self.val_confmat.to(torch.float32)  # [C,C]
        eps = 1e-9
        support = cm.sum(dim=1)                 # GT row-sum
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp                 # pred col-sum - TP
        fn = cm.sum(dim=1) - tp                 # gt row-sum - TP

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

        # Reset torchmetrics for next epoch
        for m in [self.val_iou_none, self.val_prec_none, self.val_rec_none, self.val_f1_none, self.val_iou_micro]:
            try:
                m.reset()
            except Exception:
                pass
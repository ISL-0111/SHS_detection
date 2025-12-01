# callbacks_csv.py
# Rank-0 only CSV logging for:
#  - train_epoch.csv: epoch, global_step, lr, train_loss
#  - valid_epoch.csv: epoch, global_step, lr, valid_* metrics (avg_PV_iou, macro/micro, valid_loss)
#  - metrics_epoch_per_class.csv: per-class precision/recall/iou/f1/support each epoch
#
# Requirements on the LightningModule (Mask2FormerLit already complies):
#  - exposes: number_of_classes (int), class_names (List[str], optional)
#  - sets:    last_class_stats (List[dict]) in on_validation_epoch_end
#
# Safe with DDP: only rank-0 writes.

from __future__ import annotations
import os
import csv
from typing import List, Dict, Any, Optional
import torch
import pytorch_lightning as pl


def _is_rank0(trainer) -> bool:
    return getattr(trainer, "is_global_zero", True)


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        v = v.detach()
        if v.is_cuda:
            v = v.cpu()
        try:
            return float(v.item())
        except Exception:
            return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _get_lr(trainer) -> Optional[float]:
    try:
        opt = trainer.optimizers[0]
        if opt.param_groups and "lr" in opt.param_groups[0]:
            return float(opt.param_groups[0]["lr"])
    except Exception:
        pass
    return None


class SegmentationCSVLogger(pl.Callback):
    """
    Writes three CSVs under run_dir:
      - train_epoch.csv
      - valid_epoch.csv
      - metrics_epoch_per_class.csv

    Expects the LightningModule to expose:
      - number_of_classes
      - class_names (optional; otherwise class_{i})
      - last_class_stats: list of dicts set at on_validation_epoch_end, e.g.
          [{"class_id":0,"class_name":"background","precision":...,"recall":...,"iou":...,"f1":...,"support":...}, ...]
    """
    def __init__(self, out_dir: str):
        super().__init__()
        self.run_dir = os.path.abspath(out_dir)
        os.makedirs(self.run_dir, exist_ok=True)

        self.train_csv = os.path.join(self.run_dir, "train_epoch.csv")
        self.valid_csv = os.path.join(self.run_dir, "valid_epoch.csv")
        self.perclass_csv = os.path.join(self.run_dir, "metrics_epoch_per_class.csv")

        self._train_header = os.path.exists(self.train_csv)
        self._valid_header = os.path.exists(self.valid_csv)
        self._pc_header = os.path.exists(self.perclass_csv)

    # ---------------- Train ----------------
    def on_train_epoch_end(self, trainer, pl_module):
        if not _is_rank0(trainer):
            return

        row = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "lr": _get_lr(trainer),
            "train_loss": _to_float(trainer.callback_metrics.get("train_loss")),
        }
        cols = ["epoch", "global_step", "lr", "train_loss"]

        if not self._train_header:
            with open(self.train_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerow(row)
            self._train_header = True
        else:
            with open(self.train_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writerow(row)

    # ---------------- Valid (epoch-level summary) ----------------
    def on_validation_epoch_end(self, trainer, pl_module):
        if not _is_rank0(trainer):
            return

        cm = trainer.callback_metrics
        row = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "lr": _get_lr(trainer),
            "valid_loss": _to_float(cm.get("valid_loss")),
            "valid_avg_PV_iou": _to_float(cm.get("valid_avg_PV_iou")),
            "valid_macro_fg_mIoU": _to_float(cm.get("valid_macro_fg_mIoU")),
            "valid_macro_fg_precision": _to_float(cm.get("valid_macro_fg_precision")),
            "valid_macro_fg_recall": _to_float(cm.get("valid_macro_fg_recall")),
            "valid_macro_fg_f1": _to_float(cm.get("valid_macro_fg_f1")),
            "valid_macro_all_mIoU": _to_float(cm.get("valid_macro_all_mIoU")),
            "valid_micro_mIoU": _to_float(cm.get("valid_micro_mIoU")),
        }
        vcols = list(row.keys())

        if not self._valid_header:
            with open(self.valid_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=vcols)
                w.writeheader()
                w.writerow(row)
            self._valid_header = True
        else:
            with open(self.valid_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=vcols)
                w.writerow(row)

    # ---------------- Valid (per-class table) ----------------
    # IMPORTANT: run after module.on_validation_epoch_end so last_class_stats is populated.
    def on_validation_end(self, trainer, pl_module):
        if not _is_rank0(trainer):
            return

        # Read what the module computed in its on_validation_epoch_end
        class_stats = getattr(pl_module, "last_class_stats", None)
        n_classes = int(getattr(pl_module, "number_of_classes", 0))
        class_names = getattr(pl_module, "class_names", None)
        if not class_names or len(class_names) != n_classes:
            class_names = [f"class_{i}" for i in range(n_classes)]

        pc_cols = ["epoch", "class_id", "class_name", "precision", "recall", "iou", "f1", "support"]
        if not self._pc_header:
            with open(self.perclass_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=pc_cols)
                w.writeheader()
            self._pc_header = True

        # If the model didn't produce stats, nothing to write this epoch
        if not (isinstance(class_stats, list) and len(class_stats) > 0):
            return

        epoch_idx = int(trainer.current_epoch)
        with open(self.perclass_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pc_cols)
            for i, st in enumerate(class_stats):
                cid = int(st.get("class_id", i))
                cname = st.get("class_name", (class_names[cid] if cid < len(class_names) else f"class_{cid}"))
                # normalize values to plain python types
                prec = float(st.get("precision", 0.0))
                rec  = float(st.get("recall", 0.0))
                iou  = float(st.get("iou", 0.0))
                f1   = float(st.get("f1", 0.0))
                sup  = int(st.get("support", 0))
                w.writerow({
                    "epoch": epoch_idx,
                    "class_id": cid,
                    "class_name": str(cname),
                    "precision": prec,
                    "recall": rec,
                    "iou": iou,
                    "f1": f1,
                    "support": sup,
                })
import os
import csv
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from datetime import datetime

class PVModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.class_weights = torch.tensor([0.5, 2.0, 2.0, 2.0], dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.train_metrics = {}
        self.valid_metrics = {}

        # Generate a unique CSV file per run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"metrics_log_{timestamp}.csv"
        self.first_log = True

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)

    def shared_step(self, batch, stage):
        image, mask = batch
        mask = mask.long()
        logits_mask = self.forward(image).contiguous()

        dice_loss = self.dice_loss(logits_mask, mask)
        ce_loss = self.ce_loss(logits_mask, mask)
        loss = 0.7 * dice_loss + 0.3 * ce_loss

        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        avg_dice_loss = torch.stack([x["dice_loss"] for x in outputs]).mean().item()
        avg_ce_loss = torch.stack([x["ce_loss"] for x in outputs]).mean().item()

        metrics = {
            f"{stage}_per_image_iou": per_image_iou.item(),
            f"{stage}_dataset_iou": dataset_iou.item(),
            f"{stage}_f1_score": f1_score.item(),
            f"{stage}_precision": precision.item(),
            f"{stage}_recall": recall.item(),
            f"{stage}_loss": avg_loss,
            f"{stage}_dice_loss": avg_dice_loss,
            f"{stage}_ce_loss": avg_ce_loss,
        }

        class_names = ['background', 'PV_normal', 'PV_heater', 'PV_pool']
        ious = []
        for i, class_name in enumerate(class_names):
            if i < len(per_class_iou):
                iou_tensor = per_class_iou[i]
                if torch.is_tensor(iou_tensor):
                    iou_value = iou_tensor.float().mean().item()
                else:
                    iou_value = float(iou_tensor)
                metrics[f"{stage}_iou_{class_name}"] = iou_value
                if class_name in ['PV_normal', 'PV_heater', 'PV_pool']:
                    ious.append(iou_value)

        if stage == "valid":
            metrics["valid_avg_PV_iou"] = np.mean(ious)

        self.log_dict(metrics, prog_bar=True)

        # Save metrics to internal dict
        if stage == "train":
            self.train_metrics = metrics
        elif stage == "valid":
            self.valid_metrics = metrics
            # Ensure both train and valid metrics exist before writing
            if self.train_metrics:
                self.write_csv()
                self.train_metrics = {}
                self.valid_metrics = {}


    def write_csv(self):
        epoch = self.current_epoch
        step = self.global_step

        row = {
            "epoch": epoch,
            "step": step,
        }
        row.update(self.train_metrics)
        row.update(self.valid_metrics)

        fieldnames = list(row.keys())

        write_header = self.first_log and not os.path.exists(self.log_file)
        with open(self.log_file, "a", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self.first_log = False
            writer.writerow(row)

    def training_step(self, batch, batch_idx):
        result = self.shared_step(batch, "train")
        self.training_step_outputs.append(result)
        self.log("train_loss", result["loss"], prog_bar=True)
        self.log("train_dice_loss", result["dice_loss"])
        self.log("train_ce_loss", result["ce_loss"])
        return result

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        result = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(result)
        self.log("valid_loss", result["loss"], prog_bar=True)
        return result

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        result = self.shared_step(batch, "test")
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

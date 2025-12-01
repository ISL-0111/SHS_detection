#!/usr/bin/env python3
import os, re, sys
import cv2
import numpy as np
import pandas as pd

GT_MASK_DIR   = "/shared/data/climateplus2025/Shawn_Sep.04_Attention_U-Net_Sep.07/output5k_stratified_training_db/test/masks"
PRED_MASK_DIR = "/shared/data/climateplus2025/Shawn_Sep.04_New_Mask2Former_Sep.08/predicted_masks_Mask2Former_m2f_swinS"

OUTPUT_DIR    = "/shared/data/climateplus2025/Shawn_Sep.04_New_Mask2Former_Sep.08/eval"

# Class
CLASSES = ["background", "PV_normal", "PV_heater", "PV_pool"]
NUM_CLASSES = len(CLASSES)

# PNG → Class Index (cv2 = BGR)
COLOR_BGR_FOR_CLASS = {
    0: (  0,   0,   0),  # background is Black
    1: (  0, 255,   0),  # PV_normal (RGB 0,255,0) is Green
    2: (255,   0,   0),  # PV_heater  (RGB 0,0,255 -> BGR 255,0,0) is Blue
    3: (  0,   0, 255),  # PV_pool    (RGB 255,0,0 -> BGR 0,0,255) is Red
}

def base_key(fname: str) -> str:
    """File Name Parshing"""
    name = os.path.splitext(fname)[0]
    name = re.sub(r'^(i_|m_)', '', name)
    name = re.sub(r'_pred$', '', name)
    return name

def color_png_to_labels_bgr(img: np.ndarray, num_classes: int) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # remove alpha channel
    H, W, _ = img.shape
    lab = np.full((H, W), fill_value=-1, dtype=np.int32)
    for c, bgr in COLOR_BGR_FOR_CLASS.items():
        mask = (img[:,:,0]==bgr[0]) & (img[:,:,1]==bgr[1]) & (img[:,:,2]==bgr[2])
        lab[mask] = c
    lab[lab < 0] = 0
    return lab

def gray_png_to_labels(img_gray: np.ndarray, num_classes: int) -> np.ndarray:
    lab = img_gray.astype(np.int32)
    return np.clip(lab, 0, num_classes-1)

def main(gt_dir=GT_MASK_DIR, pred_dir=PRED_MASK_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gt_files   = {base_key(f): f for f in os.listdir(gt_dir)   if f.lower().endswith(".png")}
    pred_files = {base_key(f): f for f in os.listdir(pred_dir) if f.lower().endswith(".png")}
    keys = sorted(set(gt_files.keys()) & set(pred_files.keys()))

    if not keys:
        print("[ERROR] No matched pairs."); sys.exit(1)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    # per-file records
    file_records = []

    for k in keys:
        gt_path = os.path.join(gt_dir,   gt_files[k])
        pr_path = os.path.join(pred_dir, pred_files[k])

        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pr_img = cv2.imread(pr_path, cv2.IMREAD_UNCHANGED)
        if gt_img is None or pr_img is None:
            continue

        # label map
        if pr_img.ndim >= 3:
            pred_lab = color_png_to_labels_bgr(pr_img, NUM_CLASSES)
        else:
            pred_lab = gray_png_to_labels(pr_img, NUM_CLASSES)
        gt_lab = gray_png_to_labels(gt_img, NUM_CLASSES)

        # size adjustment
        if pred_lab.shape != gt_lab.shape:
            pred_lab = cv2.resize(pred_lab, (gt_lab.shape[1], gt_lab.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # valid pixels
        y_true = gt_lab.reshape(-1)
        y_pred = pred_lab.reshape(-1)
        valid  = (y_true>=0)&(y_true<NUM_CLASSES)&(y_pred>=0)&(y_pred<NUM_CLASSES)
        y_true, y_pred = y_true[valid], y_pred[valid]

        # CM updates
        cm += np.bincount(y_true*NUM_CLASSES + y_pred,
                          minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

        # === per-file metrics (micro IoU/accuracy) ===
        # CM by files
        cm_file = np.bincount(y_true*NUM_CLASSES + y_pred,
                              minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
        tp_f = np.diag(cm_file).astype(float)
        fp_f = cm_file.sum(0) - tp_f
        fn_f = cm_file.sum(1) - tp_f
        micro_iou_file = float(np.sum(tp_f) / (np.sum(tp_f) + np.sum(fp_f) + np.sum(fn_f) + 1e-9))
        acc_file = float(np.sum(tp_f) / (cm_file.sum() + 1e-9))

        file_records.append({
            "file": gt_files[k],
            "micro_mIoU": micro_iou_file,
            "accuracy": acc_file,
            "pixels": int(cm_file.sum())
        })

    eps=1e-9
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0)-tp
    fn = cm.sum(1)-tp
    support = cm.sum(1)

    precision = tp/(tp+fp+eps)
    recall    = tp/(tp+fn+eps)
    iou       = tp/(tp+fp+fn+eps)
    f1        = 2*precision*recall/(precision+recall+eps)

    macro_all_iou  = float(np.mean(iou))
    macro_fg_iou   = float(np.mean(iou[1:])) if NUM_CLASSES>1 else macro_all_iou
    micro_iou      = float(np.sum(tp)/(np.sum(tp)+np.sum(fp)+np.sum(fn)+eps))

    print("\n=== TEST METRICS (from PNG) ===")
    print(f"micro mIoU  : {micro_iou:.4f}")
    print(f"macro(all)  : {macro_all_iou:.4f}")
    print(f"macro(FG)   : {macro_fg_iou:.4f}\n")

    print("per-class:")
    for c,name in enumerate(CLASSES):
        print(f"  [{c}] {name:10s} | IoU {iou[c]:.4f} | P {precision[c]:.4f} | R {recall[c]:.4f} "
              f"| F1 {f1[c]:.4f} | support {int(support[c])}")

    out_class_csv = os.path.join(OUTPUT_DIR, "test_metrics_per_class_from_png.csv")
    pd.DataFrame([{
        "class_id": c, "class_name": name,
        "precision": float(precision[c]), "recall": float(recall[c]),
        "iou": float(iou[c]), "f1": float(f1[c]), "support": int(support[c])
    } for c,name in enumerate(CLASSES)]).to_csv(out_class_csv, index=False)

    out_file_csv = os.path.join(OUTPUT_DIR, "test_metrics_per_file_from_png.csv")
    pd.DataFrame(file_records).to_csv(out_file_csv, index=False)

    summary = {
        "micro_mIoU": float(micro_iou),
        "macro_all_mIoU": float(macro_all_iou),
        "macro_fg_mIoU": float(macro_fg_iou),
    }
    summary_csv = os.path.join(OUTPUT_DIR, "test_metrics_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    summary_txt = os.path.join(OUTPUT_DIR, "test_metrics_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("=== TEST METRICS (from PNG) ===\n")
        f.write(f"micro mIoU  : {micro_iou:.4f}\n")
        f.write(f"macro(all)  : {macro_all_iou:.4f}\n")
        f.write(f"macro(FG)   : {macro_fg_iou:.4f}\n\n")
        f.write("per-class:\n")
        for c, name in enumerate(CLASSES):
            f.write(
                f"  [{c}] {name:10s} | IoU {iou[c]:.4f} | "
                f"P {precision[c]:.4f} | R {recall[c]:.4f} | "
                f"F1 {f1[c]:.4f} | support {int(support[c])}\n"
            )

    print("\n Saved:")
    print(f" - {out_class_csv}")
    print(f" - {out_file_csv}")
    print(f" - {summary_csv}")
    print(f" - {summary_txt}")

if __name__ == "__main__":
    main()
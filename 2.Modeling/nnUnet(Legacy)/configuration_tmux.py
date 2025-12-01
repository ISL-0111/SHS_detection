import os
import subprocess
from datetime import datetime
import time
import json
import sys
import glob
import threading

# ==========================================================
# CONFIGURATION
# ==========================================================
BASE_DIR = "/shared/data/climateplus2025/Shawn_Nov18_NNUnet_1024"
DATASET_ID = "901"
CONFIG = "2d"
GPUS = [0, 1, 2, 3, 4]
ENABLE_TTA = True


# ==========================================================
# ENV SETUP
# ==========================================================
os.environ["nnUNet_raw"] = BASE_DIR
os.environ["nnUNet_preprocessed"] = os.path.join(BASE_DIR, "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(BASE_DIR, "nnUNet_results")

os.makedirs(os.environ["nnUNet_preprocessed"], exist_ok=True)
os.makedirs(os.environ["nnUNet_results"], exist_ok=True)

print("\n======================================================================")
print("Standard nnU-Net 2D Training Pipeline")
print("======================================================================")
print(f"Raw Data Dir: {os.environ['nnUNet_raw']}")
print(f"Preprocessed Dir: {os.environ['nnUNet_preprocessed']}")
print(f"Results Dir: {os.environ['nnUNet_results']}")
print(f"GPUs Available: {GPUS}")
print(f"Configuration: {CONFIG}")
print("======================================================================\n")


# ==========================================================
# STEP 1 – PLAN & PREPROCESS
# ==========================================================
print("\n======================================================================")
print("STEP 1: Planning and Preprocessing")
print("======================================================================")

try:
    subprocess.run([
        "nnUNetv2_plan_and_preprocess",
        "-d", DATASET_ID,
        "--verify_dataset_integrity"
    ], check=True)
    print("\nPreprocessing complete.")
except subprocess.CalledProcessError as e:
    print(f"\nPreprocessing failed: {e}")
    sys.exit(1)


# ==========================================================
# STEP 2 – TRAINING (with REAL-TIME LOG STREAM)
# ==========================================================
print("\n======================================================================")
print("STEP 2: Training 5 Folds in Parallel (with live logs)")
print("======================================================================")

processes = []
log_files = []

def tail_log(filepath, fold):
    """Improved live log streamer for nnU-Net."""
    print(f"[Fold {fold}] Waiting for log file...")

    # Wait until file exists
    while not os.path.exists(filepath):
        time.sleep(0.5)

    print(f"[Fold {fold}] Live logging started.")

    with open(filepath, "r") as f:
        # DO NOT SEEK TO END; START FROM BEGINNING
        while True:
            where = f.tell()
            line = f.readline()

            if not line:
                time.sleep(0.5)
                f.seek(where)
            else:
                # print everything in real time
                print(f"[Fold {fold}] {line.strip()}")

# def tail_log(filepath, fold):
#     """Stream log lines as soon as the file exists."""
#     print(f"[Fold {fold}] Waiting for log file...")
#     while not os.path.exists(filepath):
#         time.sleep(1)

#     print(f"[Fold {fold}] Live logging started.")
#     with open(filepath, "r") as f:
#         f.seek(0, os.SEEK_END)
#         while True:
#             line = f.readline()
#             if line:
#                 if any(key in line for key in [
#                     "Epoch", "train_loss", "val_loss", "Pseudo dice", "learning rate"
#                 ]):
#                     print(f"[Fold {fold}] {line.strip()}")
#             else:
#                 time.sleep(1)


for fold, gpu in enumerate(GPUS):
    logfile = os.path.join(BASE_DIR, f"fold_{fold}_standard.log")
    log_files.append(logfile)

    print(f"\nLaunching fold {fold} on GPU {gpu}...")

    cmd = [
        "bash", "-c",
        f"CUDA_VISIBLE_DEVICES={gpu} stdbuf -oL -eL nnUNetv2_train "
        f"{DATASET_ID} {CONFIG} {fold} >> {logfile} 2>&1"
    ]

    p = subprocess.Popen(cmd)
    processes.append((p, fold, gpu))

    # Start background log streamer
    t = threading.Thread(target=tail_log, args=(logfile, fold), daemon=True)
    t.start()

    time.sleep(30)  # reduce stress on disk IO

print("\nTraining launched for all folds.\n")

# Wait for all folds
failed_folds = []
for p, fold, gpu in processes:
    p.wait()
    if p.returncode != 0:
        failed_folds.append(f"Fold {fold} (GPU {gpu})")
        print(f"[Fold {fold}] FAILED")
    else:
        print(f"[Fold {fold}] Training completed successfully.")

if failed_folds:
    print(f"\nTraining failed for: {', '.join(failed_folds)}")
    sys.exit(1)

print("\nAll folds finished successfully.")


# ==========================================================
# STEP 3 – FIND BEST CONFIGURATION
# ==========================================================
print("\n======================================================================")
print("STEP 3: Finding Best Configuration")
print("======================================================================")

try:
    subprocess.run([
        "nnUNetv2_find_best_configuration",
        DATASET_ID,
        "-c", CONFIG
    ], check=True)
    print("\nBest configuration identified.")
except subprocess.CalledProcessError as e:
    print(f"\nWarning: find_best_configuration failed: {e}")


# ==========================================================
# STEP 4 – PREDICT
# ==========================================================
print("\n======================================================================")
print("STEP 4: Prediction on Test Set")
print("======================================================================")

dataset_folders = sorted([d for d in os.listdir(BASE_DIR) if d.startswith(f"Dataset{DATASET_ID}_")])
if not dataset_folders:
    print("Dataset folder not found.")
    sys.exit(1)

dataset_folder = os.path.join(BASE_DIR, dataset_folders[0])
test_images = os.path.join(dataset_folder, "imagesTs")

if not os.path.exists(test_images):
    print(f"Test images not found at: {test_images}")
    sys.exit(1)

pred_root = os.path.join(BASE_DIR, "predictions")
os.makedirs(pred_root, exist_ok=True)

existing_versions = [d for d in os.listdir(pred_root) if d.startswith("v")]
next_ver = len(existing_versions) + 1
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_dir = os.path.join(pred_root, f"v{next_ver}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

predict_cmd = [
    "nnUNetv2_predict",
    "-i", test_images,
    "-o", output_dir,
    "-d", DATASET_ID,
    "-c", CONFIG,
    "-f", "0", "1", "2", "3", "4"
]

if not ENABLE_TTA:
    predict_cmd.append("--disable_tta")

try:
    subprocess.run(predict_cmd, check=True)
    print(f"\nPredictions saved to: {output_dir}")
except subprocess.CalledProcessError as e:
    print(f"\nPrediction failed: {e}")
    sys.exit(1)


print("\n======================================================================")
print("STEP 5: Evaluation (Fixed for nnU-Net v2 — Dice & IoU)")
print("======================================================================")

results_base = os.environ["nnUNet_results"]
result_candidates = sorted(glob.glob(os.path.join(results_base, f"Dataset{DATASET_ID}_*")))

if not result_candidates:
    print("Results folder not found.")
    sys.exit(1)

dataset_result = result_candidates[0]
trainer_dir = os.path.join(dataset_result, "nnUNetTrainer__nnUNetPlans__2d")

if not os.path.exists(trainer_dir):
    print("Trainer folder missing:", trainer_dir)
    sys.exit(1)

print(f"Reading results from: {trainer_dir}\n")

# class_names[1], [2], [3](excluding background)
class_names = ["background", "PV_normal", "PV_heater", "PV_pool"]
num_classes = len(class_names)

fold_dice = []
class_dice_sum = [0.0] * (num_classes - 1)     # excluding background → length is 3
valid_folds = 0

for fold in range(5):
    summary_json = os.path.join(trainer_dir, f"fold_{fold}", "validation", "summary.json")
    if not os.path.exists(summary_json):
        print(f"Fold {fold}: summary.json missing")
        continue

    with open(summary_json) as f:
        summary = json.load(f)

    mean_dict = summary["mean"]     # {"1": {"Dice": ...}, "2": {...}, "3": {...}}

    dice_list = []
    for cls_idx in range(1, num_classes):      # 1~3
        cls_key = str(cls_idx)
        if cls_key in mean_dict:
            dice_list.append(mean_dict[cls_key]["Dice"])
        else:
            dice_list.append(0.0)

    # accumulate for cross-fold average
    for i, d in enumerate(dice_list):
        class_dice_sum[i] += d

    fold_mean = sum(dice_list) / len(dice_list)
    fold_dice.append(fold_mean)
    valid_folds += 1

    print(f"Fold {fold} Dice:")
    for i, d in enumerate(dice_list):
        print(f"  {class_names[i+1]}: {d:.4f}")
    print(f"  Mean: {fold_mean:.4f}\n")

# compute averages
class_mean_dice = [v / valid_folds for v in class_dice_sum]
overall_mean = sum(class_mean_dice) / len(class_mean_dice)

print("\n======================================================================")
print("FINAL RESULTS - DICE")
print("======================================================================")
for i, d in enumerate(class_mean_dice):
    print(f"{class_names[i+1]}: {d:.4f}")
print(f"\nOverall Mean Dice: {overall_mean:.4f}")

# -----------------------
# IoU = Dice / (2 - Dice)
# -----------------------

class_iou = [d / (2 - d) for d in class_mean_dice]
mean_iou = sum(class_iou) / len(class_iou)

print("\n======================================================================")
print("FINAL RESULTS - IOU")
print("======================================================================")
for i, iou in enumerate(class_iou):
    print(f"{class_names[i+1]}: {iou:.4f}")
print(f"\nOverall Mean IoU: {mean_iou:.4f}")
print("======================================================================")
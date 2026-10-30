"""
prepare_yolo_subset.py
Creates a focused subset of small 416x416 images for ultra-fast YOLO training.
"""

import os
import glob
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "PRL LOT")
SUBSET_DIR = os.path.join(PROJECT_ROOT, "data", "yolo_subset")

os.makedirs(os.path.join(SUBSET_DIR, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(SUBSET_DIR, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(SUBSET_DIR, "valid", "images"), exist_ok=True)
os.makedirs(os.path.join(SUBSET_DIR, "valid", "labels"), exist_ok=True)

# Copy 150 train images + labels
train_imgs = glob.glob(os.path.join(DATASET_ROOT, "train", "images", "*.jpg"))[:150]
for img in train_imgs:
    base = os.path.splitext(os.path.basename(img))[0]
    lbl = os.path.join(DATASET_ROOT, "train", "labels", base + ".txt")
    if os.path.exists(lbl):
        shutil.copy2(img, os.path.join(SUBSET_DIR, "train", "images", os.path.basename(img)))
        shutil.copy2(lbl, os.path.join(SUBSET_DIR, "train", "labels", base + ".txt"))

# Copy 40 valid images + labels
val_imgs = glob.glob(os.path.join(DATASET_ROOT, "valid", "images", "*.jpg"))[:40]
for img in val_imgs:
    base = os.path.splitext(os.path.basename(img))[0]
    lbl = os.path.join(DATASET_ROOT, "valid", "labels", base + ".txt")
    if os.path.exists(lbl):
        shutil.copy2(img, os.path.join(SUBSET_DIR, "valid", "images", os.path.basename(img)))
        shutil.copy2(lbl, os.path.join(SUBSET_DIR, "valid", "labels", base + ".txt"))

clean_subset_path = SUBSET_DIR.replace("\\", "/")
yaml_content = f"""path: {clean_subset_path}
train: train/images
val: valid/images

nc: 2
names:
  0: space-empty
  1: space-occupied
"""

with open(os.path.join(PROJECT_ROOT, "data", "yolo_small.yaml"), "w") as f:
    f.write(yaml_content)

print("[+] Prepared YOLO small subset: 150 train, 40 valid. YAML saved to data/yolo_small.yaml")

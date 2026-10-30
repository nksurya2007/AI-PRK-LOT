"""
prepare_dataset.py
Extracts individual parking slot patches (empty vs occupied) from PKLot YOLO-annotated images.
Organizes them into train/val/test folders for high-speed CNN training and evaluation.
Also generates a canonical slots_config.json for parking lot layout visualization.
"""

import os
import glob
import json
import random
from PIL import Image

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "PRL LOT")
PATCHES_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "patches")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "slots_config.json")
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")

PATCH_SIZE = (64, 64)

# Target number of patches per class per split for balanced fast training
LIMITS = {
    'train': 4000,   # 4000 empty + 4000 occupied = 8000 total
    'valid': 1000,   # 1000 empty + 1000 occupied = 2000 total
    'test': 1000     # 1000 empty + 1000 occupied = 2000 total
}

CLASS_MAP = {
    0: 'empty',
    1: 'occupied'
}

def extract_patches():
    print("=" * 60)
    print("Starting PKLot Slot Patch Extraction & Preprocessing...")
    print(f"Source: {DATASET_ROOT}")
    print(f"Destination: {PATCHES_ROOT}")
    print("=" * 60)

    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    canonical_saved = False

    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        lbl_dir = os.path.join(DATASET_ROOT, split, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"Warning: Directory for {split} not found. Skipping...")
            continue

        out_empty = os.path.join(PATCHES_ROOT, split, "empty")
        out_occ = os.path.join(PATCHES_ROOT, split, "occupied")
        os.makedirs(out_empty, exist_ok=True)
        os.makedirs(out_occ, exist_ok=True)

        images = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        random.seed(42)
        random.shuffle(images)

        counts = {'empty': 0, 'occupied': 0}
        target_limit = LIMITS[split]

        print(f"\nProcessing {split} split ({len(images)} images found)...")

        for img_idx, img_path in enumerate(images):
            if counts['empty'] >= target_limit and counts['occupied'] >= target_limit:
                break

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base_name + ".txt")

            if not os.path.exists(lbl_path):
                continue

            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
            except Exception as e:
                continue

            # Save canonical reference configuration for parking lot dashboard
            if not canonical_saved and split == 'test':
                reference_slots = []
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                for slot_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = max(0, int((xc - bw / 2) * w))
                    y1 = max(0, int((yc - bh / 2) * h))
                    x2 = min(w, int((xc + bw / 2) * w))
                    y2 = min(h, int((yc + bh / 2) * h))
                    reference_slots.append({
                        "slot_id": slot_idx + 1,
                        "bbox": [x1, y1, x2, y2],
                        "normalized": [round(xc, 4), round(yc, 4), round(bw, 4), round(bh, 4)],
                        "ground_truth": CLASS_MAP.get(cls_id, 'unknown')
                    })

                # Save slots config JSON
                config_data = {
                    "reference_image": os.path.basename(img_path),
                    "image_width": w,
                    "image_height": h,
                    "total_slots": len(reference_slots),
                    "slots": reference_slots
                }
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                # Copy reference image to sample_data
                sample_img_dst = os.path.join(SAMPLE_DATA_DIR, "sample_parking_lot.jpg")
                img.save(sample_img_dst)
                canonical_saved = True
                print(f"  [+] Saved canonical parking layout ({len(reference_slots)} slots) to {CONFIG_PATH}")

            with open(lbl_path, 'r') as f:
                lines = f.readlines()

            for slot_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                cls_name = CLASS_MAP.get(cls_id, None)
                if cls_name is None:
                    continue

                if counts[cls_name] >= target_limit:
                    continue

                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = max(0, int((xc - bw / 2) * w))
                y1 = max(0, int((yc - bh / 2) * h))
                x2 = min(w, int((xc + bw / 2) * w))
                y2 = min(h, int((yc + bh / 2) * h))

                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue

                crop = img.crop((x1, y1, x2, y2)).resize(PATCH_SIZE, Image.Resampling.BILINEAR)
                save_dir = out_occ if cls_name == 'occupied' else out_empty
                patch_fname = f"{base_name}_s{slot_idx}.jpg"
                crop.save(os.path.join(save_dir, patch_fname), "JPEG", quality=90)
                counts[cls_name] += 1

            if img_idx % 200 == 0:
                print(f"  Progress: {split} -> Empty: {counts['empty']}/{target_limit}, Occupied: {counts['occupied']}/{target_limit}")

        print(f"Finished {split}: Empty = {counts['empty']}, Occupied = {counts['occupied']}")

    # Copy 5 diverse sample images to sample_data for live testing
    sample_imgs = glob.glob(os.path.join(DATASET_ROOT, "test", "images", "*.jpg"))[:5]
    for i, s_img in enumerate(sample_imgs):
        dst = os.path.join(SAMPLE_DATA_DIR, f"test_scene_{i+1}.jpg")
        try:
            im = Image.open(s_img)
            im.save(dst)
        except Exception:
            pass
    print(f"\n[+] Saved sample test scenes to {SAMPLE_DATA_DIR}")
    print("\n[SUCCESS] Slot patch extraction completed successfully!")

if __name__ == "__main__":
    extract_patches()

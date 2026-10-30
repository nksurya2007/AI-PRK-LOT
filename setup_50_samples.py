"""
setup_50_samples.py
1. Collects 50 diverse sample parking lot images into sample_data/
2. Sets up slots_config.json with exactly 50 parking slots (#1 to #50)
3. Tests detection and saves a verified 50-slot annotated output.
"""

import os
import glob
import json
import shutil
import cv2
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_TEST_IMAGES = os.path.join(PROJECT_ROOT, "data", "PRL LOT", "test", "images")
DATASET_TEST_LABELS = os.path.join(PROJECT_ROOT, "data", "PRL LOT", "test", "labels")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample_data")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "data", "slots_config.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup():
    print("=" * 60)
    print("Setting up 50 Sample Parking Lot Images & 50-Slot Configuration")
    print("=" * 60)

    # 1. Clear previous sample_data and copy 50 diverse images
    for f in glob.glob(os.path.join(SAMPLE_DIR, "*.*")):
        try:
            os.remove(f)
        except Exception:
            pass

    # Find 100-slot images so we can use a consistent 50-slot layout
    lbl_files = glob.glob(os.path.join(DATASET_TEST_LABELS, "*.txt"))
    target_pairs = []

    for lbl in lbl_files:
        with open(lbl, "r") as f:
            lines = f.readlines()
        if len(lines) >= 50:
            base_name = os.path.splitext(os.path.basename(lbl))[0]
            img_path = os.path.join(DATASET_TEST_IMAGES, base_name + ".jpg")
            if os.path.exists(img_path):
                target_pairs.append((img_path, lbl, lines))

    print(f"Found {len(target_pairs)} candidate parking lot images.")

    # Select 50 diverse images
    step = max(1, len(target_pairs) // 50)
    selected_50 = target_pairs[::step][:50]

    if len(selected_50) < 50:
        selected_50 = target_pairs[:50]

    print(f"Copying {len(selected_50)} sample images to {SAMPLE_DIR}...")
    for idx, (img_path, lbl_path, lines) in enumerate(selected_50, 1):
        dst_name = f"sample_parking_lot_{idx:02d}.jpg"
        dst_path = os.path.join(SAMPLE_DIR, dst_name)
        shutil.copy2(img_path, dst_path)

    print(f"[+] Successfully copied 50 parking lot images into: {SAMPLE_DIR}")

    # 2. Extract exactly 50 slots configuration from the reference lot
    ref_img_path, ref_lbl_path, ref_lines = selected_50[0]
    img = Image.open(ref_img_path)
    w, h = img.size

    slots_50 = []
    # Take exactly 50 slots
    for i, line in enumerate(ref_lines[:50]):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))

        slots_50.append({
            "slot_id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "normalized": [round(xc, 4), round(yc, 4), round(bw, 4), round(bh, 4)],
            "initial_status": "occupied" if cls_id == 1 else "empty"
        })

    config_data = {
        "description": "50 Parking Slots Layout",
        "reference_image": os.path.basename(ref_img_path),
        "image_width": w,
        "image_height": h,
        "total_slots": len(slots_50),
        "slots": slots_50
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"[+] Saved exact 50-slot configuration to: {CONFIG_PATH} (Total Slots: {len(slots_50)})")

    # 3. Test detection on sample_parking_lot_01.jpg
    from src.parking_detector import ParkingDetector
    detector = ParkingDetector(config_path=CONFIG_PATH)
    sample_img = cv2.imread(os.path.join(SAMPLE_DIR, "sample_parking_lot_01.jpg"))
    ann, stats, slot_results = detector.predict_image(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

    out_test_path = os.path.join(OUTPUT_DIR, "sample_50_slots_annotated.jpg")
    cv2.imwrite(out_test_path, ann)

    print("\n" + "=" * 45)
    print("       50-SLOT PARKING TEST RESULTS")
    print("=" * 45)
    print(f"  Total Slots Monitored: {stats['total']}")
    print(f"  Available (Vacant):    {stats['available']} (Green)")
    print(f"  Occupied (Busy):       {stats['occupied']} (Red)")
    print(f"  Occupancy Rate:        {stats['occupancy_rate']}%")
    print("=" * 45)
    print(f"[SUCCESS] Saved test verification image to: {out_test_path}")

if __name__ == "__main__":
    setup()

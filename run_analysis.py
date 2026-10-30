"""
run_analysis.py
Curates 10 DIVERSE PKLot test images (empty, mixed, busy) and runs
Ultralytics YOLOv8 inference on each, producing a full annotated gallery
and summary report.
"""

import os
import sys
import csv
import json
import shutil
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

LABEL_DIR  = os.path.join(PROJECT_ROOT, "data", "PRL LOT", "test", "labels")
IMG_DIR    = os.path.join(PROJECT_ROOT, "data", "PRL LOT", "test", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "analysis")
REPORT_PATH = os.path.join(PROJECT_ROOT, "output", "analysis_report.json")
YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Pick 10 diverse test images ──────────────────────────────────────────

def pick_diverse_images(n_per_bucket=2):
    """Returns list of (img_path, label_path, gt_vacant, gt_occupied)."""
    scenes = []
    for lf in os.listdir(LABEL_DIR):
        img_name = lf.replace(".txt", ".jpg")
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        with open(os.path.join(LABEL_DIR, lf)) as f:
            lines = [l.strip() for l in f if l.strip()]
        classes = [int(l.split()[0]) for l in lines]
        v, o = classes.count(0), classes.count(1)
        scenes.append((img_path, os.path.join(LABEL_DIR, lf), v, o))

    scenes = [(s[0], s[1], s[2], s[3]) for s in scenes if s[2]+s[3] > 0]
    scenes.sort(key=lambda x: x[2] / (x[2]+x[3]), reverse=True)

    # Buckets: 100% vacant, mostly vacant, mixed (~50%), mostly occupied, 100% occupied
    fully_vacant  = [s for s in scenes if s[2]/(s[2]+s[3]) >= 0.99]
    mostly_vacant = [s for s in scenes if 0.70 <= s[2]/(s[2]+s[3]) < 0.99]
    mixed         = [s for s in scenes if 0.45 <= s[2]/(s[2]+s[3]) < 0.55]
    mostly_occ    = [s for s in scenes if 0.10 <= s[2]/(s[2]+s[3]) < 0.30]
    fully_occ     = [s for s in scenes if s[2]/(s[2]+s[3]) < 0.05]

    picked = []
    for bucket in [fully_vacant, mostly_vacant, mixed, mostly_occ, fully_occ]:
        picked += bucket[:n_per_bucket]

    return picked[:10]

# ── 2. Run YOLO inference ───────────────────────────────────────────────────

def draw_annotated(img_bgr, boxes_data, names, img_idx):
    annotated = img_bgr.copy()
    overlay = annotated.copy()

    for b_idx, (cls_id, conf_val, x1, y1, x2, y2) in enumerate(boxes_data):
        cls_name = names.get(cls_id, "").lower()
        is_occ = "occupied" in cls_name or cls_id == 1

        if is_occ:
            color, bg_color = (0, 0, 210), (0, 0, 155)
        else:
            color, bg_color = (0, 210, 0), (0, 145, 0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"#{b_idx+1} {'Occ' if is_occ else 'Vac'} {int(conf_val*100)}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
        cv2.rectangle(annotated, (x1, max(0, y1-lh-4)), (x1+lw+4, y1), color, -1)
        cv2.putText(annotated, label, (x1+2, max(lh+2, y1-2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,255,255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)
    return annotated


def run_analysis():
    from ultralytics import YOLO

    print(f"[YOLO] Loading: {YOLO_WEIGHTS}")
    model = YOLO(YOLO_WEIGHTS)

    print("\nSelecting 10 diverse PKLot test scenes...")
    diverse = pick_diverse_images(n_per_bucket=2)
    print(f"  → {len(diverse)} scenes selected.\n")

    print("=" * 65)
    print("  Ultralytics YOLOv8  |  AI Parking Occupancy Analysis")
    print("  Images: 416×416 (small) | Model: custom-trained best.pt")
    print("=" * 65)

    all_results = []

    for idx, (img_path, lbl_path, gt_vacant, gt_occ) in enumerate(diverse, 1):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (416, 416))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = model.predict(
            source=img_rgb,
            conf=0.25,
            iou=0.45,
            max_det=300,
            imgsz=416,
            verbose=False
        )

        boxes = results[0].boxes
        names = results[0].names

        det_records = []
        vacant_pred, occ_pred = 0, 0

        if boxes is not None:
            for box in boxes:
                cls_id   = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                is_occ = names.get(cls_id, "").lower().find("occupied") >= 0

                if is_occ:
                    occ_pred += 1
                else:
                    vacant_pred += 1

                det_records.append((cls_id, conf_val, x1, y1, x2, y2))

        total_pred = vacant_pred + occ_pred
        rate_pred = round(occ_pred / max(1, total_pred) * 100, 1)
        gt_total  = gt_vacant + gt_occ
        rate_gt   = round(gt_occ / max(1, gt_total) * 100, 1)

        # Draw annotation
        ann = draw_annotated(img_bgr, det_records, names, idx)

        # HUD bar
        hud_copy = ann[:42, :].copy()
        cv2.rectangle(hud_copy, (0, 0), (416, 42), (15, 18, 24), -1)
        cv2.addWeighted(hud_copy, 0.85, ann[:42, :], 0.15, 0, ann[:42, :])
        cv2.line(ann, (0, 42), (416, 42), (50, 60, 75), 2)

        cv2.putText(ann, f"Scene {idx:02d}", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 2, cv2.LINE_AA)
        cv2.circle(ann, (98, 22), 5, (0,215,0), -1)
        cv2.putText(ann, f"FREE:{vacant_pred}", (108, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50,255,100), 2, cv2.LINE_AA)
        cv2.circle(ann, (193, 22), 5, (0,0,230), -1)
        cv2.putText(ann, f"OCC:{occ_pred}", (203, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80,100,255), 2, cv2.LINE_AA)
        cv2.putText(ann, f"TOT:{total_pred}", (275, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,180,180), 1, cv2.LINE_AA)
        cv2.putText(ann, f"RATE:{rate_pred}%", (342, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,215,255), 2, cv2.LINE_AA)

        out_name = f"scene_{idx:02d}_yolo.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), ann)

        status_icon = "🟢" if vacant_pred >= occ_pred else "🔴"
        print(f"  [{idx:02d}/10] {os.path.basename(img_path)[:40]}")
        print(f"         {status_icon}  YOLO → FREE={vacant_pred}  OCC={occ_pred}  TOTAL={total_pred}  RATE={rate_pred}%")
        print(f"         GT  → vacant={gt_vacant}  occupied={gt_occ}  ({rate_gt}%)")
        print()

        all_results.append({
            "scene": idx,
            "image": os.path.basename(img_path),
            "gt_vacant": gt_vacant,
            "gt_occupied": gt_occ,
            "gt_occupancy_rate": rate_gt,
            "pred_vacant": vacant_pred,
            "pred_occupied": occ_pred,
            "pred_total": total_pred,
            "pred_occupancy_rate": rate_pred,
            "output_file": out_name
        })

    # ── 3. Summary ─────────────────────────────────────────────────────────
    n = len(all_results)
    avg_pred_rate = round(sum(r["pred_occupancy_rate"] for r in all_results) / max(1,n), 1)
    avg_gt_rate   = round(sum(r["gt_occupancy_rate"]   for r in all_results) / max(1,n), 1)

    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Scenes analysed        : {n}")
    print(f"  Avg Predicted Rate     : {avg_pred_rate}%  |  Avg Ground-Truth Rate: {avg_gt_rate}%")
    print(f"  Annotated images  →  output/analysis/  (scene_01_yolo.jpg … scene_{n:02d}_yolo.jpg)")
    print(f"  Full report JSON  →  output/analysis_report.json")
    print("=" * 65)

    summary = {
        "scenes_analysed": n,
        "avg_predicted_occupancy_rate": avg_pred_rate,
        "avg_gt_occupancy_rate": avg_gt_rate,
        "results": all_results
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run_analysis()

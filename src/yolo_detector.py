"""
yolo_detector.py
Ultralytics YOLO integration for parking slot detection and occupancy classification.
Optimized for small images (imgsz=416).
"""

import os
import sys
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

class YOLOParkingDetector:
    def __init__(self, model_weight="yolov8n.pt", conf=0.35, imgsz=416, max_det=50, iou=0.5):
        self.conf = conf
        self.imgsz = imgsz
        self.max_det = max_det
        self.iou = iou
        self.model_weight = model_weight

        # Check for trained custom model first
        custom_trained = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt")
        if os.path.exists(custom_trained):
            self.model_weight = custom_trained
            print(f"[YOLO] Using custom trained model: {custom_trained}")
        else:
            print(f"[YOLO] Using model: {self.model_weight}")

        from ultralytics import YOLO
        self.model = YOLO(self.model_weight)

    def predict_image(self, image_np, conf_thresh=None, max_det=None):
        """
        Runs YOLO detection on an RGB image.
        Returns:
            annotated_image (BGR np.ndarray),
            summary_stats (dict),
            detections (list of dicts)
        """
        conf = conf_thresh if conf_thresh is not None else self.conf
        limit = max_det if max_det is not None else self.max_det

        # Convert RGB to BGR for cv2 drawing
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) if len(image_np.shape) == 3 else image_np
        img_h, img_w = img_bgr.shape[:2]

        results = self.model.predict(
            source=image_np,
            conf=conf,
            iou=self.iou,
            max_det=limit,
            imgsz=self.imgsz,
            verbose=False
        )

        detections = []
        empty_count = 0
        occupied_count = 0

        annotated = img_bgr.copy()
        overlay = annotated.copy()

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names

            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                cls_name = names.get(cls_id, f"class_{cls_id}").lower()
                is_occupied = ("occupied" in cls_name) or (cls_id == 1)

                if is_occupied:
                    occupied_count += 1
                    status_str = "Occupied"
                    color = (0, 0, 235)  # Red (BGR)
                    bg_color = (0, 0, 180)
                else:
                    empty_count += 1
                    status_str = "Vacant"
                    color = (0, 215, 0)  # Green (BGR)
                    bg_color = (0, 150, 0)

                detections.append({
                    "slot_id": idx + 1,
                    "status": status_str,
                    "confidence": round(confidence * 100, 1),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

                # Translucent fill
                cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
                # Border
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"#{idx+1} {status_str[:3]} {int(confidence*100)}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - lh - 4)), (x1 + lw + 4, max(lh + 4, y1)), color, -1)
                cv2.putText(annotated, label, (x1 + 2, max(lh + 1, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        total_detected = empty_count + occupied_count
        occ_rate = round((occupied_count / total_detected) * 100.0, 1) if total_detected > 0 else 0.0

        summary_stats = {
            "total": total_detected,
            "available": empty_count,
            "occupied": occupied_count,
            "occupancy_rate": occ_rate
        }

        # Draw HUD
        self._draw_hud(annotated, summary_stats)

        return annotated, summary_stats, detections

    def _draw_hud(self, img, stats):
        h, w = img.shape[:2]
        hud_h = 45
        hud_bg = img[:hud_h, :].copy()
        cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (20, 24, 30), -1)
        cv2.addWeighted(hud_bg, 0.85, img[:hud_h, :], 0.15, 0, img[:hud_h, :])
        cv2.line(img, (0, hud_h), (w, hud_h), (60, 70, 80), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_y = 28

        cv2.putText(img, "YOLOv8 DETECTOR", (10, text_y), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.circle(img, (180, 22), 6, (0, 220, 0), -1)
        cv2.putText(img, f"FREE: {stats['available']}", (192, text_y), font, 0.45, (0, 255, 100), 2, cv2.LINE_AA)

        cv2.circle(img, (270, 22), 6, (0, 0, 240), -1)
        cv2.putText(img, f"BUSY: {stats['occupied']}", (282, text_y), font, 0.45, (80, 100, 255), 2, cv2.LINE_AA)

        cv2.putText(img, f"TOTAL: {stats['total']}", (360, text_y), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

"""
parking_detector.py
Core inference engine supporting both Scikit-Learn and PyTorch models:
- Loads trained model checkpoint (.joblib or .pth)
- Loads slot bounding box configurations (slots_config.json)
- Performs batch prediction across all parking spaces
- Renders green (vacant) / red (occupied) bounding boxes, slot labels, and HUD overlay
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

class ParkingDetector:
    def __init__(self, model_path=None, config_path=None):
        if model_path is None:
            joblib_path = os.path.join(PROJECT_ROOT, "models", "best_parking_model.joblib")
            pth_path = os.path.join(PROJECT_ROOT, "models", "best_parking_model.pth")
            model_path = joblib_path if os.path.exists(joblib_path) else pth_path

        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "data", "slots_config.json")

        self.model_path = model_path
        self.config_path = config_path
        self.is_sklearn = model_path.endswith(".joblib")

        # Load Model
        self._load_model()

        # Load Slots Configuration
        self.slots = []
        if os.path.exists(config_path):
            self.load_slots_config(config_path)

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at: {self.model_path}. Train the model first!")

        if self.is_sklearn:
            import joblib
            checkpoint = joblib.load(self.model_path)
            self.model = checkpoint["model"]
            self.class_names = checkpoint.get("class_names", ["empty", "occupied"])
            self.class_to_idx = checkpoint.get("class_to_idx", {"empty": 0, "occupied": 1})
            print(f"ParkingDetector loaded Scikit-Learn model successfully: {self.model.__class__.__name__}")
        else:
            import torch
            from models.cnn_classifier import get_model
            from torchvision import transforms

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            checkpoint = torch.load(self.model_path, map_location=self.device)
            model_name = checkpoint.get("model_name", "custom_cnn")
            self.class_names = checkpoint.get("class_names", ["empty", "occupied"])
            self.class_to_idx = checkpoint.get("class_to_idx", {"empty": 0, "occupied": 1})

            self.model = get_model(model_name=model_name, num_classes=len(self.class_names), pretrained=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"ParkingDetector loaded PyTorch model successfully: {model_name} on {self.device}")

    def load_slots_config(self, config_path):
        with open(config_path, "r") as f:
            data = json.load(f)
        self.slots = data.get("slots", [])
        self.config_image_width = data.get("image_width", 416)
        self.config_image_height = data.get("image_height", 416)
        print(f"Loaded {len(self.slots)} parking slots from {config_path}")

    def extract_patch_features(self, crop_bgr):
        """Extracts features for Scikit-Learn classifier."""
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (16, 16)).flatten() / 255.0

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        h_hist = h_hist / (np.sum(h_hist) + 1e-6)
        s_hist = s_hist / (np.sum(s_hist) + 1e-6)
        v_hist = v_hist / (np.sum(v_hist) + 1e-6)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_mean = np.mean(edge_mag) / 255.0
        edge_std = np.std(edge_mag) / 255.0
        edge_max = np.max(edge_mag) / 255.0
        edge_features = np.array([edge_mean, edge_std, edge_max, np.percentile(edge_mag, 75) / 255.0])

        return np.concatenate([thumb, h_hist, s_hist, v_hist, edge_features])

    def predict_image(self, image_np, confidence_threshold=0.5):
        """
        Takes RGB numpy image.
        Returns:
            annotated_image (BGR np.ndarray),
            summary_stats (dict),
            slot_results (list of dicts)
        """
        img_h, img_w = image_np.shape[:2]
        w_scale = img_w / float(self.config_image_width) if self.config_image_height else 1.0
        h_scale = img_h / float(self.config_image_height) if self.config_image_height else 1.0

        crops_raw = []
        valid_slots = []

        # Convert to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) if len(image_np.shape) == 3 else image_np

        for s in self.slots:
            if "bbox" in s:
                x1, y1, x2, y2 = s["bbox"]
                x1 = int(x1 * w_scale)
                y1 = int(y1 * h_scale)
                x2 = int(x2 * w_scale)
                y2 = int(y2 * h_scale)
            elif "normalized" in s:
                xc, yc, bw, bh = s["normalized"]
                x1 = int((xc - bw / 2) * img_w)
                y1 = int((yc - bh / 2) * img_h)
                x2 = int((xc + bw / 2) * img_w)
                y2 = int((yc + bh / 2) * img_h)
            else:
                continue

            x1 = max(0, min(img_w - 2, x1))
            y1 = max(0, min(img_h - 2, y1))
            x2 = max(x1 + 2, min(img_w, x2))
            y2 = max(y1 + 2, min(img_h, y2))

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue

            crops_raw.append(crop)
            valid_slots.append((s.get("slot_id", len(valid_slots) + 1), (x1, y1, x2, y2)))

        if not crops_raw:
            return img_bgr, {"total": 0, "available": 0, "occupied": 0, "occupancy_rate": 0.0}, []

        # Predict
        if self.is_sklearn:
            feats = [self.extract_patch_features(c) for c in crops_raw]
            X_batch = np.array(feats, dtype=np.float32)
            probs = self.model.predict_proba(X_batch)
            preds = np.argmax(probs, axis=1)
        else:
            import torch
            import torch.nn.functional as F
            tensors = [self.transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_raw]
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

        slot_results = []
        available_count = 0
        occupied_count = 0

        annotated = img_bgr.copy()
        overlay = annotated.copy()

        empty_idx = self.class_to_idx.get("empty", 0)
        occ_idx = self.class_to_idx.get("occupied", 1)

        for i, (slot_id, (x1, y1, x2, y2)) in enumerate(valid_slots):
            pred_class = preds[i]
            is_occupied = (pred_class == occ_idx)
            conf = probs[i][pred_class]

            status_str = "Occupied" if is_occupied else "Vacant"
            if is_occupied:
                occupied_count += 1
                color = (0, 0, 235)  # Bright Red (BGR)
                bg_color = (0, 0, 180)
            else:
                available_count += 1
                color = (0, 215, 0)  # Bright Green (BGR)
                bg_color = (0, 150, 0)

            slot_results.append({
                "slot_id": slot_id,
                "status": status_str,
                "confidence": round(float(conf) * 100, 1),
                "bbox": [x1, y1, x2, y2]
            })

            # Draw translucent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)

            # Draw border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw badge label
            label = f"#{slot_id}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 4), (x1 + lw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend overlay (alpha=0.25)
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        total_slots = len(valid_slots)
        occ_rate = round((occupied_count / total_slots) * 100.0, 1) if total_slots > 0 else 0.0

        summary_stats = {
            "total": total_slots,
            "available": available_count,
            "occupied": occupied_count,
            "occupancy_rate": occ_rate
        }

        # Draw Top HUD Banner
        self._draw_hud(annotated, summary_stats)

        return annotated, summary_stats, slot_results

    def _draw_hud(self, img, stats):
        h, w = img.shape[:2]
        hud_h = 45
        hud_bg = img[:hud_h, :].copy()
        cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (20, 24, 30), -1)
        cv2.addWeighted(hud_bg, 0.85, img[:hud_h, :], 0.15, 0, img[:hud_h, :])
        cv2.line(img, (0, hud_h), (w, hud_h), (60, 70, 80), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_y = 28

        # Title
        cv2.putText(img, "SMART PARKING AI", (10, text_y), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Badges
        cv2.circle(img, (180, 22), 6, (0, 220, 0), -1)
        cv2.putText(img, f"FREE: {stats['available']}", (192, text_y), font, 0.45, (0, 255, 100), 2, cv2.LINE_AA)

        cv2.circle(img, (270, 22), 6, (0, 0, 240), -1)
        cv2.putText(img, f"BUSY: {stats['occupied']}", (282, text_y), font, 0.45, (80, 100, 255), 2, cv2.LINE_AA)

        cv2.putText(img, f"TOTAL: {stats['total']}", (360, text_y), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, f"RATE: {stats['occupancy_rate']}%", (w - 100, text_y), font, 0.45, (0, 215, 255), 2, cv2.LINE_AA)

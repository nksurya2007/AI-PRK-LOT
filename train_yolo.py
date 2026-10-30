"""
train_yolo.py
Trains Ultralytics YOLOv8 on small 416x416 parking lot images.
"""

import os
import sys
import csv
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "yolo_small.yaml")

def patch_trainer():
    """Patch Ultralytics BaseTrainer.read_results_csv and plot_results to avoid polars dependency."""
    try:
        import ultralytics.engine.trainer as trainer_mod
        def safe_read_results_csv(self):
            try:
                if not hasattr(self, 'csv') or not self.csv or not self.csv.exists():
                    return {}
                with open(self.csv, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    data = {}
                    for row in reader:
                        for k, v in row.items():
                            if k is None:
                                continue
                            k_clean = k.strip()
                            try:
                                val = float(v.strip())
                            except Exception:
                                val = v.strip() if v else ""
                            data.setdefault(k_clean, []).append(val)
                    return data
            except Exception:
                return {}
        trainer_mod.BaseTrainer.read_results_csv = safe_read_results_csv

        import ultralytics.utils.plotting as plotting_mod
        def safe_plot_results(file="path/to/results.csv", dir="", on_plot=None):
            return None
        plotting_mod.plot_results = safe_plot_results

        print("[SETUP] Patched Ultralytics BaseTrainer and plot_results successfully.")
    except Exception as e:
        print(f"[WARN] Failed to patch BaseTrainer: {e}")

def train(epochs=3, imgsz=416, batch=16, model_name="yolov8n.pt"):
    patch_trainer()
    print("=" * 60)
    print("Starting Ultralytics YOLOv8 Parking Slot Training")
    print(f"Model: {model_name} | Image Size: {imgsz}x{imgsz} | Epochs: {epochs} | Batch: {batch}")
    print(f"Dataset YAML: {DATA_YAML}")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_name)

    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=0,
        project=os.path.join(PROJECT_ROOT, "runs", "detect"),
        name="train",
        exist_ok=True,
        save=True,
        plots=True
    )

    print("\n[SUCCESS] YOLO Training completed! Best model saved to runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size (default 416)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    args = parser.parse_args()

    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model_name=args.model)

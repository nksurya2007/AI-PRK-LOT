"""
detect_image.py
Run AI parking slot classification on small images (416x416) using:
1. Ultralytics YOLO (Object Detection on space-empty and space-occupied)
2. Slot-ROI Classifier (Predefined space geometry classification)

Usage:
    python detect_image.py --image sample_data/sample_parking_lot_01.jpg --model yolo
    python detect_image.py --image sample_data/sample_parking_lot_01.jpg --model classifier
"""

import os
import sys
import argparse
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

def process_image(image_path, model_type="yolo", conf=0.25, imgsz=416, output_path=None):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        return

    print(f"Loading small image: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    # Resize to small image size if requested
    if imgsz and (img_bgr.shape[0] != imgsz or img_bgr.shape[1] != imgsz):
        img_bgr = cv2.resize(img_bgr, (imgsz, imgsz))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if model_type.lower() == "yolo":
        print(f"Running Ultralytics YOLOv8 inference (imgsz={imgsz}, conf={conf})...")
        from src.yolo_detector import YOLOParkingDetector
        detector = YOLOParkingDetector(conf=conf, imgsz=imgsz)
        annotated_bgr, stats, detections = detector.predict_image(img_rgb, conf_thresh=conf)
    else:
        print("Running Slot-ROI Classifier inference...")
        from src.parking_detector import ParkingDetector
        detector = ParkingDetector()
        annotated_bgr, stats, detections = detector.predict_image(img_rgb)

    print("\n" + "=" * 45)
    print(f"      PARKING OCCUPANCY REPORT ({model_type.upper()})")
    print("=" * 45)
    print(f"  Total Parking Slots:    {stats['total']}")
    print(f"  Available (Vacant):     {stats['available']}  (Green)")
    print(f"  Occupied (Busy):        {stats['occupied']}  (Red)")
    print(f"  Occupancy Rate:         {stats['occupancy_rate']}%")
    print("=" * 45)

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(PROJECT_ROOT, "output", f"{base_name}_{model_type}_annotated.jpg")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, annotated_bgr)
    print(f"\n[SUCCESS] Saved annotated image to: {output_path}")

    return output_path, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Parking Slot Occupancy on Images")
    parser.add_argument("--image", type=str, default="sample_data/sample_parking_lot_01.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="yolo", choices=["yolo", "classifier"], help="Model engine: yolo or classifier")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO (default 0.25)")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size (default 416 for small images)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    process_image(args.image, model_type=args.model, conf=args.conf, imgsz=args.imgsz, output_path=args.output)

"""
app.py
Streamlit Web Dashboard for AI-Based Parking Slot Occupancy Classification.
Features:
- Dual AI Engine Support:
    1. Ultralytics YOLOv8 (Dynamic Object Detection on 416x416 small images)
    2. Calibrated 50-Slot Classifier (Scikit-Learn HistGradientBoosting - 97.4% Accuracy)
- 50 Curated PKLot Sample Images (Sunny, Cloudy, Rainy conditions) + Custom Upload
- Real-Time KPI Cards (Available / Occupied / Total / Occupancy %)
- Interactive Slot Status List & Filtering
- Model Performance Analytics & Architecture Documentation
"""

import os
import sys
import json
import glob
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="AI Parking Slot Occupancy System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "data", "slots_config.json")
MODEL_JOBLIB = os.path.join(PROJECT_ROOT, "models", "best_parking_model.joblib")
MODEL_PTH = os.path.join(PROJECT_ROOT, "models", "best_parking_model.pth")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-left: 5px solid #1E88E5;
    }
    .status-free {
        color: #28a745;
        font-weight: bold;
    }
    .status-occupied {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo_detector():
    from src.yolo_detector import YOLOParkingDetector
    return YOLOParkingDetector(conf=0.25, imgsz=416, max_det=300, iou=0.45)

@st.cache_resource
def load_slot_detector():
    from src.parking_detector import ParkingDetector
    active_model = MODEL_JOBLIB if os.path.exists(MODEL_JOBLIB) else MODEL_PTH
    if not os.path.exists(active_model):
        return None
    return ParkingDetector(model_path=active_model, config_path=CONFIG_PATH)

def main():
    st.markdown('<div class="main-header">🚗 AI-Based Parking Slot Occupancy Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Capstone: 21 | Smart Transportation & Computer Vision | Ultralytics YOLOv8 & GBDT</div>', unsafe_allow_html=True)

    # Sidebar Navigation

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Module:",
        ["🎯 Real-Time Slot Detection", "📊 Model Evaluation & Metrics", "📖 Project Documentation"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("AI Engine Configuration")

    model_engine = st.sidebar.selectbox(
        "Select AI Model:",
        [
            "🚀 Ultralytics YOLOv8 (Object Detection)",
            "🎯 50-Slot Calibrated Classifier (Scikit-Learn)"
        ]
    )

    conf_threshold = st.sidebar.slider("Confidence Threshold (%)", min_value=10, max_value=90, value=25, step=5)

    show_boxes = st.sidebar.checkbox("Show Bounding Boxes / Overlays", value=True)
    show_labels = st.sidebar.checkbox("Show Slot Badges & Confidence", value=True)

    if app_mode == "🎯 Real-Time Slot Detection":
        render_detection_module(model_engine, conf_threshold, show_boxes, show_labels)

    elif app_mode == "📊 Model Evaluation & Metrics":
        render_evaluation_module()

    elif app_mode == "📖 Project Documentation":
        render_documentation_module()

def render_detection_module(model_engine, conf_threshold, show_boxes, show_labels):
    st.subheader("Live Parking Lot Occupancy Monitor")

    use_yolo = "YOLO" in model_engine

    if use_yolo:
        st.info("💡 **Ultralytics YOLOv8 Engine Active**: Running dynamic 416x416 object detection. Automatically detects parking slot boundaries and classifies `space-empty` (Green) vs `space-occupied` (Red).")
        detector = load_yolo_detector()
    else:
        st.info("💡 **Calibrated 50-Slot Classifier Active**: Utilizing fixed 50-slot layout coordinates with Histogram Gradient Boosting (97.35% test accuracy).")
        detector = load_slot_detector()
        if detector is None:
            st.warning("⚠️ Slot classifier model not found. Please train it via `python src/train_sklearn.py`.")
            return

    # Choose Input Source
    input_source = st.radio(
        "Select Input Source:",
        ["Curated 50 PKLot Sample Images (Sunny / Cloudy / Rainy)", "Upload Custom Image / Frame"],
        horizontal=True
    )

    image_to_process = None
    chosen_path_display = ""

    if input_source == "Curated 50 PKLot Sample Images (Sunny / Cloudy / Rainy)":
        sample_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "sample_parking_lot_*.jpg")))
        if not sample_files:
            sample_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "PRL LOT", "test", "images", "*.jpg"))[:50]

        if sample_files:
            file_options = {os.path.basename(f): f for f in sample_files}
            chosen_name = st.selectbox("Choose one of the 50 sample parking lot scenes:", list(file_options.keys()))
            chosen_path = file_options[chosen_name]
            chosen_path_display = chosen_name
            image_to_process = cv2.imread(chosen_path)
            if image_to_process is not None:
                image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
    else:
        uploaded_file = st.file_uploader("Upload Parking Camera Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            chosen_path_display = uploaded_file.name
            if image_to_process is not None:
                image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)

    if image_to_process is not None:
        # Resize to small image (416x416) for consistency and speed
        h, w = image_to_process.shape[:2]
        if h != 416 or w != 416:
            image_to_process = cv2.resize(image_to_process, (416, 416))

        # Run inference
        conf_ratio = conf_threshold / 100.0
        if use_yolo:
            annotated_bgr, stats, slot_results = detector.predict_image(
                image_to_process,
                conf_thresh=conf_ratio,
                max_det=300
            )
        else:
            annotated_bgr, stats, slot_results = detector.predict_image(
                image_to_process,
                confidence_threshold=conf_ratio
            )

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Real-time Metrics Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Parking Slots", stats["total"])
        col2.metric("Available / Vacant", stats["available"], delta=f"{stats['available']} Free", delta_color="normal")
        col3.metric("Occupied Slots", stats["occupied"], delta=f"-{stats['occupied']} Busy", delta_color="inverse")
        col4.metric("Occupancy Rate", f"{stats['occupancy_rate']}%")

        # Visual Status Banner
        if stats["available"] == 0:
            st.error("🚨 PARKING FULL: No vacant parking spaces available!")
        elif stats["available"] <= 5:
            st.warning(f"⚠️ LIMITED PARKING: Only {stats['available']} spaces remaining.")
        else:
            st.success(f"✅ SPACES AVAILABLE: {stats['available']} empty parking slots ready for parking.")

        # Visual layout
        st.markdown(f"### Camera Feed Analysis: `{chosen_path_display}` (416x416 Small Image)")
        col_img, col_data = st.columns([2, 1])

        with col_img:
            st.image(
                annotated_rgb,
                caption=f"Inference Overlay via {model_engine.split()[1]} (Green = Vacant, Red = Occupied)",
                use_container_width=True
            )

        with col_data:
            st.markdown("#### Slot Status Breakdown")
            filter_mode = st.radio("Filter Slots:", ["All Slots", "Vacant Only", "Occupied Only"], horizontal=True)

            filtered = slot_results
            if filter_mode == "Vacant Only":
                filtered = [s for s in slot_results if s["status"] == "Vacant"]
            elif filter_mode == "Occupied Only":
                filtered = [s for s in slot_results if s["status"] == "Occupied"]

            st.write(f"Showing **{len(filtered)}** slots:")

            for s in filtered:
                status_icon = "🟢" if s["status"] == "Vacant" else "🔴"
                st.markdown(f"**Slot #{s['slot_id']}**: {status_icon} `{s['status']}` — Conf: `{s['confidence']}%`")

def render_evaluation_module():
    st.subheader("Model Evaluation & Performance Metrics")

    tab1, tab2 = st.tabs(["🎯 Slot-ROI Classifier (97.35% Acc)", "🚀 Ultralytics YOLOv8 Detection"])

    with tab1:
        metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
            c2.metric("Macro Precision", f"{metrics['macro_precision']*100:.2f}%")
            c3.metric("Macro Recall", f"{metrics['macro_recall']*100:.2f}%")
            c4.metric("Macro F1-Score", f"{metrics['macro_f1']*100:.2f}%")

            st.markdown("---")

            col_cm, col_curves = st.columns(2)
            with col_cm:
                st.markdown("#### Confusion Matrix")
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix on PKLot Test Split", use_container_width=True)

            with col_curves:
                st.markdown("#### Learning Curves")
                if os.path.exists(curves_path):
                    st.image(curves_path, caption="Loss & Accuracy across Iterations", use_container_width=True)

            st.markdown("#### Class-Wise Performance Breakdown")
            class_wise = metrics.get("class_wise", {})
            if class_wise:
                table_rows = []
                for cls_name, vals in class_wise.items():
                    table_rows.append({
                        "Class": cls_name.capitalize(),
                        "Precision": f"{vals['precision']*100:.2f}%",
                        "Recall": f"{vals['recall']*100:.2f}%",
                        "F1-Score": f"{vals['f1_score']*100:.2f}%",
                        "Support (# Patches)": vals['support']
                    })
                st.table(table_rows)
        else:
            st.info("Metrics file not found. Run `python src/train_sklearn.py`.")

    with tab2:
        st.markdown("### Ultralytics YOLOv8 Small (416x416)")
        st.write("Trained directly on small 416x416 PKLot parking images with bounding box detection for `space-empty` (Vacant) and `space-occupied` (Occupied).")

        yolo_weights = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt")
        if os.path.exists(yolo_weights):
            st.success(f"✅ Model Weights Available: `{yolo_weights}` ({os.path.getsize(yolo_weights)/1024/1024:.2f} MB)")
        else:
            st.warning("YOLO weights not found. Run `python train_yolo.py`.")

        yolo_results_csv = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "results.csv")
        if os.path.exists(yolo_results_csv):
            st.markdown("#### Training Log Metrics (`results.csv`)")
            import csv
            with open(yolo_results_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                st.table(rows)

def render_documentation_module():
    st.subheader("Project Documentation & Architecture")
    st.markdown("""
    ### 📌 Capstone 21: AI-Based Parking Slot Occupancy Classification
    
    #### 1. Objective
    Develop an intelligent computer vision system that classifies parking spaces as **Occupied** or **Vacant** using camera footage in real-time on small images (416x416).
    
    #### 2. Models Supported:
    1. **Ultralytics YOLOv8**:
       - Dynamic bounding box detection for `space-empty` and `space-occupied`.
       - Native resolution: 416x416 small image format.
       - Runs directly on CPU / GPU with ultra-fast inference (<50ms per frame).
    
    2. **50-Slot Calibrated Classifier**:
       - Pre-configured 50-slot layout geometry (`data/slots_config.json`).
       - Feature extraction + Histogram Gradient Boosting.
       - **97.35% Test Accuracy**, 97.35% F1-score.
    
    #### 3. Dataset:
    - **PKLot Dataset**: Curated 50 sample scenes across Sunny, Cloudy, and Rainy conditions.
    """)

if __name__ == "__main__":
    main()

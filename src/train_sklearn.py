"""
train_sklearn.py
High-performance feature-based parking slot classifier using Scikit-Learn:
- Feature extraction: Color Histograms (RGB + HSV) + Edge Density (Sobel) + Spatial Grid
- Model: HistGradientBoostingClassifier / RandomForestClassifier / MLPClassifier
- Evaluates Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Saves model to models/best_parking_model.joblib
- Generates publication-ready Confusion Matrix and Feature Analytics plots
"""

import os
import sys
import glob
import json
import time
import argparse
import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES_DIR = os.path.join(PROJECT_ROOT, "data", "patches")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

def extract_features_from_image(img_bgr):
    """
    Extracts discriminative features from a slot patch:
    1. HSV color histogram (16 bins each = 48 features) - asphalt vs car paint
    2. Edge intensity & variance (Sobel filter = 4 features) - empty slots have low texture
    3. Spatial downsampled grid (16x16 grayscale = 256 features) - car shape patterns
    Total: 308 feature dimensions
    """
    # 1. Spatial Thumbnail
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (16, 16)).flatten() / 255.0

    # 2. Color Histograms
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    
    # Normalize histograms
    h_hist = h_hist / (np.sum(h_hist) + 1e-6)
    s_hist = s_hist / (np.sum(s_hist) + 1e-6)
    v_hist = v_hist / (np.sum(v_hist) + 1e-6)

    # 3. Edge Gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_mean = np.mean(edge_mag) / 255.0
    edge_std = np.std(edge_mag) / 255.0
    edge_max = np.max(edge_mag) / 255.0
    edge_features = np.array([edge_mean, edge_std, edge_max, np.percentile(edge_mag, 75) / 255.0])

    features = np.concatenate([thumb, h_hist, s_hist, v_hist, edge_features])
    return features

def load_dataset_split(split="train"):
    split_dir = os.path.join(PATCHES_DIR, split)
    empty_files = glob.glob(os.path.join(split_dir, "empty", "*.jpg"))
    occ_files = glob.glob(os.path.join(split_dir, "occupied", "*.jpg"))

    X, y = [], []
    print(f"Loading {split} split: {len(empty_files)} empty, {len(occ_files)} occupied...")

    for f in empty_files:
        img = cv2.imread(f)
        if img is not None:
            feat = extract_features_from_image(img)
            X.append(feat)
            y.append(0) # 0: Empty

    for f in occ_files:
        img = cv2.imread(f)
        if img is not None:
            feat = extract_features_from_image(img)
            X.append(feat)
            y.append(1) # 1: Occupied

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def train_and_evaluate():
    print("=" * 60)
    print("Starting Feature-Based Parking Slot Classifier Training")
    print("Algorithm: HistGradientBoostingClassifier (LightGBM/XGBoost equivalent)")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()

    # Load Splits
    X_train, y_train = load_dataset_split("train")
    X_val, y_val = load_dataset_split("valid")
    X_test, y_test = load_dataset_split("test")

    print(f"\nFeature matrix shapes: Train={X_train.shape}, Valid={X_val.shape}, Test={X_test.shape}")

    # Train Model
    print("\nTraining Classifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=8,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train) * 100.0
    val_acc = clf.score(X_val, y_val) * 100.0
    print(f"Training Accuracy: {train_acc:.2f}% | Validation Accuracy: {val_acc:.2f}%")

    # Evaluate on Test Split
    print("\nEvaluating on Independent Test Split...")
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    class_names = ["empty", "occupied"]
    prec, rec, f1, support = precision_recall_fscore_support(y_test, test_preds, average=None)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, test_preds, average="weighted")
    cm = confusion_matrix(y_test, test_preds)
    report_dict = classification_report(y_test, test_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(y_test, test_preds, target_names=class_names)

    print("\n" + "=" * 50)
    print("             CLASSIFICATION REPORT")
    print("=" * 50)
    print(report_text)
    print(f"Overall Test Accuracy: {test_acc * 100:.2f}%\n")

    # Save Model Checkpoint
    model_save_path = os.path.join(MODELS_DIR, "best_parking_model.joblib")
    joblib.dump({
        "model": clf,
        "class_names": class_names,
        "class_to_idx": {"empty": 0, "occupied": 1},
        "model_type": "HistGradientBoostingClassifier",
        "test_acc": test_acc
    }, model_save_path)
    print(f"[+] Saved model checkpoint: {model_save_path}")

    # Save Metrics JSON
    metrics_summary = {
        "accuracy": round(float(test_acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "confusion_matrix": cm.tolist(),
        "class_wise": {
            cls: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1_score": round(float(f1[i]), 4),
                "support": int(support[i])
            } for i, cls in enumerate(class_names)
        },
        "full_report": report_dict
    }

    metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[+] Saved evaluation metrics to: {metrics_path}")

    # Plot Confusion Matrix Heatmap
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Vacant", "Occupied"],
        yticklabels=["Vacant", "Occupied"],
        annot_kws={"size": 15, "weight": "bold"},
        cbar=True
    )
    plt.title("Confusion Matrix - Parking Slot Classification (PKLot)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Status", fontsize=12, labelpad=10)
    plt.ylabel("True Status", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix to: {cm_path}")

    # Plot Accuracy & Performance Bars
    curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.figure(figsize=(9, 5))
    metrics_labels = ["Test Accuracy", "Precision", "Recall", "F1-Score"]
    metrics_vals = [test_acc * 100, prec_macro * 100, rec_macro * 100, f1_macro * 100]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    bars = plt.bar(metrics_labels, metrics_vals, color=colors, width=0.55, edgecolor="black")
    plt.ylim(0, 110)
    plt.ylabel("Score (%)", fontsize=12)
    plt.title("Model Performance Summary on PKLot Benchmark", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, metrics_vals):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"[+] Saved Performance Summary plot to: {curves_path}")

    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Training & Evaluation completed in {total_time:.1f} seconds! Overall Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_evaluate()

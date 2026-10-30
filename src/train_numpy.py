"""
train_numpy.py
Trains the Deep Neural Network classifier on PKLot parking slot patches using pure NumPy.
Evaluates:
- Accuracy, Precision, Recall, F1-Score (Macro & Class-wise)
- Generates and saves Confusion Matrix heatmap
- Generates and saves Training Loss and Accuracy curves
- Exports evaluation_metrics.json for the Streamlit dashboard
"""

import os
import sys
import glob
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES_DIR = os.path.join(PROJECT_ROOT, "data", "patches")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

from src.neural_network import ParkingNeuralNet

def extract_features(img_bgr):
    """
    Extracts discriminative features from a slot patch:
    1. Spatial thumbnail (16x16 = 256)
    2. HSV color histogram (16 bins each = 48)
    3. Edge gradients (4)
    Total: 308 feature dimensions
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (16, 16)).flatten() / 255.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
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
    edge_p75 = np.percentile(edge_mag, 75) / 255.0
    edge_features = np.array([edge_mean, edge_std, edge_max, edge_p75])

    return np.concatenate([thumb, h_hist, s_hist, v_hist, edge_features])

def load_split(split_name="train", max_samples=4000):
    split_dir = os.path.join(PATCHES_DIR, split_name)
    empty_files = glob.glob(os.path.join(split_dir, "empty", "*.jpg"))[:max_samples]
    occ_files = glob.glob(os.path.join(split_dir, "occupied", "*.jpg"))[:max_samples]

    X, y = [], []
    print(f"Loading {split_name} split ({len(empty_files)} empty, {len(occ_files)} occupied)...")

    for f in empty_files:
        img = cv2.imread(f)
        if img is not None:
            X.append(extract_features(img))
            y.append(0)

    for f in occ_files:
        img = cv2.imread(f)
        if img is not None:
            X.append(extract_features(img))
            y.append(1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def calculate_metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    
    # Confusion matrix [[TN, FP], [FN, TP]]
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Class 0 (Empty)
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

    # Class 1 (Occupied)
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0

    macro_prec = (prec_0 + prec_1) / 2.0
    macro_rec = (rec_0 + rec_1) / 2.0
    macro_f1 = (f1_0 + f1_1) / 2.0

    weighted_prec = (prec_0 * (tn + fp) + prec_1 * (fn + tp)) / len(y_true)
    weighted_rec = (rec_0 * (tn + fp) + rec_1 * (fn + tp)) / len(y_true)
    weighted_f1 = (f1_0 * (tn + fp) + f1_1 * (fn + tp)) / len(y_true)

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "class_wise": {
            "empty": {"precision": round(prec_0, 4), "recall": round(rec_0, 4), "f1_score": round(f1_0, 4), "support": int(tn + fp)},
            "occupied": {"precision": round(prec_1, 4), "recall": round(rec_1, 4), "f1_score": round(f1_1, 4), "support": int(fn + tp)}
        }
    }

def main():
    print("=" * 60)
    print("Training Parking Slot Occupancy Deep Neural Network")
    print("Architecture: Multi-Layer Perceptron (308 -> 128 -> 64 -> 2)")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()

    # Load data
    X_train, y_train = load_split("train", max_samples=4000)
    X_val, y_val = load_split("valid", max_samples=1000)
    X_test, y_test = load_split("test", max_samples=1000)

    print(f"\nDataset Dimensions: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Initialize model
    model = ParkingNeuralNet(input_dim=X_train.shape[1], hidden1=128, hidden2=64, num_classes=2, lr=0.0015)

    epochs = 20
    batch_size = 64
    n_samples = X_train.shape[0]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "best_parking_model.npz")

    print("\nStarting Training Epochs...")
    for epoch in range(1, epochs + 1):
        # Shuffle train data
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Mini-batch gradient descent
        for i in range(0, n_samples, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            model.forward(X_batch, is_training=True, dropout_rate=0.2)
            model.backward(X_batch, y_batch, dropout_rate=0.2)

        # Train metrics
        train_probs = model.predict_proba(X_train)
        train_loss = model.compute_loss(train_probs, y_train)
        train_preds = np.argmax(train_probs, axis=1)
        train_acc = np.mean(train_preds == y_train) * 100.0

        # Val metrics
        val_probs = model.predict_proba(X_val)
        val_loss = model.compute_loss(val_probs, y_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_preds == y_val) * 100.0

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(best_model_path)

    print(f"\n[BEST] Validation Accuracy: {best_val_acc:.2f}%. Saved model weights to: {best_model_path}")

    # Load best weights for final test evaluation
    model.load(best_model_path)
    test_probs = model.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    metrics = calculate_metrics(y_test, test_preds)

    print("\n" + "=" * 55)
    print("               FINAL TEST SET EVALUATION")
    print("=" * 55)
    print(f"Overall Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:       {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:          {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score:        {metrics['macro_f1'] * 100:.2f}%")
    print("\nClass-Wise Breakdown:")
    for cls_name, vals in metrics["class_wise"].items():
        print(f"  [{cls_name.upper()}]: Precision={vals['precision']*100:.2f}%, Recall={vals['recall']*100:.2f}%, F1={vals['f1_score']*100:.2f}% (Support: {vals['support']})")

    # Save metrics JSON
    metrics_json_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[+] Saved metrics to {metrics_json_path}")

    # Plot Confusion Matrix Heatmap
    cm_arr = np.array(metrics["confusion_matrix"])
    cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Vacant", "Occupied"],
        yticklabels=["Vacant", "Occupied"],
        annot_kws={"size": 14, "weight": "bold"},
        cbar=True
    )
    plt.title("Confusion Matrix - Parking Slot Classification (PKLot)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Status", fontsize=12, labelpad=10)
    plt.ylabel("True Status", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix plot to {cm_plot_path}")

    # Plot Learning Curves
    curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    ep_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ep_range, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
    ax1.plot(ep_range, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)

    ax2.plot(ep_range, history["train_acc"], "b-o", label="Training Accuracy", linewidth=2)
    ax2.plot(ep_range, history["val_acc"], "g--s", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Classification Accuracy (%) vs. Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"[+] Saved Learning Curves to {curves_path}")

    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Entire training and evaluation completed in {total_time:.1f}s!")

if __name__ == "__main__":
    main()

"""
evaluate.py
Evaluates the trained parking slot model on the independent test set.
Computes:
- Overall Accuracy
- Precision, Recall, F1-Score (per class and macro/weighted average)
- Confusion Matrix Heatmap (saved as PNG)
- Classification Report (saved as text and JSON)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.cnn_classifier import get_model
from src.dataset_loader import get_dataloaders

def evaluate_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_parking_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    print("=" * 60)
    print(f"Evaluating Model Checkpoint: {checkpoint_path}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get('model_name', 'custom_cnn')
    class_names = checkpoint.get('class_names', ['empty', 'occupied'])

    # Load Data
    _, _, test_loader, _, _ = get_dataloaders(batch_size=64)

    # Initialize Model
    model = get_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metrics
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

    cm = confusion_matrix(all_targets, all_preds)
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_targets, all_preds, target_names=class_names)

    print("\n" + "=" * 45)
    print("         CLASSIFICATION REPORT")
    print("=" * 45)
    print(report_text)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%\n")

    # Output Directory
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics JSON
    metrics_summary = {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "confusion_matrix": cm.tolist(),
        "class_wise": {
            cls: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1_score": round(float(f1[i]), 4),
                "support": int(support[i])
            } for i, cls in enumerate(class_names)
        },
        "full_report": report_dict
    }

    metrics_json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[+] Saved evaluation metrics to: {metrics_json_path}")

    # Plot Confusion Matrix
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    return metrics_summary

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(7, 6))
    display_labels = [c.capitalize() for c in class_names]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        annot_kws={"size": 14, "weight": "bold"},
        cbar=True
    )
    plt.title("Confusion Matrix - Parking Slot Classification", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix heatmap to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Parking Slot Classifier")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    evaluate_model(checkpoint_path=args.checkpoint)

"""
train.py
Trains the parking slot classifier on PKLot patches.
Monitors training/validation loss & accuracy, saves the best model checkpoint,
and generates loss/accuracy training curve plots.
"""

import os
import sys
import json
import time
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.cnn_classifier import get_model
from src.dataset_loader import get_dataloaders

def train_model(model_name="custom_cnn", epochs=10, batch_size=64, lr=0.001):
    print("=" * 60)
    print(f"Starting Training: Model = {model_name} | Epochs = {epochs} | Batch Size = {batch_size} | LR = {lr}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Load Data
    train_loader, val_loader, _, class_names, class_to_idx = get_dataloaders(batch_size=batch_size)

    # Initialize Model
    model = get_model(model_name=model_name, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    models_dir = os.path.join(PROJECT_ROOT, "models")
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, "best_parking_model.pth")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0

        scheduler.step(epoch_val_loss)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        epoch_duration = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Save Best Model Checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'class_names': class_names,
                'class_to_idx': class_to_idx,
                'model_name': model_name
            }, best_model_path)
            print(f"  [BEST] Checkpoint saved with Val Acc: {best_val_acc:.2f}% -> {best_model_path}")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} mins. Best Validation Accuracy: {best_val_acc:.2f}%")

    # Save training history JSON
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot & Save Learning Curves
    plot_curves(history, os.path.join(output_dir, "training_curves.png"))

    return best_model_path

def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r--s', label='Validation Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)
    
    # Accuracy Curve
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'g--s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training & Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved training curves to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Parking Slot Classification Model")
    parser.add_argument("--model", type=str, default="custom_cnn", choices=["custom_cnn", "mobilenet_v2"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_model(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

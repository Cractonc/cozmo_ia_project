import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import CozmoNNDiscrete, ACTION_CLASS_SPECS
from dataset import CozmoDiscreteDataset

# Paths
DATA_DIR = "training_data/"
MODEL_PATH = "models/cozmo_nn_discrete_2_1.pt"
STATS_PATH = "models/norm_stats_discrete_2_1.json"
OUTPUT_DIR = "/home/adrien/cozmo_ia_project/.agents/teamwork_preview_challenger_1"
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "confusion_matrix_corrected.png")

def calculate_metrics(cm, class_names):
    metrics = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "total_true": int(np.sum(cm[i, :])),
            "total_pred": int(np.sum(cm[:, i]))
        }
    return metrics

def plot_confusion_matrix(cm, classes, title='Matrice de Confusion Corrégée (Modèle 2.1)', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Vraie Classe (Ground Truth)', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"Confusion matrix saved to {OUTPUT_IMG}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    # Load Train and Val datasets to inspect class distribution
    train_dataset = CozmoDiscreteDataset(data_dir=DATA_DIR, split='train', norm_stats_path=STATS_PATH)
    val_dataset = CozmoDiscreteDataset(data_dir=DATA_DIR, split='val', norm_stats_path=STATS_PATH)

    class_names = [spec[0] for spec in ACTION_CLASS_SPECS]
    
    # 1. Dataset stats
    train_counts = np.bincount(train_dataset.action_classes, minlength=7)
    val_counts = np.bincount(val_dataset.action_classes, minlength=7)
    
    # Recalculate weights to verify
    counts_nonzero = np.maximum(train_counts, 1)
    calculated_weights = len(train_dataset) / (7.0 * counts_nonzero)
    
    print("--- DATASET STATS ---")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print("\nClass distribution (Train vs Val):")
    for i, name in enumerate(class_names):
        print(f"Class {i} ({name:<12}): Train = {train_counts[i]:>5} ({train_counts[i]/len(train_dataset)*100.2:.1f}%) | Val = {val_counts[i]:>5} ({val_counts[i]/len(val_dataset)*100.2:.1f}%) | Weight = {calculated_weights[i]:.4f}")

    # Load Model
    model = CozmoNNDiscrete()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Inference on Val
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for imgs, sensors, classes, _, _ in val_loader:
            preds = model(imgs, sensors)
            pred_classes = torch.argmax(preds, dim=1)
            val_preds.extend(pred_classes.cpu().numpy())
            val_targets.extend(classes.cpu().numpy())
            
    val_cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(val_targets, val_preds):
        val_cm[t, p] += 1
        
    val_metrics = calculate_metrics(val_cm, class_names)
    val_acc = np.sum(np.diag(val_cm)) / len(val_dataset)
    
    # 3. Inference on Train
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    train_preds = []
    train_targets = []
    
    with torch.no_grad():
        for imgs, sensors, classes, _, _ in train_loader:
            preds = model(imgs, sensors)
            pred_classes = torch.argmax(preds, dim=1)
            train_preds.extend(pred_classes.cpu().numpy())
            train_targets.extend(classes.cpu().numpy())
            
    train_cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(train_targets, train_preds):
        train_cm[t, p] += 1
        
    train_metrics = calculate_metrics(train_cm, class_names)
    train_acc = np.sum(np.diag(train_cm)) / len(train_dataset)
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"Overall Accuracy: Train = {train_acc*100.2:.1f}% | Val = {val_acc*100.2:.1f}%")
    
    print("\nCorrected Confusion Matrix (Validation):")
    header = f"{'True/Pred':<15}" + "".join([f"{name:>13}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:<15}" + "".join([f"{val_cm[i, j]:>13d}" for j in range(7)])
        print(row)
        
    print("\nCorrected Confusion Matrix (Train):")
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:<15}" + "".join([f"{train_cm[i, j]:>13d}" for j in range(7)])
        print(row)
        
    # Plot and save
    plot_confusion_matrix(val_cm, class_names)
    
    # Save statistics as JSON
    results = {
        "dataset_stats": {
            "train_samples": int(len(train_dataset)),
            "val_samples": int(len(val_dataset)),
            "class_names": class_names,
            "train_counts": train_counts.tolist(),
            "val_counts": val_counts.tolist(),
            "calculated_weights": calculated_weights.tolist()
        },
        "validation_metrics": {
            "accuracy": float(val_acc),
            "confusion_matrix": val_cm.tolist(),
            "metrics_per_class": val_metrics
        },
        "train_metrics": {
            "accuracy": float(train_acc),
            "confusion_matrix": train_cm.tolist(),
            "metrics_per_class": train_metrics
        }
    }
    
    results_path = os.path.join(OUTPUT_DIR, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Detailed statistics saved to {results_path}")

if __name__ == '__main__':
    main()

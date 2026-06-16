import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import CozmoNNDiscrete
from dataset import CozmoDiscreteDataset

# Paths
DATA_DIR = "training_data/"
MODEL_PATH = "models/cozmo_discrete_nn_2_4.pt"
STATS_PATH = "models/norm_stats_discrete_2_4.json"
OUTPUT_IMG = "confusion_matrix_2_4.png"

def plot_confusion_matrix(cm, classes, title='Matrice de Confusion 2.4 (4 Classes)', cmap=plt.cm.Blues):
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
    if os.path.dirname(OUTPUT_IMG):
        os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    # Load Val dataset
    val_dataset = CozmoDiscreteDataset(data_dir=DATA_DIR, split='val', norm_stats_path=STATS_PATH)
    if len(val_dataset) == 0:
        print("No validation data found!")
        return

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Load Model
    model = CozmoNNDiscrete()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_targets = []

    print("Running inference on validation set for 2_3...")
    with torch.no_grad():
        for imgs, sensors, classes, _, _ in val_loader:
            preds = model(imgs, sensors)
            pred_classes = torch.argmax(preds, dim=1)
            all_preds.extend(pred_classes.cpu().numpy())
            all_targets.extend(classes.cpu().numpy())

    # Compute Confusion Matrix manually
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(all_targets, all_preds):
        if 0 <= t < 4 and 0 <= p < 4:
            cm[t, p] += 1
    
    class_names = ["Avant", "Gauche", "Droite", "Arrière"]
    
    print("\nConfusion Matrix (Raw):")
    print(cm)
    
    print("\nFormatted Confusion Matrix:")
    label = "True/Pred"
    header = f"{label:<15}" + "".join([f"{name:>12}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:<15}" + "".join([f"{cm[i, j]:>12}" for j in range(len(class_names))])
        print(row)
    print()

    plot_confusion_matrix(cm, class_names)
    print(f"Confusion matrix saved to {OUTPUT_IMG}")

if __name__ == '__main__':
    main()

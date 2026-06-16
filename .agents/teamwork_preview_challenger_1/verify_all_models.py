import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import CozmoNNDiscrete, ACTION_CLASS_SPECS
from dataset import CozmoDiscreteDataset

DATA_DIR = "training_data/"
STATS_PATH = "models/norm_stats_discrete_2_1.json"

def evaluate_model(model_path):
    if not os.path.exists(model_path):
        return None
        
    val_dataset = CozmoDiscreteDataset(data_dir=DATA_DIR, split='val', norm_stats_path=STATS_PATH)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    model = CozmoNNDiscrete()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for imgs, sensors, classes, _, _ in val_loader:
            preds = model(imgs, sensors)
            pred_classes = torch.argmax(preds, dim=1)
            preds_list.extend(pred_classes.cpu().numpy())
            targets_list.extend(classes.cpu().numpy())
            
    cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(targets_list, preds_list):
        cm[t, p] += 1
        
    accuracy = np.sum(np.diag(cm)) / len(val_dataset)
    pred_counts = np.bincount(preds_list, minlength=7)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "pred_counts": pred_counts
    }

def main():
    models_to_check = {
        "Model 2.1 (cozmo_nn_discrete_2_1.pt)": "models/cozmo_nn_discrete_2_1.pt",
        "Dryrun (cozmo_discrete_nn_dryrun.pt)": "models/cozmo_discrete_nn_dryrun.pt",
        "Test Weights (cozmo_discrete_nn_test_weights.pt)": "models/cozmo_discrete_nn_test_weights.pt"
    }
    
    class_names = [spec[0] for spec in ACTION_CLASS_SPECS]
    
    print("=== MULTI-MODEL COMPARISON ON VAL SET ===")
    for name, path in models_to_check.items():
        res = evaluate_model(path)
        if res is None:
            print(f"\n{name}: Not found")
            continue
            
        print(f"\n{name}:")
        print(f"Accuracy: {res['accuracy']*100.2:.1f}%")
        print("Predictions distribution:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name:<12}: {res['pred_counts'][i]:>5} ({res['pred_counts'][i]/np.sum(res['pred_counts'])*100.2:.1f}%)")
            
        print("Confusion Matrix:")
        header = f"{'True/Pred':<15}" + "".join([f"{c_name:>13}" for c_name in class_names])
        print(header)
        print("-" * len(header))
        for i, class_name in enumerate(class_names):
            row = f"{class_name:<15}" + "".join([f"{res['confusion_matrix'][i, j]:>13d}" for j in range(7)])
            print(row)

if __name__ == '__main__':
    main()

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import CozmoNNDiscrete, CozmoDiscreteNN
from dataset import CozmoDiscreteDataset

def verify_state_dict(model_path):
    print("=== Step 1: State Dict Verification ===")
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return None, None
    
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        print(f"Successfully loaded state dict from {model_path}.")
        print(f"Number of keys in state dict: {len(state_dict)}")
        
        # Determine architecture from keys
        has_speed_head = any("speed_head" in k for k in state_dict.keys())
        has_head_angle = any("head_angle" in k for k in state_dict.keys())
        has_output = any("output" in k for k in state_dict.keys())
        
        print(f"State dict analysis:")
        print(f" - Contains 'speed_head' key: {has_speed_head}")
        print(f" - Contains 'head_angle' key: {has_head_angle}")
        print(f" - Contains 'output' key (standard for CozmoNNDiscrete): {has_output}")
        
        if has_output:
            arch_guess = "CozmoNNDiscrete"
            model = CozmoNNDiscrete()
        elif has_speed_head or has_head_angle:
            arch_guess = "CozmoDiscreteNN"
            model = CozmoDiscreteNN()
        else:
            arch_guess = "Unknown"
            model = None
            
        print(f"Guessed architecture: {arch_guess}")
        
        if model is not None:
            # Check compatibility
            model_keys = set(model.state_dict().keys())
            sd_keys = set(state_dict.keys())
            
            missing_keys = model_keys - sd_keys
            unexpected_keys = sd_keys - model_keys
            
            print(f"Keys check against {arch_guess}:")
            print(f" - Missing keys: {len(missing_keys)}")
            if missing_keys:
                print(f"   Missing sample: {list(missing_keys)[:5]}")
            print(f" - Unexpected keys: {len(unexpected_keys)}")
            if unexpected_keys:
                print(f"   Unexpected sample: {list(unexpected_keys)[:5]}")
                
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print("Perfect state dict key match!")
                model.load_state_dict(state_dict)
                print("Successfully instantiated and loaded model state dict.")
                return model, arch_guess
            else:
                print("Warning: Key mismatch. Trying to load with strict=False.")
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print("Successfully loaded model with strict=False.")
                    return model, arch_guess
                except Exception as e:
                    print(f"Failed to load with strict=False: {e}")
                    
        return None, arch_guess
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return None, None

def run_validation_inference(model, arch_name, data_dir, stats_path):
    print("\n=== Step 2: Validation Inference ===")
    
    # Load dataset
    val_dataset = CozmoDiscreteDataset(data_dir=data_dir, split='val', norm_stats_path=stats_path)
    print(f"Validation dataset size: {len(val_dataset)}")
    if len(val_dataset) == 0:
        print("No validation data found!")
        return
        
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    class_names = ["forward", "stop", "curve_left", "curve_right", "rotate_left", "rotate_right", "backward"]
    
    with torch.no_grad():
        for batch in val_loader:
            if arch_name == "CozmoNNDiscrete":
                imgs, sensors, classes, _, _ = batch
                outputs = model(imgs, sensors)
                preds = torch.argmax(outputs, dim=1)
            elif arch_name == "CozmoDiscreteNN":
                imgs, sensors, classes, _, _ = batch
                logits, speed_scale, head_angle = model(imgs, sensors)
                preds = torch.argmax(logits, dim=1)
            else:
                print(f"Unknown architecture {arch_name}, trying default forward pass")
                imgs, sensors, classes, _, _ = batch
                outputs = model(imgs, sensors)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                preds = torch.argmax(outputs, dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(classes.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print("\n=== Step 3: Prediction Distribution ===")
    total = len(all_targets)
    print(f"Total validation samples processed: {total}")
    
    # Target distribution
    target_counts = np.bincount(all_targets, minlength=7)
    target_pcts = target_counts / total * 100
    
    # Prediction distribution
    pred_counts = np.bincount(all_preds, minlength=7)
    pred_pcts = pred_counts / total * 100
    
    # Accuracy per class
    correct_mask = (all_preds == all_targets)
    overall_accuracy = np.sum(correct_mask) / total * 100
    print(f"Overall Validation Accuracy: {overall_accuracy:.2f}%")
    
    print("\nClass-wise Distribution and Performance:")
    print(f"{'Class Name':<15} | {'Ground Truth Count':<20} | {'Ground Truth %':<15} | {'Predicted Count':<18} | {'Predicted %':<15} | {'Accuracy':<10}")
    print("-" * 105)
    
    for i, name in enumerate(class_names):
        class_gt_mask = (all_targets == i)
        class_gt_count = target_counts[i]
        class_gt_pct = target_pcts[i]
        
        class_pred_count = pred_counts[i]
        class_pred_pct = pred_pcts[i]
        
        if class_gt_count > 0:
            class_correct = np.sum(correct_mask & class_gt_mask)
            class_acc = class_correct / class_gt_count * 100
            acc_str = f"{class_acc:.2f}%"
        else:
            acc_str = "N/A"
            
        print(f"{name:<15} | {class_gt_count:<20} | {class_gt_pct:<15.2f} | {class_pred_count:<18} | {class_pred_pct:<15.2f} | {acc_str:<10}")
        
    # Confusion Matrix
    cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
        
    print("\nConfusion Matrix:")
    header = f"{'True \\ Pred':<15}" + "".join([f"{name:>13}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:<15}" + "".join([f"{cm[i, j]:>13}" for j in range(7)])
        print(row)
        
    # Write stats to file for report
    results = {
        "overall_accuracy": float(overall_accuracy),
        "total_samples": int(total),
        "target_distribution": {name: int(count) for name, count in zip(class_names, target_counts)},
        "predicted_distribution": {name: int(count) for name, count in zip(class_names, pred_counts)},
        "confusion_matrix": cm.tolist()
    }
    
    results_path = "models/verification_results_2_2.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved verification results to {results_path}")

if __name__ == "__main__":
    MODEL_PATH = "models/cozmo_discrete_nn_2_2.pt"
    STATS_PATH = "models/norm_stats_discrete_2_2.json"
    DATA_DIR = "training_data/"
    
    model, arch_name = verify_state_dict(MODEL_PATH)
    if model is not None:
        run_validation_inference(model, arch_name, DATA_DIR, STATS_PATH)
    else:
        print("Verification failed: Could not load model.")

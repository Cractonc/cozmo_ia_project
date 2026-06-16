import sys
import os
import numpy as np

# Add parent directory to sys.path so we can import model and dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset import CozmoDiscreteDataset
from model import ACTION_CLASS_SPECS

def check_distribution(data_dir):
    print(f"=== Action Class Distribution for {data_dir} ===")
    
    # Load train and val datasets
    train_dataset = CozmoDiscreteDataset(data_dir=data_dir, split='train', norm_stats_path=None)
    val_dataset = CozmoDiscreteDataset(data_dir=data_dir, split='val', norm_stats_path=None)
    
    train_classes = train_dataset.action_classes
    val_classes = val_dataset.action_classes
    
    num_classes = len(ACTION_CLASS_SPECS)
    total_train = len(train_classes)
    total_val = len(val_classes)
    
    print("\nClass Index | Class Name | Train Count | Train % | Val Count | Val % | Weight (Train)")
    print("-" * 90)
    
    for i, (name, _) in enumerate(ACTION_CLASS_SPECS):
        train_count = np.sum(train_classes == i)
        val_count = np.sum(val_classes == i)
        train_pct = (train_count / total_train * 100) if total_train > 0 else 0
        val_pct = (val_count / total_val * 100) if total_val > 0 else 0
        
        # Calculate class weight using formula: weight_c = total_samples / (num_classes * samples_c)
        if train_count > 0:
            weight = total_train / (num_classes * train_count)
            weight_str = f"{weight:.4f}"
        else:
            weight_str = "N/A (0 samples)"
            
        print(f"{i:<11} | {name:<10} | {train_count:<11} | {train_pct:<7.2f} | {val_count:<9} | {val_pct:<5.2f} | {weight_str}")
        
    print("-" * 90)
    print(f"Total       |            | {total_train:<11} | 100.00  | {total_val:<9} | 100.00 |")

if __name__ == '__main__':
    # We check both training_data and training_data_1.0
    check_distribution('training_data')
    print("\n" + "="*50 + "\n")
    check_distribution('training_data_1.0')

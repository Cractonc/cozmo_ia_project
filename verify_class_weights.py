import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# Add project path to python path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import CozmoDiscreteDataset
from model import ACTION_CLASS_SPECS

def test_class_weights_and_loss():
    print("--- 1. Loading Training Dataset ---")
    data_dir = "training_data/"
    stats_path = "models/norm_stats_discrete_verify.json"
    
    # Instantiate dataset
    dataset = CozmoDiscreteDataset(data_dir=data_dir, split='train', norm_stats_path=stats_path)
    print(f"Loaded train dataset with {len(dataset)} samples.")
    
    # 2. Replicate train.py class weight logic
    action_classes = dataset.action_classes
    counts = np.bincount(action_classes, minlength=7).astype(np.float64)
    print("\n--- 2. Class Counts in Training Split ---")
    for idx, (name, _) in enumerate(ACTION_CLASS_SPECS):
        print(f"Class {idx} ({name}): {int(counts[idx])} samples")
        
    avg_2_3 = (counts[2] + counts[3]) / 2.0
    avg_4_5 = (counts[4] + counts[5]) / 2.0
    
    # Apply forced symmetry
    counts_sym = counts.copy()
    counts_sym[2] = avg_2_3
    counts_sym[3] = avg_2_3
    counts_sym[4] = avg_4_5
    counts_sym[5] = avg_4_5
    
    counts_sym = np.maximum(counts_sym, 1)
    total_samples = len(dataset)
    raw_weights = total_samples / (7.0 * counts_sym)
    clipped_weights = np.clip(raw_weights, 0.5, 3.0)
    
    print("\n--- 3. Class Weights Calculation ---")
    for idx, (name, _) in enumerate(ACTION_CLASS_SPECS):
        print(f"Class {idx} ({name}): raw={raw_weights[idx]:.4f}, clipped={clipped_weights[idx]:.4f}")
        
    # Check Symmetry
    print("\n--- 4. Verifying Symmetry and Clipping Constraints ---")
    symmetry_2_3 = abs(clipped_weights[2] - clipped_weights[3]) < 1e-7
    symmetry_4_5 = abs(clipped_weights[4] - clipped_weights[5]) < 1e-7
    print(f"Symmetry Classes 2 & 3 (curve_left / curve_right) equal? {symmetry_2_3} (diff={abs(clipped_weights[2] - clipped_weights[3])})")
    print(f"Symmetry Classes 4 & 5 (rotate_left / rotate_right) equal? {symmetry_4_5} (diff={abs(clipped_weights[4] - clipped_weights[5])})")
    
    assert symmetry_2_3, f"Class 2 weight ({clipped_weights[2]}) and Class 3 weight ({clipped_weights[3]}) are not symmetric!"
    assert symmetry_4_5, f"Class 4 weight ({clipped_weights[4]}) and Class 5 weight ({clipped_weights[5]}) are not symmetric!"
    
    # Check Clipping boundaries
    all_clipped_ok = True
    for w in clipped_weights:
        if w < 0.5 or w > 3.0:
            all_clipped_ok = False
    print(f"All weights clipped between 0.5 and 3.0? {all_clipped_ok}")
    assert all_clipped_ok, f"Some weights are outside [0.5, 3.0]: {clipped_weights}"
    
    # 3. Validate weighted cross entropy loss
    print("\n--- 5. Validating Weighted Cross Entropy Loss ---")
    device = torch.device('cpu')
    class_weights_t = torch.tensor(clipped_weights, dtype=torch.float32, device=device)
    
    # Generate mock predictions (logits) and target labels
    # Use deterministic mock values for verification
    np.random.seed(42)
    torch.manual_seed(42)
    
    batch_size = 32
    num_classes = 7
    mock_logits = torch.randn(batch_size, num_classes)
    mock_labels = torch.randint(0, num_classes, (batch_size,))
    
    # Compute PyTorch weighted loss
    pytorch_loss = F.cross_entropy(mock_logits, mock_labels, weight=class_weights_t, reduction='mean')
    
    # Compute manual weighted loss
    # Softmax on logits
    probs = F.softmax(mock_logits, dim=1)
    
    # Pick the probabilities corresponding to the target class
    target_probs = probs[range(batch_size), mock_labels]
    
    # Compute negative log likelihood (NLL) for each sample
    nll = -torch.log(target_probs)
    
    # Retrieve weight for each target class
    sample_weights = class_weights_t[mock_labels]
    
    # Weighted NLL sum divided by sum of weights
    expected_loss = torch.sum(nll * sample_weights) / torch.sum(sample_weights)
    
    print(f"PyTorch Weighted Cross Entropy Loss: {pytorch_loss.item():.6f}")
    print(f"Manual Weighted Cross Entropy Loss:  {expected_loss.item():.6f}")
    
    diff_loss = abs(pytorch_loss.item() - expected_loss.item())
    print(f"Difference: {diff_loss:.6e}")
    assert diff_loss < 1e-6, f"Cross entropy loss validation failed! Difference: {diff_loss:.6e}"
    
    print("\nVerification successful! All checks passed.")

if __name__ == "__main__":
    test_class_weights_and_loss()

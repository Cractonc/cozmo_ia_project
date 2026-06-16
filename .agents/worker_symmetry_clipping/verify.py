import numpy as np
import torch

def calculate_weights_mock(action_classes):
    counts = np.bincount(action_classes, minlength=7).astype(np.float64)
    print("Initial counts:", counts)
    
    # Symétrie Forcée (Classes 2/3 et 4/5)
    avg_2_3 = (counts[2] + counts[3]) / 2.0
    counts[2] = avg_2_3
    counts[3] = avg_2_3
    
    avg_4_5 = (counts[4] + counts[5]) / 2.0
    counts[4] = avg_4_5
    counts[5] = avg_4_5
    
    counts = np.maximum(counts, 1)
    total_samples = len(action_classes)
    raw_weights = total_samples / (7.0 * counts)
    
    # Clipping [0.5, 3.0]
    clipped_weights = np.clip(raw_weights, 0.5, 3.0)
    
    class_weights = torch.tensor(clipped_weights, dtype=torch.float32)
    return counts, raw_weights, clipped_weights, class_weights

# Let's test with a mock action_classes array
# Classes: 0, 1, 2, 3, 4, 5, 6
# Total samples: 14
# Counts: 0: 2, 1: 2, 2: 1, 3: 3, 4: 2, 5: 0, 6: 4
action_classes = [0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 6, 6, 6, 6]
counts, raw_weights, clipped_weights, class_weights = calculate_weights_mock(action_classes)

print("Averaged counts:", counts)
print("Raw weights:", raw_weights)
print("Clipped weights:", clipped_weights)
print("Class weights tensor:", class_weights)

# Let's assert correctness of calculations:
# Expected:
# counts initially: [2, 2, 1, 3, 2, 0, 4]
# avg_2_3 = (1 + 3) / 2 = 2.0 -> counts[2]=2.0, counts[3]=2.0
# avg_4_5 = (2 + 0) / 2 = 1.0 -> counts[4]=1.0, counts[5]=1.0
# After symmetry: counts = [2, 2, 2, 2, 1, 1, 4]
# counts = max(counts, 1) -> [2, 2, 2, 2, 1, 1, 4] (all >= 1)
# raw_weights = 14 / (7 * counts) = 2.0 / counts
# raw_weights = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.5]
# clipped_weights (range [0.5, 3.0]): [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.5]
# Let's check:
assert np.allclose(counts, [2., 2., 2., 2., 1., 1., 4.])
assert np.allclose(raw_weights, [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.5])
assert np.allclose(clipped_weights, [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.5])
assert class_weights.dtype == torch.float32

print("All assertions passed successfully!")

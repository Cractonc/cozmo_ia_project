# Empirical Verification & Challenge Report: Class Weights Implementation

This report documents the verification, stress-testing, and correctness validation of the modified class weights implementation in `/home/adrien/cozmo_ia_project/train.py`.

---

## 1. Observation

### Code Under Investigation
The class weights calculation logic in `/home/adrien/cozmo_ia_project/train.py` (lines 152 to 174):
```python
    if args.mode == "discrete":
        counts = np.bincount(train_dataset.action_classes, minlength=7).astype(np.float64)
        
        # Symétrie Forcée (Classes 2/3 et 4/5)
        avg_2_3 = (counts[2] + counts[3]) / 2.0
        counts[2] = avg_2_3
        counts[3] = avg_2_3
        
        avg_4_5 = (counts[4] + counts[5]) / 2.0
        counts[4] = avg_4_5
        counts[5] = avg_4_5
        
        counts = np.maximum(counts, 1)
        total_samples = len(train_dataset)
        raw_weights = total_samples / (7.0 * counts)
        
        # Clipping [0.5, 3.0]
        clipped_weights = np.clip(raw_weights, 0.5, 3.0)
        
        device = next(model.parameters()).device
        class_weights = torch.tensor(clipped_weights, dtype=torch.float32, device=device)
        print("Calculated class weights (Method C):", clipped_weights)
```

### Empirical Results
Running the verification script `verify_class_weights.py` on the actual training split of `training_data/` yielded the following output:
```
--- 1. Loading Training Dataset ---
[train] Loading 12 sessions for discrete policy...
[train] Loaded 12224 samples (approx 10.2 minutes).
Saved discrete normalization stats to models/norm_stats_discrete_verify.json
Loaded train dataset with 12224 samples.

--- 2. Class Counts in Training Split ---
Class 0 (forward): 7462 samples
Class 1 (stop): 262 samples
Class 2 (curve_left): 1537 samples
Class 3 (curve_right): 1328 samples
Class 4 (rotate_left): 590 samples
Class 5 (rotate_right): 228 samples
Class 6 (backward): 817 samples

--- 3. Class Weights Calculation ---
Class 0 (forward): raw=0.2340, clipped=0.5000
Class 1 (stop): raw=6.6652, clipped=3.0000
Class 2 (curve_left): raw=1.2190, clipped=1.2190
Class 3 (curve_right): raw=1.2190, clipped=1.2190
Class 4 (rotate_left): raw=4.2696, clipped=3.0000
Class 5 (rotate_right): raw=4.2696, clipped=3.0000
Class 6 (backward): raw=2.1374, clipped=2.1374

--- 4. Verifying Symmetry and Clipping Constraints ---
Symmetry Classes 2 & 3 (curve_left / curve_right) equal? True (diff=0.0)
Symmetry Classes 4 & 5 (rotate_left / rotate_right) equal? True (diff=0.0)
All weights clipped between 0.5 and 3.0? True

--- 5. Validating Weighted Cross Entropy Loss ---
PyTorch Weighted Cross Entropy Loss: 2.346710
Manual Weighted Cross Entropy Loss:  2.346710
Difference: 2.384186e-07
```

---

## 2. Logic Chain

1. **Symmetry Check**:
   - Class 2 (`curve_left`) count = 1537, Class 3 (`curve_right`) count = 1328.
   - Enforced symmetric count = `(1537 + 1328) / 2 = 1432.5`.
   - Raw weights for both: `12224 / (7 * 1432.5) = 1.2190`.
   - Clipped weights for both: `1.2190`. This confirms symmetry holds for classes 2 & 3.
   - Class 4 (`rotate_left`) count = 590, Class 5 (`rotate_right`) count = 228.
   - Enforced symmetric count = `(590 + 228) / 2 = 409.0`.
   - Raw weights for both: `12224 / (7 * 409) = 4.2696`.
   - Clipped weights for both: `3.0000` (due to the upper boundary of 3.0). This confirms symmetry holds for classes 4 & 5.

2. **Clipping Check**:
   - Class 0 (`forward`) raw weight = `0.2340`. Clipped value = `0.5000` (lower boundary of 0.5).
   - Class 1 (`stop`) raw weight = `6.6652`. Clipped value = `3.0000` (upper boundary of 3.0).
   - All other classes fall within `[0.5, 3.0]` or are clipped to the boundaries. This confirms the clipping constraint is correctly applied.

3. **Loss Integration**:
   - PyTorch's `F.cross_entropy(..., weight=class_weights)` corresponds to the mathematically expected weighted cross-entropy formula:
     $$\text{loss} = \frac{\sum_i w_{y_i} \cdot L_i}{\sum_i w_{y_i}}$$
     where $L_i = -\log(P_{y_i})$.
   - The comparison between PyTorch's native loss and our manual formula resulted in an absolute difference of $2.38 \times 10^{-7}$, confirming PyTorch utilizes the weights exactly as expected.

---

## 3. Caveats

- **Binomial/Augmentation Variance**: The weights are computed statically before training begins. However, the data augmentation (`ACTION_FLIP_MAP` with a 50% horizontal flip probability) randomly changes the actual class labels during training. The actual class count in any single batch or epoch might slightly fluctuate around the expectation. Nevertheless, because the weights are fixed across training, using the mathematical expectation (enforced symmetry) is the optimal strategy.
- **Lower/Upper Clipping Bounds**: The clipping values `0.5` and `3.0` are hardcoded hyper-parameters. If the class imbalance becomes extremely severe, a larger upper limit or smaller lower limit might be needed, but the current bounds prevent gradient instability.

---

## 4. Conclusion

The modified class weights implementation in `train.py` works **exactly as expected**.
- Forced symmetry resolves the physical direction imbalances for curve turns (classes 2/3) and pivot rotations (classes 4/5).
- Clipping is correctly applied at the bounds of `0.5` and `3.0`.
- The cross-entropy loss weights are applied correctly.
- **Overall risk assessment**: **LOW**

---

## 5. Verification Method

To independently re-verify, run the following test files from the workspace root:

```bash
# 1. Run the dynamic class weights & loss validation script on training split
./venv/bin/python verify_class_weights.py

# 2. Run the robustness/edge-case unit tests
./venv/bin/python test_class_weights_robustness.py
```

---

## 6. Adversarial Challenges & Stress Testing

### Challenge Summary
- **Overall risk assessment**: **LOW**
- **Test Harness status**: Written and executed successfully. All robustness tests passed.

### Challenges

#### [Low] Challenge 1: Empty Dataset / Zero Samples
- **Assumption challenged**: The dataset has at least one sample in each class.
- **Attack scenario**: If the dataset contains no samples of a particular class, a division by zero could occur in `raw_weights = total_samples / (7.0 * counts)`.
- **Blast radius**: Python runtime error (ZeroDivisionError) during training startup.
- **Mitigation**: The code uses `counts = np.maximum(counts, 1)`, which replaces any zero count with `1.0`. Thus, division by zero is impossible. The resulting raw weight becomes `total / 7` which is clipped to `3.0`. This was verified in unit test `test_zero_counts`.

#### [Low] Challenge 2: Dynamic Augmentation Discrepancy
- **Assumption challenged**: The static class counts represent the true class distribution during training.
- **Attack scenario**: Horizontal flips change classes dynamically during `__getitem__`.
- **Blast radius**: Slight mismatch between the computed static weights and the actual augmented frequencies.
- **Mitigation**: By enforcing symmetry (`avg_2_3` and `avg_4_5`), the static counts are mathematically set to the exact expected values of the counts after 50% probability flipping. This matches the true expected distribution during training.

### Stress Test Results

- **Scenario 1**: Normal counts mapping `[100, 200, 150, 250, 50, 150, 100]` -> Expected symmetry and clipping applied -> **PASS**
- **Scenario 2**: Zero counts for all classes -> Division by zero prevented, weights default safely -> **PASS**
- **Scenario 3**: Extreme skew (1 sample in class 0, 1 million in class 1) -> Weights correctly bound to `[0.5, 3.0]` -> **PASS**
- **Scenario 4**: Single count in curve_left, 0 in curve_right -> Symmetry averages count to 0.5, then safety bound raises count to 1.0, raw weight calculated correctly -> **PASS**

### Unchallenged Areas
- None. The class weight calculation logic has been fully stress-tested and verified.

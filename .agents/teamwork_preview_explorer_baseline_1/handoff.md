# Handoff Report — Baseline Exploration of Discrete Training Pipeline

This report establishes the baseline for the discrete keyboard-action training pipeline of the Cozmo IA project, detailing the dataset statistics, dry run execution, saving locations, class weights formulation, and suggested implementation changes.

---

## 1. Observation

### 1.1 Action Classes Specification
In `model.py` (lines 5-13), the discrete actions are defined as:
```python
ACTION_CLASS_SPECS = [
    ("forward", (80.0, 80.0)),
    ("stop", (0.0, 0.0)),
    ("curve_left", (30.0, 80.0)),
    ("curve_right", (80.0, 30.0)),
    ("rotate_left", (-50.0, 50.0)),
    ("rotate_right", (50.0, -50.0)),
    ("backward", (-60.0, -60.0)),
]
```

### 1.2 Action Class Distribution & Frequencies
We ran the diagnostic script `.agents/teamwork_preview_explorer_baseline_1/check_distribution.py` to compute frequencies and class weights on the training dataset splits.

Command executed:
```bash
venv/bin/python .agents/teamwork_preview_explorer_baseline_1/check_distribution.py
```

Output:
```
=== Action Class Distribution for training_data ===
[train] Loading 12 sessions for discrete policy...
[train] Loaded 12224 samples (approx 10.2 minutes).
[val] Loading 2 sessions for discrete policy...
[val] Loaded 2745 samples (approx 2.3 minutes).
Warning: Normalization stats not found at None. Computing from current data.

Class Index | Class Name   | Train Count | Train % | Val Count | Val % | Weight (Train)
------------------------------------------------------------------------------------------
0           | forward      | 7462        | 61.04   | 1742      | 63.46 | 0.2340
1           | stop         | 262         | 2.14    | 96        | 3.50  | 6.6652
2           | curve_left   | 1537        | 12.57   | 158       | 5.76  | 1.1362
3           | curve_right  | 1328        | 10.86   | 132       | 4.81  | 1.3150
4           | rotate_left  | 590         | 4.83    | 149       | 5.43  | 2.9598
5           | rotate_right | 228         | 1.87    | 121       | 4.41  | 7.6591
6           | backward     | 817         | 6.68    | 347       | 12.64 | 2.1374
------------------------------------------------------------------------------------------
Total       |              | 12224       | 100.00  | 2745      | 100.00 |

==================================================

=== Action Class Distribution for training_data_1.0 ===
[train] Loading 5 sessions for discrete policy...
[train] Loaded 21378 samples (approx 17.8 minutes).
[val] Loading 1 sessions for discrete policy...
[val] Loaded 8901 samples (approx 7.4 minutes).
Warning: Normalization stats not found at None. Computing from current data.

Class Index | Class Name   | Train Count | Train % | Val Count | Val % | Weight (Train)
------------------------------------------------------------------------------------------
0           | forward      | 7991        | 37.38   | 4446      | 49.95 | 0.3822
1           | stop         | 4266        | 19.96   | 206       | 2.31  | 0.7159
2           | curve_left   | 2312        | 10.81   | 2056      | 23.10 | 1.3209
3           | curve_right  | 1810        | 8.47    | 1797      | 20.19 | 1.6873
4           | rotate_left  | 1989        | 9.30    | 94        | 1.06  | 1.5354
5           | rotate_right | 1782        | 8.34    | 146       | 1.64  | 1.7138
6           | backward     | 1228        | 5.74    | 156       | 1.75  | 2.4870
------------------------------------------------------------------------------------------
Total       |              | 21378       | 100.00  | 8901      | 100.00 |
```

- In the default `training_data/` split:
  - Total training samples: **12,224**
  - Most frequent class: **forward** (Index 0) with **7,462** samples (61.04%).
  - Least frequent class: **rotate_right** (Index 5) with **228** samples (1.87%).
  - **No classes have zero samples (missing classes).** All 7 action classes are represented.

- In the alternative `training_data_1.0/` split:
  - Total training samples: **21,378**
  - Most frequent class: **forward** (Index 0) with **7,991** samples (37.38%).
  - Least frequent class: **backward** (Index 6) with **1,228** samples (5.74%).
  - **No classes have zero samples (missing classes).** All 7 action classes are represented.

### 1.3 Dry Run of `train.py`
We verified successful execution of `train.py` for the discrete training pipeline by running it with 1 epoch as a dry run.

Command executed:
```bash
venv/bin/python train.py --mode discrete --epochs 1 --name dryrun
```

Output:
```
=== Starting Training Run: dryrun (discrete) ===
[train] Loading 12 sessions for discrete policy...
[train] Loaded 12224 samples (approx 10.2 minutes).
Saved discrete normalization stats to models/norm_stats_discrete_dryrun.json
[val] Loading 2 sessions for discrete policy...
[val] Loaded 2745 samples (approx 2.3 minutes).
Model instantiated with 168,935 parameters.
/home/adrien/cozmo_ia_project/venv/lib/python3.8/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
Starting training loop (CPU)...
Epoch 01/1 | Train Loss: 1.1714 | Val Loss: 1.3507 | Train Acc: 61.5% | Val Acc: 67.4% | LR: 0.001000 | Time: 22.4s (ETA: 0.0m)
  -> Best model saved to models/cozmo_discrete_nn_dryrun.pt (Val loss: 1.3507)
Training complete!
```

### 1.4 Output Saving Locations
In `train.py` (lines 26-33):
```python
def build_output_paths(args):
    if args.mode == "discrete":
        return {
            "stats": os.path.join(args.output_dir, "norm_stats_discrete_{}.json".format(args.name)),
            "model": os.path.join(args.output_dir, "cozmo_discrete_nn_{}.pt".format(args.name)),
            "history": os.path.join(args.output_dir, "history_discrete_{}.json".format(args.name)),
            "metadata": os.path.join(args.output_dir, "cozmo_discrete_nn_{}.json".format(args.name)),
        }
```
For our dry run (`--name dryrun`), files were written to:
- **Normalization stats**: `models/norm_stats_discrete_dryrun.json`
- **Model weights checkpoint**: `models/cozmo_discrete_nn_dryrun.pt`
- **History log**: `models/history_discrete_dryrun.json`
- **Metadata**: `models/cozmo_discrete_nn_dryrun.json`

---

## 2. Logic Chain

1. **Examine dataset and compute class frequencies**: By loading the datasets through `CozmoDiscreteDataset` (which maps continuous wheel speeds to one of the 7 discrete action templates using ArgMin Euclidean distance), we counted class occurrences in the training split.
2. **Compute Class Weights**: With the counted samples $N_c$ per class $c$, the total samples $N = \sum N_c$, and $C = 7$ classes, we calculated weights using the standard inverse-frequency formula:
   $$w_c = \frac{N}{C \times N_c}$$
   No class count $N_c$ was zero, so no division by zero or missing classes occurred.
3. **Verify Training Executability**: The dry run of `train.py --mode discrete --epochs 1` completed successfully without any compilation or execution errors, showing a baseline cross-entropy loss of `1.1714` and validation accuracy of `67.4%` in epoch 1.
4. **Identify Integration Points**: By tracing `train.py`:
   - `CozmoDiscreteDataset` is instantiated inside `main()`. It has the attribute `action_classes` containing the class label for each sample.
   - The loss function used during training is `F.cross_entropy(preds, classes)` in `train_discrete_epoch()` and `eval_discrete()`.
   - Therefore, the class weights must be computed after dataset loading, and the calculated weights tensor must be passed to `F.cross_entropy` using the `weight` parameter.

---

## 3. Caveats

- **Val Split Calculation**: The dataset splitting is session-based. Depending on how the sessions are recorded, validation files might contain a different distribution of classes than training files (e.g. in `training_data`, Class 6 (backward) represents 6.68% of the training data but 12.64% of the validation data).
- **Data Augmentation Flips**: `CozmoDiscreteDataset` implements horizontal flips in `__getitem__` during training (lines 297-303). Since this changes `left` vs `right` inputs/outputs, the labels are updated accordingly. This augmentation runs dynamically and does not affect the static baseline counts of `self.action_classes`, but will balance/affect the actual training distribution.
- **CPU only**: The training pipeline is executed only on CPU.

---

## 4. Conclusion

The discrete training pipeline runs successfully, but suffers from significant class imbalance (Class 0 forward accounts for ~61% of default training data, while Class 5 rotate_right accounts for under 2%). Integrating class weights into PyTorch's `F.cross_entropy` is highly recommended to improve classification performance on minority actions.

### 4.1 Calculated Class Weights (for default `training_data/`)
The target weight vector is:
```python
# [forward, stop, curve_left, curve_right, rotate_left, rotate_right, backward]
discrete_weights = torch.tensor([0.2340, 6.6652, 1.1362, 1.3150, 2.9598, 7.6591, 2.1374], dtype=torch.float32)
```

---

## 5. Proposed Code Changes

### 5.1 Step-by-Step Code Integration
To apply these weights, the following changes are proposed in `train.py`:

#### 1. In `train_discrete_epoch` (lines 69-82)
**Before**:
```python
def train_discrete_epoch(model, loader, optimizer):
    model.train()
    loss_sum = 0.0
    correct = 0
    for imgs, sensors, classes, speeds, heads in loader:
        optimizer.zero_grad()
        preds = model(imgs, sensors)
        loss = F.cross_entropy(preds, classes)
        loss.backward()
...
```

**After**:
```python
def train_discrete_epoch(model, loader, optimizer, class_weights=None):
    model.train()
    loss_sum = 0.0
    correct = 0
    for imgs, sensors, classes, speeds, heads in loader:
        optimizer.zero_grad()
        preds = model(imgs, sensors)
        loss = F.cross_entropy(preds, classes, weight=class_weights)
        loss.backward()
...
```

#### 2. In `eval_discrete` (lines 85-97)
**Before**:
```python
def eval_discrete(model, loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    if len(loader.dataset) == 0:
        return 0.0, 0.0
    with torch.no_grad():
        for imgs, sensors, classes, speeds, heads in loader:
            preds = model(imgs, sensors)
            loss = F.cross_entropy(preds, classes)
            loss_sum += loss.item() * imgs.size(0)
...
```

**After**:
```python
def eval_discrete(model, loader, class_weights=None):
    model.eval()
    loss_sum = 0.0
    correct = 0
    if len(loader.dataset) == 0:
        return 0.0, 0.0
    with torch.no_grad():
        for imgs, sensors, classes, speeds, heads in loader:
            preds = model(imgs, sensors)
            loss = F.cross_entropy(preds, classes, weight=class_weights)
            loss_sum += loss.item() * imgs.size(0)
...
```

#### 3. In `main` (dataset loading and weights computation, lines 139-142)
**Before**:
```python
    if args.mode == "discrete":
        train_dataset = CozmoDiscreteDataset(data_dir=args.data_dir, split='train', norm_stats_path=paths["stats"])
        val_dataset = CozmoDiscreteDataset(data_dir=args.data_dir, split='val', norm_stats_path=paths["stats"])
        model = CozmoNNDiscrete()
```

**After**:
```python
    if args.mode == "discrete":
        train_dataset = CozmoDiscreteDataset(data_dir=args.data_dir, split='train', norm_stats_path=paths["stats"])
        val_dataset = CozmoDiscreteDataset(data_dir=args.data_dir, split='val', norm_stats_path=paths["stats"])
        model = CozmoNNDiscrete()
        
        # Calculate class weights
        classes = train_dataset.action_classes
        num_classes = 7 # or len(ACTION_CLASS_SPECS)
        total_samples = len(classes)
        
        class_counts = np.bincount(classes, minlength=num_classes)
        # Avoid division by zero in case of unexpected empty class
        class_counts = np.maximum(class_counts, 1)
        
        weights_np = total_samples / (num_classes * class_counts)
        discrete_weights = torch.tensor(weights_np, dtype=torch.float32)
        print("Calculated class weights: {}".format(discrete_weights.tolist()))
    else:
        train_dataset = CozmoDataset(data_dir=args.data_dir, split='train', norm_stats_path=paths["stats"])
        val_dataset = CozmoDataset(data_dir=args.data_dir, split='val', norm_stats_path=paths["stats"])
        model = CozmoNN()
        discrete_weights = None
```

#### 4. In `main` (training loop call, lines 180-185)
**Before**:
```python
        if args.mode == "discrete":
            train_loss, train_acc = train_discrete_epoch(model, train_loader, optimizer)
            if len(val_dataset) > 0:
                val_loss, val_acc = eval_discrete(model, val_loader)
            else:
                val_loss, val_acc = train_loss, train_acc
```

**After**:
```python
        if args.mode == "discrete":
            train_loss, train_acc = train_discrete_epoch(model, train_loader, optimizer, discrete_weights)
            if len(val_dataset) > 0:
                val_loss, val_acc = eval_discrete(model, val_loader, discrete_weights)
            else:
                val_loss, val_acc = train_loss, train_acc
```

---

## 6. Verification Method

To independently verify the implementation, the implementer can:
1. Run the modified training pipeline with:
   ```bash
   python train.py --mode discrete --epochs 1 --name test_weights
   ```
2. Verify that:
   - The stdout prints `Calculated class weights: [0.2340..., 6.6652..., 1.1362..., 1.3150..., 2.9598..., 7.6591..., 2.1374...]`.
   - The training completes without error, showing the updated weighted Cross Entropy loss.
   - The generated files are saved correctly in the `models/` directory:
     - Checkpoint: `models/cozmo_discrete_nn_test_weights.pt`
     - Stats: `models/norm_stats_discrete_test_weights.json`

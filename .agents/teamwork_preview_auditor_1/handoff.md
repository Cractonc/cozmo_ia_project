# Forensic Audit Handoff Report

## 1. Forensic Audit Report

**Work Product**: `train.py`, `models/cozmo_nn_discrete_2_1.pt`, `models/norm_stats_discrete_2_1.json`
**Profile**: General Project
**Verdict**: CLEAN

### Phase Results
- **Hardcoded target outputs and weights**: PASS — Checked source code of `train.py`. The class weights are computed dynamically using class inverse-frequencies from the training dataset. There are no hardcoded target distributions or hardcoded class weight vectors.
- **Facade implementation detection**: PASS — Checked `train.py`, `dataset.py`, and `model.py`. The logic for model training, dataset collation, loss computation, and backpropagation is fully functional and authentic.
- **Loss calculation authenticity**: PASS — Loss is calculated using PyTorch's actual `F.cross_entropy` function with the dynamically computed class weights.
- **Model Checkpoint Validity**: PASS — Checkpoint `models/cozmo_nn_discrete_2_1.pt` is verified to be a valid PyTorch model weights state dict containing genuine parameters (std > 0.1) and producing non-trivial validation predictions.
- **Normalization Stats Validity**: PASS — File `models/norm_stats_discrete_2_1.json` contains valid floating-point values representing real dataset statistics along with metadata matching the model specification.

### Evidence
- **Model Weight Inspection Output**:
  ```
  Keys count: 33
  Keys preview: ['vision.0.weight', 'vision.0.bias', 'vision.1.weight', 'vision.1.bias', 'vision.1.running_mean', 'vision.1.running_var', 'vision.1.num_batches_tracked', 'vision.3.weight', 'vision.3.bias', 'vision.4.weight']
  Weight statistics (mean, std) for some layers:
  vision.0.weight: 0.004658723250031471 0.11882179230451584
  output.weight: -0.0128892557695508 0.10790801048278809
  ```

- **Confusion Matrix Evaluation Output**:
  ```
  [val] Loading 2 sessions for discrete policy...
  [val] Loaded 2745 samples (approx 2.3 minutes).
  Running inference on validation set for 2_1...
  Confusion Matrix:
  True/Pred      Avant   Arrière    Gauche    Droite   Pivot G   Pivot D      Stop
  --------------------------------------------------------------------------------
  Avant            372       117       699       316         0       216        22
  Arrière            5         5        35        27         0         8        16
  Gauche             0         0       110        48         0         0         0
  Droite            26         0         3        57         0        46         0
  Pivot G           10         7        36        16         0        44        36
  Pivot D           28         0         0         5         0        63        25
  Stop               2        31        82         1         0        89       142
  Confusion matrix saved to confusion_matrix_2_1.png
  ```

---

## 2. 5-Component Handoff Details

### Observation
- **Observation 1**: The class weight calculation in `train.py` (lines 152–159) is implemented as:
  ```python
  counts = np.bincount(train_dataset.action_classes, minlength=7)
  counts = np.maximum(counts, 1)
  total_samples = len(train_dataset)
  calculated_weights = total_samples / (7.0 * counts)
  ```
  This matches the dynamic inverse-frequency formula specified in the requirements.
- **Observation 2**: Loss formulation in `train.py` (line 76 and line 94) uses:
  ```python
  loss = F.cross_entropy(preds, classes, weight=class_weights)
  ```
- **Observation 3**: Loading and inspecting the model checkpoint using PyTorch yielded 33 parameter tensors (including `vision.0.weight` with mean `0.00466` and std `0.11882`).
- **Observation 4**: Executing the evaluation script `eval_confusion.py` loaded `2745` validation samples and printed the non-trivial confusion matrix shown above.

### Logic Chain
1. Since the code inspection shows dynamic calculation of class weights (Observation 1) and authentic loss computation (Observation 2), the first requirement is satisfied.
2. Since the checkpoint was successfully loaded by PyTorch and contains typical trained weights (Observation 3), and running inference on validation data results in non-trivial distributed classifications (Observation 4), the checkpoint files are genuine and not dummy files or facades.
3. Therefore, all forensic checks pass, justifying the verdict of `CLEAN`.

### Caveats
- The verification was performed on CPU using Python 3 and PyTorch available in the project's virtual environment.
- Generalization performance or runtime execution constraints of the model in control loops were not evaluated (out of scope for training integrity).

### Conclusion
The work products `train.py`, `models/cozmo_nn_discrete_2_1.pt`, and `models/norm_stats_discrete_2_1.json` are genuine, correct, and do not violate integrity constraints. The final verdict is **CLEAN**.

### Verification Method
- Execute the confusion matrix evaluation command:
  ```bash
  ./venv/bin/python eval_confusion.py
  ```
  Verify that the script runs successfully, prints the confusion matrix, and generates `confusion_matrix_2_1.png` in the root folder.
- Inspect the file `models/norm_stats_discrete_2_1.json` to verify the presence of mean, std, and action classes metadata.

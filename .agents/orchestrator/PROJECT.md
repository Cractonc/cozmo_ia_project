# Project: Cozmo Imitation Learning Class Balancing

## Architecture
- `dataset.py`: Contains `CozmoDiscreteDataset` which loads training/validation sessions from `.npz` files in `training_data/`.
- `train.py`: Contains the main training loop. The model `CozmoNNDiscrete` is trained using PyTorch Cross Entropy Loss (`F.cross_entropy`).
- `eval_confusion.py`: Generates the confusion matrix.
- `model.py`: Defines the neural network architectures.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Baseline & Exploration | Verify validation split, check class distributions in training files. | None | DONE |
| 2 | Dynamic Class Weights & Loss | Implement class weights calculation and integrate it into cross entropy loss in `train.py`. | M1 | DONE |
| 3 | Model 2.1 Training | Train the model `2_1` using the updated training script (15 epochs). | M2 | DONE |
| 4 | Validation Confusion Matrix | Generate and verify the confusion matrix `confusion_matrix_2_1.png` on the validation set. | M3 | DONE |
| 5 | Forced Symmetry & Clipping | Modify class weights logic to average mirrors (2/3 and 4/5) and clip weights between 0.5 and 3.0. | M2 | DONE |
| 6 | Model 2.2 Training | Train the model `2_2` for 15 epochs using the new weights. | M5 | DONE |
| 7 | Validation Confusion Matrix 2.2 | Generate the confusion matrix for 2.2 at the specified path and verify it. | M6 | DONE |

## Interface Contracts
### Class Weights Logic (Model 2.2)
- Average class indices 2 & 3: `avg_2_3 = (counts[2] + counts[3]) / 2.0` (set both counts to `avg_2_3`)
- Average class indices 4 & 5: `avg_4_5 = (counts[4] + counts[5]) / 2.0` (set both counts to `avg_4_5`)
- Raw weight formula: `raw_weights = total_samples / (7.0 * counts)`
- Weight clipping: `clipped_weights = np.clip(raw_weights, 0.5, 3.0)`
- Pass `clipped_weights` to PyTorch CrossEntropyLoss.


## Code Layout
- `dataset.py` - Dataset loaders.
- `train.py` - Training loops and CLI entry point.
- `model.py` - Network architectures.
- `eval_confusion.py` - Confusion matrix generation script.
- `models/` - Target directory for model files.

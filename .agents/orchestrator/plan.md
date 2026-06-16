# Plan - Cozmo Class Balancing (Model 2.2)

## Objective
To implement forced symmetry on class indices 2/3 and 4/5, apply clipping to the loss weights, train model 2.2 for 15 epochs, and generate the final confusion matrix under the specified path.

## Steps
1. **Implement Forced Symmetry and Clipping**:
   - Spawn a Worker to modify `train.py` around line 153.
   - Cast `counts` to float, average the mirror classes: counts[2]/counts[3] and counts[4]/counts[5].
   - Compute `raw_weights = total_samples / (7.0 * counts)`.
   - Apply `np.clip(raw_weights, 0.5, 3.0)`.
   - Ensure these weights are converted to PyTorch tensor and passed to `F.cross_entropy` in training/eval functions.
2. **Train Model 2.2**:
   - Run `./venv/bin/python train.py --mode discrete --name 2_2 --epochs 15`.
   - Verify that `models/cozmo_discrete_nn_2_2.pt` is generated.
3. **Generate Confusion Matrix for 2.2**:
   - Modify or parameterize `eval_confusion.py` to point to model 2.2 and output the matrix to `/home/adrien/.gemini/antigravity/brain/5418179d-c86f-4b55-9fc9-325fac478202/confusion_matrix_2_2.png`.
   - Run the script and confirm the image file is created.
4. **Verify and Audit**:
   - Use Challenger to verify predictions and check diagonal dominance.
   - Use Forensic Auditor to ensure clean implementation.


# Handoff Report — Model 2.1 Verification

This report documents the empirical verification of the dynamic class weights implementation and the trained model 2.1 (`models/cozmo_nn_discrete_2_1.pt`) predictions.

---

## 1. Observation

### Command Execution
To perform this validation, we ran two custom verification scripts:
1. `verify_model.py` — Evaluated `models/cozmo_nn_discrete_2_1.pt` on both the train and validation splits.
2. `verify_all_models.py` — Evaluated and compared all three available discrete models (`cozmo_nn_discrete_2_1.pt`, `cozmo_discrete_nn_dryrun.pt`, and `cozmo_discrete_nn_test_weights.pt`).

Commands used:
```bash
PYTHONPATH=. ./venv/bin/python .agents/teamwork_preview_challenger_1/verify_model.py
PYTHONPATH=. ./venv/bin/python .agents/teamwork_preview_challenger_1/verify_all_models.py
```

### Script Output (Model 2.1 Statistics)
The validation run on Model 2.1 returned:
* **Train Dataset Size**: 12,224 samples
* **Val Dataset Size**: 2,745 samples
* **Calculated Static Class Weights**:
  * `forward` (class 0): 0.2340
  * `stop` (class 1): 6.6652
  * `curve_left` (class 2): 1.1362
  * `curve_right` (class 3): 1.3150
  * `rotate_left` (class 4): 2.9598
  * `rotate_right` (class 5): 7.6591
  * `backward` (class 6): 2.1374
* **Val Set Accuracy**: 27.3% (Train Accuracy: 34.4%)
* **Val Set Predictions Distribution**:
  * `forward`: 443 predictions (16.2%)
  * `stop`: 160 predictions (5.8%)
  * `curve_left`: 965 predictions (35.2%)
  * `curve_right`: 470 predictions (17.2%)
  * `rotate_left`: 0 predictions (0.0%)
  * `rotate_right`: 466 predictions (17.0%)
  * `backward`: 241 predictions (8.8%)

### Corrected Validation Confusion Matrix (Model 2.1)
```
True/Pred            forward         stop   curve_left  curve_right  rotate_left  rotate_right     backward
------------------------------------------------------------------------------------------------------------
forward                  372          117          699          316            0           216           22
stop                       5            5           35           27            0             8           16
curve_left                 0            0          110           48            0             0            0
curve_right               26            0            3           57            0            46            0
rotate_left               10            7           36           16            0            44           36
rotate_right              28            0            0            5            0            63           25
backward                   2           31           82            1            0            89          142
```

### Comparison with Other Models (Val Set Accuracy & Prediction Counts)
* **Dryrun Model (`cozmo_discrete_nn_dryrun.pt`)**:
  * Accuracy: 67.6%
  * Predictions: 2,418 `forward` (88.3%), 327 `backward` (11.9%), other 5 classes = 0.
* **Test Weights Model (`cozmo_discrete_nn_test_weights.pt`)**:
  * Accuracy: 23.3%
  * Predictions: 412 `forward` (15.0%), 41 `stop` (1.5%), 617 `curve_left` (22.5%), 1,284 `curve_right` (46.9%), 0 `rotate_left` (0.0%), 229 `rotate_right` (8.4%), 162 `backward` (5.9%).

---

## 2. Logic Chain

1. **Inference Functionality**: `verify_model.py` and `verify_all_models.py` loaded `models/cozmo_nn_discrete_2_1.pt` and executed forward passes on 2,745 validation samples without error. This proves that the model runs inference successfully.
2. **Prediction Diversity**: The prediction counts for Model 2.1 are: `forward` (16.2%), `stop` (5.8%), `curve_left` (35.2%), `curve_right` (17.2%), `rotate_left` (0.0%), `rotate_right` (17.0%), and `backward` (8.8%). 
   * This confirms that the model does **not** collapse to a single class (unlike the Dryrun model, which only predicts `forward` and `backward`).
   * The leftmost columns (`curve_left` and `curve_right`) are active with 965 and 470 predictions, respectively.
   * The rightmost columns (`rotate_right`, `stop`, and `backward`) are active with 466, 160, and 241 predictions, respectively.
3. **Swapped Labels Bug in `eval_confusion.py`**:
   * `eval_confusion.py` (line 72) lists: `class_names = ["Avant", "Arrière", "Gauche", "Droite", "Pivot G", "Pivot D", "Stop"]`.
   * This maps index 1 to "Arrière" and index 6 to "Stop".
   * However, `ACTION_CLASS_SPECS` in `model.py` (lines 5-13) and `norm_stats_discrete_2_1.json` defines class 1 as `"stop"` and class 6 as `"backward"`.
   * Therefore, `eval_confusion.py` plotted "Arrière" (index 1) for the "stop" predictions, and "Stop" (index 6) for the "backward" predictions. Our corrected confusion matrix fixes this labeling bug.
4. **Complete Collapse of `rotate_left`**:
   * We observed that `rotate_left` (class 4) has exactly 0 predictions on both the training and validation sets in both Model 2.1 and the Test Weights model.
   * During training, 50% random horizontal flipping is performed on images and sensors, which mathematically maps `rotate_left` inputs to `rotate_right` targets (and vice-versa). Consequently, the physical input feature distributions for both classes become symmetric.
   * However, the class weights are computed statically on the *unflipped* dataset. Because `rotate_right` has far fewer original samples (228) than `rotate_left` (590), it is assigned a much higher weight (`7.6591` vs `2.9598`).
   * In a symmetric input space, the loss optimizer minimizes the weighted cross-entropy by heavily favoring the class with the higher weight (`rotate_right`), forcing its output logits to always exceed those of `rotate_left`. As a result, the model never outputs `rotate_left`.

---

## 3. Caveats

* **Simulator/Physical Testing**: The verification was conducted entirely in offline evaluation (on the pre-recorded validation split). Closed-loop physical performance has not been tested.
* **Overfitting / Low Accuracy**: While Model 2.1 shows much better prediction diversity than the Dryrun, its accuracy on the validation set is quite low (27.3%), and the training loss is significantly lower than validation loss, indicating overfitting.
* **Lack of Temporal Context**: The discrete policy predicts actions based on single frames and sensor readings. A single static frame may contain insufficient cues to distinguish rotation directions.

---

## 4. Conclusion

* **Inference and Diversity**: The model successfully runs inference. Dynamic class weights succeeded in preventing collapse to the majority class (`forward`), keeping 6 out of 7 classes highly active, including the curve classes (leftmost) and the stop/backward/rotate_right classes (rightmost).
* **Systematic Bias**: The model completely fails to predict `rotate_left` (0 predictions) due to a methodological conflict between horizontal flip data augmentation and static asymmetric class weights.
* **Actionable Recommendations**:
  1. **Fix `eval_confusion.py` labeling**: Correct the class names array to `["Avant", "Stop", "Gauche", "Droite", "Pivot G", "Pivot D", "Arrière"]` to match the actual model output indices.
  2. **Symmetrize Class Weights**: When using horizontal flip augmentation, the class weights for left/right actions should be symmetrized (averaged) to prevent the loss function from introducing artificial directional bias:
     * `weight(rotate_left) = weight(rotate_right) = average_weight`
     * `weight(curve_left) = weight(curve_right) = average_weight`
     This will activate `rotate_left` and remove the steering bias.

---

## 5. Verification Method

To independently reproduce these findings and verify the model predictions, run the following commands:

```bash
# 1. Run the corrected verification script
PYTHONPATH=. ./venv/bin/python .agents/teamwork_preview_challenger_1/verify_model.py

# 2. View the generated corrected confusion matrix image
# Path: /home/adrien/cozmo_ia_project/.agents/teamwork_preview_challenger_1/confusion_matrix_corrected.png

# 3. View the detailed metrics JSON
# Path: /home/adrien/cozmo_ia_project/.agents/teamwork_preview_challenger_1/validation_results.json
```

*Conditions for Invalidation*:
* Modifying the model file (`models/cozmo_nn_discrete_2_1.pt`).
* Modifying the validation dataset files inside `training_data/`.

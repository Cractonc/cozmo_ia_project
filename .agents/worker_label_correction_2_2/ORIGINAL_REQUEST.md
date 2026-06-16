## 2026-06-16T00:59:31+02:00
Objective: Correct the class label order in `eval_confusion.py` to match `ACTION_CLASS_SPECS` index mapping, and regenerate the confusion matrix for model 2.2.
Input:
- Script to modify: `/home/adrien/cozmo_ia_project/eval_confusion.py`
- Model file: `/home/adrien/cozmo_ia_project/models/cozmo_discrete_nn_2_2.pt`
- Stats file: `/home/adrien/cozmo_ia_project/models/norm_stats_discrete_2_2.json`
- Output image path: `/home/adrien/.gemini/antigravity/brain/5418179d-c86f-4b55-9fc9-325fac478202/confusion_matrix_2_2.png`

Requirements:
1. In `/home/adrien/cozmo_ia_project/eval_confusion.py` (around line 73), verify the class names list order. It must correspond EXACTLY to the indices of `ACTION_CLASS_SPECS` from `model.py`:
   - Index 0: forward -> "Avant"
   - Index 1: stop -> "Stop"
   - Index 2: curve_left -> "Gauche"
   - Index 3: curve_right -> "Droite"
   - Index 4: rotate_left -> "Pivot G"
   - Index 5: rotate_right -> "Pivot D"
   - Index 6: backward -> "Arrière"
   So `class_names = ["Avant", "Stop", "Gauche", "Droite", "Pivot G", "Pivot D", "Arrière"]`.
   If it is currently `["Avant", "Arrière", "Gauche", "Droite", "Pivot G", "Pivot D", "Stop"]` (where Stop and Arrière are swapped), change it back to the correct original order.
2. Run `eval_confusion.py` to regenerate `/home/adrien/.gemini/antigravity/brain/5418179d-c86f-4b55-9fc9-325fac478202/confusion_matrix_2_2.png` with the correct labels.
3. Confirm that the regenerated confusion matrix prints the correct raw and formatted confusion matrix to stdout.
4. Verify that `/home/adrien/.gemini/antigravity/brain/5418179d-c86f-4b55-9fc9-325fac478202/confusion_matrix_2_2.png` exists and is non-empty.

Output:
Write a handoff report at `/home/adrien/cozmo_ia_project/.agents/worker_label_correction_2_2/handoff.md`. Include details of changes, the command run, stdout log showing the corrected formatted table, and confirmation of image generation.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

# Cotton Disease Recognition Training: My Notes & Reflections (Jan 2026)

## 1. What Happened (Initial Observations)
- Training stopped early at epoch 15 (should have gone to 42) because of early stopping.
- Loss and accuracy curves: both train/val loss stayed high and fluctuated, not much improvement. Accuracy was stuck between 30% and 47%.
- Classification report: test accuracy only ~38% (barely better than random for 4 classes). Some classes (like "bacterial_blight") had 0 precision/recall. Only "fussarium_wilt" was detected well (F1: 0.66).

## 2. Why Did This Happen? (Analysis)
- Early stopping: validation loss didn't improve for the set patience period, so training stopped.
- Underfitting: model wasn't learning enough (train/val metrics both poor).
- Class imbalance: some classes not detected at all, so model biased toward more common classes.

## 3. What I Changed (Actions)
- Increased early stopping patience: 15 → 30 (lets model try longer before stopping).
- Lowered initial learning rate: 0.001 → 0.0005 (slower, more stable learning).
- Added class weights to loss: now calculated from training data to help rare classes.
- Increased scheduler patience: 3 → 5 (less aggressive LR drops).

## 4. Parameter Change Table (Before/After)

| Parameter/Setting         | Before         | After         | Why I Changed It / What It Does                  |
|--------------------------|----------------|---------------|-------------------------------------------------|
| early_stop_patience      | 15             | 30            | More epochs to improve before stopping           |
| learning_rate            | 0.001          | 0.0005        | More stable, gradual learning                    |
| class weights in loss    | Not used       | Auto-computed | Helps with class imbalance, rare class recall    |
| scheduler patience       | 3              | 5             | Less aggressive LR reduction, more time to learn |

## 5. What To Try Next (My TODOs)
- Retrain with these changes and watch the new curves and report.
- If still underfitting or class imbalance, try:
  - Stronger data augmentation (flips, rotations, color jitter, etc.)
  - Bigger/more complex model
  - More regularization if overfitting (dropout, L2)
  - Check for bad/mislabeled images in the dataset

## 6. Quick Summary Table

| Issue                  | Evidence                                   | What I Did / Should Do Next           |
|------------------------|--------------------------------------------|---------------------------------------|
| Early stopping         | Flat/oscillating val loss, low patience    | Increased patience to 30              |
| Underfitting           | High loss, low accuracy, low F1            | Lowered LR, consider bigger model     |
| Class imbalance        | 0 recall for some classes                  | Added class weights to loss           |
| Poor generalization    | Low macro/weighted F1, low val accuracy    | Data augmentation, regularization     |

---

# Training Phase 1 — TensorFlow Port of ShunyaNet (Paddy Disease) 🌾

First pass where ShunyaNet was ported to TensorFlow/Keras and trained on the paddy disease dataset. Intent: verify end‑to‑end pipeline, sanity‑check model, and baseline performance before optimization.

---
## What was done
- Ported ShunyaNet to TensorFlow/Keras and wired the full training loop.
- Used the paddy dataset in `Data/PaddyDisease/` with splits: `train/`, `val/`, `test/` and 10 classes.
- Implemented the same augmentation/normalization pipeline used elsewhere (crop/flip/rotation/color jitter/blur), adapted for TF.
- Added checkpoints, history logging, and basic evaluation exports.

## Key Training Settings (Phase 1)
- Framework: TensorFlow/Keras
- Backbone: ShunyaNet (custom CNN)
- Input size: 224×224 RGB
- Batch size: 2
- Epochs: planned ~40+ (early stop enabled)
- Optimizer: AdamW
- LR: 0.001 with ReduceLROnPlateau
- Weight decay: 1e‑5
- Loss: SparseCategoricalCrossentropy
- Early stopping: patience ~12 on `val_loss`
- Augmentation: crop, horizontal/vertical flip, rotation, color jitter, optional blur
- Regularization: DropBlock + weight decay

## Observations (Runtime & Behavior)
- One epoch took ~3 hours due to large dataset + small batch size + heavy augmentations.
- Throughput in TensorFlow (on this setup) was noticeably slower than the PyTorch runs for the same data.
- PyTorch achieved better effective wall‑clock utilization and converged faster under similar settings in later phases.

## Outputs (Where to find)
- Checkpoints 
[checkpoints](checkpoints)

![Confusion Matrix Epoch1](TrainingPhase1-Tensorflow/results/confusion_matrix_epoch_1.png)

> Note: Exact numbers (accuracy/loss) are not restated here to avoid mismatches; consult the artifacts in `output/results/` for the precise metrics from Phase 1.

## Quick Takeaways
- The TensorFlow port works end‑to‑end, but training is slow per epoch (~3h).
- PyTorch later showed better performance/throughput in this project context.
- Next steps: optimize TF input pipeline (tf.data prefetch/cache, fused ops), increase batch size if feasible, and consider mixed precision/XLA where compatible.

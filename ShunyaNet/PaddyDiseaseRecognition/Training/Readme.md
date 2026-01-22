# Paddy Disease Recognition – Consolidated Training Report

This document consolidates all training phases, key results, observations, and next steps for the Paddy Disease Recognition system. Runs were executed on CPU (Intel i7 12th Gen).

- Classes: 10 paddy disease categories (e.g., bacterial_leaf_blight, blast, brown_spot, …, tungro)
- Hardware: Intel i7 12th Gen CPU (no GPU)

Quick navigation (click to open phase summaries):
- Phase 1 — TensorFlow port and baseline → [TrainingPhase1-Tensorflow/Readme.md](./TrainingPhase1-Tensorflow/Readme.md)
- Phase 2 — PyTorch run (checkpoints/results) → [TrainingPhase2-Pytorch/](./TrainingPhase2-Pytorch/)
- Phase 3 — PyTorch fast run (partial, power interruption) → [TrainingPhase3-Pytorch/Readme.md](ShunyaNet/PaddyDiseaseRecognition/Training/TrainingPhase3-Pytorch )
- Phase 4 — PyTorch continued (test-per-epoch focus) → [TrainingPhase4-Pytorch/Readme.md](./TrainingPhase4-Pytorch/Readme.md)

Summary of outcomes by phase
- Phase 1 (TensorFlow baseline)
  - Goal: Verify the TF/Keras port of ShunyaNet on paddy disease dataset; establish baseline.
  - Settings: 224×224 input; batch size 2; AdamW; LR 0.001 with ReduceLROnPlateau; weight decay 1e‑5; early stopping on val_loss; augmentations (crop/flip/rotation/color jitter/blur); DropBlock.
  - Runtime: ~3 hours per epoch (large dataset + heavy augmentations + small batch size).
  - Observations: End‑to‑end works; slower throughput vs PyTorch observed later; metrics refer to artifacts.

- Phase 2 (PyTorch, checkpoints/results)
  - Artifacts present (best_model, results) but run notes not captured in a phase README.
  - Intent: Use PyTorch for better throughput; continue baseline with similar augmentations.

- Phase 3 (PyTorch fast run, interrupted)
  - Test (checkpoint epoch 5): Loss ≈ 2.2769, Accuracy ≈ 0.2080
  - Confusion matrices show heavy misclassification into a few dominant classes (e.g., tungro, normal); minority classes with near‑zero recall.
  - Run interrupted due to power; partial convergence.

- Phase 4 (PyTorch continued, test‑per‑epoch)
  - Test (epoch 10): Loss ≈ 2.2788, Accuracy ≈ 0.1508
  - Class‑wise: higher recall for hispa/tungro; many classes with near‑zero recall; strong class bias.
  - Multiple test confusion matrices (epoch 1 and 12) show predictions concentrated into few labels.

Cross-phase observations
- Throughput: PyTorch delivered faster epochs than TensorFlow on the same machine; better suited for continued experimentation here.
- Class imbalance and representation gaps: consistent collapse into 1–3 labels; several classes show very low recall.
- Underfitting on test: losses ~2.27–2.28; low accuracy even as epochs increase; suggests need for better balancing and stronger regularization or longer, stable training.
- Interruptions: Some runs were cut short (power), limiting convergence.

What helped
- Switching to PyTorch improved iteration speed and made frequent evaluation feasible.
- Per‑epoch confusion matrices and text reports clarified class‑wise failure modes.

What hurt
- Heavy imbalance + uniform augmentation likely reinforced majority classes.
- Small batch sizes and CPU‑only training increased epoch times, limiting total training cycles.
- Training interruptions reduced chances to reach planned schedules/patience.

Action plan (next iterations)
- Data balance and sampling
  - Audit per‑class counts; adopt class‑balanced sampler (WeightedRandomSampler) for training.
  - Targeted augmentation for low‑recall classes; lighter transforms for dominant classes.
  - Consider simple oversampling of minority classes; optionally use mixup/cutmix.

- Loss and optimization
  - Add class‑weighted CrossEntropy or Focal Loss; try label smoothing for stability.
  - Explore cosine annealing with warm restarts; tune LR/batch for CPU constraints.

- Model and calibration
  - Add/adjust dropout in classifier head; review width/depth tradeoffs.
  - Post‑hoc calibration (temperature scaling) to reduce over‑confidence.

- Training/runtime robustness
  - Frequent checkpoints and resume‑on‑failure;
  - Extend training to planned epochs on stable power; adjust early‑stopping patience.

- Evaluation diagnostics
  - Track per‑class PR curves; log misclassified samples per class.
  - Maintain confusion matrices at multiple eval epochs (train/val/test).

System info
- Hardware used: Intel i7 12th Gen CPU (no GPU)



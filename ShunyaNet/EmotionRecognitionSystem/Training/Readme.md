# Emotion Recognition – Consolidated Training Report

This document consolidates all training phases, key results, observations, and next steps for the Emotion Recognition system trained on CPU (Intel i7 12th Gen).

- Classes: anger, contempt, disgust, fear, happy, neutral, sad, surprise
- Hardware: Intel i7 12th Gen CPU (no GPU)

Quick navigation (click to open phase summaries):
- Phase 1 – Baseline and first pass: Training setup, results, and early observations → [TrainingPhase1/Readme.md](./TrainingPhase1/Readme.md)
- Phase 2 – Extended training (interrupted): Best test accuracy so far, partial convergence → [TrainingPhase2/README.md](./TrainingPhase2/README.md)
- Phase 3 – Continued experiments: Validation-focused snapshots, instability analysis → [TrainingPhase3/README.md](./TrainingPhase3/README.md)
- Phase 4 – Class-specific insights: Bias and augmentation issues, corrective actions → [trainingPhase4/README.md](./trainingPhase4/README.md)

Summary of outcomes by phase
- Phase 1 (baseline)
  - Epochs: 52 planned; Early stopping patience: 35; Batch size: 16; CPU-only.
  - Test: Loss 2.0376, Accuracy 0.2091
  - Notes: Early stopping at epoch 35; accuracy ~0.21. Strong bias toward “neutral” and “surprise”; many classes with 0 precision/recall. Training loss unstable early on. Clear class imbalance indicators.

- Phase 2 (extended training, interrupted)
  - Intended: 60 epochs, patience 35; training interrupted (power issue). Checkpoint ~epoch 25.
  - Test (epoch 25): Loss 1.9663, Accuracy 0.2547 (peak so far)
  - Notes: Predictions collapse to “neutral” (very high recall) and sometimes “disgust.” Many classes near-zero recall. Loss improved; accuracy peaked near eval epoch ~20. Partial learning due to interruption.

- Phase 3 (continued experiments)
  - Artifacts: checkpoint_epoch_20.pth; validation reports at eval epochs 5/10/20.
  - Validation highlights:
    - Epoch 5 — Val Loss 1.9381; Val Acc 0.3010 (neutral recall 0.8359; surprise recall 0.8238; happy F1 0.3261)
    - Epoch 10 — Val Loss 2.0733; Val Acc 0.1155 (fear recall 0.9401; others ~0)
    - Epoch 20 — Val Loss 2.0003; Val Acc 0.2676 (neutral recall 0.9141; sad recall 0.9968; others ~0)
  - Notes: High volatility; predictions concentrate in few classes; best validation near eval epochs 5 and 20 but does not generalize broadly.

- Phase 4 (class-specific analysis and issues)
  - Intended: 60 epochs; interruptions occurred. Checkpoint: checkpoint_epoch_45.pth.
  - Validation (selected):
    - Epoch 5 — Val Loss 4.5602; Val Acc 0.1010 (disgust/sad moderate recall; others ~0)
    - Epoch 10 — Val Loss 13.0181; Val Acc 0.1738 (happy recall 1.0; precision low)
    - Epoch 20 — Val Loss 7.1332; Val Acc 0.1766 (neutral recall 1.0; precision low)
    - Epoch 45 — Val Loss 2.2666; Val Acc 0.1390 (anger/neutral modest recall)
  - Test (epoch 45): Loss 2.2669; Accuracy 0.1418
  - Notes: Pronounced bias and instability. Uniform augmentation appears to reinforce majority-class dominance; minority classes underrepresented.

Cross-phase observations
- Systematically biased predictions toward a few dominant classes (often “neutral”; occasionally “disgust”, “sad”, or “anger” depending on epoch), with near-zero recall for the rest.
- Strong symptoms of class imbalance and/or representation gaps. Uniform augmentations likely helped majority classes more than minority ones.
- Training interruptions (power issues) prevented runs from reaching intended convergence, likely hurting generalization.
- Validation volatility across epochs suggests optimization and calibration issues (e.g., overly confident logits, mismatch in normalization, or insufficient regularization in the head).

What helped
- Longer training (Phase 2) improved test accuracy to ~25% despite interruptions.
- Intermediate checkpoints with confusion matrices and per-class reports improved diagnostics.

What hurt
- Uniform augmentation across imbalanced classes amplified bias.
- Lack of class-aware sampling/weighting and early training interruptions.

Action plan (next iterations)
- Data balance and sampling
  - Audit per-class counts across train/val/test; adjust splits if needed.
  - Use WeightedRandomSampler to oversample minority classes during training.
  - Consider simple oversampling or targeted augmentation for minority classes only.

- Loss functions and optimization
  - Try class-weighted CrossEntropy or Focal Loss to emphasize minority/hard classes.
  - Add label smoothing for stability; evaluate cosine annealing with warm restarts.
  - Introduce mixup/cutmix to regularize and reduce overfitting to dominant patterns.

- Augmentation and normalization
  - Make augmentation class-specific (stronger for minority, lighter for majority).
  - Recheck normalization statistics against the actual dataset distribution.

- Model head and calibration
  - Add/adjust dropout in the classifier head; review classifier width.
  - Calibrate logits (e.g., temperature scaling) after training.

- Training runtime and robustness
  - Implement resume-on-failure with frequent autosave checkpoints.
  - Run to full planned epochs when power/runtime is stable; extend patience if needed.

- Evaluation diagnostics
  - Track per-class PR curves and confusion matrices at multiple eval epochs.
  - Log misclassified samples per class and analyze recurring failure modes.
  - Add threshold analyses to understand decision boundaries.




Purpose
- Quick reference explaining the relationship between number of images (N), epochs (E), batch size (B), and iterations (steps) per epoch (I).

Key definitions and formulas
- N = total number of training images (dataset size).
- B = batch size (number of images processed together before a gradient update).
- E = number of epochs (how many times the training loop sees the whole dataset).
- I = iterations (batches) per epoch.
  - If using a dataloader that does not drop the last incomplete batch: I = ceil(N / B).
  - If drop_last=True: I = floor(N / B).
- Total gradient updates (total steps) across training = I * E.
- Total images presented to the model (counting repeats) = N * E (not reduced by B).

Basic intuitions
- Larger B -> fewer iterations I per epoch (because each iteration contains more samples).
- Smaller B -> more iterations I per epoch.
- Increasing E increases total exposures to data and total updates (proportional to E).
- The product I * E determines how many weight updates occur (important for convergence).

What changes when you change one variable
1) Change B (batch size) while keeping N and E fixed
   - Increase B:
     - I decreases (fewer updates per epoch).
     - Each update uses more samples (less noisy gradient estimates).
     - Requires more GPU memory and typically gives faster wall-clock throughput up to hardware limits.
     - May require learning-rate adjustments (e.g., linear-scaling rule: LR_new = LR_old * (B_new / B_old)).
     - Can reduce number of optimization steps for same number of epochs -> may affect convergence/generalization.
   - Decrease B:
     - I increases (more updates per epoch).
     - More stochastic/noisy gradients which can help generalization but may harm stability.
     - Lower memory footprint, but potentially slower throughput and higher communication overhead on distributed setups.

2) Change E (number of epochs) while keeping N and B fixed
   - Increase E:
     - More exposures to data, more total updates -> better opportunity to converge.
     - Higher risk of overfitting if training beyond the point where validation loss plateaus or increases.
   - Decrease E:
     - Fewer updates -> may underfit if too small.

Combined effects and common scenarios
- Keep total steps constant (I * E fixed):
  - If you increase B (I decreases) and increase E proportionally so that I*E is constant, total number of weight updates stays the same. However, each update's gradient variance changes (bigger batches = lower variance), so optimization dynamics and generalization can change even with same total steps.
- Keep total epochs constant, change B:
  - Changing B with fixed E changes total updates and can change convergence speed and generalization.
- Increase both B and E:
  - More updates (if E increased more than I decreased) and less noisy gradients per update; could converge faster but risk overfitting.
- Decrease both B and E:
  - Fewer updates and more noisy gradients; likely underfitting and slower convergence.

Practical guidance
- Monitor validation loss/accuracy; tune E until validation saturates or begins to degrade (early stopping recommended).
- If GPU memory is the limit, start with the largest B that fits and adjust learning rate accordingly.
- For small batch sizes (<32) expect noisier training; reduce LR or use adaptive optimizers.
- For extremely large batch sizes, consider warmup schedules and scaled learning rates to avoid optimization issues.
- If you want to keep the same number of optimization steps when changing B, adjust E so I*E remains approximately constant.

Edge cases & notes
- Data augmentation: the notion of "exposure" changes because each epoch presents transformed versions; total unique-image-like samples can be much larger than N.
- If using class-imbalanced sampling or weighted samplers, N and I formulas still hold but effective per-class exposures differ.
- Repeat/oversampling (to balance classes) increases effective N and thus I.
- Deterministic behavior: if you drop the last batch, tiny changes to B can drop or add samples and slightly change training trajectories.

Short checklist when changing B or E
- If you increase batch size: test if current LR needs scaling; check GPU memory/time per epoch; watch validation for generalization change.
- If you increase epochs: guard with early stopping and weight decay; watch for overfitting.
- When tuning, change one variable at a time and keep a log of I and total steps (I*E).

References (practical rules)
- Iterations per epoch = ceil(N / B) (unless last batch dropped).
- Total updates = iterations_per_epoch * epochs.
- Linear learning-rate scaling: LR proportional to B (empirical rule; validate on your task).

End of note.

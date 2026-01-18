# Resuming CNN Training After an Interruption — My Notes

TL;DR
- Yes, I can resume training exactly where it stopped as long as I have a proper checkpoint.
- I must restore: model weights + optimizer state + the epoch index (and scheduler/scaler if I used them).
- Start from the correct next epoch (watch out for 0-indexing) so learning-rate schedules and logs continue properly.

What I need in a checkpoint
- Model architecture (or a way to rebuild it identically)
- Weights (parameters)
- Optimizer state (e.g., Adam’s moments). Without this, training “feels” different and can destabilize convergence
- The epoch I reached (and ideally the global step)
- If used: LR scheduler state, AMP GradScaler state, random seeds for reproducibility

TensorFlow / Keras
Saving during the first run
- Use ModelCheckpoint to save the full model (not just weights) so the optimizer state is included

```python
import tensorflow as tf

# Build/compile model as usual
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/epoch-{epoch:02d}.keras',  # or .tf/.h5; .keras is preferred in TF 2.12+
    save_weights_only=False,  # save full model incl. optimizer state
    save_freq='epoch'
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TOTAL_EPOCHS,
    callbacks=[ckpt_cb]
)
```

Resuming after an interruption (example: last completed epoch index = 3 → resume from epoch 4)
```python
# Load the exact checkpoint saved at the end of epoch 3
model = tf.keras.models.load_model('checkpoints/epoch-03.keras')

# If the model was compiled on save, the optimizer state is restored too.
# Continue training from initial_epoch=3 (0-indexed → this starts the next epoch, i.e., epoch 4)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=3,          # last completed epoch index
    epochs=TOTAL_EPOCHS,      # overall target end epoch index
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/epoch-{epoch:02d}.keras',
            save_weights_only=False,
            save_freq='epoch'
        )
    ]
)
```
Notes
- initial_epoch is 0-indexed and means “skip training until this epoch index.” So setting it to 3 resumes at epoch 4.
- If I only saved weights (save_weights_only=True), I can still resume but the optimizer state is reset; LR schedules may not align—prefer full-model saves.

PyTorch
Saving during the first run
- I must save a dict with model.state_dict(), optimizer.state_dict(), and the epoch. Include scheduler and AMP scaler if used

```python
import torch

# ... build model/optimizer/scheduler/scaler ...

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # forward/backward/step...
        pass

    # Save checkpoint at the end of the epoch
    ckpt = {
        'epoch': epoch + 1,  # store the next epoch index to run
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if 'scaler' in locals() else None,
    }
    torch.save(ckpt, f'checkpoints/epoch-{epoch:02d}.pth')
```

Resuming after an interruption
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recreate model/optimizer/scheduler/scaler in the same way as before
model = build_model().to(device)
optimizer = build_optimizer(model)
scheduler = build_scheduler(optimizer)  # if used
scaler = torch.cuda.amp.GradScaler()    # if used

ckpt = torch.load('checkpoints/epoch-03.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
if ckpt.get('scheduler_state_dict') and scheduler:
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
if ckpt.get('scaler_state_dict') and 'scaler' in locals():
    scaler.load_state_dict(ckpt['scaler_state_dict'])

start_epoch = ckpt.get('epoch', 0)  # we stored "next epoch" above, so this starts at 4 when last completed was 3
model.train()

for epoch in range(start_epoch, num_epochs):
    for batch in train_loader:
        # training step
        pass
    if scheduler: scheduler.step()
```
Notes
- If I saved epoch as the last completed epoch (not next), then set start_epoch = ckpt['epoch'] + 1. I prefer saving the “next epoch” value to avoid off-by-one.
- If I trained with DataParallel/Distributed, I may need to map keys or wrap the model consistently on resume.

fast.ai (when applicable)
- If I used SaveModelCallback, I can reload the last .pth and continue: learn.load('best'); then call fit/fit_one_cycle again.
- Some versions surface a start_epoch mechanism via callbacks; either way, reloading the saved model and continuing training is the usual flow.

Why resuming works (and why the optimizer state matters)
- Optimizers like Adam keep moving averages of gradients/second moments. If I don’t restore them, training “restarts” mathematically and can jump or stall
- Restoring the epoch index ensures LR schedules (cosine decay, step decay, etc.) pick up exactly where they should

Pitfalls I avoid
- Architecture drift: any layer/shape change breaks state_dict loading or makes weights incompatible
- Different optimizer/scheduler settings on resume: causes LR or momentum mismatches
- Wrong initial_epoch/start_epoch: can repeat an epoch or skip one unintentionally
- Only saving weights in Keras: optimizer state is lost; LR schedule misalignment likely
- RNG/augmentations drift: set seeds if I need strict reproducibility

Quick checklist I follow
- During first training: enable robust checkpoints (full model for Keras; full dict for PyTorch)
- Verify the saved epoch index convention (completed vs next)
- On resume: load model + optimizer (+ scheduler/scaler), set start/initial epoch correctly
- Sanity check the first resumed epoch’s LR and logs; confirm continuity (no big spikes or resets)

Naming convention tips
- Keras: checkpoints/epoch-00.keras, epoch-01.keras, …
- PyTorch: checkpoints/epoch-00.pth, epoch-01.pth, …
- Add a latest symlink or copy for quick resume (e.g., checkpoints/latest.pth)

Validation after resume
- Plot or check metrics (loss, accuracy, LR) before and after the interruption. The curves should continue smoothly
- Run a quick eval on the validation set to confirm performance matches expectations

That’s it. With good checkpoints (weights + optimizer state + epoch), I can always pick up from the exact point the run was interrupted and continue training my CNN without wasting previous compute.


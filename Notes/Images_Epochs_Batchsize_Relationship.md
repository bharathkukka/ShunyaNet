# 🧠 Model Training Notes (My Understanding)

## Dataset & Training Loop Basics
- `N` → Total images in my training dataset  
- `B` → Batch size = how many images go into the model at once before 1 gradient update  
- `E` → Epochs = how many times the model sees all `N` images  
- `I` → Iterations per epoch = number of batches in 1 epoch  

### How to calculate `I`
- If I **use all images** (last small batch included):  
  `I = ceil(N / B)`
- If I **drop last batch** (`drop_last=True`):  
  `I = floor(N / B)`

### Key takeaways
- Model weights update `I × E` times in full training  
- Model sees `N × E` images (batch size does NOT reduce exposures)  
- Larger `B` → `I` decreases, training is stable but needs more memory  
- Smaller `B` → `I` increases, training is noisy but might generalize better  
- More `E` → more learning but risk of overfitting  
- Less `E` → model may underfit  

---

## Optimization & Learning Behavior
- **Gradient Update / Step** → 1 weight update using 1 batch  
- **Learning Rate (LR)** → controls how big each update is  
  - If `B` increases, LR might need scaling:  
    `LR_new = LR_old × (B_new / B_old)`
- **Loss Function** → tells the model how wrong its predictions are  
  - ex: `CrossEntropyLoss` (classification), `MSE` (regression)

- **Optimizer** → updates weights using gradients  
  - ex: `SGD`, `Adam`, `RMSprop`

- **Scheduler** → changes LR during training  
  - ex: `ReduceLROnPlateau`, `CosineDecay`, `StepLR`

- **Convergence** → model stops improving much, training saturates  
- **Overfitting** → training loss ↓ but validation loss ↑  
- **Underfitting** → training loss is high, model didn’t learn enough  

---

## Regularization (to avoid overfitting)
- **Weight Decay** → penalizes large weights (L2 regularization)
- **Dropout** → randomly turns off neurons to avoid memorization
- **Data Augmentation** → makes new variations of images every epoch  

---

## Model Performance Terms
- **Validation Set** → data model sees *only for testing*, not training  
- **Accuracy** → % of correct predictions  
- **Precision / Recall / F1-score** → better metrics for imbalanced data  
- **Confusion Matrix** → shows class-wise correct vs wrong predictions  

---

## Hardware & Training Speed Terms
- **Throughput** → how many images processed per second
- **VRAM** → GPU memory needed to fit batch size
- **Mixed Precision Training** → uses float16 + float32 to train faster & use less memory  
  - ex: `AMP (Automatic Mixed Precision)`

- **Distributed Training** → training across multiple GPUs
  - ex: `DataParallel`, `DDP`

---

## Dataset Loader Terms
- **DataLoader / tf.data pipeline** → loads data in batches  
- **Shuffle** → randomizes order of images each epoch  
- **num_workers** → loads images faster using CPU threads  
- **Prefetch** → prepares next batch while model trains current batch  
- **Caching** → stores dataset in memory for faster epochs  

---

## Training Safety & Control
- **Early Stopping** → stop training when validation stops improving  
- **Warmup** → start LR small and increase slowly to avoid unstable jumps  
- **Checkpoints** → save model weights at intervals  
  - ex: `model.h5`, `ckpt`, `SavedModel`

- **Inference** → using trained model to predict new images  
- **Fine-tuning** → retrain last layers on my custom task  

---

## Final Understanding Summary
| If I change | What happens |
|-----------|-------------|
| `B ↑` | `I ↓`, memory ↑, stable gradients, LR may ↑ |
| `B ↓` | `I ↑`, memory ↓, noisy gradients, LR may ↓ |
| `E ↑` | more updates, better learning, overfit risk |
| `E ↓` | fewer updates, underfit risk |

---

## Notes 
- Batch size only affects updates count, **not** total images seen  
- Validation set guides me when to stop, training set guides weights  
- Always log: `batch size`, `epochs`, `iterations`, `LR`, `loss`, `val_loss`

---


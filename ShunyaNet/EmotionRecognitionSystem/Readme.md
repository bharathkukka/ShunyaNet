# Emotion Recognition System 😃

A deep learning project to classify facial emotions from images using a custom CNN backbone (ShunyaNet). This project is designed for clarity, reproducibility, and strong performance.

## 🚀 Project Summary
- **Goal:** Detect and classify emotions from face images.
- **Classes:** 8 (anger, contempt, disgust, fear, happy, neutral, sad, surprise)
- **Framework:** PyTorch
- **Backbone:** ShunyaNet (combines best blocks from ResNet, Inception, DenseNet, etc.)
- **Input:** 96x96 RGB images
- **Output:** Emotion class label
---
## 🧩 How It Works (Workflow)
1. **Data Preparation**
   - Images are organized in `Data/Emotions/` by split (`train/`, `val/`, `test/`) and class.
   - `PreProcessing.py` loads images, applies augmentations (crop, flip, rotation, color jitter, blur), and normalizes them.
2. **Model Setup**
   - `ShunyaNetArchitecture.py` defines the ShunyaNet model, combining advanced CNN blocks for robust feature extraction.
   - DropBlock, SE, Inception, ResidualDense, MBConv, Ghost, Attention, and more are used.
3. **Training**
   - `main.py` sets up training with AdamW optimizer, CrossEntropyLoss, and learning rate scheduling.
   - Early stopping and checkpointing are used for best results.
   - Training/validation metrics and confusion matrices are saved in `output/results/`.
4. **Evaluation**
   - Best model is loaded and tested on the test set.
   - Results (accuracy, confusion matrix, classification report) are saved and printed.

## ⚙️ Key Training Settings
| Parameter         | Value/Setting                |
|-------------------|------------------------------|
| Batch Size        | 16                           |
| Epochs            | 52                           |
| Optimizer         | AdamW                        |
| Learning Rate     | 0.001 (ReduceLROnPlateau)    |
| Weight Decay      | 1e-5                         |
| Early Stopping    | Patience 12 (on val_loss)    |
| Augmentation      | Crop, flip, rotation, jitter |
| Regularization    | DropBlock, weight decay      |
| Input Size        | 96x96                        |

## 📊 Outputs & Results
- **Checkpoints:** Saved in `output/checkpoints/` (best model, periodic checkpoints)
- **Results:**
  - Confusion matrices (`output/results/`)
  - Training curves (loss/accuracy)
  - Classification report (precision, recall, F1)
- **Visuals:**
  - Data split diagrams in `Data/`

### 📈 Evaluation Metrics & Loss
| Metric                | Description                                      | Where to Find / How Computed                |
|-----------------------|--------------------------------------------------|---------------------------------------------|
| **Test Accuracy**     | Overall correct predictions on test set          | Printed at end of `main.py`, in results     |
| **Validation Accuracy**| Best val accuracy during training                | Printed/saved in `output/results/`          |
| **Test Loss**         | CrossEntropyLoss on test set                     | Printed at end of `main.py`, in results     |
| **Validation Loss**   | CrossEntropyLoss on validation set (best epoch)  | Printed/saved in `output/results/`          |
| **Confusion Matrix**  | Visual of true vs. predicted classes             | PNGs in `output/results/`                   |
| **Loss/Accuracy Curves** | Plots of training/validation loss & accuracy   | `training_history.png` in `output/results/` |

---

## 🗂️ Project Structure & Main Files
| File/Folder             | Purpose                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| `main.py`               | Main training, validation, and testing script. Handles all workflow.     |
| `PreProcessing.py`      | Custom PyTorch Dataset & image transforms (augmentation, normalization). |
| `ShunyaNetArchitecture.py` | Full ShunyaNet model definition (all blocks, classifier, etc).       |
| `Data/`                 | Contains data split diagrams and visualizations.                        |
| `output/`               | Stores checkpoints and results (confusion matrices, best model, etc).    |

# Emotion Recognition System (ShunyaNet)

This module implements an end-to-end image-based emotion recognition system, built on top of the custom ShunyaNet architecture. It covers dataset loading, preprocessing/augmentation, model definition, training with checkpoints and early stopping, and evaluation with confusion matrices and classification reports.

The code is organized to work directly with a folder-structured dataset and runs on CUDA (NVIDIA GPU), Apple Silicon (MPS), or CPU automatically.

---

## Contents
- Project structure
- Dataset format
- Data preprocessing and augmentation
- Model architecture (ShunyaNet)
- Training pipeline
- Evaluation and outputs
- Quick start
- Inference example
- Resuming from checkpoints
- Configuration and customization
- Performance tips (macOS MPS, batch size)
- Troubleshooting

---

## Project structure
```
ShunyaNet/
  EmotionRecognitionSystem/
    PreProcessing.py           # Dataset and transforms
    train_emotion_classifier.py# Training & evaluation entrypoint
    output/
      checkpoints/             # Auto-created: best and periodic checkpoints
      results/                 # Auto-created: plots and reports
    README.md                  # This file
    requirements.txt           # Python dependencies for this module
```
The dataset is expected under the repository-level `Data/Emotions` directory:
```
Data/
  Emotions/
    train/
      anger/
      contempt/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
    val/
      ... (same class subfolders as train)
    test/
      ... (same class subfolders as train)
```

---

## Dataset format
- A standard image classification directory layout.
- Each split (`train`, `val`, `test`) contains one subfolder per class.
- Supported image formats: .jpg, .jpeg, .png, .bmp, .webp.
- Hidden files and folders (starting with `.`) are ignored.
- Class names are inferred alphabetically from the `train/` subfolders and used consistently for metrics and reports.

---

## Data preprocessing and augmentation
Implemented in `PreProcessing.GenericImageDataset`.

Target input size: 96x96 RGB

- Train-time transforms (augment=True):
  - RandomResizedCrop(96x96, scale=(0.8, 1.0))
  - RandomHorizontalFlip()
  - RandomRotation(10 degrees)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
  - GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
  - ToTensor()
  - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

- Val/Test transforms (augment=False):
  - Resize(96x96)
  - ToTensor()
  - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Notes:
- Images are always loaded in RGB.
- Normalization uses ImageNet statistics, which harmonizes training for most CNNs.

---

## Model architecture (ShunyaNet)
Defined in `ShunyaNet/ShunyaNetArchitecture.py` and composed of modern CNN components:
- Stem: Conv + BN + Swish (stride 2)
- Inception-like multi-branch feature extraction
- SE (Squeeze-and-Excitation) channel attention
- Residual Dense Block
- MBConv (Mobile inverted bottleneck) with SE
- Ghost module for cheap feature generation
- Selective Kernel convolution
- Dual Attention (CBAM: channel + spatial)
- CSP-Inception fusion
- ReZero Residual Block
- Global Context Block
- MHSA (Multi-Head Self-Attention)
- DropBlock regularization
- Classifier head (AdaptiveAvgPool + Dropout + Linear)
- Attention pooling head (Conv-based attention + Linear)

The final logits are the average of the standard classifier and the attention-pooling classifier, improving robustness.

---

## Training pipeline
Implemented in `train_emotion_classifier.py`:
- Device selection: CUDA > MPS (Apple Silicon) > CPU.
- Reproducibility: seeds set for Python, NumPy, and PyTorch; cudnn flags guarded to CUDA-only.
- Dataloaders: configurable batch size, up to 4 workers; pin_memory enabled on CUDA.
- Loss: CrossEntropyLoss.
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5 by default).
- Scheduler: ReduceLROnPlateau on validation loss (factor=0.5, patience=3).
- Early stopping: monitored on validation loss with patience=12 and min_delta=0.
- Metrics tracked per epoch: train/val loss and accuracy, learning rate.
- Checkpointing:
  - Best model by validation accuracy: `output/checkpoints/best_model.pth`.
  - Periodic checkpoints every 5 epochs: `output/checkpoints/checkpoint_epoch_{N}.pth`.
- Visualizations & reports:
  - Training curves: `output/results/training_history.png`.
  - Confusion matrix (best val epoch): `output/results/confusion_matrix_epoch_{N}.png`.
  - Test confusion matrix: `output/results/test_confusion_matrix.png`.
  - Classification report: `output/results/classification_report.txt`.
  - CSV of history: `output/results/training_history.csv`.

---

## Evaluation and outputs
After training, the script automatically:
1) Loads the best checkpoint.
2) Evaluates on the test split.
3) Produces a confusion matrix and a detailed classification report (precision/recall/F1 per class and overall accuracy).

Outputs live under `ShunyaNet/EmotionRecognitionSystem/output/`.

---

## Quick start
1) Install dependencies (preferably in a virtual environment) using `requirements.txt`.
2) Ensure your dataset is at `Data/Emotions` with `train/`, `val/`, `test/` subfolders per class.
3) Run the training script as a module from the repository root to avoid path issues.
4) Inspect `output/checkpoints/` and `output/results/` for artifacts after training.

If you are on Apple Silicon and want to use MPS (default when available), no extra configuration is needed.

---

## Inference example
The snippet below shows how to load the best model and predict a single image. It mirrors the test-time preprocessing used in training (Resize + Normalize).

```python
import os, torch
from PIL import Image
from torchvision import transforms
import importlib

# Imports
ShunyaNet = importlib.import_module('ShunyaNet.ShunyaNetArchitecture').ShunyaNet
GenericImageDataset = importlib.import_module('ShunyaNet.EmotionRecognitionSystem.PreProcessing').GenericImageDataset

# Paths
base_dir = os.path.dirname(__file__)
ckpt_path = os.path.join(base_dir, 'output', 'checkpoints', 'best_model.pth')

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location=device)
class_names = ckpt.get('class_names', [])
num_classes = len(class_names) if class_names else 8

# Model
model = ShunyaNet(num_classes=num_classes).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Transforms (test-time)
tfms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction
img = Image.open('/path/to/your/image.jpg').convert('RGB')
img_t = tfms(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(img_t)
    pred = torch.argmax(logits, dim=1).item()
label = class_names[pred] if class_names else str(pred)
print('Predicted:', label)
```

---

## Resuming from checkpoints
- Best model: `output/checkpoints/best_model.pth` includes `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, epoch, and `class_names`.
- To resume training, load these states before continuing your training loop.

---

## Configuration and customization
Edit `Config` in `train_emotion_classifier.py`:
- `data_dir`: path to your dataset root (having `train/`, `val/`, `test/`).
- `target_size`: `(96, 96)` by default; adjust with care (affects receptive fields and MHSA cost).
- `batch_size`, `num_epochs`, `learning_rate`, `weight_decay`.
- `dropblock_prob`, `dropblock_size`.
- Early stopping and scheduler parameters.
- Output directories are anchored to this script’s folder and are created automatically.

Model-level tweaks (in `ShunyaNetArchitecture.py`):
- To reduce memory: remove or downsample before MHSA, reduce channels, or disable some attention blocks.
- To speed up training: lower input size, batch size, or disable expensive augmentations.

---

## Performance tips
- Batch size: If you encounter out-of-memory (OOM), reduce `Config.batch_size` (e.g., to 8 or 4).
- macOS MPS: Most operations are supported. If you hit an unsupported op, PyTorch may fall back to CPU; performance can vary. You can also force CPU to compare behavior.
- DataLoader workers: 2–4 workers is usually good. For notebook or stdin-executed contexts, `num_workers=0` can avoid multiprocessing issues.

---

## Troubleshooting
- Import errors: Run the script as a module from the repository root to ensure Python can resolve `ShunyaNet.*` imports.
- No images found: Verify class folders and extensions under `Data/Emotions/<split>/<class>/`.
- Mismatch in class order: Class names are sorted alphabetically from `train/`; ensure the same set exists in `val/` and `test/`.
- Slow training on CPU/MPS: Reduce batch size, disable some augmentations, or train fewer epochs for quick iteration.
- Plotting issues in headless environments: The script writes plots to files; no display backend is required.

---

## License and attribution
This repository contains a custom CNN architecture (ShunyaNet) and training pipeline authored by the project owner. External libraries used are listed in `requirements.txt` and are subject to their respective licenses.


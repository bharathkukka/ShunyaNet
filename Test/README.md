# Test Directory

This directory contains test scripts for the ShunyaNet emotion recognition system.

## Files

### 1. `emotion_dataset.py`
Dataset class for loading emotion images with preprocessing and augmentation.

**Features:**
- Dynamic image loading and preprocessing
- Support for train/val/test splits
- Configurable data augmentation
- Compatible with PyTorch DataLoader

**Usage:**
```python
from emotion_dataset import EmotionDataset

dataset = EmotionDataset(
    root_dir='../Data/EmotionRecognitionSystem/8Emotions/',
    split='train',
    target_size=(96, 96),
    augment=True
)
```

### 2. `train_emotion_model.py`
Simple training script for emotion recognition using a basic CNN model.

**Requirements:**
- Dataset directory: `../Data/EmotionRecognitionSystem/8Emotions/`
- Directory structure:
  ```
  8Emotions/
  ├── train/
  │   ├── class1/
  │   ├── class2/
  │   └── ...
  ├── val/
  │   └── ...
  └── test/
      └── ...
  ```

**Usage:**
```bash
python train_emotion_model.py
```

### 3. `colab_emotion_classifier_combined.py`
Comprehensive training script with the full ShunyaNet architecture.

**Features:**
- Complete ShunyaNet architecture implementation
- Advanced data augmentation
- Training with validation
- Confusion matrix generation
- Classification report
- Model checkpointing
- Learning rate scheduling

**Configuration:**
- Modify the `Config` class to adjust:
  - `data_dir`: Path to dataset
  - `batch_size`: Batch size for training
  - `num_epochs`: Number of training epochs
  - `learning_rate`: Initial learning rate

**Usage:**
```bash
python colab_emotion_classifier_combined.py
```

### 4. `Validation.py`
Validation script for evaluating trained models on test/validation datasets.

**Features:**
- Load trained model from checkpoint
- Evaluate on test or validation set
- Generate confusion matrix
- Generate classification report
- Save results to files

**Configuration:**
- Edit `ValidationConfig` class:
  - `model_checkpoint`: Path to trained model checkpoint
  - `data_dir`: Path to dataset
  - `split`: Which split to validate on ('test' or 'val')

**Usage:**
```bash
python Validation.py
```

### 5. `preprocess_emotion_images.py`
Static preprocessing script for creating preprocessed image dataset.

**Note:** This script is optional. The `emotion_dataset.py` class performs dynamic preprocessing, so you typically don't need to run this script.

## Setup

### Install Dependencies

```bash
# From the repository root
pip install -r ShunyaNet/requirements.txt
```

Or install individually:
```bash
pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn tqdm pandas scipy tensorflow
```

### Dataset Structure

Ensure your dataset follows this structure:
```
Data/
└── EmotionRecognitionSystem/
    └── 8Emotions/
        ├── train/
        │   ├── emotion1/
        │   │   ├── img1.jpg
        │   │   └── ...
        │   ├── emotion2/
        │   └── ...
        ├── val/
        │   └── ...
        └── test/
            └── ...
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn tqdm
   ```

2. **Prepare your dataset** in the correct directory structure.

3. **Train a model:**
   ```bash
   # Simple CNN model
   python train_emotion_model.py
   
   # Or use full ShunyaNet architecture
   python colab_emotion_classifier_combined.py
   ```

4. **Validate the trained model:**
   ```bash
   python Validation.py
   ```

## Expected Output

After training, you'll find:
- Model checkpoints in `checkpoints/` or `output/checkpoints/`
- Training history plots in `results/` or `output/results/`
- Confusion matrices
- Classification reports

After validation, you'll find:
- Confusion matrix: `validation_results/confusion_matrix.png`
- Classification report: `validation_results/classification_report.txt`

## Troubleshooting

### Missing Dependencies
If you get `ModuleNotFoundError`, install the missing package:
```bash
pip install <package-name>
```

### Dataset Not Found
Ensure the dataset path in the script configuration matches your actual dataset location.

### Out of Memory
- Reduce `batch_size` in the configuration
- Reduce image `target_size`
- Use CPU instead of GPU if GPU memory is limited

### Import Errors
Ensure you're running the scripts from the `Test/` directory or adjust the import paths accordingly.

## Notes

- All scripts are configured to use CUDA if available, otherwise CPU
- Data augmentation is only applied to training data, not validation/test data
- Model checkpoints are saved automatically during training
- The best model is saved based on validation accuracy

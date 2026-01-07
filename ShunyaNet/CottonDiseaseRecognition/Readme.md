# Cotton Disease Recognition 🌱

A deep learning project to classify cotton leaf images into four disease categories using a custom CNN backbone (ShunyaNet). This project is designed for clarity, reproducibility, and strong performance.

## 🚀 Project Summary
- **Goal:** Detect and classify cotton leaf diseases from images.
- **Classes:** 4 (bacterial blight, curl virus, fusarium wilt, healthy)
- **Framework:** PyTorch
- **Backbone:** ShunyaNet (combines best blocks from ResNet, Inception, DenseNet, etc.)
- **Input:** 224x224 RGB images
- **Output:** Disease class label
---
## 🧩 How It Works (Workflow)
1. **Data Preparation**
   - Images are organized in `Data/CottonDisease/` by split (`train/`, `val/`, `test/`) and class.
   - `PreProcessing.py` loads images, applies augmentations (crop, flip, rotation, color jitter, blur), and normalizes them.
2. **Model Setup**
   - `ShunyaNetArch.py` defines the ShunyaNet model, combining advanced CNN blocks for robust feature extraction.
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
| Epochs            | 42                           |
| Optimizer         | AdamW                        |
| Learning Rate     | 0.0001 (ReduceLROnPlateau)   |
| Weight Decay      | 1e-5                         |
| Early Stopping    | Patience 35 (on val_loss)    |
| Augmentation      | Crop, flip, rotation, jitter |
| Regularization    | DropBlock, weight decay      |
| Input Size        | 224x224                      |

## 📊 Outputs & Results  
  [Visit Here for detailed results and visualizations](ShunyaNet/CottonDiseaseRecognition/Training)
- **Results:**
  - **Test Accuracy:** 48.9%
  - **Test Loss:** 1.33
  - **Classification Report:**
    ```
    precision    recall  f1-score   support
    bacterial_blight     0.38       0.50      0.43        46
    curl_virus           0.55       0.28      0.37        43
    fussarium_wilt       0.51       0.72      0.60        43
    healthy              0.61       0.45      0.52        44

    accuracy                                 0.49       176
    macro avg            0.51       0.49      0.48       176
    weighted avg         0.51       0.49      0.48       176
    ```
  - **Confusion matrices:**
    - ![Test Confusion Matrix](ShunyaNet/CottonDiseaseRecognition/Training/TrainingPhaseFinal-output/results/test_confusion_matrix.png)  
    
  - **Training curves:** (loss/accuracy)
    - ![Training History](ShunyaNet/CottonDiseaseRecognition/Training/TrainingPhaseFinal-output/results/training_history.png)

---


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
| File/Folder         | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `main.py`           | Main training, validation, and testing script. Handles all workflow.     |
| `PreProcessing.py`  | Custom PyTorch Dataset & image transforms (augmentation, normalization). |
| `ShunyaNetArch.py`  | Full ShunyaNet model definition (all blocks, classifier, etc).           |
| `Data/`             | Contains data split diagrams, training results, and visualizations.      |
| `output/`           | Stores checkpoints and results (confusion matrices, best model, etc).    |

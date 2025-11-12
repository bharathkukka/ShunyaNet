# Paddy Disease Recognition - Dataset Documentation

## ⚠️ Important: Test Data Structure Issue

### The Problem

The test data in this dataset follows a **Kaggle competition format**, which creates a structural mismatch:

| Dataset Split | Structure | Labels Available? |
|--------------|-----------|-------------------|
| **Train** | `train/disease_name/image.jpg` | ✅ Yes (folder names) |
| **Validation** | `val/disease_name/image.jpg` | ✅ Yes (folder names) |
| **Test** | `test/image.jpg` (flat structure) | ❌ No |

### Why This Matters

1. **Cannot evaluate on test set** - No ground truth labels means you can't calculate accuracy, confusion matrix, or any performance metrics
2. **Different structure** - Train/val are organized by disease folders, test is a flat directory
3. **Competition format** - Test labels are intentionally hidden for Kaggle submission

### The Solution

#### ✅ Correct Approach: Use Train/Val Split

```
Training Data (9,371 images)
    ↓ Train model
Validation Data (1,036 images)
    ↓ Evaluate & tune hyperparameters
    ↓ Calculate metrics (accuracy, F1, etc.)
    ↓ Generate confusion matrix
    ↓ Select best model
Test Data (3,469 images)
    ↓ Generate predictions only
    ↓ Create submission.csv
    ↓ Submit to Kaggle (if competition)
```

## Dataset Statistics

### Current Split (After Running DataArrange.py)

- **Training Images:** 9,371 (90%)
- **Validation Images:** 1,036 (10%)
- **Test Images:** 3,469 (unlabeled)
- **Total Images:** 13,876

### Class Distribution

| Disease | Train | Val | Total |
|---------|-------|-----|-------|
| bacterial_leaf_blight | 432 | 47 | 479 |
| bacterial_leaf_streak | 342 | 38 | 380 |
| bacterial_panicle_blight | 304 | 33 | 337 |
| blast | 1,565 | 173 | 1,738 |
| brown_spot | 869 | 96 | 965 |
| dead_heart | 1,298 | 144 | 1,442 |
| downy_mildew | 558 | 62 | 620 |
| hispa | 1,435 | 159 | 1,594 |
| normal | 1,588 | 176 | 1,764 |
| tungro | 980 | 108 | 1,088 |

## How to Use

### 1. Analyze Dataset

```bash
python DataArrange.py
```

This will:
- Create validation split if not exists (10% of training data)
- Show detailed statistics for train/val/test sets
- Display warnings about unlabeled test data
- Verify CSV label files

### 2. Training Your Model

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load labeled train and val datasets
train_dataset = datasets.ImageFolder('Data/PaddyDiseases/Dataset/train', 
                                     transform=train_transform)
val_dataset = datasets.ImageFolder('Data/PaddyDiseases/Dataset/val', 
                                   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train model
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader)  # Use VAL, not TEST!
    print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}')
```

### 3. Evaluating Your Model

**✅ DO THIS - Use Validation Set:**
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate on VALIDATION set
val_preds, val_labels = get_predictions(model, val_loader)
accuracy = accuracy_score(val_labels, val_preds)
print(f"Validation Accuracy: {accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(val_labels, val_preds)
plot_confusion_matrix(cm, class_names)
```

**❌ DON'T DO THIS - Can't Evaluate on Test:**
```python
# This WON'T WORK - test set has no labels!
# test_acc = evaluate(model, test_loader)  # ❌ No labels to compare!
```

### 4. Making Predictions on Test Set

```python
from test_data_utils import UnlabeledTestDataset, predict_test_set, create_submission_file

# Load unlabeled test data
test_dataset = UnlabeledTestDataset('Data/PaddyDiseases/Dataset/test', 
                                    transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get class names
class_names = train_dataset.classes

# Generate predictions
predictions = predict_test_set(model, test_loader, device, class_names)

# Create submission file
create_submission_file(predictions, 'submission.csv', class_names)
```

## Files in This Directory

- **`DataArrange.py`** - Main script to analyze and organize dataset
- **`test_data_utils.py`** - Helper utilities for handling unlabeled test data
- **`README.md`** - This file

## Key Takeaways

### ✅ DO:
- Use **train** set for training
- Use **validation** set for evaluation, metrics, and model selection
- Use **test** set only for generating predictions
- Report your model's performance using **validation accuracy**
- Generate confusion matrices and metrics on **validation set**

### ❌ DON'T:
- Try to evaluate on test set (no labels available)
- Use test accuracy in your reports (it doesn't exist!)
- Expect test data to be organized like train/val

## Questions?

### Q: Why can't I evaluate on test set?
**A:** Test images have no labels. They're in a Kaggle competition format where labels are intentionally hidden.

### Q: What if I need to report test accuracy?
**A:** Report **validation accuracy** instead. The validation set is your evaluation set.

### Q: How do I know if my model is good?
**A:** Evaluate on the validation set. High validation accuracy = good model.

### Q: What's the test set for then?
**A:** Generating predictions for Kaggle submission or deployment, not for evaluation.

### Q: Can I create my own labeled test set?
**A:** Yes! You could further split the training data into train/val/test (e.g., 80/10/10), but the validation set already serves this purpose.

## Need Help?

Check the example scripts in `test_data_utils.py` for complete working examples of:
- Loading unlabeled test data
- Generating predictions
- Creating submission files

---

**Remember:** The validation set IS your test set for evaluation purposes! 🎯


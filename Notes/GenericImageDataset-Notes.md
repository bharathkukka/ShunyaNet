# 📒 Notes: GenericImageDataset Class

---

## Overview
The `GenericImageDataset` class is a flexible PyTorch dataset loader for image classification tasks. It is designed to work with datasets organized into 'train', 'val', and 'test' splits, where each split contains folders for each class and each class folder contains images (e.g., JPG files). This class can be used for any image classification dataset with this structure, such as emotion recognition, disease detection, or object classification.

---

## Key Features
- **Split Handling:** Loads images from a specified split ('train', 'val', or 'test') using the `split` argument.
- **Class Detection:** Automatically detects class folders and assigns integer labels based on folder order.
- **Flexible Image Size:** Resizes all images to a user-defined `target_size` (default: 96x96).
- **Data Augmentation:** Applies a rich set of augmentations (random flips, rotations, color jitter, affine transforms, cropping, blur) when `augment=True` (recommended for training only).
- **Normalization:** Normalizes images using ImageNet mean and standard deviation, making it compatible with most pretrained models.
- **Generic Usage:** Can be used for any dataset with the same folder structure, not limited to emotions.

---

## Folder Structure Assumed
```
root_dir/
    train/
        class1/
        class2/
        ...
    val/
        class1/
        class2/
        ...
    test/
        class1/
        class2/
        ...
```
Each class folder contains images (JPG format).

---

## Constructor Arguments
- `root_dir`: Path to the root directory containing the dataset splits.
- `split`: Which split to load ('train', 'val', or 'test'). Defaults to 'train'.
- `target_size`: Desired image size (width, height). Defaults to (96, 96).
- `augment`: Whether to apply data augmentation. Use `True` for training, `False` for validation/test.

---

## How It Works
1. **Initialization:**
   - Scans the specified split folder for class directories.
   - Collects all image file paths and assigns integer labels based on class folder order.
2. **Transform Pipeline:**
   - Resizes images to `target_size`.
   - Applies augmentations if `augment=True` (training only).
   - Converts images to PyTorch tensors.
   - Normalizes using ImageNet statistics.
3. **Data Loading:**
   - `__getitem__` loads and transforms each image, returning `(image_tensor, label)`.
   - `__len__` returns the number of samples in the split.

---

## Example Usage
```python
from torch.utils.data import DataLoader
from 3-DatasetPreProcessing import GenericImageDataset

# Training set (with augmentation)
train_dataset = GenericImageDataset(root_dir, split='train', target_size=(96, 96), augment=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation set (no augmentation)
val_dataset = GenericImageDataset(root_dir, split='val', target_size=(96, 96), augment=False)
val_loader = DataLoader(val_dataset, batch_size=32)

# Test set (no augmentation)
test_dataset = GenericImageDataset(root_dir, split='test', target_size=(96, 96), augment=False)
test_loader = DataLoader(test_dataset, batch_size=32)
```

---

## Best Practices
- **Augmentation:** Only use augmentation for training data, not for validation or test sets.
- **Image Format:** If your images are not JPG, update the file extension check in the code.
- **Normalization:** Use ImageNet normalization for compatibility with pretrained models.
- **Generic Use:** Works for any dataset with the same folder structure.

---

## Customization & Extensibility
- **Image Formats:** Add more extensions (e.g., PNG) in the file check.
- **Augmentations:** Edit the `_build_transform` method to add/remove augmentations.
- **Grayscale Support:** Change `.convert('RGB')` to `.convert('L')` for grayscale images.
- **Class Detection:** Uses sorted folder names for reproducibility.

---

## Error Handling
- Skips non-directory files in the split folder.
- For robustness, you can add try/except in `__getitem__` to handle unreadable images.

---

## Integration
- Use this class in your training, validation, and test scripts for any image classification project.
- Works seamlessly with PyTorch's DataLoader and training loops.

---

## Summary
The `GenericImageDataset` class is a powerful, reusable tool for loading and preprocessing image datasets for classification. It supports flexible splits, rich augmentations, and is easy to adapt for new datasets and tasks.


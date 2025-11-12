"""
DataArrange.py
Script to count and organize information about the Paddy Disease dataset

IMPORTANT NOTE ABOUT TEST DATA:
================================
The test folder contains unlabeled images (Kaggle competition format).
This means:
- Test images are NOT organized by disease folders
- No ground truth labels available for evaluation
- Cannot calculate accuracy/metrics on test set
- Test set is only for generating predictions for submission

SOLUTION:
=========
We use the train/val split for model development:
- Train set: For training the model
- Validation set: For hyperparameter tuning and model selection
- Test set: For generating final predictions (no evaluation possible)

If you need a labeled evaluation set, use the validation set!
"""

import os
import shutil
import random
from pathlib import Path


def count_images_in_directory(directory_path, extensions=None):
    """
    Count all image files in a directory (including subdirectories)

    Args:
        directory_path: Path to the directory
        extensions: List of valid image extensions (default: common image formats)

    Returns:
        int: Total count of image files
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    count = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                count += 1
    return count


def count_images_by_class(directory_path, extensions=None):
    """
    Count images in each subdirectory (class) of a directory

    Args:
        directory_path: Path to the directory containing class subdirectories
        extensions: List of valid image extensions

    Returns:
        dict: Dictionary with class names as keys and image counts as values
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    class_counts = {}

    if not os.path.exists(directory_path):
        return class_counts

    # Get all subdirectories
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            count = 0
            for file in os.listdir(item_path):
                if Path(file).suffix.lower() in extensions:
                    count += 1
            class_counts[item] = count

    return class_counts


def create_validation_split(dataset_root, val_split=0.1, seed=42):
    """
    Create a validation set by moving val_split percentage of training data

    Args:
        dataset_root: Root directory of the dataset
        val_split: Percentage of training data to use for validation (default: 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    train_path = os.path.join(dataset_root, 'train')
    val_path = os.path.join(dataset_root, 'val')

    if not os.path.exists(train_path):
        print("Error: Training directory not found!")
        return

    # Check if validation directory already exists
    if os.path.exists(val_path):
        response = input(f"\nValidation directory already exists at: {val_path}\nDo you want to recreate it? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
        else:
            print(f"Removing existing validation directory...")
            shutil.rmtree(val_path)

    print("\n" + "=" * 70)
    print("CREATING VALIDATION SET")
    print("=" * 70)
    print(f"Validation split: {val_split * 100}%\n")

    # Create validation directory
    os.makedirs(val_path, exist_ok=True)

    total_moved = 0

    # Process each class directory
    for class_name in os.listdir(train_path):
        class_train_path = os.path.join(train_path, class_name)

        if not os.path.isdir(class_train_path) or class_name.startswith('.'):
            continue

        # Create corresponding class directory in validation set
        class_val_path = os.path.join(val_path, class_name)
        os.makedirs(class_val_path, exist_ok=True)

        # Get all image files in this class
        image_files = [f for f in os.listdir(class_train_path)
                      if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}]

        # Calculate number of images to move to validation
        num_val = max(1, int(len(image_files) * val_split))

        # Randomly select images for validation
        val_images = random.sample(image_files, num_val)

        # Move selected images to validation directory
        for img_file in val_images:
            src = os.path.join(class_train_path, img_file)
            dst = os.path.join(class_val_path, img_file)
            shutil.move(src, dst)

        total_moved += num_val
        print(f"  {class_name:<30} : {len(image_files):>4} -> Train: {len(image_files) - num_val:>4}, Val: {num_val:>4}")

    print(f"\n  Total images moved to validation: {total_moved}")
    print("=" * 70)
    print("Validation set created successfully!\n")


def verify_csv_labels(dataset_root):
    """
    Verify that the CSV labels file exists and matches the dataset structure

    NOTE: CSV files are NOT needed for model development!
    Since train/val data is organized in folders, PyTorch's ImageFolder
    can read labels directly from folder names.

    This function is just for informational purposes.

    Args:
        dataset_root: Root directory of the dataset

    Returns:
        bool: True if CSV exists and can be used
    """
    csv_path = os.path.join(Path(dataset_root).parent, 'train.csv')

    if not os.path.exists(csv_path):
        print("\n" + "=" * 70)
        print("CSV LABEL FILE")
        print("=" * 70)
        print("CSV file not found (train.csv)")
        print("\n✓ This is OK! CSV is NOT needed for model development.")
        print("✓ Labels are already in folder structure.")
        print("✓ Use PyTorch's ImageFolder to load data.")
        print("=" * 70)
        return False

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        print("\n" + "=" * 70)
        print("CSV LABEL FILE (INFORMATIONAL ONLY)")
        print("=" * 70)
        print(f"CSV File: {csv_path}")
        print(f"Total labeled images: {len(df)}")
        print(f"\nLabel distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label:<30} : {count:>6} images")

        print("\n⚠️  NOTE: You DON'T need this CSV for model development!")
        print("   Your data is organized in folders → Use ImageFolder")
        print("=" * 70)

        return True
    except ImportError:
        print("\n[INFO] pandas not installed (but you don't need it anyway!)")
        print("       CSV files are not needed for model development.")
        return False
    except Exception as e:
        print(f"\n[INFO] Could not read CSV file: {e}")
        print("       (But CSV is not needed for model development anyway!)")
        return False


def analyze_dataset(dataset_root):
    """
    Analyze the complete dataset structure and count all images

    Args:
        dataset_root: Root directory of the dataset
    """
    print("=" * 70)
    print("PADDY DISEASE DATASET ANALYSIS")
    print("=" * 70)
    print(f"\nDataset Location: {dataset_root}\n")

    train_path = os.path.join(dataset_root, 'train')
    val_path = os.path.join(dataset_root, 'val')
    test_path = os.path.join(dataset_root, 'test')

    total_images = 0

    # Analyze training set
    if os.path.exists(train_path):
        print("-" * 70)
        print("TRAINING SET")
        print("-" * 70)
        train_class_counts = count_images_by_class(train_path)

        if train_class_counts:
            for class_name, count in sorted(train_class_counts.items()):
                print(f"  {class_name:<30} : {count:>6} images")

            train_total = sum(train_class_counts.values())
            print(f"\n  {'Total Training Images':<30} : {train_total:>6}")
            total_images += train_total
        else:
            print("  No training data found!")
    else:
        print("\nTraining directory not found!")

    # Analyze validation set
    if os.path.exists(val_path):
        print("\n" + "-" * 70)
        print("VALIDATION SET")
        print("-" * 70)
        val_class_counts = count_images_by_class(val_path)

        if val_class_counts:
            for class_name, count in sorted(val_class_counts.items()):
                print(f"  {class_name:<30} : {count:>6} images")

            val_total = sum(val_class_counts.values())
            print(f"\n  {'Total Validation Images':<30} : {val_total:>6}")
            total_images += val_total
        else:
            print("  No validation data found!")
    else:
        print("\nValidation directory not found!")

    # Analyze test set
    if os.path.exists(test_path):
        print("\n" + "-" * 70)
        print("TEST SET (UNLABELED)")
        print("-" * 70)
        test_total = count_images_in_directory(test_path)
        print(f"  {'Total Test Images':<30} : {test_total:>6}")
        print("\n  ⚠️  WARNING: Test images are UNLABELED (Kaggle competition format)")
        print("      - Cannot evaluate model performance on test set")
        print("      - Test set is for generating predictions only")
        print("      - Use VALIDATION set for model evaluation!")
        total_images += test_total
    else:
        print("\nTest directory not found!")

    # Summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    if os.path.exists(train_path):
        train_total = sum(count_images_by_class(train_path).values())
        print(f"  Training Images    : {train_total:>6}")
    if os.path.exists(val_path):
        val_total = sum(count_images_by_class(val_path).values())
        print(f"  Validation Images  : {val_total:>6}")
    if os.path.exists(test_path):
        test_total = count_images_in_directory(test_path)
        print(f"  Test Images        : {test_total:>6}")
    print(f"  {'TOTAL IMAGES':<20}: {total_images:>6}")
    print("=" * 70)


if __name__ == "__main__":
    # Dataset root path (relative to current file)
    current_dir = Path(__file__).parent
    dataset_root = current_dir.parent.parent / "Data" / "PaddyDiseases" / "Dataset"

    # Convert to string for compatibility
    dataset_root = str(dataset_root)

    # Check if validation directory exists
    val_path = os.path.join(dataset_root, 'val')

    if not os.path.exists(val_path):
        print("Validation directory not found. Creating validation set with 10% of training data...\n")
        create_validation_split(dataset_root, val_split=0.1)
    else:
        print("Validation directory already exists.\n")

    # Analyze the dataset
    analyze_dataset(dataset_root)

    # Verify CSV labels
    verify_csv_labels(dataset_root)

    # Print recommendations
    print("\n" + "=" * 70)
    print("WORKFLOW FOR MODEL DEVELOPMENT")
    print("=" * 70)
    print("\n📚 PHASE 1: MODEL DEVELOPMENT (Use Train + Val)")
    print("-" * 70)
    print("  ✓ TRAIN set     → Train your model")
    print("  ✓ VALIDATION set → Evaluate & tune your model")
    print("                    - Calculate accuracy, F1-score, confusion matrix")
    print("                    - Select best model based on val performance")
    print("                    - THIS IS YOUR EVALUATION METRIC!")
    print()
    print("🔮 PHASE 2: FINAL PREDICTIONS (Use Test)")
    print("-" * 70)
    print("  ✓ TEST set      → Generate predictions (blind predictions!)")
    print("                    - You WON'T know if predictions are correct")
    print("                    - Submit to Kaggle for scoring")
    print("                    - Kaggle will tell you the accuracy")
    print()
    print("💡 SUMMARY:")
    print("-" * 70)
    print("  • During development: Test data is USELESS (no labels)")
    print("  • For final submission: Test data is ESSENTIAL (Kaggle scoring)")
    print("  • Your reported accuracy: Use VALIDATION accuracy")
    print()
    print("📁 WHAT YOU NEED:")
    print("-" * 70)
    print("  ✓ Train folder (organized by disease) - YES")
    print("  ✓ Val folder (organized by disease)   - YES")
    print("  ✗ Test folder (for now)               - NO (ignore it)")
    print("  ✗ CSV files (train.csv)               - NO (not needed!)")
    print()
    print("  → Use PyTorch's ImageFolder - it reads labels from folder names!")
    print("=" * 70)


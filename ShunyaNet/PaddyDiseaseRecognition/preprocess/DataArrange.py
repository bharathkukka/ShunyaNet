"""
NOTE ABOUT TEST DATA (KAGGLE FORMAT):
------------------------------------
Test folder images do not have labels.
So:
- They are not grouped into disease folders
- I don't have correct answers (ground truth)
- I cannot calculate accuracy, F1, confusion matrix, or any metrics
- This folder is only useful to generate predictions and upload to Kaggle for scoring later

WHAT I WILL DO INSTEAD:
----------------------
I will use train + validation folders for model building.
Meaning:
- Train folder → model learns from this
- Validation folder → I test model here and tune parameters
- Test folder → ignore for now, only used when submitting predictions to Kaggle
"""

import os
import shutil
import random
from pathlib import Path


def count_images_in_directory(directory_path, extensions=None):
    """
    This function counts image files inside a folder and all its subfolders.
    If I don't give extensions, it will assume common image formats automatically.
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
    This function checks each subfolder inside the given directory and counts images in them.
    I will use this to know how many images belong to each disease/class folder.
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    class_counts = {}

    if not os.path.exists(directory_path):
        return class_counts

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
    This function will take 10% (default) of images from each disease folder in train
    and move them into a new 'val' folder, so I can evaluate my model while training.

    Why?
    → Because test data is unlabeled, so validation folder becomes my real test for accuracy.
    """
    random.seed(seed)

    train_path = os.path.join(dataset_root, 'train')
    val_path = os.path.join(dataset_root, 'val')

    if not os.path.exists(train_path):
        print("Train folder missing. I cannot split without it.")
        return

    if os.path.exists(val_path):
        response = input("\nValidation folder already exists. Do I want to delete and make it again? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("I chose not to recreate. Keeping the existing validation folder.")
            return
        else:
            print("Deleting old validation folder to create a fresh split...")
            shutil.rmtree(val_path)

    print("\nStarting validation split process...\n")

    total_moved = 0

    for class_name in os.listdir(train_path):
        class_train_path = os.path.join(train_path, class_name)

        if not os.path.isdir(class_train_path) or class_name.startswith('.'):
            continue

        class_val_path = os.path.join(val_path, class_name)
        os.makedirs(class_val_path, exist_ok=True)

        image_files = [f for f in os.listdir(class_train_path)
                      if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}]

        num_val = max(1, int(len(image_files) * val_split))

        val_images = random.sample(image_files, num_val)

        for img_file in val_images:
            src = os.path.join(class_train_path, img_file)
            dst = os.path.join(class_val_path, img_file)
            shutil.move(src, dst)

        total_moved += num_val
        print(f"{class_name} → moved {num_val} images to validation")

    print(f"\nTotal images shifted into validation folder: {total_moved}\n")


def verify_csv_labels(dataset_root):
    """
    This function only checks if a train.csv file exists.
    But I don't actually need CSV for training because my images are already inside labeled folders.
    This check is just to confirm Kaggle provided CSV or not.
    """
    csv_path = Path(dataset_root).parent.parent / 'train.csv'

    if not os.path.exists(csv_path):
        print("\nNo CSV file found. That's fine because folder names already act as labels.")
        return False

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        print("\nCSV exists, but I will NOT use it for training.")
        print("Because PyTorch ImageFolder can read folder names as labels directly.")

        return True
    except:
        print("\nCSV file is there, but I don't need it for model development.")
        return False


def analyze_dataset(dataset_root):
    """
    This function prints image count details for train, val, and test folders.
    This helps me confirm dataset structure before training the model.
    """
    print("\nChecking dataset structure and image counts...\n")

    train_path = os.path.join(dataset_root, 'train')
    val_path = os.path.join(dataset_root, 'val')
    test_path = os.path.join(dataset_root, 'test')

    total_images = 0

    if os.path.exists(train_path):
        train_class_counts = count_images_by_class(train_path)
        train_total = sum(train_class_counts.values())
        print(f"Total images in train folder: {train_total}")
        total_images += train_total
    else:
        print("Train folder not found!")

    if os.path.exists(val_path):
        val_class_counts = count_images_by_class(val_path)
        val_total = sum(val_class_counts.values())
        print(f"Total images in validation folder: {val_total}")
        total_images += val_total
    else:
        print("Validation folder not found!")

    if os.path.exists(test_path):
        test_total = count_images_in_directory(test_path)
        print(f"Total images in test folder (unlabeled): {test_total}")
        total_images += test_total
    else:
        print("Test folder not found!")

    print(f"\nOverall total image files in dataset: {total_images}\n")


if __name__ == "__main__":
    """
    This block decides if validation folder needs to be created.
    Then it runs dataset analysis and prints workflow reminders for myself.
    """

    current_dir = Path(__file__).parent
    dataset_root = current_dir.parent.parent / "Data" / "PaddyDiseases" / "Dataset"

    dataset_root = str(dataset_root)

    val_path = os.path.join(dataset_root, 'val')

    if not os.path.exists(val_path):
        print("Validation folder missing → creating a 80/10/10 split from train data...\n")
        create_validation_split(dataset_root, val_split=0.1)
    else:
        print("Validation folder already present. No need to create again.\n")

    analyze_dataset(dataset_root)

    verify_csv_labels(dataset_root)

    print("\nMODEL BUILDING PLAN (MY OWN WORKFLOW):")
    print("-------------------------------------")
    print("1. Train model using images from train folder")
    print("2. Check accuracy and tune model using validation folder")
    print("3. Ignore test folder until I want to generate Kaggle submission predictions")
    print("4. No need for CSV files while training because labels come from folder names")

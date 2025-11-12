"""
test_data_utils.py
Utility functions for handling unlabeled test data in Kaggle competition format

This module provides tools to:
1. Load unlabeled test images for prediction
2. Generate submission files in the correct format
3. Handle the mismatch between labeled train/val and unlabeled test data
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image


class UnlabeledTestDataset(Dataset):
    """
    PyTorch Dataset for unlabeled test images

    Use this for making predictions on the test set.
    Since test images have no labels, this dataset only returns images and their IDs.
    """

    def __init__(self, test_dir: str, transform=None):
        """
        Args:
            test_dir: Path to test directory containing unlabeled images
            transform: Optional transform to be applied on images
        """
        self.test_dir = test_dir
        self.transform = transform

        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        self.image_files = sorted([
            f for f in os.listdir(test_dir)
            if Path(f).suffix.lower() in valid_extensions and not f.startswith('.')
        ])

        print(f"Found {len(self.image_files)} test images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, img_name


def create_submission_file(predictions: Dict[str, str],
                          output_path: str,
                          class_names: List[str] = None):
    """
    Create a submission CSV file from predictions

    Args:
        predictions: Dictionary mapping image_id to predicted label
        output_path: Path where to save the submission CSV
        class_names: Optional list of valid class names for validation

    Example:
        predictions = {
            '200001.jpg': 'blast',
            '200002.jpg': 'normal',
            ...
        }
        create_submission_file(predictions, 'submission.csv')
    """
    # Create DataFrame
    df = pd.DataFrame([
        {'image_id': img_id, 'label': label}
        for img_id, label in sorted(predictions.items())
    ])

    # Validate predictions if class names provided
    if class_names:
        invalid_labels = set(df['label']) - set(class_names)
        if invalid_labels:
            print(f"Warning: Found invalid labels: {invalid_labels}")
            print(f"Valid labels are: {class_names}")

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())


def predict_test_set(model, test_loader, device, class_names: List[str]) -> Dict[str, str]:
    """
    Generate predictions for the entire test set

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for unlabeled test images
        device: torch device (cuda/cpu)
        class_names: List of class names in the same order as model outputs

    Returns:
        Dictionary mapping image_id to predicted class name
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Map predictions to class names
            for img_id, pred_idx in zip(image_ids, predicted):
                predictions[img_id] = class_names[pred_idx.item()]

    return predictions


def get_class_names_from_train(train_dir: str) -> List[str]:
    """
    Extract class names from training directory structure

    Args:
        train_dir: Path to training directory with subdirectories for each class

    Returns:
        Sorted list of class names
    """
    class_names = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')
    ]
    return sorted(class_names)


def load_csv_labels(csv_path: str) -> pd.DataFrame:
    """
    Load the train.csv file with image labels

    Args:
        csv_path: Path to train.csv

    Returns:
        DataFrame with image_id, label, variety, age columns
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} labeled images from CSV")
    print(f"Columns: {df.columns.tolist()}")
    return df


# Example usage and workflow
if __name__ == "__main__":
    print("=" * 70)
    print("TEST DATA UTILITIES - USAGE EXAMPLES")
    print("=" * 70)

    print("\n1. LOADING UNLABELED TEST DATA:")
    print("-" * 70)
    print("""
from test_data_utils import UnlabeledTestDataset
from torchvision import transforms

# Define transforms (same as used during training)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create test dataset
test_dataset = UnlabeledTestDataset('Data/PaddyDiseases/Dataset/test', 
                                    transform=test_transform)

# Create dataloader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    """)

    print("\n2. GENERATING PREDICTIONS:")
    print("-" * 70)
    print("""
from test_data_utils import predict_test_set, get_class_names_from_train

# Get class names
class_names = get_class_names_from_train('Data/PaddyDiseases/Dataset/train')

# Generate predictions
predictions = predict_test_set(model, test_loader, device, class_names)
    """)

    print("\n3. CREATING SUBMISSION FILE:")
    print("-" * 70)
    print("""
from test_data_utils import create_submission_file

# Create submission CSV
create_submission_file(predictions, 'submission.csv', class_names)
    """)

    print("\n" + "=" * 70)
    print("IMPORTANT NOTES:")
    print("=" * 70)
    print("✓ Test data has NO LABELS - cannot evaluate accuracy")
    print("✓ Use VALIDATION set to evaluate your model")
    print("✓ Use TEST set only to generate predictions for submission")
    print("✓ The validation set IS YOUR EVALUATION SET")
    print("=" * 70)


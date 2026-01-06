import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image


class UnlabeledTestDataset(Dataset):
    """
    This is a PyTorch Dataset class for test images that don't have labels.
    Since there are no labels, it will only return:
    → the image tensor
    → the image filename (which acts as its ID)
    """

    def __init__(self, test_dir: str, transform=None):
        """
        What I need to give it:
        - test_dir: the folder where all unlabeled test images are stored
        - transform: same preprocessing steps I used while training (resize, normalize, etc.)
        """
        self.test_dir = test_dir
        self.transform = transform

        # Only select valid image files and sort them for consistency
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        self.image_files = sorted([
            f for f in os.listdir(test_dir)
            if Path(f).suffix.lower() in valid_extensions and not f.startswith('.')
        ])

        print(f"Total test images detected: {len(self.image_files)}")

    def __len__(self):
        # Returns number of test images
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)

        # Load image and convert to RGB format
        image = Image.open(img_path).convert('RGB')

        # Apply transforms if I defined any
        if self.transform:
            image = self.transform(image)

        # Returns image tensor + filename as image ID
        return image, img_name


def create_submission_file(predictions: Dict[str, str],
                          output_path: str,
                          class_names: List[str] = None):
    """
    This function creates a Kaggle submission CSV file using model predictions.

    What it expects:
    - predictions: a dictionary like { image_filename : predicted_disease }
    - output_path: where the CSV file should be saved
    - class_names: optional check to make sure predictions only contain valid disease names
    """

    # Convert predictions dictionary into a table format using DataFrame
    df = pd.DataFrame([
        {'image_id': img_id, 'label': label}
        for img_id, label in sorted(predictions.items())
    ])

    # If I gave class_names, check for any wrong prediction labels
    if class_names:
        invalid_labels = set(df['label']) - set(class_names)
        if invalid_labels:
            print("Some predicted labels do not match my train folder class names.")
            print(f"Invalid labels found: {invalid_labels}")
            print(f"Correct/valid class labels should be one of: {class_names}")

    # Save final submission file
    df.to_csv(output_path, index=False)
    print(f"Submission CSV saved at: {output_path}")
    print(f"Total predictions written: {len(df)}")

    # Show how many times each disease/class was predicted
    print("\nPrediction count per class:")
    print(df['label'].value_counts())


def predict_test_set(model, test_loader, device, class_names: List[str]) -> Dict[str, str]:
    """
    This function runs the trained model on all test images and returns predictions.

    What I give it:
    - model: my trained CNN model
    - test_loader: DataLoader that loads unlabeled test images
    - device: CPU or GPU (if available)
    - class_names: list of disease names in same order as model output neurons
    """

    model.eval()  # Set model to inference mode
    predictions = {}

    with torch.no_grad():  # No gradients needed during prediction
        for images, image_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Convert predicted index into actual disease name
            for img_id, pred_idx in zip(image_ids, predicted):
                predictions[img_id] = class_names[pred_idx.item()]

    return predictions


def get_class_names_from_train(train_dir: str) -> List[str]:
    """
    This reads all subfolder names inside train directory.
    Each folder name = one disease/class label.
    I will use this list to map model outputs correctly.
    """
    class_names = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')
    ]
    return sorted(class_names)


def load_csv_labels(csv_path: str) -> pd.DataFrame:
    """
    This loads the train.csv file if I want to inspect it.
    But I DON'T need this for model training since my labels are already folder-based.
    This is only for checking what Kaggle originally provided.
    """
    df = pd.read_csv(csv_path)
    print(f"CSV contains {len(df)} labeled images")
    print(f"Available columns in CSV: {df.columns.tolist()}")
    return df


if __name__ == "__main__":
    """
    Below are sample steps to remember how I will use these utilities.
    This is just printed as a reference for myself.
    """

    print("MY TEST DATA WORKFLOW:")
    print("--------------------------------")

    print("\nStep 1: Load unlabeled test images")
    print("""
from test_data_utils import UnlabeledTestDataset
from torchvision import transforms

# Use the same preprocessing I used during training
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load test images (no labels)
test_dataset = UnlabeledTestDataset('Data/PaddyDiseases/Dataset/test',
                                   transform=test_transform)

# Create DataLoader (batch-wise loading)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    """)

    print("\nStep 2: Generate predictions using trained model")
    print("""
from test_data_utils import predict_test_set, get_class_names_from_train

# Get disease names from train folder structure
class_names = get_class_names_from_train('Data/PaddyDiseases/Dataset/train')

# Run model on test images
predictions = predict_test_set(model, test_loader, device, class_names)
    """)

    print("\nStep 3: Create Kaggle submission file from predictions")
    print("""
from test_data_utils import create_submission_file

# Convert predictions into submission.csv
create_submission_file(predictions, 'submission.csv', class_names)
    """)

    print("\nNOTES I SHOULD REMEMBER:")
    print("--------------------------------")
    print("- Test images have no labels, so I cannot evaluate accuracy here")
    print("- Validation folder is what I will use to check model performance")
    print("- Test folder is only for creating prediction file and submitting to Kaggle")
    print("- CSV is not needed for training because folder names already provide labels")
    print("--------------------------------")

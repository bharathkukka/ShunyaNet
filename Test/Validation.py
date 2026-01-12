"""
Validation script for ShunyaNet models.
This script can be used to validate trained models on test/validation datasets.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from tqdm import tqdm
import numpy as np

# Import the dataset class
from emotion_dataset import EmotionDataset

# Configuration
class ValidationConfig:
    # Paths
    model_checkpoint = '../ShunyaNet/EmotionRecognitionSystem/output/checkpoints/best_model.pth'
    data_dir = '../Data/EmotionRecognitionSystem/8Emotions/'
    output_dir = './validation_results'
    
    # Model parameters
    target_size = (96, 96)
    batch_size = 32
    
    # Dataset split to validate ('test' or 'val')
    split = 'test'


def validate_model(model, data_loader, device, class_names):
    """
    Validate the model on the given dataset.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for the validation/test dataset
        device: torch device (cuda/cpu)
        class_names: List of class names
        
    Returns:
        accuracy: Overall accuracy
        all_preds: List of predictions
        all_labels: List of true labels
    """
    print(f"Running validation on device: {device}")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    return accuracy, all_preds, all_labels


def plot_confusion_matrix(all_labels, all_preds, class_names, output_path):
    """
    Generate and save confusion matrix plot.
    
    Args:
        all_labels: True labels
        all_preds: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
    """
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def save_classification_report(all_labels, all_preds, class_names, accuracy, output_path):
    """
    Generate and save classification report.
    
    Args:
        all_labels: True labels
        all_preds: Predicted labels
        class_names: List of class names
        accuracy: Overall accuracy
        output_path: Path to save the report
    """
    cr = classification_report(all_labels, all_preds, 
                               target_names=class_names, digits=4)
    
    with open(output_path, 'w') as f:
        f.write(f"Validation Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cr)
    
    print(f"Classification report saved to: {output_path}")
    print("\nClassification Report:")
    print(cr)


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: torch device (cuda/cpu)
        
    Returns:
        model: Loaded model
        class_names: Class names from checkpoint
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get class names
    class_names = checkpoint.get('class_names', [])
    num_classes = len(class_names)
    
    # Import ShunyaNet architecture
    # Note: This assumes the architecture file is available
    # You may need to adjust the import based on your project structure
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from ShunyaNet.EmotionRecognitionSystem.ShunyaNetArchitecture import ShunyaNet
        
        model = ShunyaNet(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded successfully from: {checkpoint_path}")
        print(f"Best validation accuracy during training: {checkpoint.get('best_val_acc', 'N/A')}")
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {class_names}")
        
        return model, class_names
    except ImportError as e:
        print(f"Error importing ShunyaNet architecture: {e}")
        print("Attempting to load with colab_emotion_classifier_combined ShunyaNet...")
        
        # Fallback: try to use the ShunyaNet from colab_emotion_classifier_combined
        from colab_emotion_classifier_combined import ShunyaNet
        
        model = ShunyaNet(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, class_names


def main():
    # Create output directory
    os.makedirs(ValidationConfig.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    print("\nLoading model...")
    try:
        model, class_names = load_model_from_checkpoint(
            ValidationConfig.model_checkpoint, 
            device
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. You have trained a model first")
        print("2. The checkpoint path in ValidationConfig is correct")
        print(f"   Current path: {ValidationConfig.model_checkpoint}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load the dataset
    print(f"\nLoading {ValidationConfig.split} dataset...")
    try:
        dataset = EmotionDataset(
            ValidationConfig.data_dir,
            split=ValidationConfig.split,
            target_size=ValidationConfig.target_size,
            augment=False
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=ValidationConfig.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"Loaded {len(dataset)} samples from {ValidationConfig.split} set")
    except FileNotFoundError as e:
        print(f"Error: Dataset directory not found: {ValidationConfig.data_dir}")
        print("Please ensure the dataset path in ValidationConfig is correct")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Run validation
    print("\nStarting validation...")
    accuracy, all_preds, all_labels = validate_model(
        model, data_loader, device, class_names
    )
    
    # Generate and save confusion matrix
    cm_path = os.path.join(ValidationConfig.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)
    
    # Generate and save classification report
    report_path = os.path.join(ValidationConfig.output_dir, 'classification_report.txt')
    save_classification_report(all_labels, all_preds, class_names, accuracy, report_path)
    
    print(f"\nValidation complete!")
    print(f"Results saved to: {ValidationConfig.output_dir}")


if __name__ == "__main__":
    main()

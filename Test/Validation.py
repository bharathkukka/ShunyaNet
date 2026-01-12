import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import sys
import random
import numpy as np
from torch.backends import cudnn
from torchvision import transforms
import argparse
import importlib.util

# Ensure project root is on sys.path so that the 'ShunyaNet' package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Robust file-based import fallback for ShunyaNetArchitecture and PreProcessing
ARCH_PATH = os.path.join(PROJECT_ROOT, 'ShunyaNet', 'EmotionRecognitionSystem', 'ShunyaNetArchitecture.py')
PREP_PATH = os.path.join(PROJECT_ROOT, 'ShunyaNet', 'EmotionRecognitionSystem', 'PreProcessing.py')

def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_arch_mod = _import_from_path('shunya_arch', ARCH_PATH)
_prep_mod = _import_from_path('shunya_prep', PREP_PATH)
ShunyaNet = _arch_mod.ShunyaNet
GenericImageDataset = _prep_mod.GenericImageDataset

# Set device (prefer CUDA, then MPS, else CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Only set cudnn flags when CUDA is available
    if torch.cuda.is_available():
        try:
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass

# Configuration
class Config:
    # Dataset parameters
    data_dir = '/Users/bharathgoud/PycharmProjects/Shunya-00/Data/Emotions'
    target_size = (96, 96)

    # Training parameters
    num_classes = 8  # default/reference; model will derive from dataset at runtime
    batch_size = 16
    num_epochs = 52
    learning_rate = 0.0005  # Lower initial learning rate
    weight_decay = 1e-5
    seed = 42
    # Early stopping
    early_stop_patience = 45  # Increased patience
    early_stop_min_delta = 0.0  # consider as improvement only if val_loss decreases by > min_delta

    # Model parameters
    dropblock_prob = 0.1
    dropblock_size = 5

    # Paths for saving (anchored to this script directory)
    _base_dir = os.path.dirname(__file__)
    # Point to output/checkpoints as requested
    checkpoint_dir = os.path.join(_base_dir, 'output', 'checkpoints')
    # Results under Test/output/results
    results_dir    = os.path.join(_base_dir, 'output', 'results')

# Create directories for checkpoints and results
os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.results_dir, exist_ok=True)

# # 1. Data augmentation for training set
# train_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor()
# ])
# val_transform = transforms.ToTensor()

# Load datasets
def load_data():
    print("Loading datasets...")
    train_dataset = GenericImageDataset(
        Config.data_dir,
        split='train',
        target_size=Config.target_size,
        augment=True,
        # transform=train_transform  # Add data augmentation
    )

    val_dataset = GenericImageDataset(
        Config.data_dir,
        split='val',
        target_size=Config.target_size,
        augment=False,
        # transform=val_transform
    )

    test_dataset = GenericImageDataset(
        Config.data_dir,
        split='test',
        target_size=Config.target_size,
        augment=False
    )

    # DataLoader ergonomics
    workers = 0  # Avoid multiprocessing to prevent pickling issues with dynamically imported modules
    pin = torch.cuda.is_available()

    # Create Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Get class names
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names

class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float).to(device)

# Model definition (increase complexity)
class ImprovedModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Evaluation function for test set
def evaluate(model, test_loader, criterion, class_names, checkpoint_label=""):
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            test_correct += torch.eq(predicted, labels).long().sum().item()
            test_total += labels.size(0)

            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / max(1, test_total)
    test_acc = test_correct / max(1, test_total)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Test Set Confusion Matrix (Accuracy: {test_acc:.4f})')
    plt.tight_layout()
    cm_filename = 'test_confusion_matrix'
    if checkpoint_label:
        cm_filename += f'_{checkpoint_label}'
    cm_filename += '.png'
    plt.savefig(os.path.join(Config.results_dir, cm_filename))
    plt.close()

    # Generate classification report
    cr = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("Classification Report:")
    print(cr)

    # Save classification report to file
    report_filename = 'test_classification_report'
    if checkpoint_label:
        report_filename += f'_{checkpoint_label}'
    report_filename += '.txt'
    with open(os.path.join(Config.results_dir, report_filename), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cr)

    return test_acc, cm

# Evaluation function for arbitrary split
def evaluate_split(model, data_loader, criterion, class_names, split_name="test", checkpoint_label=""):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating ({split_name})"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += torch.eq(predicted, labels).long().sum().item()
            total_samples += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)

    print(f"{split_name.capitalize()} Loss: {avg_loss:.4f}, {split_name.capitalize()} Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    title_extra = f" ({checkpoint_label})" if checkpoint_label else ""
    plt.title(f'{split_name.capitalize()} Confusion Matrix{title_extra} (Acc: {accuracy:.4f})')
    plt.tight_layout()
    cm_filename = f'{split_name}_confusion_matrix_eval'
    if checkpoint_label:
        cm_filename += f'_{checkpoint_label}'
    cm_filename += '.png'
    plt.savefig(os.path.join(Config.results_dir, cm_filename))
    plt.close()

    # Classification report
    cr = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(f"\n{split_name.capitalize()} Classification Report:")
    print(cr)

    # Save text report
    report_filename = f'{split_name}_classification_report_eval'
    if checkpoint_label:
        report_filename += f'_{checkpoint_label}'
    report_filename += '.txt'
    with open(os.path.join(Config.results_dir, report_filename), 'w') as f:
        f.write(f"{split_name.capitalize()} Loss: {avg_loss:.4f}, {split_name.capitalize()} Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cr)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': cr,
    }


def load_model_from_checkpoint(checkpoint_path, class_names):
    """Reload a ShunyaNet model from a checkpoint.

    Supports two formats:
    1) Full checkpoint dict with 'model_state_dict' and optional metadata.
    2) Raw state_dict saved directly as model.state_dict().
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine whether this is a full checkpoint or a raw state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        ckpt_label = f"epoch_{checkpoint.get('epoch', 'unknown')}"
        saved_class_names = checkpoint.get('class_names')
        if saved_class_names is not None:
            class_names = saved_class_names
    else:
        state_dict = checkpoint
        ckpt_label = os.path.splitext(os.path.basename(checkpoint_path))[0]

    num_classes = len(class_names)
    model = ShunyaNet(
        num_classes=num_classes,
        dropblock_prob=Config.dropblock_prob,
        dropblock_size=Config.dropblock_size
    ).to(device)

    model.load_state_dict(state_dict)
    print("Model weights loaded successfully.")

    return model, class_names, ckpt_label


def evaluate_all_checkpoints():
    """Iterate over all .pth checkpoints in Config.checkpoint_dir, evaluate on val, and
    for the last checkpoint also evaluate on test. Save all artifacts under a
    dedicated subfolder in Config.results_dir.
    """
    # Prepare output directory for this evaluation run
    eval_root = os.path.join(Config.results_dir, 'checkpoints_eval')
    os.makedirs(eval_root, exist_ok=True)

    # Discover checkpoint files
    all_files = [f for f in os.listdir(Config.checkpoint_dir) if f.endswith('.pth')]
    if not all_files:
        print(f"No .pth files found in checkpoint dir: {Config.checkpoint_dir}")
        return

    # Sort files so that the "last" checkpoint is deterministic
    all_files.sort()

    # Load data once
    train_loader, val_loader, test_loader, class_names = load_data()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluate each checkpoint on validation set
    for idx, fname in enumerate(all_files):
        ckpt_path = os.path.join(Config.checkpoint_dir, fname)
        print("\n" + "=" * 80)
        print(f"Evaluating checkpoint {idx+1}/{len(all_files)}: {ckpt_path}")

        model, class_names, ckpt_label = load_model_from_checkpoint(ckpt_path, class_names)

        # Create per-checkpoint subfolder
        ckpt_out_dir = os.path.join(eval_root, ckpt_label)
        os.makedirs(ckpt_out_dir, exist_ok=True)

        # Temporarily redirect Config.results_dir for this checkpoint's artifacts
        original_results_dir = Config.results_dir
        ckpt_out_dir = os.path.join(eval_root, ckpt_label)
        Config.results_dir = ckpt_out_dir
        os.makedirs(ckpt_out_dir, exist_ok=True)
        try:
            # Validation evaluation
            val_metrics = evaluate_split(
                model,
                val_loader,
                criterion,
                class_names,
                split_name="val",
                checkpoint_label=ckpt_label,
            )
            # If this is the checkpoint_epoch_25.pth, also run full test evaluation
            if os.path.basename(ckpt_path) == 'checkpoint_epoch_45.pth':
                print("\nRunning full test-set evaluation for checkpoint_epoch_25.pth...")
                test_acc, _ = evaluate(model, test_loader, criterion, class_names, checkpoint_label=ckpt_label)
                print(f"Test accuracy for {ckpt_label}: {test_acc:.4f}")
        finally:
            Config.results_dir = original_results_dir

    print("\nAll checkpoints evaluated. Per-checkpoint results stored under:")
    print(f"  {eval_root}")


if __name__ == "__main__":
    # Pure evaluation entry point: no training.
    # This will scan all .pth files under Config.checkpoint_dir and evaluate them.
    set_seed(Config.seed)
    evaluate_all_checkpoints()

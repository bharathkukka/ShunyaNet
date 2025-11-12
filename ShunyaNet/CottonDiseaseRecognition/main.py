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

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import using string-based imports to handle numeric prefixes in module names
import importlib
ShunyaNet = importlib.import_module("ShunyaNet.ShunyaNetArchitecture").ShunyaNet
GenericImageDataset = importlib.import_module("ShunyaNet.CottonDiseaseRecognition.PreProcessing").GenericImageDataset

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
    data_dir = '/Users/bharathgoud/PycharmProjects/Shunya-00/Data/CottonDisease'
    target_size = (224, 224)

    # Training parameters
    num_classes = 4 # default/reference; model will derive from dataset at runtime
    batch_size = 32
    num_epochs = 40
    learning_rate = 0.001
    weight_decay = 1e-5
    seed = 42
    # Early stopping
    early_stop_patience = 10
    early_stop_min_delta = 0.0  # consider as improvement only if val_loss decreases by > min_delta

    # Model parameters
    dropblock_prob = 0.1
    dropblock_size = 5

    # Paths for saving (anchored to this script directory)
    _base_dir = os.path.dirname(__file__)
    checkpoint_dir = os.path.join(_base_dir, 'output', 'checkpoints')
    results_dir    = os.path.join(_base_dir, 'output', 'results')

# Create directories for checkpoints and results
os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.results_dir, exist_ok=True)

# Load datasets
def load_data():
    print("Loading datasets...")
    train_dataset = GenericImageDataset(
        Config.data_dir,
        split='train',
        target_size=Config.target_size,
        augment=True
    )

    val_dataset = GenericImageDataset(
        Config.data_dir,
        split='val',
        target_size=Config.target_size,
        augment=False
    )

    test_dataset = GenericImageDataset(
        Config.data_dir,
        split='test',
        target_size=Config.target_size,
        augment=False
    )

    # DataLoader ergonomics
    workers = min(4, os.cpu_count() or 1)
    pin = torch.cuda.is_available()  # pinning is only beneficial for CUDA

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

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, class_names):
    print("Starting training...")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    stopped_epoch = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += torch.eq(predicted, labels).long().sum().item()
            train_total += labels.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}",
                                 acc=f"{train_correct/train_total:.4f}")

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_all = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                # Statistics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += torch.eq(predicted, labels).long().sum().item()
                val_total += labels.size(0)

                # Store predictions and labels for confusion matrix
                val_preds.extend(predicted.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

                val_bar.set_postfix(loss=f"{loss.item():.4f}",
                                   acc=f"{val_correct/val_total:.4f}")

        epoch_val_loss = val_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)

        # Update learning rate
        scheduler.step(epoch_val_loss)

        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Track improvement for early stopping (based on val_loss)
        if epoch_val_loss < best_val_loss - Config.early_stop_min_delta:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save best model (based on val_acc)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_path = os.path.join(Config.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names
            }, checkpoint_path)
            print(f"  New best model saved with validation accuracy: {best_val_acc:.4f}")

            # Generate and save confusion matrix for best model
            cm = confusion_matrix(val_labels_all, val_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Epoch {epoch+1}, Val Acc: {epoch_val_acc:.4f})')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.results_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
            plt.close()

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(Config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_acc': epoch_train_acc,
                'val_acc': epoch_val_acc,
                'class_names': class_names
            }, checkpoint_path)
            print(f"  Checkpoint saved at epoch {epoch+1}")

        # Early stopping check
        if epochs_no_improve >= Config.early_stop_patience:
            stopped_epoch = epoch + 1
            print(f"Early stopping triggered at epoch {stopped_epoch}: no val_loss improvement for {Config.early_stop_patience} epochs.")
            break

    # Plot and save training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig(os.path.join(Config.results_dir, 'training_history.png'))
    plt.close()

    # Save history to CSV
    try:
        import csv
        csv_path = os.path.join(Config.results_dir, 'training_history.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr'])
            for i in range(len(history['train_loss'])):
                writer.writerow([
                    i + 1,
                    history['train_loss'][i],
                    history['val_loss'][i],
                    history['train_acc'][i],
                    history['val_acc'][i],
                    history['lr'][i]
                ])
    except Exception as e:
        print(f"Warning: failed to write training history CSV: {e}")

    # Do not add non-list values to history to keep types consistent
    return history, best_val_acc

# Evaluation function for test set
def evaluate(model, test_loader, criterion, class_names):
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
    plt.savefig(os.path.join(Config.results_dir, 'test_confusion_matrix.png'))
    plt.close()

    # Generate classification report
    cr = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("Classification Report:")
    print(cr)

    # Save classification report to file
    with open(os.path.join(Config.results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cr)

    return test_acc, cm


def main():
    # Set seeds
    set_seed(Config.seed)

    # Load Data
    train_loader, val_loader, test_loader, class_names = load_data()

    # Initialize model
    print("Initializing ShunyaNet...")
    num_classes = len(class_names)
    model = ShunyaNet(
        num_classes=num_classes,
        dropblock_prob=Config.dropblock_prob,
        dropblock_size=Config.dropblock_size
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    # Train the model
    history, best_val_acc = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        Config.num_epochs,
        class_names
    )

    # Load best model for evaluation
    checkpoint = torch.load(os.path.join(Config.checkpoint_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model with validation accuracy: {checkpoint['best_val_acc']:.4f}")

    # Evaluate on test set
    test_acc, confusion_mat = evaluate(model, test_loader, criterion, class_names)

    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Results saved in {Config.results_dir}")

if __name__ == "__main__":
    main()

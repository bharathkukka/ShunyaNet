
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import sys
import random
import numpy as np
import csv
from tqdm import tqdm

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # Import the ShunyaNet architecture and preprocessing
# from Paddy_Disease_Recognition.ShunyaNetTensorflow import ShunyaNet
# from Paddy_Disease_Recognition.preprocessing_tf import GenericImageDataset
import importlib
ShunyaNet = importlib.import_module("ShunyaNet.PaddyDiseaseRecognition.ShunyaNetTensorflow").ShunyaNet
GenericImageDataset = importlib.import_module("ShunyaNet.PaddyDiseaseRecognition.preprocessing").GenericImageDataset
# Set GPU memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) configured: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("Using CPU")

# Reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Configuration
class Config:
    # Dataset parameters
    data_dir = 'Data/PaddyDisease'
    target_size = (224, 224)

    # Training parameters
    num_classes = 10  # default/reference; model will derive from dataset at runtime
    batch_size = 32
    num_epochs = 35
    learning_rate = 0.001
    weight_decay = 1e-5
    seed = 42

    # Early stopping
    early_stop_patience = 12
    early_stop_min_delta = 0.0  # consider as improvement only if val_loss decreases by > min_delta

    # Model parameters
    dropblock_prob = 0.1
    dropblock_size = 5

    # Learning rate scheduler parameters
    lr_scheduler_factor = 0.5
    lr_scheduler_patience = 3

    # Paths for saving (anchored to this script directory)
    _base_dir = os.path.dirname(__file__)
    checkpoint_dir = os.path.join(_base_dir, 'output', 'checkpoints')
    results_dir = os.path.join(_base_dir, 'output', 'results')


# Create directories for checkpoints and results
os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.results_dir, exist_ok=True)


# Load datasets
def load_data():
    """Load and prepare training, validation, and test datasets."""
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

    # Create TensorFlow datasets
    train_loader = train_dataset.get_dataset(
        batch_size=Config.batch_size,
        shuffle=True
    )

    val_loader = val_dataset.get_dataset(
        batch_size=Config.batch_size,
        shuffle=False
    )

    test_loader = test_dataset.get_dataset(
        batch_size=Config.batch_size,
        shuffle=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Get class names
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

    return (
        train_loader, val_loader, test_loader, class_names,
        len(train_dataset), len(val_dataset), len(test_dataset)
    )


# Custom training loop with metrics tracking
class TrainMetrics:
    """Helper class to track training metrics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0.0
        self.correct = 0
        self.total = 0

    def update(self, loss_value, predictions, labels):
        batch_size = tf.shape(labels)[0].numpy()
        self.loss += float(loss_value) * batch_size
        self.correct += np.sum(predictions == labels.numpy())
        self.total += batch_size

    def get_loss(self):
        return self.loss / max(1, self.total)

    def get_accuracy(self):
        return self.correct / max(1, self.total)


# Training function
def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, class_names,
          train_samples, val_samples):
    """Train the model."""
    print("Starting training...")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(Config.checkpoint_dir, 'best_model')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        train_metrics = TrainMetrics()

        train_bar = tqdm(train_loader, desc="Training")
        for images, labels in train_bar:
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss_value = loss_fn(labels, logits)

            # Backward pass
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update metrics
            predictions = tf.argmax(logits, axis=1).numpy()
            train_metrics.update(loss_value, predictions, labels)

            train_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                acc=f"{train_metrics.get_accuracy():.4f}"
            )

        epoch_train_loss = train_metrics.get_loss()
        epoch_train_acc = train_metrics.get_accuracy()

        # Validation phase
        val_metrics = TrainMetrics()
        val_preds = []
        val_labels_all = []

        val_bar = tqdm(val_loader, desc="Validation")
        for images, labels in val_bar:
            logits = model(images, training=False)
            loss_value = loss_fn(labels, logits)

            # Update metrics
            predictions = tf.argmax(logits, axis=1).numpy()
            val_metrics.update(loss_value, predictions, labels)

            # Store for confusion matrix
            val_preds.extend(predictions)
            val_labels_all.extend(labels.numpy())

            val_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                acc=f"{val_metrics.get_accuracy():.4f}"
            )

        epoch_val_loss = val_metrics.get_loss()
        epoch_val_acc = val_metrics.get_accuracy()

        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        current_lr = optimizer.learning_rate
        if isinstance(current_lr, tf.Variable):
            current_lr = float(current_lr.numpy())
        else:
            current_lr = float(current_lr)
        history['lr'].append(current_lr)

        # Print epoch results
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Learning rate scheduling (ReduceLROnPlateau equivalent)
        if epoch_val_loss < best_val_loss - Config.early_stop_min_delta:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

            # Reduce learning rate after patience epochs
            if epochs_no_improve > 0 and epochs_no_improve % Config.lr_scheduler_patience == 0:
                new_lr = current_lr * Config.lr_scheduler_factor
                optimizer.learning_rate.assign(new_lr)
                print(f"Reduced learning rate to {new_lr:.6f}")

        # Save best model (based on val_acc)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            model.save(best_model_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")

            # Generate and save confusion matrix for best model
            cm = confusion_matrix(val_labels_all, val_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix (Epoch {epoch+1}, Val Acc: {epoch_val_acc:.4f})')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.results_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
            plt.close()

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(Config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}')
            model.save(checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # Early stopping check
        if epochs_no_improve >= Config.early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}: "
                  f"no val_loss improvement for {Config.early_stop_patience} epochs.")
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
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.results_dir, 'training_history.png'))
    plt.close()

    # Save history to CSV
    try:
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

    return history, best_val_acc


# Evaluation function for test set
def evaluate(model, test_loader, loss_fn, class_names):
    """Evaluate model on test set."""
    print("\nEvaluating on test set...")

    test_metrics = TrainMetrics()
    all_preds = []
    all_labels = []

    test_bar = tqdm(test_loader, desc="Testing")
    for images, labels in test_bar:
        logits = model(images, training=False)
        loss_value = loss_fn(labels, logits)

        # Update metrics
        predictions = tf.argmax(logits, axis=1).numpy()
        test_metrics.update(loss_value, predictions, labels)

        # Store predictions and labels
        all_preds.extend(predictions)
        all_labels.extend(labels.numpy())

        test_bar.set_postfix(
            loss=f"{loss_value:.4f}",
            acc=f"{test_metrics.get_accuracy():.4f}"
        )

    test_loss = test_metrics.get_loss()
    test_acc = test_metrics.get_accuracy()

    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Test Set Confusion Matrix (Accuracy: {test_acc:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.results_dir, 'test_confusion_matrix.png'))
    plt.close()

    # Generate classification report
    cr = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(cr)

    # Save classification report to file
    with open(os.path.join(Config.results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cr)

    return test_acc, cm


def main():
    """Main training pipeline."""
    # Set seeds
    set_seed(Config.seed)

    # Load Data
    train_loader, val_loader, test_loader, class_names, train_samples, val_samples, test_samples = load_data()

    # Initialize model
    print("\nInitializing ShunyaNet...")
    num_classes = len(class_names)
    model = ShunyaNet(
        num_classes=num_classes,
        dropblock_prob=Config.dropblock_prob,
        dropblock_size=Config.dropblock_size
    )

    # Build model by calling it once
    dummy_input = tf.random.normal((1, Config.target_size[0], Config.target_size[1], 3))
    _ = model(dummy_input, training=False)

    print(f"Model built successfully!")
    print(f"Total trainable parameters: {model.count_params():,}")

    # Loss function
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Optimizer with weight decay
    optimizer = keras.optimizers.AdamW(
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    # Train the model
    history, best_val_acc = train(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        Config.num_epochs,
        class_names,
        train_samples,
        val_samples
    )

    # Load best model for evaluation
    best_model_path = os.path.join(Config.checkpoint_dir, 'best_model')
    model = keras.models.load_model(best_model_path)
    print(f"\nLoaded best model with validation accuracy: {best_val_acc:.4f}")

    # Evaluate on test set
    test_acc, confusion_mat = evaluate(model, test_loader, loss_fn, class_names)

    print(f"\n" + "=" * 50)
    print(f"Training completed.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Results saved in {Config.results_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()


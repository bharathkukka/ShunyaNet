"""TensorFlow CNN on CIFAR-10

Train and evaluate a convolutional neural network on the CIFAR-10 dataset using tf.keras.
Provides command-line arguments for hyperparameters, data augmentation, model saving, and plotting.

Features:
- Loads CIFAR-10 from tf.keras.datasets
- Normalizes inputs to [0,1]
- Optional data augmentation via Keras preprocessing layers
- Configurable model depth and optimizer params
- Early stopping + ModelCheckpoint (optional)
- Evaluation: accuracy, classification report, confusion matrix
- Plots: training accuracy/loss curves + confusion matrix (optional save)

Example (quick smoke test):
    python Lab/tensorflow.py --epochs 1 --batch_size 64 --augment --verbose

Recommended (full):
    python Lab/tensorflow.py --epochs 25 --batch_size 128 --lr 0.001 --augment --model_save_path cifar10_cnn.keras --plot_curves curves.png --plot_confusion cm.png

"""
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Suppress excessive TF logging (set externally if desired)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility (best-effort)."""
    tf.keras.utils.set_random_seed(seed)


def load_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 and return ((x_train, y_train), (x_test, y_test))."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # y_* are shape (N,1); flatten to (N,) for sklearn
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    return (x_train, y_train), (x_test, y_test)


def preprocess_images(x: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0,1] float32."""
    x = x.astype('float32') / 255.0
    return x


def build_augmentation_layer() -> tf.keras.Sequential:
    """Return a Keras Sequential of augmentation layers."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ], name='augmentation')


def conv_block(filters: int, kernel_size: int = 3, dropout: float = 0.0) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(dropout) if dropout > 0 else tf.keras.layers.Identity(),
    ])


def build_model(
    depth: int = 3,
    base_filters: int = 32,
    dropout: float = 0.25,
    augment: bool = False,
) -> tf.keras.Model:
    """Construct CNN model with variable depth.

    depth: number of conv blocks
    base_filters: filters in first block (doubles each block)
    dropout: dropout after pooling
    augment: include augmentation submodel at input
    """
    inputs = tf.keras.Input(shape=(32, 32, 3), name='images')
    x = inputs
    if augment:
        aug = build_augmentation_layer()
        x = aug(x)
    filters = base_filters
    for i in range(depth):
        x = conv_block(filters, dropout=dropout)(x)
        filters *= 2
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name='cifar10_cnn')
    return model


def compile_model(model: tf.keras.Model, lr: float = 1e-3) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


def plot_training(history: tf.keras.callbacks.History, save_path: str | None = None) -> None:
    plt.figure(figsize=(10, 4))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Training curves saved to {save_path}")
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: str | None = None) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Confusion matrix saved to {save_path}")
    plt.show()


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.seed is not None:
        set_seed(args.seed)
    (x_train, y_train), (x_test, y_test) = load_data()

    # Show dataset head
    if args.verbose:
        print("Dataset sample (first 3 images pixel ranges):")
        print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
        for i in range(3):
            print(f"Sample {i} label={CLASS_NAMES[y_train[i]]}")

    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    # Validation split from training set
    val_split = args.val_split
    num_val = int(x_train.shape[0] * val_split)
    x_val, y_val = x_train[:num_val], y_train[:num_val]
    x_train2, y_train2 = x_train[num_val:], y_train[num_val:]

    model = build_model(
        depth=args.depth,
        base_filters=args.base_filters,
        dropout=args.dropout,
        augment=args.augment,
    )
    compile_model(model, lr=args.lr)
    if args.verbose:
        model.summary()

    callbacks: list[tf.keras.callbacks.Callback] = []
    if args.early_stop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True, monitor='val_loss'))
    if args.model_save_path:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(args.model_save_path, save_best_only=True, monitor='val_loss'))
    if args.lr_schedule:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=max(1, args.patience//2), factor=0.5))

    history = model.fit(
        x_train2, y_train2,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1 if args.verbose else 0,
    )

    # Evaluation on test set
    test_probs = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(test_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    if args.verbose:
        print(f"Test Accuracy: {acc:.4f}")
        print("Classification Report:\n", report)

    if args.plot_curves:
        plot_training(history, save_path=args.plot_curves)
    if args.plot_confusion:
        plot_confusion_matrix(cm, CLASS_NAMES, save_path=args.plot_confusion)

    if args.save_report:
        with open(args.save_report, 'w') as f:
            f.write(f"Accuracy: {acc}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
        print(f"[INFO] Report saved to {args.save_report}")

    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'history': history.history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TensorFlow CIFAR-10 CNN Trainer')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--depth', type=int, default=3, help='Number of convolutional blocks')
    parser.add_argument('--base_filters', type=int, default=32, help='Filters in first conv block')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate inside conv blocks')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of training used for validation')
    parser.add_argument('--early_stop', action='store_true', help='Enable EarlyStopping callback')
    parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping / LR schedule')
    parser.add_argument('--lr_schedule', action='store_true', help='Enable ReduceLROnPlateau learning rate schedule')
    parser.add_argument('--model_save_path', type=str, help='Path to save best model (.keras or .h5)')
    parser.add_argument('--plot_curves', type=str, help='Path to save training curves plot')
    parser.add_argument('--plot_confusion', type=str, help='Path to save confusion matrix plot')
    parser.add_argument('--save_report', type=str, help='File to save evaluation report')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output and model summary')
    return parser.parse_args()


def main():
    args = parse_args()
    results = train_and_evaluate(args)
    # Print final metric for quick reference
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")


if __name__ == '__main__':
    main()


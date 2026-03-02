"""CIFAR-10 CNN Training (TensorFlow)

Standalone training script avoiding name-conflict with the TensorFlow package.
Run:
  python Lab/cifar10_tf.py --epochs 1 --batch_size 128 --augment --verbose
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)

def load_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    return (x_train, y_train), (x_test, y_test)

def preprocess_images(x: np.ndarray) -> np.ndarray:
    return x.astype('float32') / 255.0

def build_augmentation_layer() -> tf.keras.Sequential:
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

def build_model(depth: int = 3, base_filters: int = 32, dropout: float = 0.25, augment: bool = False) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = inputs
    if augment:
        x = build_augmentation_layer()(x)
    filters = base_filters
    for _ in range(depth):
        x = conv_block(filters, dropout=dropout)(x)
        filters *= 2
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

def compile_model(model: tf.keras.Model, lr: float = 1e-3) -> None:
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def plot_training(history: tf.keras.callbacks.History, save_path: str | None = None) -> None:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history: plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    if 'val_accuracy' in history.history: plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
    if save_path: plt.savefig(save_path); print(f'[INFO] Curves saved to {save_path}')
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, save_path: str | None = None) -> None:
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix'); plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45); plt.yticks(ticks, CLASS_NAMES)
    thresh = cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j]>thresh else 'black')
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
    if save_path: plt.savefig(save_path); print(f'[INFO] Confusion matrix saved to {save_path}')
    plt.show()

def train_and_evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.seed is not None: set_seed(args.seed)
    (x_train, y_train), (x_test, y_test) = load_data()
    if args.verbose:
        print('x_train:', x_train.shape, 'y_train:', y_train.shape)
        print('Sample labels:', [CLASS_NAMES[y_train[i]] for i in range(3)])
    x_train = preprocess_images(x_train); x_test = preprocess_images(x_test)
    num_val = int(x_train.shape[0] * args.val_split)
    x_val, y_val = x_train[:num_val], y_train[:num_val]
    x_train2, y_train2 = x_train[num_val:], y_train[num_val:]
    model = build_model(args.depth, args.base_filters, args.dropout, args.augment)
    compile_model(model, args.lr)
    if args.verbose: model.summary()
    callbacks: list[tf.keras.callbacks.Callback] = []
    if args.early_stop: callbacks.append(tf.keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True, monitor='val_loss'))
    if args.model_save_path: callbacks.append(tf.keras.callbacks.ModelCheckpoint(args.model_save_path, save_best_only=True, monitor='val_loss'))
    if args.lr_schedule: callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=max(1,args.patience//2), factor=0.5))
    history = model.fit(x_train2, y_train2, validation_data=(x_val,y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=1 if args.verbose else 0)
    probs = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    if args.verbose:
        print(f'Test Accuracy: {acc:.4f}')
        print(report)
    if args.plot_curves: plot_training(history, args.plot_curves)
    if args.plot_confusion: plot_confusion_matrix(cm, args.plot_confusion)
    if args.save_report:
        with open(args.save_report,'w') as f:
            f.write(f'Accuracy: {acc}\n\n{report}\n\nConfusion Matrix:\n{cm}')
        print(f'[INFO] Report saved to {args.save_report}')
    return {'accuracy': acc, 'confusion_matrix': cm, 'classification_report': report, 'history': history.history}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CIFAR-10 CNN Trainer (TensorFlow)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--base_filters', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.25)
    p.add_argument('--augment', action='store_true')
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--early_stop', action='store_true')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr_schedule', action='store_true')
    p.add_argument('--model_save_path', type=str)
    p.add_argument('--plot_curves', type=str)
    p.add_argument('--plot_confusion', type=str)
    p.add_argument('--save_report', type=str)
    p.add_argument('--seed', type=int)
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    results = train_and_evaluate(args)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")

if __name__ == '__main__':
    main()

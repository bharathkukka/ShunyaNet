"""
Example demonstrating basic usage of ShunyaNet components without requiring data files.
This shows that all modules are correctly set up and functional.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath('../ShunyaNet'))

print("=" * 70)
print("ShunyaNet Usage Example")
print("=" * 70)
print()

# Example 1: Import and instantiate PyTorch ShunyaNet
print("1. PyTorch ShunyaNet (Emotion Recognition)")
print("-" * 70)

from EmotionRecognitionSystem.ShunyaNetArchitecture import ShunyaNet

# Create model for 8 emotion classes
model_emotion = ShunyaNet(num_classes=8, dropblock_prob=0.1, dropblock_size=5)
print(f"✓ Created ShunyaNet for emotion recognition (8 classes)")

# Test forward pass with dummy data
dummy_input = torch.randn(2, 3, 96, 96)  # Batch of 2 images, 96x96 RGB
output = model_emotion(dummy_input)
print(f"✓ Forward pass successful")
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Output logits sample: {output[0][:3].detach().numpy()}")
print()

# Example 2: Count model parameters
print("2. Model Statistics")
print("-" * 70)

total_params = sum(p.numel() for p in model_emotion.parameters())
trainable_params = sum(p.numel() for p in model_emotion.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print()

# Example 3: Import Cotton Disease Recognition model
print("3. PyTorch ShunyaNet (Cotton Disease Recognition)")
print("-" * 70)

from CottonDiseaseRecognition.ShunyaNetArch import ShunyaNet as ShunyaNetCotton

model_cotton = ShunyaNetCotton(num_classes=4)
print(f"✓ Created ShunyaNet for cotton disease (4 classes)")

dummy_input_cotton = torch.randn(1, 3, 224, 224)  # 224x224 RGB
output_cotton = model_cotton(dummy_input_cotton)
print(f"✓ Forward pass successful")
print(f"  Input shape: {dummy_input_cotton.shape}")
print(f"  Output shape: {output_cotton.shape}")
print()

# Example 4: Import TensorFlow ShunyaNet
print("4. TensorFlow ShunyaNet (Paddy Disease Recognition)")
print("-" * 70)

import tensorflow as tf
from PaddyDiseaseRecognition.ShunyaNetTensorflow import ShunyaNet as ShunyaNetTF

# Suppress TensorFlow warnings
import logging
tf.get_logger().setLevel(logging.ERROR)

model_paddy = ShunyaNetTF(num_classes=10)
print(f"✓ Created TensorFlow ShunyaNet for paddy disease (10 classes)")

# Build the model by calling it with sample input
dummy_input_tf = tf.random.normal((1, 224, 224, 3))
output_tf = model_paddy(dummy_input_tf, training=False)
print(f"✓ Forward pass successful")
print(f"  Input shape: {dummy_input_tf.shape}")
print(f"  Output shape: {output_tf.shape}")
print()

# Example 5: Dataset class usage
print("5. Dataset Class Structure")
print("-" * 70)

from emotion_dataset import EmotionDataset
from torchvision import transforms

print("✓ EmotionDataset class imported successfully")
print("  Expected usage:")
print("    dataset = EmotionDataset(")
print("        root_dir='path/to/data',")
print("        split='train',")
print("        target_size=(96, 96),")
print("        augment=True")
print("    )")
print()

# Example 6: Show available transforms
print("6. Available Data Augmentations")
print("-" * 70)

transforms_list = [
    "transforms.Resize(target_size)",
    "transforms.RandomHorizontalFlip()",
    "transforms.RandomRotation(10)",
    "transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)",
    "transforms.RandomVerticalFlip()",
    "transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))",
    "transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0))",
    "transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))",
    "transforms.ToTensor()",
    "transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
]

for t in transforms_list:
    print(f"  • {t}")
print()

# Summary
print("=" * 70)
print("Summary")
print("=" * 70)
print("✓ All ShunyaNet architectures are functional")
print("✓ PyTorch models: Emotion Recognition, Cotton Disease Recognition")
print("✓ TensorFlow models: Paddy Disease Recognition")
print("✓ Dataset classes are ready for use")
print()
print("To train models, ensure you have:")
print("  1. Dataset in the correct directory structure")
print("  2. Sufficient compute resources (GPU recommended)")
print("  3. All dependencies installed (see Test/README.md)")
print()
print("Next steps:")
print("  • Run 'python smoke_test.py' to verify all components")
print("  • Prepare your dataset following the structure in Test/README.md")
print("  • Run 'python train_emotion_model.py' or 'python colab_emotion_classifier_combined.py'")
print("  • After training, run 'python Validation.py' to evaluate the model")
print()

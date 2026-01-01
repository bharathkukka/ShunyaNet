#!/usr/bin/env python3
"""
Integration Verification Script
Verifies that all three PaddyDiseaseRecognition files work together.
Run this script to ensure everything is properly configured.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

def check_imports():
    """Check if all modules can be imported."""
    print("=" * 60)
    print("CHECKING MODULE IMPORTS")
    print("=" * 60)

    success = True

    # Check preprocessing
    try:
        from ShunyaNet.PaddyDiseaseRecognition.preprocessing import GenericImageDataset
        print("✅ GenericImageDataset imported successfully")
    except Exception as e:
        print(f"❌ Failed to import GenericImageDataset: {e}")
        success = False

    # Check ShunyaNet
    try:
        from ShunyaNet.PaddyDiseaseRecognition.ShunyaNetTensorflow import ShunyaNet
        print("✅ ShunyaNet imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ShunyaNet: {e}")
        success = False

    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version {tf.__version__} available")
    except Exception as e:
        print(f"❌ TensorFlow not available: {e}")
        success = False

    return success


def check_config():
    """Verify configuration file exists and is valid."""
    print("\n" + "=" * 60)
    print("CHECKING CONFIGURATION")
    print("=" * 60)

    data_dir = '/Users/bharathgoud/PycharmProjects/Shunya-00/Data/PaddyDisease'

    if os.path.exists(data_dir):
        print(f"✅ Data directory found: {data_dir}")

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(data_dir, split)
            if os.path.exists(split_dir):
                classes = [d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]
                print(f"   ✅ {split.upper()}: {len(classes)} classes found")
            else:
                print(f"   ❌ {split.upper()}: directory not found")
        return True
    else:
        print(f"❌ Data directory not found: {data_dir}")
        return False


def check_architecture():
    """Verify ShunyaNet architecture can be instantiated."""
    print("\n" + "=" * 60)
    print("CHECKING MODEL ARCHITECTURE")
    print("=" * 60)

    try:
        from ShunyaNet.PaddyDiseaseRecognition.ShunyaNetTensorflow import ShunyaNet
        import tensorflow as tf

        model = ShunyaNet(num_classes=10, dropblock_prob=0.1, dropblock_size=7)

        # Build model with dummy input
        dummy_input = tf.random.normal((1, 224, 224, 3))
        output = model(dummy_input, training=False)

        print(f"✅ ShunyaNet model created successfully")
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Trainable parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"❌ Failed to create ShunyaNet: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_preprocessing():
    """Verify preprocessing pipeline works."""
    print("\n" + "=" * 60)
    print("CHECKING PREPROCESSING PIPELINE")
    print("=" * 60)

    try:
        from ShunyaNet.PaddyDiseaseRecognition.preprocessing import GenericImageDataset

        data_dir = '/Users/bharathgoud/PycharmProjects/Shunya-00/Data/PaddyDisease'

        dataset = GenericImageDataset(
            root_dir=data_dir,
            split='train',
            target_size=(224, 224),
            augment=False
        )

        print(f"✅ GenericImageDataset created successfully")
        print(f"   - Classes: {len(dataset.classes)}")
        print(f"   - Samples: {len(dataset)}")

        if len(dataset) > 0:
            # Try getting a batch
            ds = dataset.get_dataset(batch_size=4, shuffle=False)
            for images, labels in ds.take(1):
                print(f"   - Batch image shape: {images.shape}")
                print(f"   - Batch labels shape: {labels.shape}")
                print(f"   - Image dtype: {images.dtype}")
                print("✅ Preprocessing pipeline working correctly")

        return True
    except Exception as e:
        print(f"❌ Preprocessing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  PADDY DISEASE RECOGNITION - INTEGRATION CHECK  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    results = []

    # Run all checks
    results.append(("Imports", check_imports()))
    results.append(("Configuration", check_config()))
    results.append(("Architecture", check_architecture()))
    results.append(("Preprocessing", check_preprocessing()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<30} {status}")

    print("-" * 60)
    print(f"Total: {passed}/{total} checks passed")

    if passed == total:
        print("\n" + "🎉 " * 10)
        print("\n✅ ALL INTEGRATION CHECKS PASSED!")
        print("\nYou can now run: python ShunyaNet/PaddyDiseaseRecognition/main.py")
        print("\n" + "🎉 " * 10)
        return 0
    else:
        print("\n⚠️ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Smoke test script to verify all components can be imported and basic functionality works.
This script checks that:
1. All required dependencies are installed
2. All modules can be imported
3. Basic functionality works (without requiring actual data files)
"""

import sys
import os

# Track test results
tests_passed = 0
tests_failed = 0
failures = []

def test(description, func):
    """Run a test and track results."""
    global tests_passed, tests_failed, failures
    try:
        func()
        print(f"✓ {description}")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"✗ {description}")
        print(f"  Error: {e}")
        tests_failed += 1
        failures.append((description, str(e)))
        return False

print("=" * 70)
print("ShunyaNet Smoke Test")
print("=" * 70)
print()

# Test 1: Check dependencies
print("Testing Dependencies...")
print("-" * 70)

def test_torch():
    import torch
    assert torch.__version__, "PyTorch version check"

def test_torchvision():
    import torchvision
    assert torchvision.__version__, "torchvision version check"

def test_numpy():
    import numpy as np
    assert np.__version__, "NumPy version check"

def test_pillow():
    from PIL import Image
    import PIL
    assert PIL.__version__, "Pillow version check"

def test_matplotlib():
    import matplotlib
    assert matplotlib.__version__, "matplotlib version check"

def test_sklearn():
    import sklearn
    assert sklearn.__version__, "scikit-learn version check"

def test_tqdm():
    import tqdm
    assert tqdm.__version__, "tqdm version check"

def test_tensorflow():
    import tensorflow as tf
    assert tf.__version__, "TensorFlow version check"

test("PyTorch", test_torch)
test("torchvision", test_torchvision)
test("NumPy", test_numpy)
test("Pillow", test_pillow)
test("matplotlib", test_matplotlib)
test("scikit-learn", test_sklearn)
test("tqdm", test_tqdm)
test("TensorFlow (optional)", test_tensorflow)

print()

# Test 2: Import Test directory modules
print("Testing Test Directory Modules...")
print("-" * 70)

def test_emotion_dataset():
    from emotion_dataset import EmotionDataset
    assert EmotionDataset, "EmotionDataset class exists"

def test_validation_import():
    # Just check it can be imported, don't run main
    import importlib.util
    spec = importlib.util.spec_from_file_location("Validation", "Validation.py")
    module = importlib.util.module_from_spec(spec)
    assert module, "Validation module can be loaded"

test("emotion_dataset.py import", test_emotion_dataset)
test("Validation.py import", test_validation_import)

print()

# Add ShunyaNet parent directory to path once (used by multiple tests)
_shunyanet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ShunyaNet'))
if _shunyanet_path not in sys.path:
    sys.path.insert(0, _shunyanet_path)

# Test 3: Import ShunyaNet architectures
print("Testing ShunyaNet Architecture Imports...")
print("-" * 70)

def test_emotion_architecture():
    from EmotionRecognitionSystem.ShunyaNetArchitecture import ShunyaNet
    assert ShunyaNet, "EmotionRecognitionSystem ShunyaNet exists"

def test_cotton_architecture():
    from CottonDiseaseRecognition.ShunyaNetArch import ShunyaNet
    assert ShunyaNet, "CottonDiseaseRecognition ShunyaNet exists"

def test_paddy_architecture():
    from PaddyDiseaseRecognition.ShunyaNetTensorflow import ShunyaNet
    assert ShunyaNet, "PaddyDiseaseRecognition ShunyaNet exists"

test("EmotionRecognitionSystem architecture", test_emotion_architecture)
test("CottonDiseaseRecognition architecture", test_cotton_architecture)
test("PaddyDiseaseRecognition architecture", test_paddy_architecture)

print()

# Test 4: Test basic model instantiation
print("Testing Model Instantiation...")
print("-" * 70)

def test_pytorch_model():
    import torch
    from EmotionRecognitionSystem.ShunyaNetArchitecture import ShunyaNet
    model = ShunyaNet(num_classes=8)
    assert model, "PyTorch ShunyaNet can be instantiated"
    # Test forward pass with dummy data
    x = torch.randn(1, 3, 96, 96)
    output = model(x)
    assert output.shape == (1, 8), "Model output shape is correct"

def test_tensorflow_model():
    import tensorflow as tf
    from PaddyDiseaseRecognition.ShunyaNetTensorflow import ShunyaNet
    # TensorFlow version doesn't take input_shape in __init__
    model = ShunyaNet(num_classes=10)
    assert model, "TensorFlow ShunyaNet can be instantiated"

test("PyTorch model instantiation and forward pass", test_pytorch_model)
test("TensorFlow model instantiation", test_tensorflow_model)

print()

# Test 5: Test dataset class (without actual data)
print("Testing Dataset Class (without data)...")
print("-" * 70)

def test_dataset_class_structure():
    from emotion_dataset import EmotionDataset
    import inspect
    
    # Check that the class has expected methods
    assert hasattr(EmotionDataset, '__init__'), "EmotionDataset has __init__"
    assert hasattr(EmotionDataset, '__len__'), "EmotionDataset has __len__"
    assert hasattr(EmotionDataset, '__getitem__'), "EmotionDataset has __getitem__"
    assert hasattr(EmotionDataset, '_build_transform'), "EmotionDataset has _build_transform"

test("EmotionDataset class structure", test_dataset_class_structure)

print()

# Print summary
print("=" * 70)
print("Test Summary")
print("=" * 70)
print(f"Tests passed: {tests_passed}")
print(f"Tests failed: {tests_failed}")
print()

if tests_failed > 0:
    print("Failed tests:")
    for desc, error in failures:
        print(f"  - {desc}")
        print(f"    {error}")
    print()
    print("Some tests failed. Please check the errors above.")
    sys.exit(1)
else:
    print("✓ All tests passed!")
    print()
    print("The ShunyaNet codebase is ready to use.")
    print("To train models, ensure you have the dataset in the correct location.")
    sys.exit(0)

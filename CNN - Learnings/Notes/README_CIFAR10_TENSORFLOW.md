# CIFAR-10 TensorFlow CNN

This project now includes a TensorFlow implementation for training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.

## Files
- `Lab/cifar10_tf.py`: Main training/evaluation script (avoid naming conflict with `tensorflow` package).
- `Lab/cifar10_tensorflow_lib.py`: Original more feature-rich version (renamed from `tensorflow.py` to prevent shadowing). Prefer `cifar10_tf.py` for quick runs.

## Usage
### Quick Smoke Test (1 epoch)
```bash
python Lab/cifar10_tf.py --epochs 1 --batch_size 128 --augment --verbose
```

### Typical Training
```bash
python Lab/cifar10_tf.py \
  --epochs 25 \
  --batch_size 128 \
  --lr 0.001 \
  --depth 3 \
  --base_filters 32 \
  --dropout 0.25 \
  --augment \
  --early_stop \
  --patience 7 \
  --lr_schedule \
  --model_save_path cifar10_best.keras \
  --plot_curves curves.png \
  --plot_confusion cm.png \
  --save_report eval_report.txt \
  --verbose
```

## Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| --epochs | Training epochs | 5 |
| --batch_size | Mini-batch size | 128 |
| --lr | Learning rate | 1e-3 |
| --depth | Number of convolutional blocks | 3 |
| --base_filters | Filters in first block (doubles each block) | 32 |
| --dropout | Dropout inside conv blocks | 0.25 |
| --augment | Enable data augmentation | False |
| --val_split | Fraction of training set for validation | 0.1 |
| --early_stop | Enable EarlyStopping | False |
| --patience | Patience for early stop / LR schedule | 5 |
| --lr_schedule | Enable ReduceLROnPlateau | False |
| --model_save_path | Path to save best model | None |
| --plot_curves | Save training curves image | None |
| --plot_confusion | Save confusion matrix image | None |
| --save_report | Save text evaluation report | None |
| --seed | Random seed | None |
| --verbose | Verbose + model summary | False |

## Dependencies
Add (or ensure) the following packages:
```
tensorflow
scikit-learn
matplotlib
numpy
```
Install:
```bash
pip install -r requirements.txt
```

## Notes
- Do not name your own script `tensorflow.py` inside the same environment; it shadows the actual TensorFlow package.
- First epoch accuracy will be low; train for ~25+ epochs for ~80%+ test accuracy.
- Data augmentation helps generalization.

## Next Steps / Improvements
- Add mixed precision training (tf.keras.mixed_precision.set_global_policy('mixed_float16')).
- Integrate TensorBoard logging.
- Add learning rate warmup and cosine decay.
- Implement cutmix or mixup augmentation.
- Add unit tests for model construction.


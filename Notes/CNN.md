# CNN Model Training – Notes & Best Practices

## 1. Architecture & Layers

| Component | Description | Source / Inspiration | Activation Used | Alternatives |
|---|---|---|---|---|
| Input Layer | Accepts image tensor input | Standard CNN pipeline | – | Different input sizes, normalization |
| Convolutional Layer (Conv) | Extracts spatial features using filters | LeNet, ResNet, Inception | ReLU | Leaky ReLU, ELU, Tanh, GELU |
| Pooling Layer | Downsamples feature maps | AlexNet, VGG | – | Max, Average, Global, Adaptive |
| Flatten Layer | Converts 2D feature maps to 1D vector | Standard CNN flow | – | GAP (Global Avg Pooling) |
| Fully Connected Layer (Dense/FC) | Learns high-level feature combinations | LeNet, VGG | Softmax / Sigmoid | Linear, SVM head |
| Dropout Layer | Prevents co-adaptation, reduces overfitting | Regularization method | – | DropBlock, SpatialDropout |
| Batch Normalization (BatchNorm) | Stabilizes training, reduces internal shift | Modern CNN training | – | LayerNorm, GroupNorm, InstanceNorm |
| Residual / Skip Connection | Helps gradient flow in deep networks | ResNet blocks | ReLU | Highway nets, Dense connections |
| Inception Module | Multi-scale parallel conv blocks | InceptionNet | ReLU | Xception, ResNeXt, EfficientNet blocks |
| Bottleneck Layer | 1×1 Conv for channel reduction | ResNet bottleneck | ReLU | Linear, Swish |
| Bottleneck / Feature Compression | Reduces parameters, speeds training | MobileNet, ResNet | ReLU | Swish, Mish |
| UpSampling / Transposed Conv | Used for segmentation and generative CNNs | U-Net, GANs | ReLU | Bilinear, nearest, sub-pixel |
| Depthwise Conv | Applies convolution per channel | MobileNet | ReLU | Standard conv |
| Pointwise Conv | 1×1 conv to combine channels | MobileNet | ReLU | Linear, GELU |
| Attention Block | Focuses on important regions/features | Vision Transformers, Attention CNNs | Sigmoid | SE, CBAM, Self-Attention |
| Activation Functions | Introduces non-linearity | Standard DL practice | ReLU | Leaky ReLU, Tanh, Sigmoid, ELU, Softmax, Tanh |

---

## 2. Hyperparameters & Configuration

- **Filters / Kernels** – Number of feature extractors
- **Kernel Size / Filter Size** – Size of convolution window (e.g., 3×3, 5×5)
- **Stride** – Step size of filter movement
- **Padding** – `Valid` (no padding), `Same` (zero padding)
- **Dilation** – Skips pixels to increase receptive field
- **Receptive Field** – Area of input influencing a feature
- **Depth / Channels** – Number of channels in feature maps
- **Learning Rate** – Controls update step size
- **Batch Size** – Number of samples per gradient update
- **Epochs** – Full dataset passes
- **Iterations / Steps per Epoch** – `ceil(N/B)` or `floor(N/B)`
- **Momentum** – Helps accelerate gradient direction
- **Weight Decay** – L2 regularization
- **Drop Last** – Drops incomplete batch in dataloader
- **Image Normalization / Standardization** – Mean-std scaling
- **Dataset Split Ratio** – Train/Val/Test configuration
- **Train/Val/Test Split Ratio** – Defines data distribution for model evaluation

---

## 3. The Training Process

- **Forward Pass** – Input flows through the network
- **Loss Computation** – Difference between prediction & ground truth
- **Backward Pass** – Gradients calculated using backpropagation
- **Backpropagation** – Chain rule based gradient flow
- **Gradient Descent** – Updates weights using gradients
- **Optimizers** – `Adam`, `SGD`, `RMSprop`, `Adagrad`, `Adadelta`
- **Loss Function** – `Cross-Entropy`, `MSE`, `Focal Loss`, `Hinge Loss`
- **Weights & Biases** – Learnable parameters
- **Learning Rate Scheduler** – `StepLR`, `ReduceLROnPlateau`, `CosineAnnealing`, `OneCycleLR`
- **Gradient Clipping** – Prevents exploding gradients
- **Gradient Clipping** – Limits gradient norm to avoid instability
- **Mixed Precision Training** – FP16 / BF16 training for speed and memory efficiency
- **Early Stopping** – Stops training when validation stops improving
- **Checkpointing** – Saves model state during training
- **Data Augmentation** – Random transformations (flip, crop, brightness, rotation, etc.)
- **Vanishing / Exploding Gradients** – Gradient instability in deep nets
- **Gradient Clipping** – Prevents exploding gradients
- **Validation Loop** – Evaluates model on validation set
- **Forward/Backward Pass** – Core computation steps
- **Gradient Clipping** – Prevents exploding gradients
- **Mixed Precision Training (FP16/BF16)** – Faster training, lower memory
- **Validation Loop** – Evaluates performance after each epoch
- **Gradient Clipping** – Prevents exploding gradients
- **Gradient Clipping** – Prevents exploding gradients
- **Mixed Precision Training** – Uses FP16/BF16 to reduce memory and increase speed
- **Validation Loop** – Runs evaluation on validation data every epoch

---

## 4. Transfer Learning & Refinement

- **Pre-trained Model** – Models trained on large datasets
- **Transfer Learning** – Reusing learned weights for new tasks
- **Fine-tuning** – Unfreezing selected layers and retraining
- **Freezing / Unfreezing Layers** – Controls which layers update
- **Weight Initialization** – `Xavier`, `He`, random normal/uniform
- **Feature Extraction** – Using CNN as fixed backbone
- **Classifier Head Replacement** – Modifying final layer for new classes
- **Domain Adaptation** – Adapting model to different data distribution
- **Pretrained Weights Source** – ImageNet, custom checkpoints, or domain-specific sources
- **Head Replacement / Classifier Head** – Replacing final layer for new task
- **Domain Adaptation** – Handling distribution shift between source and target data

---

## 5. Performance & Metrics

- **Inference** – Model prediction phase
- **Overfitting / Underfitting** – Model generalization issues
- **Accuracy** – Overall correct predictions
- **Validation Accuracy** – Accuracy on validation set
- **Precision / Recall** – Class-wise correctness and coverage
- **F1-Score** – Balance between precision and recall
- **Confusion Matrix** – True vs predicted distribution
- **IoU (Intersection over Union)** – Used in segmentation
- **NMS (Non-Max Suppression)** – Used in object detection
- **Top-K Accuracy** – Checks if correct class is in top K predictions
- **ROC-AUC Score** – Performance across thresholds
- **Confidence Score / Calibration** – Model certainty and probability alignment

---

## 6. Optimization for Deployment (Edge/Production)

- **Model Exporting** – ONNX, SavedModel, H5
- **Model Conversion** – TensorRT, CoreML, TFLite
- **Quantization** – Reduces precision (e.g., FP32 → INT8)
- **Pruning** – Removes less important weights
- **Depthwise Separable Convolutions** – Efficient conv operations
- **Attention Mechanisms** – SE, CBAM, self-attention
- **Latency** – Time per inference
- **Throughput** – Predictions per second
- **FPS (Frames Per Second)** – Used for real-time evaluation
- **Memory Footprint** – RAM/storage used by model
- **Model Versioning** – Tracking model updates
- **Hardware Acceleration** – GPU, TPU, NPU support
- **TensorRT / CoreML / TFLite Conversion** – Optimizes model for target device
- **FPS (Frames Per Second)** – Measures real-time inference speed
- **Memory Footprint / Memory Usage** – Tracks RAM/storage consumption
- **Model Versioning** – Maintains model updates and history
- **Hardware Acceleration** – Uses GPU/TPU/NPU for optimized inference

---

## Summary

Training a CNN efficiently depends on:
- Choosing the right architecture
- Tuning hyperparameters
- Using stable training techniques
- Evaluating with proper metrics
- Optimizing for deployment on target hardware

---


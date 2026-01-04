# ShunyaNet: A Unified CNN Backbone Combining the Best of Modern Architectures
## Project Overview
ShunyaNet is a custom-designed convolutional neural network (CNN) backbone that unifies the strongest architectural blocks from leading models such as ResNet, Inception, DenseNet, EfficientNet, and others. The goal is to combine the best ideas from modern deep learning research into a single, flexible, and high-performing architecture.

### Applications
ShunyaNet was used to train models for three real-world tasks:
1. **Emotion Recognition** (8 emotion classes, PyTorch)
2. **Cotton Disease Detection** (4 classes, PyTorch)
3. **Paddy Disease Detection** (10 classes, TensorFlow/Keras, via PyTorch-to-TensorFlow conversion)

### Framework Portability
The architecture was first implemented and trained in PyTorch for emotion and cotton disease recognition. To demonstrate portability and reproducibility, the same design was re-implemented in TensorFlow/Keras and successfully trained for paddy disease detection. This highlights ShunyaNet’s flexibility and the importance of cross-framework compatibility in modern AI workflows.

---

## ShunyaNet Architecture Overview       
#### [For More Details](Notes/ShunyaNet-Class-Docstrings.md) 

| Architecture Block      | Purpose                                 | Source Model              | Activation Function used | Possible Alternatives         | Why this one was chosen                                      |
|------------------------|-----------------------------------------|---------------------------|-------------------------|-------------------------------|--------------------------------------------------------------|
| Swish                  | Smooth, non-monotonic activation for better gradients | EfficientNet, MobileNetV2 | Swish                   | ReLU, GELU, Mish              | Outperforms ReLU in deep nets, better gradient flow          |
| DropBlock2D            | Structured dropout for regularization   | ResNet, EfficientNet      | -                       | Standard Dropout              | Forces spatial robustness, better for CNNs                   |
| InceptionBlock         | Multi-scale feature extraction          | Inception (GoogLeNet)     | ReLU/Swish               | Standard Conv, ResBlock       | Captures fine & coarse features in parallel                  |
| SEBlock                | Channel-wise attention (feature recalibration) | SENet, EfficientNet       | Swish/ReLU               | CBAM, ECA, Squeeze-Excite     | Lightweight, boosts accuracy with minimal cost               |
| ResidualDenseBlock     | Combines residual & dense connections   | DenseNet, ResNet          | ReLU/Swish               | Standard Residual, DenseBlock  | Maximizes feature reuse, improves gradient flow              |
| MBConv                 | Efficient feature extraction (mobile bottleneck) | MobileNetV2, EfficientNet | Swish                    | Standard Conv, GhostModule     | Fewer params, high accuracy, mobile-friendly                 |
| GhostModule            | Cheap feature generation (efficient convolutions) | GhostNet                  | ReLU/Swish               | Standard Conv, MBConv         | Reduces computation/memory, maintains performance            |
| DualAttention (CBAM)   | Channel & spatial attention             | CBAM                      | Sigmoid (attention)      | SEBlock, ECA, BAM             | Lightweight, improves focus on relevant features             |
| SKConv                 | Dynamic receptive field selection       | SKNet                     | ReLU/Softmax (attention) | Inception, Standard Conv       | Adapts to object scale, flexible feature extraction          |
| ReZeroResidualBlock    | Stabilizes very deep residual networks  | ReZero                    | Linear + learnable alpha | Standard Residual, Pre-activation | Enables very deep nets, fast convergence                 |
| CSPInception           | Efficient feature splitting/merging     | CSPNet, YOLOv4            | ReLU/Swish               | Standard Inception, CSPResNet  | Better gradient flow, less computation                       |
| GlobalContextBlock     | Adds global context to each spatial location | GCNet, ConvNeXt           | ReLU/Swish               | Squeeze-Excite, Non-local      | Models long-range dependencies, holistic understanding       |
| MHSA (Multi-Head Self-Attention) | Models global dependencies via attention | Transformer, ViT          | Softmax (attention)      | Standard Conv, Non-local       | Captures complex relationships, state-of-the-art in vision   |
| AttentionPooling       | Importance-based feature pooling        | Attention pooling (NLP/CV) | Softmax/Sigmoid          | GlobalAvgPool, MaxPool         | Focuses on key regions, improves aggregation for prediction  |

---  

## Training Details

|  Parameter         | [Emotion Recognition](ShunyaNet/EmotionRecognitionSystem)                            | [Cotton Disease Recognition ](ShunyaNet/CottonDiseaseRecognition) | [Paddy Disease Recognition](ShunyaNet/PaddyDiseaseRecognition) |
|--------------------|----------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------|
| **PC Specifications** | Intel i7-10750H 12th Gen, 16GB RAM, Windows 11 | Intel i7-10750H 12th Gen, 16GB RAM, Windows 11                    | Intel i7-10750H 12th Gen, 16GB RAM, Windows 11                 |
| **Python Version** | Python 3.14.x                          | Python 3.14.x                                                     | Python 3.11.x                                                  |
| **Framework**      | PyTorch 1.13                           | PyTorch 1.13                                                      | TensorFlow 2.10 (Keras)                                        |
| **Batch Size**     | 16                                     | 16                                                                | 2                                                              |
| **Epochs**         | 52                                     | 42                                                                | 42                                                             |
| **Iterations per Epoch** | 23229/16 = ~1452                       | 1366/16 = ~86                                                     | 8323/2 = ~ 4162                                                |
| **Train Images**   | 23229                                  | 1365                                                              | 8323                                                           |
| **Validation Images** | 2900                                   | 168                                                               | 1036                                                           |
| **Test Images**    | 2913                                   | 176                                                               | 1048                                                           |
| **Number of Classes** | 8                                      | 4                                                                 | 10                                                             |
| **Loss Function**  | CrossEntropyLoss                       | CrossEntropyLoss                                                  | SparseCategoricalCrossentropy                                  |
| **Optimizer**      | AdamW                                  | AdamW                                                             | AdamW                                                          |
| **Learning Rate**  | 0.001 (cosine annealing)               | 0.001 (cosine annealing)                                          | 0.001 (ReduceLROnPlateau)                                      |
| **Data Augmentation** | Flip, Crop, Color Jitter               | Crop, Flip, Rotation, Color Jitter, Blur                          | Flip, Rotation, Zoom, Brightness/Contrast                      |
| **Regularization** | DropBlock2D, Weight Decay (1e-5)       | DropBlock2D, Weight Decay (1e-5)                                  | Dropout (0.3), Weight Decay (1e-5)                             |
| **Input Image Size** | 96x96                                  | 224x224                                                           | 224x224                                                        |
| **Early Stopping** | Yes                         | Yes                                                               | Yes                                                            |
| **Model Checkpointing** | Best val accuracy, checkpoints every 5 epochs | Best val accuracy, checkpoints every 5 epochs                     | Best val accuracy, checkpoints every 5 epochs                  |
| **Transfer Learning** | No                                     | No                                                                | No                                                             |
| **Total Training Time** | 20                                     | 30                                                                | 34                                                             |



---
### Final Outcome
ShunyaNet achieved strong, generalizable performance across all three tasks, validating the effectiveness of combining advanced blocks from multiple architectures. The project serves as a blueprint for building robust, portable, and high-performing CNN backbones for diverse computer vision challenges.

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

| Architecture Block      | Purpose / What it does                                 | Source Model              | Activation Function used | Possible Alternatives         | Why this one was chosen                                      |
|------------------------|--------------------------------------------------------|---------------------------|-------------------------|-------------------------------|--------------------------------------------------------------|
| Swish                  | Smooth, non-monotonic activation for better gradients  | EfficientNet, MobileNetV2 | Swish                   | ReLU, GELU, Mish              | Outperforms ReLU in deep nets, better gradient flow          |
| DropBlock2D            | Structured dropout for regularization                  | ResNet, EfficientNet      | -                       | Standard Dropout              | Forces spatial robustness, better for CNNs                   |
| InceptionBlock         | Multi-scale feature extraction                         | Inception (GoogLeNet)     | ReLU/Swish               | Standard Conv, ResBlock       | Captures fine & coarse features in parallel                  |
| SEBlock                | Channel-wise attention (feature recalibration)         | SENet, EfficientNet       | Swish/ReLU               | CBAM, ECA, Squeeze-Excite     | Lightweight, boosts accuracy with minimal cost               |
| ResidualDenseBlock     | Combines residual & dense connections                  | DenseNet, ResNet          | ReLU/Swish               | Standard Residual, DenseBlock  | Maximizes feature reuse, improves gradient flow              |
| MBConv                 | Efficient feature extraction (mobile bottleneck)       | MobileNetV2, EfficientNet | Swish                    | Standard Conv, GhostModule     | Fewer params, high accuracy, mobile-friendly                 |
| GhostModule            | Cheap feature generation (efficient convolutions)      | GhostNet                  | ReLU/Swish               | Standard Conv, MBConv         | Reduces computation/memory, maintains performance            |
| DualAttention (CBAM)   | Channel & spatial attention                           | CBAM                      | Sigmoid (attention)      | SEBlock, ECA, BAM             | Lightweight, improves focus on relevant features             |
| SKConv                 | Dynamic receptive field selection                      | SKNet                     | ReLU/Softmax (attention) | Inception, Standard Conv       | Adapts to object scale, flexible feature extraction          |
| ReZeroResidualBlock    | Stabilizes very deep residual networks                 | ReZero                    | Linear + learnable alpha | Standard Residual, Pre-activation | Enables very deep nets, fast convergence                 |
| CSPInception           | Efficient feature splitting/merging                    | CSPNet, YOLOv4            | ReLU/Swish               | Standard Inception, CSPResNet  | Better gradient flow, less computation                       |
| GlobalContextBlock     | Adds global context to each spatial location           | GCNet, ConvNeXt           | ReLU/Swish               | Squeeze-Excite, Non-local      | Models long-range dependencies, holistic understanding       |
| MHSA (Multi-Head Self-Attention) | Models global dependencies via attention      | Transformer, ViT          | Softmax (attention)      | Standard Conv, Non-local       | Captures complex relationships, state-of-the-art in vision   |
| AttentionPooling       | Importance-based feature pooling                       | Attention pooling (NLP/CV) | Softmax/Sigmoid          | GlobalAvgPool, MaxPool         | Focuses on key regions, improves aggregation for prediction  |

---  

### Final Outcome
ShunyaNet achieved strong, generalizable performance across all three tasks, validating the effectiveness of combining advanced blocks from multiple architectures. The project serves as a blueprint for building robust, portable, and high-performing CNN backbones for diverse computer vision challenges.
  
---  
## Training Details

| Component / Parameter         | Emotion Recognition                                 | Cotton Disease Recognition                          | Paddy Disease Recognition                       |
|------------------------------|-----------------------------------------------------|-----------------------------------------------------|--------------------------------------------------|
| **PC Specifications**        | Intel i7-10750H 12th Gen, 16GB RAM, Windows 11      | Intel i7-10750H 12th Gen, 16GB RAM, Windows 11      | Intel i7-10750H 12th Gen, 16GB RAM, Windows 11   |
| **Python Version**           | Python 3.10                                         | Python 3.10                                         | Python 3.10                                       |
| **Framework**                | PyTorch 1.13                                        | PyTorch 1.13                                        | TensorFlow 2.10 (Keras)                          |
| **Batch Size**               | 16                                                  | 16                                                  | 8                                                |
| **Epochs**                   | 34                                                  | 34                                                  | 30                                               |
| **Iterations per Epoch**     | train_loader length (depends on dataset/batch)      | train_loader length (depends on dataset/batch)      | train_loader length (depends on dataset/batch)   |
| **Train Images**             | len(train_dataset)                                  | len(train_dataset)                                  | len(train_dataset)                               |
| **Validation Images**        | len(val_dataset)                                    | len(val_dataset)                                    | len(val_dataset)                                 |
| **Test Images**              | len(test_dataset)                                   | len(test_dataset)                                   | len(test_dataset)                                |
| **Number of Classes**        | 8                                                   | 4                                                   | 10                                               |
| **Loss Function**            | CrossEntropyLoss                                    | CrossEntropyLoss                                    | SparseCategoricalCrossentropy                    |
| **Reported Loss (best/val)** | Printed in logs                                     | Printed in logs                                     | Printed in logs                                  |
| **Reported Accuracy**        | Printed in logs                                     | Printed in logs                                     | Printed in logs                                  |
| **Optimizer**                | AdamW                                               | AdamW                                               | AdamW                                            |
| **Learning Rate**            | 0.001 (cosine annealing)                            | 0.001 (cosine annealing)                            | 0.001 (ReduceLROnPlateau)                        |
| **Data Augmentation**        | Flip, Crop, Color Jitter                            | Crop, Flip, Rotation, Color Jitter, Blur            | Flip, Rotation, Zoom, Brightness/Contrast        |
| **Regularization**           | DropBlock2D, Weight Decay (1e-5)                    | DropBlock2D, Weight Decay (1e-5)                    | Dropout (0.3), Weight Decay (1e-5)               |
| **Input Image Size**         | 224x224                                             | 224x224                                             | 224x224                                          |
| **Precision**                | FP32                                                | FP32                                                | FP32                                             |
| **Early Stopping**           | Yes (logic present)                                 | Yes (logic present)                                 | Yes (logic present)                              |
| **Model Checkpointing**      | Best val accuracy, checkpoints every 5 epochs       | Best val accuracy, checkpoints every 5 epochs       | Best val accuracy, checkpoints every 5 epochs    |
| **Transfer Learning**        | No                                                  | No                                                  | No                                               |
| **Total Training Time**      | Not specified (depends on hardware/dataset)         | Not specified (depends on hardware/dataset)         | Not specified (depends on hardware/dataset)      |


---
### Final Outcome
ShunyaNet achieved strong, generalizable performance across all three tasks, validating the effectiveness of combining advanced blocks from multiple architectures. The project serves as a blueprint for building robust, portable, and high-performing CNN backbones for diverse computer vision challenges.
  
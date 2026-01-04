# ShunyaNet: A Unified CNN Backbone Combining the Best of Modern Architectures
## Project Overview
ShunyaNet is a custom-designed convolutional neural network (CNN) backbone that unifies the strongest architectural blocks from leading models such as ResNet, Inception, DenseNet, EfficientNet, and others. The goal is to combine the best ideas from modern deep learning research into a single, flexible, and high-performing architecture.

### Why ShunyaNet Was Created
While many state-of-the-art CNNs excel in specific tasks, each has unique strengths and limitations. ShunyaNet was created to:
- Integrate the most effective architectural innovations (e.g., residual connections, multi-scale feature extraction, attention mechanisms) into one backbone.
- Provide a robust, modular foundation for diverse computer vision tasks.
- Enable easy experimentation and transfer across domains.

### Problems Solved
- **Fragmentation of best practices:** ShunyaNet brings together proven blocks from multiple architectures, reducing the need to choose between them.
- **Task versatility:** The unified design allows the same backbone to be used for very different image classification problems.
- **Framework portability:** By supporting both PyTorch and TensorFlow/Keras, ShunyaNet demonstrates how advanced architectures can be ported across deep learning ecosystems.

### Applications
ShunyaNet was used to train models for three real-world tasks:
1. **Emotion Recognition** (8 emotion classes, PyTorch)
2. **Cotton Disease Detection** (4 classes, PyTorch)
3. **Paddy Disease Detection** (10 classes, TensorFlow/Keras, via PyTorch-to-TensorFlow conversion)

### Framework Portability
The architecture was first implemented and trained in PyTorch for emotion and cotton disease recognition. To demonstrate portability and reproducibility, the same design was re-implemented in TensorFlow/Keras and successfully trained for paddy disease detection. This highlights ShunyaNet’s flexibility and the importance of cross-framework compatibility in modern AI workflows.

### Final Outcome
ShunyaNet achieved strong, generalizable performance across all three tasks, validating the effectiveness of combining advanced blocks from multiple architectures. The project serves as a blueprint for building robust, portable, and high-performing CNN backbones for diverse computer vision challenges.

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


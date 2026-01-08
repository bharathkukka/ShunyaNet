# Sequential vs Functional API in PyTorch Architectures

## What is Sequential?
- `nn.Sequential` is a container module in PyTorch that allows you to stack layers in a linear order.
- You define the order of layers, and data flows through them one after another.
- Example:
```python
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
```
- Good for simple models where each layer's output is the next layer's input.

## What is Functional?
- Functional API means you define the forward pass manually in the `forward()` method.
- You can use layers in any order, reuse outputs, branch, merge, or apply custom logic.
- Example:
```python
def forward(self, x):
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    return x2 + x1  # skip connection
```
- Needed for complex architectures (like Inception, Residual, Attention blocks).

## Notes on Functional API
- Functional API is not a separate module, but a way of writing the `forward()` method in your custom `nn.Module` class.
- You can:
    - Pass data through layers in any order.
    - Use outputs from earlier layers later in the network (skip connections).
    - Merge, split, or branch data flows (e.g., multi-path blocks).
    - Apply custom operations, conditions, or logic (e.g., if-else, loops).
    - Combine outputs from different blocks (ensembles, attention, etc.).
- This flexibility is essential for modern architectures like ResNet, DenseNet, Transformers, etc.
- Example of branching and merging:
```python
def forward(self, x):
    branch1 = self.conv1(x)
    branch2 = self.conv2(x)
    merged = torch.cat([branch1, branch2], dim=1)
    out = self.final_conv(merged)
    return out
```
- You can also use PyTorch's functional operations (from `torch.nn.functional`) for activations, pooling, etc., directly in the forward pass.
- Functional API makes your model more expressive and customizable, but requires you to manage tensor shapes and connections yourself.

## How I Used These in My Architecture
- In my ShunyaNet, I used both techniques:
    - For simple blocks (stem, classifier), I used `nn.Sequential` to stack layers.
    - For custom blocks (Inception, ResidualDense, MBConv, etc.), I used the functional approach in their `forward()` methods to control data flow, add skip connections, and combine outputs.
- Example from my code:
    - The stem and classifier are defined using `nn.Sequential`.
    - Blocks like InceptionBlock, ResidualDenseBlock, MBConv, etc., use functional logic in their `forward()`.
    - The main `forward()` in ShunyaNet chains these blocks together, sometimes combining outputs (ensemble of classifier and attention pooling).

## Why Use Both?
- Sequential is simple and readable for straightforward layer stacks.
- Functional is flexible and powerful for advanced architectures (branching, merging, attention, skip connections).
- Mixing both lets me build complex models efficiently.

## Summary
- Use Sequential for simple, linear stacks.
- Use Functional for anything more complex.
- My architecture combines both for best results.
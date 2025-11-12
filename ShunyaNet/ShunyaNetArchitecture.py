import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish Activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# DropBlock Regularization
class DropBlock2D(nn.Module):
    """DropBlock regularization for 2D feature maps."""
    def __init__(self, block_size, drop_prob):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < gamma).float()
        block_mask = F.max_pool2d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        out = x * (1 - block_mask)
        scale = block_mask.numel() / (block_mask.sum() + 1e-6)  # avoid division by zero
        out = out * scale
        return out

# Inception Block
class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Residual Dense Block
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, in_channels, kernel_size=1)
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        return x + x3

# MBConv Block
class MBConv(nn.Module):
    def __init__(self, in_channels, activation=None, expansion_factor=6):
        super().__init__()
        if activation is None:
            activation = Swish()
        hidden_dim = in_channels * expansion_factor
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            activation,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            activation,
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )
        self.se = SEBlock(in_channels)
    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return out + x

# Ghost Module
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, ratio=2):
        super().__init__()
        if activation is None:
            activation = Swish()
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_channels),
            activation
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            activation
        )
    def forward(self, x):
        primary = self.primary_conv(x)
        cheap = self.cheap_operation(primary)
        return torch.cat([primary, cheap], dim=1)

# Dual Attention Block (CBAM)
class DualAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        ca = self.channel_att(x)
        x = x * ca
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        sa = self.spatial_att(torch.cat([avg, max_], dim=1))
        return x * sa

# Selective Kernel Convolution
class SKConv(nn.Module):
    def __init__(self, in_channels, activation=None, M=2, G=8, r=16):
        super().__init__()
        if activation is None:
            activation = Swish()
        d = max(in_channels // r, 32)
        self.M = M
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3 + 2 * i, padding=1 + i, groups=G),
                nn.BatchNorm2d(in_channels),
                activation
            ) for i in range(M)
        ])
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, d, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, in_channels * M, 1)
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)
        attn = self.fc(torch.sum(feats, dim=1)).view(x.size(0), self.M, -1, 1, 1)
        attn = self.softmax(attn)
        out = torch.sum(feats * attn, dim=1)
        return out

# ReZero Residual Block
class ReZeroResidualBlock(nn.Module):
    def __init__(self, channels, activation=None):
        super().__init__()
        if activation is None:
            activation = Swish()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            activation,
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x + self.alpha * self.block(x)

# CSP-Inception Block
class CSPInception(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.split = in_channels // 2
        self.inception_path = nn.Sequential(
            nn.Conv2d(self.split, self.split, 1),
            nn.Conv2d(self.split, self.split, 3, padding=1),
            nn.Conv2d(self.split, self.split, 5, padding=2)
        )
        self.concat_conv = nn.Conv2d(self.split * 2, in_channels, 1)
    def forward(self, x):
        x1, x2 = torch.split(x, self.split, dim=1)
        out = self.inception_path(x1)
        out = torch.cat([x2, out], dim=1)
        out = self.concat_conv(out)
        return out

# Global Context Block (ConvNeXt/GCNet)
class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, activation=None):
        super().__init__()
        if activation is None:
            activation = Swish()
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            activation,
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
    def forward(self, x):
        # Global context
        context = self.pool(x)
        # Transform context
        context = self.block(context)
        # Apply context back to input
        return x + context.expand_as(x)

# Multi-Head Self-Attention (MHSA, optional)
class MHSA(nn.Module):
    def __init__(self, in_channels, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (C // self.heads) ** 0.5, dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        out = self.proj(out)
        return out + x

# Attention Pooling (for classifier head)
class AttentionPooling(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.attn = nn.Conv2d(in_channels, 1, 1)
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, x):
        w = torch.softmax(self.attn(x).view(x.size(0), -1), dim=1).view(x.size(0), 1, x.size(2), x.size(3))
        x = (x * w).sum(dim=[2, 3])
        return self.fc(x)

# ShunyaNet: Combined Architecture
class ShunyaNet(nn.Module):
    def __init__(self, num_classes=10, dropblock_prob=0.1, dropblock_size=7, activation=None):
        super().__init__()
        if activation is None:
            activation = Swish()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation
        )
        self.inception = InceptionBlock(64)
        self.se = SEBlock(128)
        self.res_dense = ResidualDenseBlock(128)
        self.mbconv = MBConv(128, activation=activation)
        self.ghost = GhostModule(128, 128, activation=activation)
        self.sk = SKConv(128, activation=activation)
        self.dual_att = DualAttention(128)
        self.csp_inception = CSPInception(128)
        self.rezero = ReZeroResidualBlock(128, activation=activation)
        self.global_context = GlobalContextBlock(128, activation=activation)
        self.mhsa = MHSA(128)
        self.dropblock = DropBlock2D(dropblock_size, dropblock_prob)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.attn_pool = AttentionPooling(128, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.stem(x)
        x = self.inception(x)
        x = self.se(x)
        x = self.res_dense(x)
        x = self.mbconv(x)
        x = self.ghost(x)
        x = self.sk(x)
        x = self.dual_att(x)
        x = self.csp_inception(x)
        x = self.rezero(x)
        x = self.global_context(x)
        x = self.mhsa(x)
        x = self.dropblock(x)
        # Option 1: Standard classifier
        out1 = self.classifier(x)
        # Option 2: Attention pooling classifier
        out2 = self.attn_pool(x)
        return (out1 + out2) / 2  # Ensemble output

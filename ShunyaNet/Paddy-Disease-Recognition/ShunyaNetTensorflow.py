import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Swish Activation
class Swish(layers.Layer):
    def call(self, x):
        return x * tf.nn.sigmoid(x)

# DropBlock Regularization
class DropBlock2D(layers.Layer):
    """DropBlock regularization for 2D feature maps."""
    def __init__(self, block_size, drop_prob):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        mask = tf.cast(tf.random.uniform([batch_size, height, width, 1]) < gamma, tf.float32)

        # Apply max pooling to create blocks
        block_mask = tf.nn.max_pool2d(mask, ksize=self.block_size, strides=1, padding='SAME')

        out = x * (1 - block_mask)
        scale = tf.cast(tf.size(block_mask), tf.float32) / (tf.reduce_sum(block_mask) + 1e-6)
        out = out * scale
        return out

# Inception Block
class InceptionBlock(layers.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = layers.Conv2D(32, kernel_size=1, padding='same')

        self.branch2_conv1 = layers.Conv2D(32, kernel_size=1, padding='same')
        self.branch2_conv2 = layers.Conv2D(32, kernel_size=3, padding='same')

        self.branch3_conv1 = layers.Conv2D(32, kernel_size=1, padding='same')
        self.branch3_conv2 = layers.Conv2D(32, kernel_size=5, padding='same')

        self.branch4_pool = layers.MaxPooling2D(pool_size=3, strides=1, padding='same')
        self.branch4_conv = layers.Conv2D(32, kernel_size=1, padding='same')

    def call(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2_conv1(x)
        branch2 = self.branch2_conv2(branch2)

        branch3 = self.branch3_conv1(x)
        branch3 = self.branch3_conv2(branch3)

        branch4 = self.branch4_pool(x)
        branch4 = self.branch4_conv(branch4)

        return tf.concat([branch1, branch2, branch3, branch4], axis=-1)

# Squeeze-and-Excitation Block
class SEBlock(layers.Layer):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Dense(in_channels // reduction, activation='relu')
        self.fc2 = layers.Dense(in_channels, activation='sigmoid')

    def call(self, x):
        # Global average pooling
        se = self.pool(x)
        # Squeeze spatial dimensions and apply FC layers
        se = tf.squeeze(se, axis=[1, 2]) if len(se.shape) == 4 else se
        se = self.fc1(se)
        se = self.fc2(se)
        # Reshape and multiply with input
        se = tf.reshape(se, [-1, 1, 1, tf.shape(se)[-1]])
        return x * se

# Residual Dense Block
class ResidualDenseBlock(layers.Layer):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv1 = layers.Conv2D(growth_rate, kernel_size=3, padding='same')
        self.conv2 = layers.Conv2D(growth_rate, kernel_size=3, padding='same')
        self.conv3 = layers.Conv2D(in_channels, kernel_size=1, padding='same')

    def call(self, x):
        x1 = tf.nn.relu(self.conv1(x))
        x2 = tf.nn.relu(self.conv2(tf.concat([x, x1], axis=-1)))
        x3 = self.conv3(tf.concat([x, x1, x2], axis=-1))
        return x + x3

# MBConv Block
class MBConv(layers.Layer):
    def __init__(self, in_channels, activation=None, expansion_factor=6):
        super().__init__()
        if activation is None:
            activation = Swish()
        hidden_dim = in_channels * expansion_factor

        self.conv1 = layers.Conv2D(hidden_dim, kernel_size=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.activation1 = activation

        self.conv2 = layers.Conv2D(hidden_dim, kernel_size=3, padding='same', groups=hidden_dim)
        self.bn2 = layers.BatchNormalization()
        self.activation2 = activation

        self.conv3 = layers.Conv2D(in_channels, kernel_size=1, padding='same')
        self.bn3 = layers.BatchNormalization()

        self.se = SEBlock(in_channels)

    def call(self, x, training=None):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.activation2(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        out = self.se(out)
        return out + x

# Ghost Module
class GhostModule(layers.Layer):
    def __init__(self, in_channels, out_channels, activation=None, ratio=2):
        super().__init__()
        if activation is None:
            activation = Swish()
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels

        self.primary_conv = keras.Sequential([
            layers.Conv2D(init_channels, kernel_size=1, use_bias=False, padding='same'),
            layers.BatchNormalization(),
            activation
        ])

        self.cheap_operation = keras.Sequential([
            layers.Conv2D(new_channels, kernel_size=3, padding='same', groups=init_channels, use_bias=False),
            layers.BatchNormalization(),
            activation
        ])

    def call(self, x):
        primary = self.primary_conv(x)
        cheap = self.cheap_operation(primary)
        return tf.concat([primary, cheap], axis=-1)

# Dual Attention Block (CBAM)
class DualAttention(layers.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = keras.Sequential([
            layers.GlobalAveragePooling2D(keepdims=True),
            layers.Conv2D(in_channels // 8, kernel_size=1, padding='same'),
            layers.ReLU(),
            layers.Conv2D(in_channels, kernel_size=1, padding='same'),
            layers.Activation('sigmoid')
        ])

        self.spatial_att = keras.Sequential([
            layers.Conv2D(1, kernel_size=7, padding='same'),
            layers.Activation('sigmoid')
        ])

    def call(self, x):
        ca = self.channel_att(x)
        x = x * ca
        avg = tf.reduce_mean(x, axis=1, keepdims=True)
        max_ = tf.reduce_max(x, axis=1, keepdims=True)
        sa = self.spatial_att(tf.concat([avg, max_], axis=1))
        return x * sa

# Selective Kernel Convolution
class SKConv(layers.Layer):
    def __init__(self, in_channels, activation=None, M=2, G=8, r=16):
        super().__init__()
        if activation is None:
            activation = Swish()
        d = max(in_channels // r, 32)
        self.M = M
        self.in_channels = in_channels
        self.G = G

        self.convs = []
        for i in range(M):
            kernel_size = 3 + 2 * i
            padding = 1 + i
            self.convs.append(keras.Sequential([
                layers.Conv2D(in_channels, kernel_size=kernel_size, padding=padding, groups=G),
                layers.BatchNormalization(),
                activation
            ]))

        self.fc = keras.Sequential([
            layers.GlobalAveragePooling2D(keepdims=True),
            layers.Conv2D(d, kernel_size=1, padding='same'),
            layers.ReLU(),
            layers.Conv2D(in_channels * M, kernel_size=1, padding='same')
        ])

    def call(self, x):
        feats = tf.stack([conv(x) for conv in self.convs], axis=1)
        attn = self.fc(tf.reduce_sum(feats, axis=1))
        attn = tf.reshape(attn, [tf.shape(x)[0], self.M, -1, 1, 1])
        attn = tf.nn.softmax(attn, axis=1)
        out = tf.reduce_sum(feats * attn, axis=1)
        return out

# ReZero Residual Block
class ReZeroResidualBlock(layers.Layer):
    def __init__(self, channels, activation=None):
        super().__init__()
        if activation is None:
            activation = Swish()
        self.conv1 = layers.Conv2D(channels, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.activation = activation
        self.conv2 = layers.Conv2D(channels, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.alpha = tf.Variable(tf.zeros([1]), trainable=True)

    def call(self, x, training=None):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        return x + self.alpha * out

# CSP-Inception Block
class CSPInception(layers.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.split = in_channels // 2
        self.inception_path = keras.Sequential([
            layers.Conv2D(self.split, kernel_size=1, padding='same'),
            layers.Conv2D(self.split, kernel_size=3, padding='same'),
            layers.Conv2D(self.split, kernel_size=5, padding='same')
        ])
        self.concat_conv = layers.Conv2D(in_channels, kernel_size=1, padding='same')

    def call(self, x):
        x1, x2 = tf.split(x, 2, axis=-1)
        out = self.inception_path(x1)
        out = tf.concat([x2, out], axis=-1)
        out = self.concat_conv(out)
        return out

# Global Context Block (ConvNeXt/GCNet)
class GlobalContextBlock(layers.Layer):
    def __init__(self, in_channels, activation=None):
        super().__init__()
        if activation is None:
            activation = Swish()
        self.in_channels = in_channels
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.block = keras.Sequential([
            layers.Conv2D(in_channels, kernel_size=1, padding='same'),
            activation,
            layers.Conv2D(in_channels, kernel_size=1, padding='same')
        ])

    def call(self, x):
        # Global context
        context = self.pool(x)
        # Transform context
        context = self.block(context)
        # Apply context back to input
        return x + context

# Multi-Head Self-Attention (MHSA)
class MHSA(layers.Layer):
    def __init__(self, in_channels, heads=4):
        super().__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.qkv = layers.Conv2D(in_channels * 3, kernel_size=1, padding='same')
        self.proj = layers.Conv2D(in_channels, kernel_size=1, padding='same')

    def call(self, x):
        B = tf.shape(x)[0]
        C = self.in_channels
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, H * W, 3, self.heads, C // self.heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = tf.cast((C // self.heads) ** 0.5, tf.float32)
        attn = tf.matmul(q, k, transpose_b=True) / scale
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, H, W, C])
        out = self.proj(out)
        return out + x

# Attention Pooling (for classifier head)
class AttentionPooling(layers.Layer):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.attn = layers.Conv2D(1, kernel_size=1, padding='same')
        self.fc = layers.Dense(num_classes)

    def call(self, x):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]

        w = self.attn(x)
        w = tf.reshape(w, [B, -1])
        w = tf.nn.softmax(w, axis=1)
        w = tf.reshape(w, [B, 1, H, W])

        x = tf.reduce_sum(x * w, axis=[1, 2])
        return self.fc(x)

# ShunyaNet: Combined Architecture
class ShunyaNet(keras.Model):
    def __init__(self, num_classes=10, dropblock_prob=0.1, dropblock_size=7, activation=None):
        super().__init__()
        if activation is None:
            activation = Swish()

        self.stem = keras.Sequential([
            layers.Conv2D(64, kernel_size=3, stride=2, padding=1),
            layers.BatchNormalization(),
            activation
        ])

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

        self.classifier = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(num_classes)
        ])

        self.attn_pool = AttentionPooling(128, num_classes)

        self.num_classes = num_classes

    def call(self, x, training=None):
        x = self.stem(x, training=training)
        x = self.inception(x)
        x = self.se(x)
        x = self.res_dense(x)
        x = self.mbconv(x, training=training)
        x = self.ghost(x)
        x = self.sk(x)
        x = self.dual_att(x)
        x = self.csp_inception(x)
        x = self.rezero(x, training=training)
        x = self.global_context(x)
        x = self.mhsa(x)
        x = self.dropblock(x, training=training)

        # Option 1: Standard classifier
        out1 = self.classifier(x)
        # Option 2: Attention pooling classifier
        out2 = self.attn_pool(x)

        # Ensemble output
        return (out1 + out2) / 2

"""
TensorFlow implementation of image preprocessing and dataset loading.
Converted from PyTorch torchvision implementation.
"""
import os
import tensorflow as tf
from tensorflow.image import random_brightness, random_contrast, random_saturation


class GenericImageDataset:
    """TensorFlow-based generic image dataset loader with augmentation support."""

    def __init__(self, root_dir, split='train', target_size=(128, 128), augment=False):
        self.root_dir = os.path.join(root_dir, split)
        self.target_size = target_size
        self.augment = augment

        # Filter out hidden files/folders that start with a dot
        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if not d.startswith('.') and os.path.isdir(os.path.join(self.root_dir, d))
        ])

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        # Accept common image extensions (case-insensitive)
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp',
                   '.JPG', '.JPEG', '.PNG')

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_dir):
                low = fname.lower()
                if low.endswith(valid_exts) and not fname.startswith('.'):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def _load_image(self, image_path):
        """Load image from file path."""
        image = tf.io.read_file(image_path)
        # Try to decode as JPEG first, then PNG
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except:
            image = tf.image.decode_png(image, channels=3)
        return image

    def _augment_image(self, image):
        """Apply data augmentation for training."""
        # Random resized crop (scale between 0.8 and 1.0 of target size)
        height, width = self.target_size

        # Random size
        random_scale = tf.random.uniform([], 0.8, 1.0)
        new_height = tf.cast(tf.cast(height, tf.float32) / random_scale, tf.int32)
        new_width = tf.cast(tf.cast(width, tf.float32) / random_scale, tf.int32)

        # Resize to new size
        image = tf.image.resize(image, [new_height, new_width])

        # Random crop to target size
        image = tf.image.random_crop(image, [height, width, 3])

        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Color jitter: brightness, contrast, saturation
        image = random_brightness(image, 0.2)
        image = random_contrast(image, 0.8, 1.2)
        image = random_saturation(image, 0.8, 1.2)

        # Gaussian blur approximation using average pooling
        if tf.random.uniform([]) < 0.5:
            image = tf.nn.avg_pool2d(
                tf.expand_dims(image, 0),
                ksize=3,
                strides=1,
                padding='SAME'
            )
            image = tf.squeeze(image, 0)

        return image

    def _preprocess_image(self, image_path, label):
        """Load and preprocess image."""
        # Load image
        image = self._load_image(image_path)
        image = tf.cast(image, tf.float32)

        # Apply augmentation if training
        if self.augment:
            image = self._augment_image(image)
        else:
            # For validation/test, simple resize
            image = tf.image.resize(image, self.target_size)

        # Normalize to [0, 1]
        image = image / 255.0

        # ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - mean) / std  # type: ignore

        return image, label

    def get_dataset(self, batch_size=16, shuffle=True):
        """Create and return a TensorFlow tf.data.Dataset."""
        # Extract file paths and labels
        image_paths = [s[0] for s in self.samples]
        labels = [s[1] for s in self.samples]

        # Create dataset from slices
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # Shuffle if needed
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.samples))

        # Map preprocessing function with parallel processing
        dataset = dataset.map(
            lambda path, label: self._preprocess_image(path, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def __len__(self):
        return len(self.samples)


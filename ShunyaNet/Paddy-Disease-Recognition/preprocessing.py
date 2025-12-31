import os
import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf

class GenericImageDataset:
    """
    Generic image dataset for loading images from directory structure.
    Compatible with TensorFlow/Keras.
    """
    def __init__(self, root_dir, split='train', target_size=(96, 96), augment=False):
        self.root_dir = os.path.join(root_dir, split)
        self.target_size = target_size
        self.augment = augment

        # Filter out hidden files/folders that start with a dot
        self.classes = sorted([d for d in os.listdir(self.root_dir)
                              if not d.startswith('.') and os.path.isdir(os.path.join(self.root_dir, d))])

        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        self.samples = []
        # Accept common image extensions (case-insensitive)
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_dir):
                low = fname.lower()
                if low.endswith(valid_exts) and not fname.startswith('.'):
                    self.samples.append((os.path.join(class_dir, fname), label))

        # Normalization constants (ImageNet mean and std)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        # Read image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.float32)

        # Apply augmentation or deterministic preprocessing
        if self.augment:
            image = self._augment_image(image)
        else:
            # For val/test, keep deterministic resize (no crop/flip)
            image = self._resize_image(image, self.target_size)

        # Normalize (ImageNet normalization)
        image = image / 255.0
        image = (image - self.mean) / self.std

        return image.astype(np.float32)

    def _resize_image(self, image, target_size):
        """Resize image to target size."""
        pil_image = Image.fromarray((image).astype(np.uint8))
        pil_image = pil_image.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(pil_image, dtype=np.float32)

    def _augment_image(self, image):
        """Apply data augmentation to image."""
        pil_image = Image.fromarray(image.astype(np.uint8))

        # Random Resized Crop
        width, height = pil_image.size
        scale_factor = np.random.uniform(0.8, 1.0)
        crop_size = int(min(width, height) * scale_factor)
        x = np.random.randint(0, width - crop_size)
        y = np.random.randint(0, height - crop_size)
        pil_image = pil_image.crop((x, y, x + crop_size, y + crop_size))
        pil_image = pil_image.resize(self.target_size, Image.Resampling.BILINEAR)

        # Random Horizontal Flip
        if np.random.random() > 0.5:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Random Rotation
        angle = np.random.uniform(-10, 10)
        pil_image = pil_image.rotate(angle, expand=False, fillcolor='white')

        # Color Jitter
        image_array = np.array(pil_image, dtype=np.float32)
        brightness_factor = np.random.uniform(0.8, 1.2)
        contrast_factor = np.random.uniform(0.8, 1.2)
        saturation_factor = np.random.uniform(0.8, 1.2)

        image_array = image_array * brightness_factor
        image_array = np.clip(image_array, 0, 255)

        # Simple contrast adjustment
        image_array = (image_array - 128) * contrast_factor + 128
        image_array = np.clip(image_array, 0, 255)

        # Convert back to PIL for saturation (via HSV)
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        import colorsys
        pil_array = np.array(pil_image, dtype=np.float32) / 255.0

        # Simple saturation adjustment in RGB space
        hsv_image = np.zeros_like(pil_array)
        for i in range(pil_array.shape[0]):
            for j in range(pil_array.shape[1]):
                r, g, b = pil_array[i, j]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                s = s * saturation_factor
                s = np.clip(s, 0, 1)
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                hsv_image[i, j] = [r, g, b]

        image_array = hsv_image * 255.0
        pil_image = Image.fromarray(image_array.astype(np.uint8))

        # Gaussian Blur
        if np.random.random() > 0.5:
            sigma = np.random.uniform(0.1, 2.0)
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=sigma))

        return np.array(pil_image, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._preprocess_image(img_path)
        return image, label

    def create_tf_dataset(self, batch_size=32, shuffle=True):
        """
        Create a TensorFlow dataset from this data loader.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset

        Returns:
            tf.data.Dataset: A TensorFlow dataset
        """
        def generator():
            indices = np.arange(len(self))
            if shuffle:
                np.random.shuffle(indices)
            for idx in indices:
                image, label = self[idx]
                yield image, label

        output_signature = (
            tf.TensorSpec(shape=self.target_size + (3,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


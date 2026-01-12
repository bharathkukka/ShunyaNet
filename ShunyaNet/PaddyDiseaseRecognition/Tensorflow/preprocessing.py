"""
This file loads images using TensorFlow, and applies preprocessing + augmentation if I enable it.
Earlier I implemented this using PyTorch (torchvision), now I converted that into TensorFlow/Keras format
so I can train models without changing my dataset folder structure.

Important:
- My dataset is already organized into disease folders, so labels come from folder names.
- No CSV is required for training or validation.
"""

import os
import tensorflow as tf
from tensorflow.image import random_brightness, random_contrast, random_saturation


class GenericImageDataset:
    """
    This class loads images from train/val folders and prepares them for model training.
    It also supports basic augmentations if I want to increase training data variety.
    """

    def __init__(self, root_dir, split='train', target_size=(224, 224), augment=False):
        """
        What I give it:
        - root_dir     : main dataset folder path
        - split        : which subset I want to load (train or val)
        - target_size  : final image size my model should receive (default 224x224)
        - augment      : if True, random augmentations will run while loading images
        """
        self.root_dir = os.path.join(root_dir, split)
        self.target_size = target_size
        self.augment = augment

        # Read all disease/class folders inside train or val path, ignore hidden folders
        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if not d.startswith('.') and os.path.isdir(os.path.join(self.root_dir, d))
        ])

        # Assign a number index to each disease folder name, this helps TensorFlow map labels
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Store (image_path, label_index) pairs here
        self.samples = []

        # Valid image file formats I want to allow
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp',
                      '.JPG', '.JPEG', '.PNG')

        # Loop through each disease folder and collect image file paths
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(valid_exts) and not fname.startswith('.'):
                    self.samples.append((os.path.join(class_dir, fname), label))

        print(f"Images loaded from {split} folder: {len(self.samples)}")
        print(f"Disease classes detected: {self.classes}")

    def _load_image(self, image_path):
        """
        Reads image file and decodes it into a 3-channel RGB image.
        First try JPEG decode, if that fails, try PNG decode.
        """
        image = tf.io.read_file(image_path)
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except:
            image = tf.image.decode_png(image, channels=3)
        return image

    def _augment_image(self, image):
        """
        Applies augmentations randomly, only if I enabled augment=True.
        This makes training more robust because images will not always look the same.
        """
        h, w = self.target_size

        # Randomly scale image and then crop back to target size
        scale = tf.random.uniform([], 0.8, 1.0)
        new_h = tf.cast(tf.cast(h, tf.float32) / scale, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) / scale, tf.int32)

        image = tf.image.resize(image, [new_h, new_w])
        image = tf.image.random_crop(image, [h, w, 3])

        # Random flip
        image = tf.image.random_flip_left_right(image)

        # Small color variations (brightness, contrast, saturation)
        image = random_brightness(image, 0.2)
        image = random_contrast(image, 0.8, 1.2)
        image = random_saturation(image, 0.8, 1.2)

        # Optional light blur using avg pooling, randomly applied
        if tf.random.uniform([]) < 0.5:
            image = tf.nn.avg_pool2d(tf.expand_dims(image, 0), 3, 1, padding='SAME')
            image = tf.squeeze(image, 0)

        return image

    def _preprocess_image(self, image_path, label):
        """
        This loads the image and resizes it.
        If augmentation is OFF → normal resize.
        If augmentation is ON → random augment + resize.
        Then normalize to ImageNet mean/std so model gets clean input.
        """
        image = self._load_image(image_path)
        image = tf.cast(image, tf.float32)

        # If augment is disabled, directly resize
        if self.augment:
            image = self._augment_image(image)
        else:
            image = tf.image.resize(image, self.target_size)

        # Convert pixel values to 0-1 range
        image = image / 255.0

        # Normalize using ImageNet standard values
        mean = tf.constant([0.485, 0.456, 0.406], tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], tf.float32)
        image = (image - mean) / std

        return image, label

    def get_dataset(self, batch_size=16, shuffle=True):
        """
        This converts my collected image paths into a TensorFlow dataset pipeline.
        It will shuffle if shuffle=True, then batch images, and prefetch for faster loading.
        """
        # Separate image paths and labels into 2 lists
        image_paths = [s[0] for s in self.samples]
        labels = [s[1] for s in self.samples]

        # Create TensorFlow dataset from these lists
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        if shuffle:
            # Shuffle images so model doesn't learn in order
            dataset = dataset.shuffle(buffer_size=len(self.samples))

        # Apply preprocessing while loading images, use parallel calls for speed
        dataset = dataset.map(
            lambda path, label: self._preprocess_image(path, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch images and load next batches while model trains
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def __len__(self):
        # Just returns total image samples stored
        return len(self.samples)


"""
Where I will provide my dataset path?

I give it when I initialize the class, like:

dataset = GenericImageDataset(
    root_dir="/Users/Bharath/Datasets/Paddy",
    split="train",
    target_size=(224,224),
    augment=True
)

So, the dataset path is passed directly into root_dir.
"""

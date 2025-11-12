import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GenericImageDataset(Dataset):

    def __init__(self, root_dir, split='train', target_size=(224, 224), augment=False):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.target_size = target_size

        # Filter out hidden files/folders that start with a dot
        self.classes = sorted([d for d in os.listdir(self.root_dir)
                              if not d.startswith('.') and os.path.isdir(os.path.join(self.root_dir, d))])

        # Build samples list: [(image_path, label), ...]
        self.samples = []
        IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_dir):
                f_lower = fname.lower()
                if f_lower.endswith(IMG_EXTS) and not fname.startswith('.'):
                    self.samples.append((os.path.join(class_dir, fname), label))

        self.transform = self._build_transform(target_size, augment)

    def _build_transform(self, target_size, augment):
        """
        Build the transformation pipeline similar to Emotion PreProcessing:
        - For training (augment=True): RandomResizedCrop to target_size + horizontal flip + rotation + color jitter (hue=0.05) + gaussian blur
        - For val/test (augment=False): simple Resize(target_size) without center crop
        - Converts to tensor and normalizes
        """
        tfms = []

        if augment:
            # Match PreProcessing.py choices: no vertical flip or affine; smaller hue jitter
            tfms += [
                transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ]
        else:
            # Deterministic resize to exact target size (may distort aspect ratio), mirroring PreProcessing.py
            tfms += [
                transforms.Resize(target_size),
            ]

        tfms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        return transforms.Compose(tfms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

    # def get_class_counts(self):
    #     """Return the number of samples per class."""
    #     class_counts = {}
    #     for _, label in self.samples:
    #         class_name = self.classes[label]
    #         class_counts[class_name] = class_counts.get(class_name, 0) + 1
    #     return class_counts

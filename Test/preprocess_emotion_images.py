import os
from PIL import Image
from torchvision import transforms

# NOTE: If you use PreProcessing.py for dynamic preprocessing, you do NOT need to run this script or use preprocessed images. Use the original dataset and let EmotionDataset handle preprocessing on-the-fly.

# Source and destination directories
SRC_ROOT = '../Data/EmotionRecognitionSystem/8Emotions/'
DST_ROOT = '../Data/EmotionRecognitionSystem/8Emotions-Preprocessed/'
SPLITS = ['train', 'val', 'test']
TARGET_SIZE = (96, 96)

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
val_test_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

os.makedirs(DST_ROOT, exist_ok=True)
for split in SPLITS:
    src_split_dir = os.path.join(SRC_ROOT, split)
    dst_split_dir = os.path.join(DST_ROOT, split)
    os.makedirs(dst_split_dir, exist_ok=True)
    for emotion in os.listdir(src_split_dir):
        src_emotion_dir = os.path.join(src_split_dir, emotion)
        dst_emotion_dir = os.path.join(dst_split_dir, emotion)
        if not os.path.isdir(src_emotion_dir):
            continue
        os.makedirs(dst_emotion_dir, exist_ok=True)
        for fname in os.listdir(src_emotion_dir):
            if fname.lower().endswith('.jpg'):
                src_path = os.path.join(src_emotion_dir, fname)
                dst_path = os.path.join(dst_emotion_dir, fname)
                try:
                    img = Image.open(src_path).convert('RGB')
                    if split == 'train':
                        tensor_img = train_transform(img)
                    else:
                        tensor_img = val_test_transform(img)
                    # Convert tensor back to PIL Image for saving
                    # Undo normalization for saving
                    img_np = tensor_img.mul(0.5).add(0.5).clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
                    img_out = Image.fromarray(img_np)
                    img_out.save(dst_path)
                except Exception as e:
                    print(f'Error processing {src_path}: {e}')
print('Preprocessing complete. Preprocessed images saved to:', DST_ROOT)

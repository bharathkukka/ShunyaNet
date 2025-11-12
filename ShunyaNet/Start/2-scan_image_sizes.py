import os
from PIL import Image

# Set the root directory for your dataset
DATASET_ROOT = '/Users/bharathgoud/PycharmProjects/Shunya-00/Data/PaddyDiseases/Dataset'
SPLITS = ['train', 'val']

# All common image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.JPG', '.JPEG', '.PNG'}

sizes = {}
total_images = 0
error_count = 0

for split in SPLITS:
    split_dir = os.path.join(DATASET_ROOT, split)
    if not os.path.exists(split_dir):
        print(f'Warning: {split_dir} does not exist, skipping...')
        continue

    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue

        for fname in os.listdir(class_dir):
            # Check if file has a valid image extension
            if any(fname.endswith(ext) for ext in IMAGE_EXTENSIONS):
                fpath = os.path.join(class_dir, fname)
                try:
                    with Image.open(fpath) as img:
                        size = img.size  # (width, height)
                        sizes[size] = sizes.get(size, 0) + 1
                        total_images += 1
                except Exception as e:
                    print(f'Error reading {fpath}: {e}')
                    error_count += 1

print('=' * 70)
print('IMAGE SIZE DISTRIBUTION REPORT')
print('=' * 70)
print(f'\nTotal images scanned: {total_images}')
print(f'Unique sizes found: {len(sizes)}')
if error_count > 0:
    print(f'Errors encountered: {error_count}')
print('\nSize distribution (sorted by frequency):')
print('-' * 70)

for size, count in sorted(sizes.items(), key=lambda x: -x[1]):
    percentage = (count / total_images) * 100 if total_images > 0 else 0
    print(f'  {size[0]:>4}x{size[1]:<4} : {count:>6} images ({percentage:>5.1f}%)')

print('=' * 70)


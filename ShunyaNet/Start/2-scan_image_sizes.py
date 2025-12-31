import os
from PIL import Image

DATASET_ROOT = '/Users/bharathgoud/PycharmProjects/Shunya-00/Data/CottonDisease'
SPLITS = ['train', 'val', 'test']

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp',
                   '.JPG', '.JPEG', '.PNG'}

sizes = {}
total_images = 0
error_count = 0

for split in SPLITS:
    split_dir = os.path.join(DATASET_ROOT, split)
    if not os.path.exists(split_dir):
        print(f'Warning: {split_dir} not found, skipping...')
        continue

    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue

        for fname in os.listdir(class_dir):
            ext = os.path.splitext(fname)[1]
            if ext in IMAGE_EXTENSIONS:
                fpath = os.path.join(class_dir, fname)
                try:
                    with Image.open(fpath) as img:
                        size = img.size  # (width, height)
                        sizes[size] = sizes.get(size, 0) + 1
                        total_images += 1
                except Exception as e:
                    print(f'Error reading {fpath}: {e}')
                    error_count += 1

print('=' * 60)
print('IMAGE SIZE DISTRIBUTION REPORT')
print('=' * 60)
print(f'\nTotal images scanned: {total_images}')
print(f'Unique resolutions found: {len(sizes)}')
if error_count > 0:
    print(f'Corrupted/Unreadable images: {error_count}')

print('\nResolution distribution (sorted by count):')
print('-' * 60)

for (w, h), count in sorted(sizes.items(), key=lambda x: -x[1]):
    percent = (count / total_images) * 100 if total_images else 0
    print(f'{w:>4}x{h:<4} : {count:>5} images ({percent:>5.1f}%)')

print('=' * 60)

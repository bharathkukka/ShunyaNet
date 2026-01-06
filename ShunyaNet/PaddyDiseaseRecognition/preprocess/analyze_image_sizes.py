"""
This script helps me understand the image resolutions in my dataset.
Why I need this?
→ My CNN model expects a fixed input size, but real datasets contain mixed sizes.
→ Before training, I should know the distribution so I can choose the best resize value.

What this script checks:
1. Width, height, and aspect ratio of all images in train and val folders.
2. Most frequent image sizes.
3. Whether resizing is needed or not.
4. A recommended size based on average dimensions.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np


def analyze_image_sizes(dataset_path, dataset_name="Dataset"):
    """
    This function scans a folder that contains class subfolders (each disease type).
    It opens every image and stores its size details.
    No labels are read here — only dimensions.
    """
    sizes = []
    widths = []
    heights = []
    aspect_ratios = []
    class_sizes = defaultdict(list)

    # These are the image formats I want the script to accept
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    print(f"\nScanning {dataset_name} for image sizes...")

    # Go inside each disease/class folder
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)

        # Ignore if it's not a folder or if it's a hidden system folder
        if not os.path.isdir(class_path) or class_dir.startswith('.'):
            continue

        # Read each image inside the folder
        for img_file in os.listdir(class_path):
            if Path(img_file).suffix.lower() not in valid_extensions:
                continue

            img_path = os.path.join(class_path, img_file)

            try:
                # Open image and collect width/height
                with Image.open(img_path) as img:
                    width, height = img.size
                    sizes.append((width, height))
                    widths.append(width)
                    heights.append(height)
                    aspect_ratios.append(width / height)
                    class_sizes[class_dir].append((width, height))
            except Exception as e:
                # Some images may be corrupted — print a warning and move on
                print(f"Could not read {img_path}, skipping it. Error: {e}")

    # Return everything I collected in a dictionary so I can reuse it later
    return {
        'sizes': sizes,
        'widths': widths,
        'heights': heights,
        'aspect_ratios': aspect_ratios,
        'class_sizes': class_sizes,
        'total_images': len(sizes)
    }


def print_statistics(stats, dataset_name):
    """
    This prints all useful numbers about width, height, aspect ratio.
    Basically a quick summary to tell me what my dataset looks like.
    """
    sizes = stats['sizes']
    widths = stats['widths']
    heights = stats['heights']
    aspect_ratios = stats['aspect_ratios']

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} — IMAGE SIZE REPORT")
    print("=" * 60)

    # Count how many times each resolution appears
    unique_sizes = Counter(sizes)

    print(f"\nTotal images checked: {stats['total_images']}")
    print(f"Different resolutions found: {len(unique_sizes)}")

    # If only one resolution exists, then resizing is optional
    if len(unique_sizes) == 1:
        size = list(unique_sizes.keys())[0]
        print(f"\nAll images are same resolution: {size[0]}x{size[1]}")
    else:
        print("\nImages are not uniform in size → resizing will be needed")

    # Basic stats for width
    print("\nWidth details:")
    print(f"- Smallest width  : {min(widths)} px")
    print(f"- Largest width   : {max(widths)} px")
    print(f"- Average width   : {np.mean(widths):.1f} px")
    print(f"- Middle value    : {np.median(widths):.0f} px")

    # Basic stats for height
    print("\nHeight details:")
    print(f"- Smallest height : {min(heights)} px")
    print(f"- Largest height  : {max(heights)} px")
    print(f"- Average height  : {np.mean(heights):.1f} px")
    print(f"- Middle value    : {np.median(heights):.0f} px")

    # Aspect ratio summary
    print("\nAspect ratio details (Width/Height):")
    print(f"- Lowest  : {min(aspect_ratios):.3f}")
    print(f"- Highest : {max(aspect_ratios):.3f}")
    print(f"- Average : {np.mean(aspect_ratios):.3f}")

    # Print top 10 most repeated resolutions
    print("\nMost common resolutions:")
    for i, (size, count) in enumerate(unique_sizes.most_common(10), 1):
        percent = (count / stats['total_images']) * 100
        print(f"{i}. {size[0]}x{size[1]} → {count} images ({percent:.1f}%)")

    if len(unique_sizes) > 10:
        print(f"...and {len(unique_sizes) - 10} more unique sizes exist\n")


def plot_size_distribution(train_stats, val_stats, output_dir):
    """
    This creates plots comparing train vs validation resolutions.
    The output image will be saved so I can view it later.
    (This part is optional — if it fails, training can still happen.)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with multiple small plots (only for internal visualization)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Image Size Distribution Analysis', fontsize=16, fontweight='bold')

    # Plot width comparison
    axes[0, 0].hist(train_stats['widths'], bins=50, alpha=0.7, label='Train')
    axes[0, 0].hist(val_stats['widths'], bins=50, alpha=0.7, label='Val')
    axes[0, 0].set_title('Width Distribution')
    axes[0, 0].legend()

    # Plot height comparison
    axes[0, 1].hist(train_stats['heights'], bins=50, alpha=0.7, label='Train')
    axes[0, 1].hist(val_stats['heights'], bins=50, alpha=0.7, label='Val')
    axes[0, 1].set_title('Height Distribution')
    axes[0, 1].legend()

    # Plot aspect ratio comparison
    axes[0, 2].hist(train_stats['aspect_ratios'], bins=50, alpha=0.7, label='Train')
    axes[0, 2].hist(val_stats['aspect_ratios'], bins=50, alpha=0.7, label='Val')
    axes[0, 2].set_title('Aspect Ratio Distribution')
    axes[0, 2].legend()

    # Scatter for train images
    axes[1, 0].scatter(train_stats['widths'], train_stats['heights'], alpha=0.5, s=10)
    axes[1, 0].set_title('Width vs Height (Train)')

    # Scatter for val images
    axes[1, 1].scatter(val_stats['widths'], val_stats['heights'], alpha=0.5, s=10)
    axes[1, 1].set_title('Width vs Height (Val)')

    # Bar plot for top 10 resolutions
    size_counts = Counter(train_stats['sizes']).most_common(10)
    labels = [f"{w}x{h}" for (w, h), _ in size_counts]
    counts = [c for _, c in size_counts]

    axes[1, 2].bar(np.arange(len(labels)), counts, label="Train")
    axes[1, 2].set_title('Top 10 Most Common Sizes')
    axes[1, 2].set_xticks(np.arange(len(labels)))
    axes[1, 2].set_xticklabels(labels, rotation=45)

    plt.tight_layout()

    # Save visualization result
    output_path = os.path.join(output_dir, 'image_size_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved at: {output_path}")

    plt.close()


def compare_datasets(train_stats, val_stats):
    """
    This prints side-by-side comparison so I know if val split still holds similar image properties.
    """
    print("\n" + "=" * 60)
    print("Train vs Validation — Quick Comparison")
    print("=" * 60)

    print(f"Total images     : Train = {train_stats['total_images']}, Val = {val_stats['total_images']}")
    print(f"Unique sizes     : Train = {len(Counter(train_stats['sizes']))}, Val = {len(Counter(val_stats['sizes']))}")
    print(f"Avg width        : Train = {np.mean(train_stats['widths']):.1f}, Val = {np.mean(val_stats['widths']):.1f}")
    print(f"Avg height       : Train = {np.mean(train_stats['heights']):.1f}, Val = {np.mean(val_stats['heights']):.1f}")
    print(f"Avg aspect ratio : Train = {np.mean(train_stats['aspect_ratios']):.3f}, Val = {np.mean(val_stats['aspect_ratios']):.3f}")


def print_recommendations(train_stats, val_stats):
    """
    This gives me a suggested resize value based on average resolution.
    This is not perfect science, but a good rule to choose a size.
    """
    print("\n" + "=" * 60)
    print("Resize Recommendation for CNN Input")
    print("=" * 60)

    # If all images are same, I *can* skip resizing, but most datasets are mixed
    if len(Counter(train_stats['sizes'])) == 1 and len(Counter(val_stats['sizes'])) == 1:
        size = list(Counter(train_stats['sizes']).keys())[0]
        print(f"\nDataset already uniform: {size[0]}x{size[1]} → I can use this directly")
    else:
        # Compute average width/height from both splits
        avg_w = (np.mean(train_stats['widths']) + np.mean(val_stats['widths'])) / 2
        avg_h = (np.mean(train_stats['heights']) + np.mean(val_stats['heights'])) / 2

        print(f"\nAverage resolution ~ {avg_w:.0f}x{avg_h:.0f}")

        # Simple decision rule for picking resize value
        if avg_w < 224 or avg_h < 224:
            print("Suggested size: 224x224")
        elif avg_w < 384 or avg_h < 384:
            print("Suggested size: 256x256 or 384x384")
        else:
            print("Suggested size: 384x384 or 512x512")


if __name__ == "__main__":
    """
    This is the actual execution flow.
    I give dataset paths here, run analysis, compare splits, save visualization, print suggestions.
    """

    # Base path of my dataset (relative to project structure)
    current_dir = Path(__file__).parent
    dataset_root = current_dir.parent.parent / "Data" / "PaddyDiseases"

    # Define exact train and validation folders
    train_path = dataset_root / "train"
    val_path = dataset_root / "val"

    # Folder where plots will be stored
    output_dir = current_dir / "Data"

    print("\nRunning image resolution check on dataset...")

    # Get stats from train folder
    train_stats = analyze_image_sizes(str(train_path), "Train Folder")
    print_statistics(train_stats, "Train Folder")

    # Get stats from validation folder
    val_stats = analyze_image_sizes(str(val_path), "Validation Folder")
    print_statistics(val_stats, "Validation Folder")

    # Compare both splits to make sure they look similar in size properties
    compare_datasets(train_stats, val_stats)

    # Generate and save visualization plots
    plot_size_distribution(train_stats, val_stats, str(output_dir))

    # Print my resize suggestions for CNN input
    print_recommendations(train_stats, val_stats)

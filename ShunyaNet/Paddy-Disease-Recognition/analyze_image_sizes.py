"""
analyze_image_sizes.py
Script to analyze image sizes in train and validation datasets
"""

import os
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np


def analyze_image_sizes(dataset_path, dataset_name="Dataset"):
    """
    Analyze all image sizes in a dataset directory

    Args:
        dataset_path: Path to dataset directory (with class subdirectories)
        dataset_name: Name of the dataset for display

    Returns:
        dict: Dictionary with size statistics
    """
    sizes = []
    widths = []
    heights = []
    aspect_ratios = []
    class_sizes = defaultdict(list)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    print(f"\nAnalyzing {dataset_name}...")

    # Iterate through all class directories
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)

        if not os.path.isdir(class_path) or class_dir.startswith('.'):
            continue

        # Process each image in the class directory
        for img_file in os.listdir(class_path):
            if Path(img_file).suffix.lower() not in valid_extensions:
                continue

            img_path = os.path.join(class_path, img_file)

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    sizes.append((width, height))
                    widths.append(width)
                    heights.append(height)
                    aspect_ratios.append(width / height)
                    class_sizes[class_dir].append((width, height))
            except Exception as e:
                print(f"  Warning: Could not read {img_path}: {e}")

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
    Print detailed statistics about image sizes

    Args:
        stats: Statistics dictionary from analyze_image_sizes
        dataset_name: Name of the dataset
    """
    sizes = stats['sizes']
    widths = stats['widths']
    heights = stats['heights']
    aspect_ratios = stats['aspect_ratios']

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - IMAGE SIZE ANALYSIS")
    print("=" * 70)

    # Count unique sizes
    unique_sizes = Counter(sizes)

    print(f"\nTotal images analyzed: {stats['total_images']}")
    print(f"Unique image sizes: {len(unique_sizes)}")

    # Check if all images have the same size
    if len(unique_sizes) == 1:
        size = list(unique_sizes.keys())[0]
        print(f"\n✅ ALL IMAGES HAVE THE SAME SIZE: {size[0]}x{size[1]}")
    else:
        print(f"\n⚠️  IMAGES HAVE DIFFERENT SIZES")

    # Width statistics
    print("\n" + "-" * 70)
    print("WIDTH STATISTICS")
    print("-" * 70)
    print(f"  Min width:     {min(widths):>6} px")
    print(f"  Max width:     {max(widths):>6} px")
    print(f"  Mean width:    {np.mean(widths):>6.1f} px")
    print(f"  Median width:  {np.median(widths):>6.0f} px")
    print(f"  Std dev:       {np.std(widths):>6.1f} px")

    # Height statistics
    print("\n" + "-" * 70)
    print("HEIGHT STATISTICS")
    print("-" * 70)
    print(f"  Min height:    {min(heights):>6} px")
    print(f"  Max height:    {max(heights):>6} px")
    print(f"  Mean height:   {np.mean(heights):>6.1f} px")
    print(f"  Median height: {np.median(heights):>6.0f} px")
    print(f"  Std dev:       {np.std(heights):>6.1f} px")

    # Aspect ratio statistics
    print("\n" + "-" * 70)
    print("ASPECT RATIO STATISTICS")
    print("-" * 70)
    print(f"  Min ratio:     {min(aspect_ratios):>6.3f}")
    print(f"  Max ratio:     {max(aspect_ratios):>6.3f}")
    print(f"  Mean ratio:    {np.mean(aspect_ratios):>6.3f}")
    print(f"  Median ratio:  {np.median(aspect_ratios):>6.3f}")

    # Most common sizes
    print("\n" + "-" * 70)
    print("TOP 10 MOST COMMON SIZES")
    print("-" * 70)
    for i, (size, count) in enumerate(unique_sizes.most_common(10), 1):
        percentage = (count / stats['total_images']) * 100
        print(f"  {i:>2}. {size[0]:>4}x{size[1]:<4} : {count:>5} images ({percentage:>5.1f}%)")

    if len(unique_sizes) > 10:
        print(f"\n  ... and {len(unique_sizes) - 10} more unique sizes")


def plot_size_distribution(train_stats, val_stats, output_dir):
    """
    Create visualization plots for image size distributions

    Args:
        train_stats: Statistics for training set
        val_stats: Statistics for validation set
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Image Size Distribution Analysis', fontsize=16, fontweight='bold')

    # Width distribution
    axes[0, 0].hist(train_stats['widths'], bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
    axes[0, 0].hist(val_stats['widths'], bins=50, alpha=0.7, label='Val', color='orange', edgecolor='black')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Width Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Height distribution
    axes[0, 1].hist(train_stats['heights'], bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
    axes[0, 1].hist(val_stats['heights'], bins=50, alpha=0.7, label='Val', color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Height Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Aspect ratio distribution
    axes[0, 2].hist(train_stats['aspect_ratios'], bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
    axes[0, 2].hist(val_stats['aspect_ratios'], bins=50, alpha=0.7, label='Val', color='orange', edgecolor='black')
    axes[0, 2].set_xlabel('Aspect Ratio (W/H)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Aspect Ratio Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Scatter plot: Width vs Height (Train)
    axes[1, 0].scatter(train_stats['widths'], train_stats['heights'], alpha=0.5, s=10, color='blue')
    axes[1, 0].set_xlabel('Width (pixels)')
    axes[1, 0].set_ylabel('Height (pixels)')
    axes[1, 0].set_title('Width vs Height (Train)')
    axes[1, 0].grid(True, alpha=0.3)

    # Scatter plot: Width vs Height (Val)
    axes[1, 1].scatter(val_stats['widths'], val_stats['heights'], alpha=0.5, s=10, color='orange')
    axes[1, 1].set_xlabel('Width (pixels)')
    axes[1, 1].set_ylabel('Height (pixels)')
    axes[1, 1].set_title('Width vs Height (Val)')
    axes[1, 1].grid(True, alpha=0.3)

    # Size frequency (top 20 sizes)
    train_size_counts = Counter(train_stats['sizes']).most_common(20)
    val_size_counts = Counter(val_stats['sizes']).most_common(20)

    x_labels = [f"{w}x{h}" for (w, h), _ in train_size_counts[:10]]
    train_counts = [count for _, count in train_size_counts[:10]]
    val_size_dict = {size: count for size, count in val_size_counts}
    val_counts = [val_size_dict.get(size, 0) for size, _ in train_size_counts[:10]]

    x = np.arange(len(x_labels))
    width = 0.35

    axes[1, 2].bar(x - width/2, train_counts, width, label='Train', color='blue', alpha=0.7)
    axes[1, 2].bar(x + width/2, val_counts, width, label='Val', color='orange', alpha=0.7)
    axes[1, 2].set_xlabel('Image Size')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Top 10 Most Common Sizes')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, 'image_size_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.close()


def compare_datasets(train_stats, val_stats):
    """
    Compare train and validation dataset statistics

    Args:
        train_stats: Statistics for training set
        val_stats: Statistics for validation set
    """
    print("\n" + "=" * 70)
    print("TRAIN vs VALIDATION COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Train':<20} {'Val':<20}")
    print("-" * 70)
    print(f"{'Total images':<30} {train_stats['total_images']:<20} {val_stats['total_images']:<20}")
    print(f"{'Unique sizes':<30} {len(Counter(train_stats['sizes'])):<20} {len(Counter(val_stats['sizes'])):<20}")
    print(f"{'Min width':<30} {min(train_stats['widths']):<20} {min(val_stats['widths']):<20}")
    print(f"{'Max width':<30} {max(train_stats['widths']):<20} {max(val_stats['widths']):<20}")
    print(f"{'Mean width':<30} {np.mean(train_stats['widths']):<20.1f} {np.mean(val_stats['widths']):<20.1f}")
    print(f"{'Min height':<30} {min(train_stats['heights']):<20} {min(val_stats['heights']):<20}")
    print(f"{'Max height':<30} {max(train_stats['heights']):<20} {max(val_stats['heights']):<20}")
    print(f"{'Mean height':<30} {np.mean(train_stats['heights']):<20.1f} {np.mean(val_stats['heights']):<20.1f}")
    print(f"{'Mean aspect ratio':<30} {np.mean(train_stats['aspect_ratios']):<20.3f} {np.mean(val_stats['aspect_ratios']):<20.3f}")


def print_recommendations(train_stats, val_stats):
    """
    Print recommendations based on image size analysis

    Args:
        train_stats: Statistics for training set
        val_stats: Statistics for validation set
    """
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR MODEL TRAINING")
    print("=" * 70)

    train_unique = len(Counter(train_stats['sizes']))
    val_unique = len(Counter(val_stats['sizes']))

    if train_unique == 1 and val_unique == 1:
        size = list(Counter(train_stats['sizes']).keys())[0]
        print(f"\n✅ All images are the same size: {size[0]}x{size[1]}")
        print("   → No resizing needed in transforms!")
        print("   → Can use this size directly for model input")
    else:
        print(f"\n⚠️  Images have varying sizes:")
        print(f"   Train: {train_unique} unique sizes")
        print(f"   Val: {val_unique} unique sizes")
        print("\n   → You MUST resize images in transforms!")
        print("   → Common choices: 224x224, 256x256, 384x384")

        # Suggest a size based on mean dimensions
        mean_w = (np.mean(train_stats['widths']) + np.mean(val_stats['widths'])) / 2
        mean_h = (np.mean(train_stats['heights']) + np.mean(val_stats['heights'])) / 2

        print(f"\n   📊 Average image size: {mean_w:.0f}x{mean_h:.0f}")

        if mean_w < 224 or mean_h < 224:
            print("   → Recommend: 224x224 (will upscale small images)")
        elif mean_w < 384 or mean_h < 384:
            print("   → Recommend: 256x256 or 384x384")
        else:
            print("   → Recommend: 384x384 or 512x512")

    print("\n" + "-" * 70)
    print("SAMPLE TRANSFORM CODE:")
    print("-" * 70)
    print("""
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize to fixed size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Same size as train
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    """)
    print("=" * 70)


if __name__ == "__main__":
    # Dataset paths
    current_dir = Path(__file__).parent
    dataset_root = current_dir.parent.parent / "Data" / "PaddyDiseases" / "Dataset"

    train_path = dataset_root / "train"
    val_path = dataset_root / "val"
    output_dir = current_dir / "Data"

    print("=" * 70)
    print("IMAGE SIZE ANALYSIS FOR PADDY DISEASE DATASET")
    print("=" * 70)

    # Analyze train set
    train_stats = analyze_image_sizes(str(train_path), "Training Set")
    print_statistics(train_stats, "Training Set")

    # Analyze validation set
    val_stats = analyze_image_sizes(str(val_path), "Validation Set")
    print_statistics(val_stats, "Validation Set")

    # Compare datasets
    compare_datasets(train_stats, val_stats)

    # Create visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    try:
        plot_size_distribution(train_stats, val_stats, str(output_dir))
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")

    # Print recommendations
    print_recommendations(train_stats, val_stats)


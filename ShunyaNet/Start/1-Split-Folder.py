import splitfolders
import os

# Set your actual input and output folder paths here
input_folder = "/Users/bharathgoud/PycharmProjects/Shunya-00/Data/CottonDiseases"
output_folder = "/Users/bharathgoud/PycharmProjects/Shunya-00/Data/CottonDisease"

# Check if input folder exists
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"Input folder not found: {input_folder}")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.8, 0.10, 0.10))
print("Dataset split into train, validation, and test sets.")

# Creating paths for the train, val, and test folders
train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "val")
test_folder = os.path.join(output_folder, "test")

# Function to count images in each subfolder and total images
def count_images_in_subfolders(folder_path):
    subfolder_counts = {}
    total_count = 0
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return subfolder_counts, total_count
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):  # Ensure it's a directory
            image_count = len([
                file for file in os.listdir(subfolder_path)
                if file.lower().endswith(("jpg", "jpeg", "png", "bmp"))
            ])
            subfolder_counts[subfolder] = image_count
            total_count += image_count
    return subfolder_counts, total_count

train_counts, train_total = count_images_in_subfolders(train_folder)
val_counts, val_total = count_images_in_subfolders(val_folder)
test_counts, test_total = count_images_in_subfolders(test_folder)

# Print the total counts and counts for each subfolder
print(f"\nTotal number of images in training set: {train_total}")
print("Number of images in training set subfolders:")
for subfolder, count in train_counts.items():
    print(f"  {subfolder}: {count}")

print(f"\nTotal number of images in validation set: {val_total}")
print("Number of images in validation set subfolders:")
for subfolder, count in val_counts.items():
    print(f"  {subfolder}: {count}")

print(f"\nTotal number of images in test set: {test_total}")
print("Number of images in test set subfolders:")
for subfolder, count in test_counts.items():
    print(f"  {subfolder}: {count}")

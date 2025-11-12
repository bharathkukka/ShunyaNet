# Notes on Split-Folder.py

## Purpose
This script is used to split an image dataset into training, validation, and test sets using the `splitfolders` library. It also counts the number of images in each set and prints the counts for each emotion category.

---

## How to Use
1. **Set Input/Output Paths**
   - Update `input_folder` and `output_folder` variables at the top of the script to point to your dataset and desired output location.
   - Example:
     ```python
     input_folder = "./Data/EmotionRecognitionSystem/8Emotions"
     output_folder = "./Data/EmotionRecognitionSystem/Split"
     ```

2. **Run the Script**
   - Make sure you have installed the `splitfolders` library:
     ```bash
     pip install split-folders
     ```
   - Run the script:
     ```bash
     python 1-Split-Folder.py
     ```

3. **What Happens**
   - The script checks if the input folder exists.
   - It creates the output folder if it doesn't exist.
   - It splits the dataset into train (80%), val (10%), and test (10%) folders.
   - It counts the number of images in each subfolder (emotion category) for train, val, and test sets.
   - It prints the total and per-category image counts for each set.

---

## Explanation of Key Parts
- **splitfolders.ratio**
  - Splits the dataset into train/val/test folders based on the ratio provided.
  - Uses a random seed for reproducibility.

- **Counting Images**
  - The function `count_images_in_subfolders` goes through each emotion folder and counts images with extensions jpg, jpeg, png, bmp (case-insensitive).
  - Prints warnings if any expected folder is missing.

- **Output Structure**
  - After running, the output folder will have:
    - `train/`
    - `val/`
    - `test/`
  - Each of these will have subfolders for each emotion (anger, happy, etc.)

---

## Troubleshooting
- If you get `FileNotFoundError`, check that your input path is correct and exists.
- If you want to change the split ratio, modify the `ratio` argument in `splitfolders.ratio`.
- If you add new image formats, update the extensions in `count_images_in_subfolders`.

---

## Example Output
```
Dataset split into train, validation, and test sets.

Total number of images in training set: 8000
Number of images in training set subfolders:
  anger: 1000
  happy: 1000
  ...

Total number of images in validation set: 1000
Number of images in validation set subfolders:
  anger: 125
  happy: 125
  ...

Total number of images in test set: 1000
Number of images in test set subfolders:
  anger: 125
  happy: 125
  ...
```

---

## Good Practices
- Always check your input/output paths before running.
- Use a fixed seed for reproducibility.
- Check the printed counts to verify the split is as expected.

---

## Reference
- [splitfolders documentation](https://pypi.org/project/split-folders/)

---

# End of Notes


import os
import glob
import random
import shutil
from tqdm import tqdm


SOURCE_FOLDER = "/net/tscratch/people/plgmjung/p_dyplomowa/ASTAdataset/Processed/Processed"
#SOURCE_FOLDER = "ASTAdataset/Processed/Processed"   # deleted after running
DEST_FOLDER = "/net/tscratch/people/plgmjung/p_dyplomowa/ASTA_dataset_split"
#DEST_FOLDER = "ASTA_dataset_split"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


IMAGE_SUFFIX = ".fits_full.png"
MASK_SUFFIX = "_mask.png"


def find_file_pairs(source_dir, img_suffix, mask_suffix):
    """Finds all pairs picture - mask."""
    pairs = []
    
    mask_files = glob.glob(os.path.join(source_dir, f"*{mask_suffix}"))
    
    if not mask_files:
        print(f"Failure. Couldn't find any masks '{mask_suffix}' w {source_dir}")
        return []
        
    print(f"Found {len(mask_files)} mask pixes. Looking for matching images...")

    for mask_path in tqdm(mask_files, desc="Finding pairs"):
        base_name_with_path = mask_path.replace(mask_suffix, "")
        img_path = base_name_with_path + img_suffix
        
        if os.path.exists(img_path):
            pairs.append((img_path, mask_path))
        else:
            print(f"Warning: Found mask {mask_path}, but no matching image {img_path}")
            
    print(f"Found {len(pairs)} complete image-mask pairs.")
    return pairs

def split_and_copy_files(pairs, dest_dir, train_r, val_r):
    """Creates folder structure and copies files."""
    
    # 1. Shuffle pairs
    random.shuffle(pairs)
    
    # 2. Calculate split points
    total_pairs = len(pairs)
    train_split = int(total_pairs * train_r)
    val_split = int(total_pairs * (train_r + val_r))
    
    # 3. Split lists
    train_pairs = pairs[:train_split]
    val_pairs = pairs[train_split:val_split]
    test_pairs = pairs[val_split:]
    
    print(f"\nSplit plan:")
    print(f"  Training set (train): {len(train_pairs)} pairs")
    print(f"  Validation set (val): {len(val_pairs)} pairs")
    print(f"  Test set (test):     {len(test_pairs)} pairs")
    print(f"  Total:               {len(train_pairs) + len(val_pairs) + len(test_pairs)} pairs")

    subsets = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    if os.path.exists(dest_dir):
        print(f"\nWarning: Destination folder {dest_dir} already exists. It may contain old files.")
    
    for subset_name, subset_pairs in subsets.items():
        # Paths to images and masks folders
        img_dir = os.path.join(dest_dir, subset_name, 'images')
        mask_dir = os.path.join(dest_dir, subset_name, 'masks')
        
        # Create folders if they don't exist
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        print(f"\nCopying files to '{subset_name}'...")
        for img_path, mask_path in tqdm(subset_pairs, desc=f"Copying {subset_name}"):
            # Copy image
            shutil.copy2(img_path, os.path.join(img_dir, os.path.basename(img_path)))
            # Copy mask
            shutil.copy2(mask_path, os.path.join(mask_dir, os.path.basename(mask_path)))
            
    print("\n--- Done! ---")
    print(f"New, split dataset is ready at: {dest_dir}")
def main():
    # Step 1: Find pairs
    file_pairs = find_file_pairs(SOURCE_FOLDER, IMAGE_SUFFIX, MASK_SUFFIX)
    
    if not file_pairs:
        print("No pairs found. Aborting.")
        return
        
    # Step 2: Split and copy
    split_and_copy_files(file_pairs, DEST_FOLDER, TRAIN_RATIO, VAL_RATIO)

if __name__ == "__main__":
    main()
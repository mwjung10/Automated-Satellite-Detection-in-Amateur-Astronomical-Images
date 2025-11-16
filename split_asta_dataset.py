import os
import glob
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm

#SOURCE_FOLDER = "/net/tscratch/people/plgmjung/p_dyplomowa/ASTAdataset/Processed/Processed"
SOURCE_FOLDER = "ASTAdataset/Processed"   # deleted after running
#DEST_FOLDER = "/net/tscratch/people/plgmjung/p_dyplomowa/ASTA_dataset_split"
DEST_FOLDER = "ASTA_dataset_split"

IMAGE_SUFFIX = "_red.fits_full.png"
MASK_SUFFIX = "_red_mask.png"

TRAIN_RATIO = 0.8  
VAL_RATIO = 0.1    
# -------------------------

def find_file_pairs(source_folder):
    pairs = []
    
    mask_files = glob.glob(os.path.join(source_folder, f"*{MASK_SUFFIX}"))
    
    if not mask_files:
        print(f"Error not found masks'*{MASK_SUFFIX}'")
        print(f"       in folder: {source_folder}")
        return []

    print(f"Found {len(mask_files)} mask files. Verifying pairs...")
    
    for mask_path in tqdm(mask_files, desc="Searching for file pairs"):
        # E.g., "ML1_..._red_mask.png" -> "ML1_..._red"
        base_name = os.path.basename(mask_path).replace(MASK_SUFFIX, "")
        
        # Reconstruct expected image file name
        # E.g., "ML1_..._red" -> "ML1_..._red.fits_full.png"
        image_name = f"{base_name}{IMAGE_SUFFIX}"
        image_path = os.path.join(source_folder, image_name)
        
        if os.path.exists(image_path):
            pairs.append((image_path, mask_path, base_name))
        else:
            print(f"[Warning] Found a mask but no matching image: {mask_path}")
            print(f"          Expected image: {image_name}")

    if not pairs:
        print(f"[ERROR] No matching image-mask pairs found.")
        print("       Check the variables SOURCE_FOLDER, IMAGE_SUFFIX, and MASK_SUFFIX.")
        
    return pairs

def convert_mask_to_yolo_txt(mask_path, img_shape_for_norm, dest_txt_path):
    """
    Converts mask .png to YOLO .txt format.
    Returns True if successful, False if the mask was empty.
    """
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[Warning] Cannot read mask: {mask_path}")
            return False

        h, w = img_shape_for_norm[:2] # We use the ORIGINAL image dimensions
        
        # Find contours (polygons) on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # print(f"[Info] Mask {os.path.basename(mask_path)} is empty.")
            return False # Mask was empty, do not create .txt file

        yolo_lines = []
        for cnt in contours:
            # Simplify contour to reduce number of points
            cnt = cnt.reshape(-1, 2).astype(float)
            
            # Normalize coordinates (x/w, y/h)
            cnt[:, 0] /= w
            cnt[:, 1] /= h
            
            # Flatten list of points to format x1 y1 x2 y2...
            polygon_points = cnt.flatten()
            
            # Create a YOLO line: "0 x1 y1 x2 y2 ..."
            # "0" is the class index (we have only one class - "streak")
            line = f"0 {' '.join(polygon_points.astype(str))}"
            yolo_lines.append(line)

        with open(dest_txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
            
        return True 
        
    except Exception as e:
        print(f"[ERROR] Cannot process mask {mask_path}: {e}")
        return False

def process_and_split_dataset(pairs, dest_folder):
    """
    Splits the dataset and converts .png masks to .txt labels.
    """
    subfolders = ['train', 'val', 'test']
    for sub in subfolders:
        os.makedirs(os.path.join(dest_folder, sub, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_folder, sub, 'labels'), exist_ok=True)
            
    random.shuffle(pairs)
    
    total_files = len(pairs)
    train_end = int(total_files * TRAIN_RATIO)
    val_end = train_end + int(total_files * VAL_RATIO)
    
    datasets = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }

    for dataset_name, dataset_pairs in datasets.items():
        print(f"Dataset: {dataset_name} ({len(dataset_pairs)} files)")
        
        img_dir = os.path.join(dest_folder, dataset_name, 'images')
        label_dir = os.path.join(dest_folder, dataset_name, 'labels')
        
        valid_files_count = 0
        
        for img_src_path, mask_src_path, base_name in tqdm(dataset_pairs, desc=f"Konwertowanie {dataset_name}"):
            
            try:
                # Load the original image to get its dimensions for normalization
                img_for_shape = cv2.imread(img_src_path)
                if img_for_shape is None:
                    print(f"[Warning] Cannot load image {img_src_path} to check dimensions. Skipping.")
                    continue
                
                # --- Destination file name ---
                # E.g. "ML1..._red.png" for image
                # E.g. "ML1..._red.txt" for label
                new_img_name = f"{base_name}.png"
                new_label_name = f"{base_name}.txt"
                
                img_dest_path = os.path.join(img_dir, new_img_name)
                label_dest_path = os.path.join(label_dir, new_label_name)

                success = convert_mask_to_yolo_txt(mask_src_path, img_for_shape.shape, label_dest_path)
                
                # 2. Copy the image only if mask conversion succeeded
                if success:
                    shutil.copyfile(img_src_path, img_dest_path)
                    valid_files_count += 1
                else:
                    # Mask was empty, so we do not copy either the image or the label
                    pass
                    
            except Exception as e:
                print(f"[ERROR] Cannot process pair for {base_name}: {e}")
        print(f"Finished {dataset_name}. Created {valid_files_count} valid pairs (image + .txt label).")

def main():
    print("--- Starting ASTA dataset CONVERSION script to YOLO format ---")
    
    if os.path.exists(DEST_FOLDER):
        print(f"[WARNING] Destination folder {DEST_FOLDER} already exists.")
        print("                 It is recommended to remove it (rm -rf) before continuing.")
    
    file_pairs = find_file_pairs(SOURCE_FOLDER)

    if not file_pairs:
        print("No files found. Exiting.")
        return
        
    print(f"\nFound {len(file_pairs)} matching image-mask pairs.")
    
    process_and_split_dataset(file_pairs, DEST_FOLDER)
    
    print("\n--- All operations completed successfully! ---")
    print(f"New dataset is ready at: {DEST_FOLDER}")
    print("Created .txt files in 'labels' folders.")

if __name__ == "__main__":
    main()
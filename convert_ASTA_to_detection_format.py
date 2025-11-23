''''
Script that converts ASTA mask images to YOLO detection format.
'''
import random
import cv2
import numpy as np
import os
import glob
import shutil

INPUT_FOLDER = 'ASTAdataset/Processed'

OUTPUT_BASE_FOLDER = 'ASTA_YOLO_Ready'

CLASS_ID = 0 # ID satellite trails
MIN_AREA = 50 # min area of detected object to consider, in order to filter noise

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2
TEST_RATIO  = 0.1


def prepare_asta_for_yolo():
    mask_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*_mask.png')))
    
    if not mask_files:
        print("Could not find any mask files in the input folder.")
        return

    random.seed(42) 
    random.shuffle(mask_files)
    
    total_files = len(mask_files)
    print(f"Found {total_files} files. Shuffling and processing...")

    train_end = int(total_files * TRAIN_RATIO)
    val_end = train_end + int(total_files * VAL_RATIO)
    
    counts = {'train': 0, 'val': 0, 'test': 0}

    for i, mask_path in enumerate(mask_files):
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'

        images_dir = os.path.join(OUTPUT_BASE_FOLDER, split, 'images')
        labels_dir = os.path.join(OUTPUT_BASE_FOLDER, split, 'labels')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        image_path = mask_path.replace('_mask.png', '.fits_full.png')
        
        if not os.path.exists(image_path):
            print(f"WARNING: Missing image for mask: {os.path.basename(mask_path)}")
            continue
            
        base_name = os.path.basename(mask_path).replace('_mask.png', '')
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        height, width = mask.shape
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yolo_lines = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            
            yolo_lines.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        new_image_path = os.path.join(images_dir, base_name + '.png')
        shutil.copy(image_path, new_image_path)
        if yolo_lines:  
            label_path = os.path.join(labels_dir, base_name + '.txt')
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            
        counts[split] += 1
            
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total_files} files...")

    print(f"DONE! Dataset created in: {OUTPUT_BASE_FOLDER}")
    print("Statistics:")
    print(f"  Train: {counts['train']} images")
    print(f"  Val:   {counts['val']} images")
    print(f"  Test:  {counts['test']} images")

def save_verification_images(num_samples=10):
    debug_dir = os.path.join(OUTPUT_BASE_FOLDER, 'debug_visualizations')

    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)
    
    images_dir = os.path.join(OUTPUT_BASE_FOLDER, 'train', 'images')
    labels_dir = os.path.join(OUTPUT_BASE_FOLDER, 'train', 'labels')
    
    all_images = glob.glob(os.path.join(images_dir, "*.png"))
    
    if not all_images:
        print("No images")
        return

    samples = random.sample(all_images, min(len(all_images), num_samples))
    
    for img_path in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h_img, w_img = img.shape[:2]
        
        base_name = os.path.basename(img_path).replace('.png', '.txt')
        label_path = os.path.join(labels_dir, base_name)
        
        box_color = (0, 255, 0)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                # format: class x_center y_center w h
                xc, yc, wn, hn = map(float, parts[1:])
                
                # Convert to pixel coordinates
                x_center_px = xc * w_img
                y_center_px = yc * h_img
                width_px = wn * w_img
                height_px = hn * h_img
                
                x1 = int(x_center_px - (width_px / 2))
                y1 = int(y_center_px - (height_px / 2))
                x2 = int(x_center_px + (width_px / 2))
                y2 = int(y_center_px + (height_px / 2))
                
                # Drawing the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                
                # Adding a small marker in the center
                cv2.circle(img, (int(x_center_px), int(y_center_px)), 2, (0, 0, 255), -1)

        else:
            # If label file is missing, write it in red on the image
            print(f"Missing label for: {base_name}")
            cv2.putText(img, "MISSING LABEL", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # Save file to disk
        output_filename = f"vis_{os.path.basename(img_path)}"
        output_path = os.path.join(debug_dir, output_filename)
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_filename}")
    print(f"\nDone! Check the folder: {debug_dir}")

if __name__ == "__main__":
    #prepare_asta_for_yolo()
    save_verification_images(num_samples=10)

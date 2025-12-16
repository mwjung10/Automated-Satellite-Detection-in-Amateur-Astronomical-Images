import os
import glob
import random
import shutil
from tqdm import tqdm

SOURCE_FOLDER = "Amateur_Synthetic_Dataset_v2"
DEST_FOLDER = "Syntetic_Dataset_Split_v2"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def split_dataset():
    images_dir = os.path.join(SOURCE_FOLDER, "images")
    labels_dir = os.path.join(SOURCE_FOLDER, "labels")
    
    if not os.path.exists(images_dir):
        print(f"ERROR: {images_dir} not found!")
        return
    
    if not os.path.exists(labels_dir):
        print(f"ERROR: {labels_dir} not found!")
        return
    
    image_files = glob.glob(os.path.join(images_dir, "*.*"))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    random.seed(42)
    random.shuffle(image_files)
    
    total = len(image_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DEST_FOLDER, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DEST_FOLDER, split, 'labels'), exist_ok=True)
    
    def copy_split(file_list, split_name):
        print(f"\nCopying {split_name} set...")
        for img_path in tqdm(file_list):
            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            
            dest_img = os.path.join(DEST_FOLDER, split_name, 'images', filename)
            shutil.copy2(img_path, dest_img)
            
            label_path = os.path.join(labels_dir, base_name + '.txt')
            if os.path.exists(label_path):
                dest_label = os.path.join(DEST_FOLDER, split_name, 'labels', base_name + '.txt')
                shutil.copy2(label_path, dest_label)
           
    
    copy_split(train_files, 'train')
    copy_split(val_files, 'val')
    copy_split(test_files, 'test')
    
    print(f"\n✓ Dataset split completed!")
    print(f"Output folder: {DEST_FOLDER}")
    print(f"  - train: {len(train_files)} images")
    print(f"  - val: {len(val_files)} images")
    print(f"  - test: {len(test_files)} images")

if __name__ == "__main__":
    split_dataset()



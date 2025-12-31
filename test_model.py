from ultralytics import YOLO
import torch
import os
import glob
import cv2
import numpy as np

DATASET_YAML = 'C:/STUDIA/PracaDyplomowa/Automated-Satellite-Detection-in-Amateur-Astronomical-Images/test_dataset.yaml'

IMAGES_DIR = 'C:/STUDIA/PracaDyplomowa/Automated-Satellite-Detection-in-Amateur-Astronomical-Images/images_to_test/images'

MODEL_PATH = 'best_augumentation_streaks.pt'


def get_max_image_size(images_dir):
    """
    Returns the maximum dimension of the largest image in the directory,
    rounded up to the nearest multiple of 32 (YOLO requirement).
    """
    image_files = glob.glob(os.path.join(images_dir, '*.*'))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.fits']
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]
    
    max_dim = 640 
    
    if not image_files:
        print("No images found! Using default imgsz=640.")
        return 640

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            current_max = max(h, w)
            if current_max > max_dim:
                max_dim = current_max
    
    final_size = int(np.ceil(max_dim / 32) * 32)
    print(f"Found maximum dimension: {max_dim}. Setting imgsz={final_size}")
    return final_size

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    inference_size = get_max_image_size(IMAGES_DIR)

    try:
        model = YOLO(MODEL_PATH)

        print(f"Start of the test (imgsz={inference_size})...")
        
        metrics = model.val(
            data=DATASET_YAML,
            split='test',
            device=device,
            imgsz=inference_size, 
            batch=1,              
            rect=True,                      
            plots=True,
            project='runs/detect',
            name='local_test_highres'
        )

        print("\n" + "="*40)
        print(f"Precision:  {metrics.box.mp:.4f}")
        print(f"Recall:     {metrics.box.mr:.4f}")
        print(f"mAP@50:     {metrics.box.map50:.4f}")
        print(f"mAP@50-95:  {metrics.box.map:.4f}")
        print("="*40)
        
        print(f"\Finish! {metrics.save_dir}")

    except Exception as e:
        print(f"\n[Error] {e}")

if __name__ == '__main__':
    main()
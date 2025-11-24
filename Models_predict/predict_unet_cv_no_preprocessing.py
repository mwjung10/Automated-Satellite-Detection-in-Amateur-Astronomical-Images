import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
import torch

MODEL_PATH = '../best_augumentation_streaks.pt'
IMAGES_FOLDER = '../images_to_predict'
RESULTS_FOLDER = 'prediction_after_aug_training/no_preprocessing_results'

def analyze_mask_with_cv(mask, original_image):
    image_with_contours = original_image.copy()
    image_with_hough = original_image.copy()
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  [CV Analysis] Found {len(contours)} contours.")
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 20: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
    
    if lines is not None:
        print(f"  [CV Analysis] Found {len(lines)} Hough lines.")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
    return image_with_contours, image_with_hough

def main():
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. {e}")
        return

    image_files = sorted(glob.glob(os.path.join(IMAGES_FOLDER, '*.*')))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.fits']
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]

    print(f"Found {len(image_files)} images.")

    for img_path in image_files:
        print(f"\n--- Processing: {os.path.basename(img_path)} ---")
        
        original_image = cv2.imread(img_path)
        if original_image is None: 
            print("Skipped: Cannot load image.")
            continue

        h, w = original_image.shape[:2]
        
        inference_size = max(h, w)
        inference_size = int(np.ceil(inference_size / 32) * 32)
        
        print(f"  [Model] Predicting with imgsz={inference_size} (Original: {w}x{h})")
        
        results = model.predict(
            original_image, 
            conf=0.01,        
            imgsz=inference_size, 
            verbose=False
        ) 
        
        result = results[0]
        
        
        if result.masks is None and (result.boxes is None or len(result.boxes) == 0):
            print("  No objects found (masks or boxes).")
            continue
            
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        if result.masks is not None:
            print(f"  Found {len(result.masks)} masks (Segmentation).")
            
            masks_data = result.masks.data.cpu().numpy()
            for i, mask_tensor in enumerate(masks_data):
                mask_single = (mask_tensor * 255).astype("uint8")
                
                if mask_single.shape[:2] != (h, w):
                    mask_single = cv2.resize(mask_single, (w, h), interpolation=cv2.INTER_NEAREST)
                
                final_mask = cv2.bitwise_or(final_mask, mask_single)
                
        elif result.boxes is not None:
            print(f"  Found {len(result.boxes)} boxes (Detection).")
            print("  -> Creating artificial mask from bounding boxes...")
            
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                cv2.rectangle(final_mask, (x1, y1), (x2, y2), 255, -1)

        
        mask_path = os.path.join(RESULTS_FOLDER, f"{os.path.splitext(os.path.basename(img_path))[0]}_MASK.png")
        cv2.imwrite(mask_path, final_mask)
        
        img_cnt, img_hough = analyze_mask_with_cv(final_mask, original_image)
        
        cv2.imwrite(os.path.join(RESULTS_FOLDER, f"analyzed_{os.path.basename(img_path)}_cnt.png"), img_cnt)
        cv2.imwrite(os.path.join(RESULTS_FOLDER, f"analyzed_{os.path.basename(img_path)}_hough.png"), img_hough)

    print("\nDone.")

if __name__ == "__main__":
    main()
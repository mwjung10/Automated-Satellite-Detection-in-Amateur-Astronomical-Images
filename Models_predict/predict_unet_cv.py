import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO

# path to unet model trained on ASTA dataset (yolov8-seg)
MODEL_PATH = '../best.pt'

IMAGES_FOLDER = '../images_to_predict' 
RESULTS_FOLDER = 'prediction_unet_hybrid_FINAL'

def analyze_mask_with_cv(mask, original_image):
    """
    Analyzes the binary mask using two methods:
    1. findContours to detect "blobs"
    2. HoughLinesP to detect line segments
    """
    image_with_contours = original_image.copy()
    image_with_hough = original_image.copy()
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"  [findContours] Found {len(contours)} potential objects.")
    for cnt in contours:
        # Filter out small objects (noise) that U-Net might have missed
        if cv2.contourArea(cnt) > 20: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green

    # --- Method 2: HoughLinesP (Line detection) ---
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
    
    if lines is not None:
        print(f"  [Hough] Found {len(lines)} line segments.")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_hough, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red
    else:
        print("  [Hough] No lines found.")
        
    return image_with_contours, image_with_hough

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Trained U-Net model not found at {MODEL_PATH}")
        return
        
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        
    model = YOLO(MODEL_PATH)
    print(f"U-Net model (segmentation) successfully loaded from {MODEL_PATH}")
    
    image_files = sorted(glob.glob(os.path.join(IMAGES_FOLDER, '*.jpg')))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(IMAGES_FOLDER, '*.png')))

    print(f"Found {len(image_files)} images in {IMAGES_FOLDER} to process...")

    for img_path in image_files:
        print(f"--- Processing: {os.path.basename(img_path)}")
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"  Skipped: Unable to load image {img_path}")
            continue

        # --- STEP 1: U-Net (YOLOv8-Seg) acts as an "AI filter" ---
        # Use a lower confidence threshold to catch even faint streaks
        results = model.predict(original_image, conf=0.05, verbose=False) 
        
        if results[0].masks is None:
            print("  [U-Net] No objects (masks) found in this image.")
            continue
            
        # Create an empty mask with the size of the image
        final_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
        
        # Combine all found masks into one (if U-Net found >1)
        for mask_data in results[0].masks:
            # Convert mask to a binary image 0-255
            mask_binary = (mask_data.data.cpu().numpy()[0] * 255).astype("uint8")
            if mask_binary.shape != final_mask.shape:
                mask_binary = cv2.resize(
                    mask_binary, 
                    (final_mask.shape[1], final_mask.shape[0]), 
                    interpolation=cv2.INTER_NEAREST 
                )
            final_mask = cv2.bitwise_or(final_mask, mask_binary)
            
        # Save the mask itself (for visualization and debugging)
        mask_save_path = os.path.join(RESULTS_FOLDER, f"{os.path.splitext(os.path.basename(img_path))[0]}_MASK.png")
        cv2.imwrite(mask_save_path, final_mask)
        
        # --- STEP 2: Run CV analysis (Hough and findContours) on the CLEAN mask ---
        img_contours, img_hough = analyze_mask_with_cv(final_mask, original_image)
        
        # Save both result images
        contours_save_path = os.path.join(RESULTS_FOLDER, f"{os.path.splitext(os.path.basename(img_path))[0]}_CONTOURS.png")
        hough_save_path = os.path.join(RESULTS_FOLDER, f"{os.path.splitext(os.path.basename(img_path))[0]}_HOUGH.png")
        
        cv2.imwrite(contours_save_path, img_contours)
        cv2.imwrite(hough_save_path, img_hough)

    print(f"Results saved in: {RESULTS_FOLDER}")

if __name__ == "__main__":
    main()
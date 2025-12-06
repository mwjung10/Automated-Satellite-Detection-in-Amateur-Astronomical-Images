import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO


MODEL_PATH = '../best_augumentation_streaks.pt'
IMAGES_FOLDER = '../images_to_predict'
RESULTS_FOLDER = 'prediction_results'
ENABLE_PREPROCESSING = True # to enable or disable preprocessing step
CONFIDENCE_THRESHOLD = 0.25


def preprocess_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        contrast_img = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(contrast_img, (7, 7), 0)
        preprocessed_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        return preprocessed_img
    except Exception as e:
        print(f"Warning: Preprocessing failed ({e}), using original image.")
        return image

def analyze_mask_with_cv(mask, original_image):
    image_with_contours = original_image.copy()
    image_with_hough = original_image.copy()
    
    # 1. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 20: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 2. Hough transform on the mask
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
    return image_with_contours, image_with_hough

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Failed to load model. {e}")
        return
    
    image_files = sorted(glob.glob(os.path.join(IMAGES_FOLDER, '*')))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.fits']
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]

    if not image_files:
        print(f"No images found in folder: {IMAGES_FOLDER}")
        return

    print(f"Found {len(image_files)} images.")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(filename)[0]
        print(f"\n--- Processing: {filename} ---")
        
        original_image = cv2.imread(img_path)
        if original_image is None:
            continue
        
        if ENABLE_PREPROCESSING:
            img_to_process = preprocess_image(original_image)
        else:
            img_to_process = original_image

        h, w = img_to_process.shape[:2]
        inference_size = max(h, w)
        inference_size = int(np.ceil(inference_size / 32) * 32)

    
        results = model.predict(
            source=img_to_process,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=inference_size,
            verbose=False,
            save=False 
        )
        
        result = results[0]

        annotated_frame = result.plot() # YOLO draws boxes on the image
        
        yolo_box_name = f"{filename_no_ext}_boxesyolo.jpg"
        cv2.imwrite(os.path.join(RESULTS_FOLDER, yolo_box_name), annotated_frame)
        print(f"  Saved YOLO preview: {yolo_box_name}")
        mask_from_boxes = np.zeros((h, w), dtype=np.uint8)
        
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"  Found {len(result.boxes)} objects. Generating mask from boxes")
            
            for i, box in enumerate(result.boxes):
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
            
                cv2.rectangle(mask_from_boxes, (x1, y1), (x2, y2), 255, -1)
                
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0].item())]
                print(f"    Object {i+1} ({label}): Conf={conf:.2f}, Box={coords.tolist()}")
            
            mask_filename = f"{filename_no_ext}_MASK_FROM_BOX.png"
            cv2.imwrite(os.path.join(RESULTS_FOLDER, mask_filename), mask_from_boxes)
            
            img_cnt, img_hough = analyze_mask_with_cv(mask_from_boxes, original_image)
            
            cv2.imwrite(os.path.join(RESULTS_FOLDER, f"{filename_no_ext}_analysis_cnt.png"), img_cnt)
            cv2.imwrite(os.path.join(RESULTS_FOLDER, f"{filename_no_ext}_analysis_hough.png"), img_hough)
            print(f"  Saved mask analysis results in: {RESULTS_FOLDER}")
            
        else:
            print("  No objects detected -> Mask is black.")

    print(f"\n--- Done. Results in: {RESULTS_FOLDER} ---")

if __name__ == "__main__":
    main()
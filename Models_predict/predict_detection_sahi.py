import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

MODEL_PATH = '../best_syn_aug_rtdert.pt'
IMAGES_FOLDER = '../images_to_predict/'
RESULTS_FOLDER = 'prediction_images/preprocessing_results'

SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2
CONFIDENCE_THRESHOLD = 0.25

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast_img, (7, 7), 0) 

    preprocessed_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    return preprocessed_img

def analyze_mask_with_cv(mask, original_image):
    image_with_contours = original_image.copy()
    image_with_hough = original_image.copy()
    
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours.")
    for cnt in contours:
        if cv2.contourArea(cnt) > 20: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
    
    if lines is not None:
        print(f"Found {len(lines)} Hough lines.")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("No Hough lines found.")
            
    return image_with_contours, image_with_hough

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
        
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        
    print(f"Loading SAHI model wrapper for: {MODEL_PATH}")
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    )
    
    image_files = sorted(glob.glob(os.path.join(IMAGES_FOLDER, '*.*')))
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.fits']
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]

    print(f"Found {len(image_files)} images in {IMAGES_FOLDER} to process...")

    for img_path in image_files:
        print(f"\n--- Processing: {os.path.basename(img_path)}")
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"  Skipped: Unable to load image {img_path}")
            continue
        
        h, w = original_image.shape[:2]

        preprocessed_image = preprocess_image(original_image)
        
        preprocessed_save_path = os.path.join(RESULTS_FOLDER, f"{os.path.splitext(os.path.basename(img_path))[0]}_PREPROCESSED.png")
        cv2.imwrite(preprocessed_save_path, preprocessed_image)

        print(f"  [Model] Predicting with SAHI (slice={SLICE_HEIGHT}x{SLICE_WIDTH})...")
        
    
        image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
        
        result = get_sliced_prediction(
            image_rgb,
            detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            verbose=0
        )
        
        object_prediction_list = result.object_prediction_list

        if len(object_prediction_list) == 0:
            print("No objects found.")
            continue
        
        print(f"Found {len(object_prediction_list)} boxes (Detection).")

        
        final_mask = np.zeros((h, w), dtype=np.uint8)

        for prediction in object_prediction_list:
            bbox = prediction.bbox.to_xyxy() 
            x1, y1, x2, y2 = map(int, bbox)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cv2.rectangle(final_mask, (x1, y1), (x2, y2), 255, -1)

        mask_path = os.path.join(RESULTS_FOLDER, f"{os.path.splitext(os.path.basename(img_path))[0]}_MASK.png")
        cv2.imwrite(mask_path, final_mask)
        
        img_cnt, img_hough = analyze_mask_with_cv(final_mask, original_image)
        
        cv2.imwrite(os.path.join(RESULTS_FOLDER, f"analyzed_{os.path.basename(img_path)}_cnt.png"), img_cnt)
        cv2.imwrite(os.path.join(RESULTS_FOLDER, f"analyzed_{os.path.basename(img_path)}_hough.png"), img_hough)

    print(f"\n--- Processing completed. Results in: {RESULTS_FOLDER}")

if __name__ == "__main__":
    main()
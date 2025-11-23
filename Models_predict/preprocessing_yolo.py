"""
Steps for predicing images:
1. Add prepocessing of images: grayscale, CLAHE
2. Use Yolo on preprocessed images
"""

import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = '../runs/detect/yolov8n_streaks_baseline/weights/best.pt'

IMAGES_TO_PREDICT_FOLDER = '../images_to_predict'
RESULTS_SAVE_DIR = 'runs/detect/predict_clahe'


def apply_clahe_pipeline(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Median filter to reduce noise
    gray_filtered = cv2.medianBlur(gray, 3)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_filtered)
    
    # Convert back to BGR
    final_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    return final_bgr


def predcit_all_images_in_folder():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Could not find model file at '{MODEL_PATH}'.")
        return
        
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
    if not os.path.isdir(IMAGES_TO_PREDICT_FOLDER):
        print(f"ERROR: Folder '{IMAGES_TO_PREDICT_FOLDER}' does not exist.")
        return

    # Find all images (JPG, PNG) in the folder
    allowed_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_files = []
    for ext in allowed_extensions:
        image_files.extend(glob.glob(os.path.join(IMAGES_TO_PREDICT_FOLDER, ext)))
        
    print(f"Found {len(image_files)} images to process.\n")

    for img_path in image_files:
        print(f"--- Processing file: {img_path}")
        
        # Step A: Load the .JPG/.PNG image
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print("  Skipped: failed to load image.")
            continue

        # Step B: Apply your preprocessing pipeline (CLAHE, etc.)
        print("  Applying preprocessing pipeline (CLAHE)...")
        processed_image = apply_clahe_pipeline(image_bgr)
        
        # Step C: Run YOLO prediction
        print("  Running YOLO prediction...")
        model.predict(
            processed_image,
            save=True,
            save_txt=True,
            project=RESULTS_SAVE_DIR, 
            name=os.path.basename(img_path), 
            conf=0.5   
        )
        print("  Prediction completed and saved.\n")

    print(f"--- Finished processing all images. ---")
    print(f"Results (images with boxes) have been saved in the folder '{RESULTS_SAVE_DIR}'.")


def predict_one_image(name):
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Could not find model file at '{MODEL_PATH}'.")
        return
        
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    test_image_path = os.path.join(IMAGES_TO_PREDICT_FOLDER, name)
    print(f"Processing single image: {test_image_path}")
    
    image_bgr = cv2.imread(test_image_path)
    if image_bgr is None:
        print("  Skipped: failed to load image.")
        return

    processed_image = apply_clahe_pipeline(image_bgr)
    
    print("  Running YOLO prediction...")
    results = model.predict(
        processed_image,
        save=True,
        save_txt=True,
        project=RESULTS_SAVE_DIR, 
        name=name + '_preprocess_pred',
        conf=0.1
    )
    result = results[0]
    
    print("\n--- Prediction finished ---")
    if len(result.boxes) == 0:
        print("Model did not find any objects.")
    else:
        print(f"Model found {len(result.boxes)} objects!")
        for box in result.boxes:
            confidence = box.conf[0]
            print(f"  - Confidence: {confidence:.2f}")
    
    print(f"Results (image with boxes) have been saved in the folder '{RESULTS_SAVE_DIR}'.")

    

if __name__ == "__main__":
    predict_one_image('DSC02533.jpg')
    predict_one_image('DSC02545.jpg')
    predict_one_image('DSC02547.jpg')
    predict_one_image('DSC01979.jpg')
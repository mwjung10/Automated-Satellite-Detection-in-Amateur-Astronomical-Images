import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO

MODEL_PATH = '../best_augumentation_streaks.pt'
IMAGES_FOLDER = '../images_to_predict_cut'
RESULTS_FOLDER = 'prediction_images_cut/preprocessing_results'

def preprocess_image(image):
    """
    Applies CLAHE and Adaptive Thresholding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast_img, (7, 7), 0)  # this one works good

    # thresh_img = cv2.adaptiveThreshold(
    #     contrast_img, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     19, 3 
    # )
    
    preprocessed_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    return preprocessed_img
#----------------------------------------------------------------------------------------------------
    # not so good!!
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    
    # tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # _, binary = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
    
    # return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#----------------------------------------------------------------------------------------------------
    # not so good as well !!
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)

    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # cl = clahe.apply(l)

    # limg = cv2.merge((cl, a, b))
    # enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    # thresh_img = cv2.adaptiveThreshold(
    #     gray, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     19, 3 
    # )

    # preprocessed_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    # return preprocessed_img
#----------------------------------------------------------------------------------------------------
    # not so good !!
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # #edges = cv2.Canny(gray_image, 100, 200)
    # normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
    # blurred_image = cv2.GaussianBlur(normalized_image, (3, 3), 0)
    # #blurred_image = cv2.medianBlur(normalized_image, 5)
    # _, binary_img = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # preprocessed_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    # return preprocessed_img
#----------------------------------------------------------------------------------------------------
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray_image, 100, 200)
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image_2 = image.copy()
    # cv2.drawContours(image_2, contours, -1, (0, 255, 0), 2)
    # return image_2
#----------------------------------------------------------------------------------------------------
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    # enhanced = clahe.apply(gray_image)

    # blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # edges = cv2.Canny(blurred, 50, 200)
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # (OOptional)
    # # valid_contours = []
    # # for cnt in contours:
    # #     if cv2.arcLength(cnt, True) > 50: 
    # #         valid_contours.append(cnt)

    # image_2 = image.copy()
    # cv2.drawContours(image_2, contours, -1, (0, 255, 0), 2)
    
    # return image_2




def analyze_mask_with_cv(mask, original_image):
    image_with_contours = original_image.copy()
    image_with_hough = original_image.copy()
    
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
        
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
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
        
        inference_size = max(h, w)
        inference_size = int(np.ceil(inference_size / 32) * 32)

        preprocessed_image = preprocess_image(original_image)
        
        preprocessed_save_path = os.path.join(RESULTS_FOLDER, f"{os.path.splitext(os.path.basename(img_path))[0]}_PREPROCESSED.png")
        cv2.imwrite(preprocessed_save_path, preprocessed_image)

        print(f"  [Model] Predicting on preprocessed image (imgsz={inference_size})...")
        results = model.predict(
            preprocessed_image, 
            conf=0.01, 
            imgsz=inference_size, 
            verbose=False
        ) 
        
        result = results[0]
        
        # Sprawdzenie czy cokolwiek znaleziono
        if result.masks is None and (result.boxes is None or len(result.boxes) == 0):
            print("No objects found (masks or boxes).")
            continue
            
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        if result.masks is not None:
            print(f"Found {len(result.masks)} masks (Segmentation).")
            
            masks_data = result.masks.data.cpu().numpy()
            for i, mask_tensor in enumerate(masks_data):
                mask_single = (mask_tensor * 255).astype("uint8")
                
                if mask_single.shape[:2] != (h, w):
                    mask_single = cv2.resize(mask_single, (w, h), interpolation=cv2.INTER_NEAREST)
                
                final_mask = cv2.bitwise_or(final_mask, mask_single)
                
        elif result.boxes is not None:
            print(f"Found {len(result.boxes)} boxes (Detection).")
            
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

    print(f"\n--- Processing completed. Results in: {RESULTS_FOLDER}")

if __name__ == "__main__":
    main()
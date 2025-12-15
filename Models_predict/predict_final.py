import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO


MODEL_PATH = '../best_augumentation_streaks.pt'
IMAGES_FOLDER = '../images_to_predict_short'
RESULTS_FOLDER = 'prediction_results_short/no_preprocessing'
ENABLE_PREPROCESSING = True # to enable or disable preprocessing step
CONFIDENCE_THRESHOLD = 0.25

def clip_line_to_rect(vx, vy, x0, y0, w, h):
    """
    Calculates the intersection points of a line defined by a point (x0, y0) and direction vector (vx, vy)
    with the rectangle defined by width w and height h (
    """
    if abs(vx) < 1e-9: vx = 1e-9
    if abs(vy) < 1e-9: vy = 1e-9

    # T calculations for intersections with rectangle edges
    # Edges vertical (x=0 and x=w-1)
    # position = start + time * direction
    # x = x0 + t*vx, y = y0 + t*vy
    
    t_left = (0 - x0) / vx
    t_right = (w - 1 - x0) / vx
    
    # Edges horizontal (y=0 and y=h-1)
    t_top = (0 - y0) / vy
    t_bottom = (h - 1 - y0) / vy

    intersections = []
    
    # Check each of the 4 potential intersections
    for t in [t_left, t_right, t_top, t_bottom]:
        x = x0 + t * vx
        y = y0 + t * vy
        
        if -1.0 <= x <= w and -1.0 <= y <= h:
            intersections.append((int(x), int(y)))

    # Remove duplicates and sort (to have points in some order)
    intersections = sorted(list(set(intersections)))
    
    # We should find 2 points. If the line touches only a corner, there may be 1.
    if len(intersections) >= 2:
        # We take the two most distant points (in case more were found due to rounding)
        return intersections[0], intersections[-1]
    
    return None

def get_streaks_endpoints_and_angle(box, image):
    x1, y1, x2, y2 = map(int, box)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    roi = image[y1:y2, x1:x2] # region of interest
    
    if roi.size == 0:
        return None

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    
    if points is not None:
        # Fit line to points
        # vx, vy - direction vector
        # x, y - a point on the line
        [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = vx.item()
        vy = vy.item()
        x = x.item()
        y = y.item()
        
        angle = np.degrees(np.arctan2(vy, vx))
        if angle < 0:
            angle += 180

        rows, cols = gray_roi.shape
        
        endpoints_local = clip_line_to_rect(vx, vy, x, y, cols, rows)
        
        if endpoints_local:
            p1_local, p2_local = endpoints_local
            
            p1_global = (x1 + p1_local[0], y1 + p1_local[1])
            p2_global = (x1 + p2_local[0], y1 + p2_local[1])

            return p1_global, p2_global, angle
        
    return None

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
                endpoints = get_streaks_endpoints_and_angle(coords, original_image)
                if endpoints is not None:
                    p1, p2, angle = endpoints
                    print(f" Endpoints: {p1}, {p2}, Angle: {angle} degrees")
            
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
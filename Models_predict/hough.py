import cv2
import numpy as np
import os
import glob

IMAGES_FOLDER = '../all_images_to_predict' 
RESULTS_FOLDER = 'Houghruns/detect/predict_hough_baseline'
# ------------------

def detect_lines_with_hough(image_path, save_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Couldn't read {image_path}")
        return

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    # Probabilistic Hough Transform
    # We search for lines with a minimum length of 50 pixels
    # and a maximum gap of 10 pixels between segments to be considered a single line.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        print(f"  Found {len(lines)} lines.")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw in red
        cv2.imwrite(save_path, image_bgr)
        original_save_path = os.path.join(RESULTS_FOLDER, "original_" + os.path.basename(image_path))
        cv2.imwrite(original_save_path, cv2.imread(image_path))
        return lines
    else:
        print("  No lines found.")
        return None
    
    

def predict_all_images():
    detected_lines = 0
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
     
    image_files = glob.glob(os.path.join(IMAGES_FOLDER, '*.jpg'))
    print(f"Found {len(image_files)} images to process...")
    
    for img_path in image_files:
        print(f"Processing: {img_path}")
        save_name = os.path.basename(img_path)
        save_path = os.path.join(RESULTS_FOLDER, save_name)
        lines = detect_lines_with_hough(img_path, save_path)
        if lines is not None:
            detected_lines += len(lines)
        
    print(f"\nTotal lines detected across all images: {detected_lines}")
    print(f"\nFinished. Results saved in: {RESULTS_FOLDER}")


def predict_one_image(image_path):
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    image_path = os.path.join(IMAGES_FOLDER, image_path)
        
    print(f"Processing single image: {image_path}")
    save_name = os.path.basename(image_path)
    save_path = os.path.join(RESULTS_FOLDER, save_name)
    detect_lines_with_hough(image_path, save_path)
    
    print(f"Finished. Result saved in: {save_path}")

if __name__ == "__main__":
    '''
    predict_one_image('DSC02533.jpg')  # WORKED!!
    predict_one_image('DSC02545.jpg')
    predict_one_image('DSC02547.jpg')
    predict_one_image('DSC01979.jpg')
    '''
    predict_all_images()

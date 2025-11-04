import os
import shutil
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = 'best.pt'

OUTPUT_DIR = Path('error_analysis')

SET_TO_ANALYZE = 'test' 

DATASET_PATH = Path('StreaksYoloDataset')

CONF_THRESHOLD = 0.1

def analyze_errors_detailed():
    print("--- Starting detailed error analysis ---")
    
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model not found! Check path: {MODEL_PATH}")
        return
        
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    image_dir = DATASET_PATH / SET_TO_ANALYZE / 'images'
    label_dir = DATASET_PATH / SET_TO_ANALYZE / 'labels'

    if not image_dir.exists() or not label_dir.exists():
        print(f"[ERROR] 'images' or 'labels' folders not found in: {DATASET_PATH / SET_TO_ANALYZE}")
        return

    fn_dir = OUTPUT_DIR / 'false_negatives'
    fp_dir = OUTPUT_DIR / 'false_positives'
    os.makedirs(fn_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)
    
    print(f"False Negative images will be saved to: {fn_dir}")
    print(f"False Positive images will be saved to: {fp_dir}")
    
    image_files = list(image_dir.glob('*.jpg'))
    if not image_files:
        image_files.extend(list(image_dir.glob('*.png'))) 
    if not image_files:
        image_files.extend(list(image_dir.glob('*.jpeg')))
    if not image_files:
        print(f"[ERROR] No images (.jpg or .png) found in: {image_dir}")
        return

    print(f"Found {len(image_files)} images to analyze in '{SET_TO_ANALYZE}'...")

    fn_count = 0
    fp_count = 0

    for i, img_path in enumerate(image_files):
        print(f"  Analyzing {i+1}/{len(image_files)}: {img_path.name}", end='\r')
        
        label_path = label_dir / (img_path.stem + '.txt')
        ground_truth_count = 0
        if label_path.exists():
            with open(label_path, 'r') as f:
                ground_truth_count = sum(1 for line in f if line.strip())

        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        prediction_count = len(results[0].boxes)

        if ground_truth_count > 0 and prediction_count == 0:
            fn_count += 1
            shutil.copy(img_path, fn_dir / img_path.name)

        if ground_truth_count == 0 and prediction_count > 0:
            fp_count += 1
            save_path = fp_dir / img_path.name
            results[0].save(filename=str(save_path))

    print("\n--- Analysis finished ---")
    print(f"\nFound {fn_count} images with 'False Negative' errors (missed streaks).")
    print(f"Files saved in: {fn_dir}")
    
    print(f"\nFound {fp_count} images with 'False Positive' errors (false alarms).")
    print(f"Files saved in: {fp_dir}")
    
    if fn_count == 0 and fp_count == 0:
        print(f"\nNo FN/FP errors found at confidence threshold {CONF_THRESHOLD}.")
        print("If your metrics (Recall < 1.0) indicate that errors exist,")
        print("please ensure the MODEL_PATH and DATASET_PATH are correct.")


if __name__ == "__main__":
    analyze_errors_detailed()



# This file saves images that produced false negative and false positive errors
import os
import shutil
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = 'runs/detect/yolov8n_streaks_baseline/weights/best.pt'

DATASET_PATH = Path('StreaksYoloDataset')

SET_TO_ANALYZE = 'test'

OUTPUT_DIR = Path('error_analysis_results')

def analyze_errors():
    print(f"Ładowanie modelu z: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    image_dir = DATASET_PATH / SET_TO_ANALYZE / 'images'
    label_dir = DATASET_PATH / SET_TO_ANALYZE / 'labels'

    fn_dir = OUTPUT_DIR / 'false_negatives'
    fp_dir = OUTPUT_DIR / 'false_positives'
    os.makedirs(fn_dir, exist_ok=True)
    os.makedirs(fp_dir, exist_ok=True)

    print(f"Analyzing images in: {image_dir}")
    
    image_files = list(image_dir.glob('*.jpg'))
    if not image_files:
        image_files = list(image_dir.glob('*.png')) 

    fn_count = 0
    fp_count = 0

    for i, img_path in enumerate(image_files):
        print(f"  Analyzing {i+1}/{len(image_files)}: {img_path.name}", end='\r')
        
        label_path = label_dir / (img_path.stem + '.txt')
        ground_truth_count = 0
        if label_path.exists():
            with open(label_path, 'r') as f:
                ground_truth_count = len(f.readlines())

        results = model.predict(img_path, verbose=False)
        prediction_count = len(results[0].boxes)

        if ground_truth_count > 0 and prediction_count == 0:
            fn_count += 1
            shutil.copy(img_path, fn_dir / img_path.name)


        if ground_truth_count == 0 and prediction_count > 0:
            fp_count += 1
            save_path = fp_dir / img_path.name
            results[0].save(filename=str(save_path))

    print("\n--- Analysis done ---")
    print(f"Found {fn_count} images with 'False Negative' errors (missed streaks).")
    print(f"Files saved in: {fn_dir}")
    print(f"\nFound {fp_count} images with 'False Positive' errors (false alarms).")
    print(f"Files saved in: {fp_dir}")

if __name__ == "__main__":
    analyze_errors()

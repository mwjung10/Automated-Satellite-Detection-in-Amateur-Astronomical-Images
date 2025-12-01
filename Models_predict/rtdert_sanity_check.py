import os
import glob
from ultralytics import RTDETR, YOLO
import cv2

MODEL_PATH = '../best_rtdetr.pt' 
IMAGES_FOLDER = '../images_to_predict_cut'
OUTPUT_FOLDER = 'debug_native_output'

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"Model loading: {MODEL_PATH}")
    try:
        model = RTDETR(MODEL_PATH)  
    except Exception as e:
        print(f"CRITICAL ERROR: Cannot load model. Check the path! {e}")
        return
    
    image_files = sorted(glob.glob(os.path.join(IMAGES_FOLDER, '*.*'))) 
    print(f"Testing on {len(image_files)} images...")

    for img_path in image_files:
        print(f"Processing: {os.path.basename(img_path)}")
        
       
        results = model.predict(
            img_path, 
            conf=0.3,       # Low threshold for testing
            imgsz=640,      # Default size
            save=True,      # <--- THIS IS IMPORTANT: Let the model draw the results itself
            project=OUTPUT_FOLDER, # Where to save
            name='results', # Subfolder
            exist_ok=True
        )
        
        for r in results:
            print(f"  Found objects: {len(r.boxes)}")
            if len(r.boxes) > 0:
                for box in r.boxes:
                    print(f"    -> Box: {box.xyxy[0].cpu().numpy().astype(int)} | Conf: {box.conf.item():.2f}")

    print(f"\nDone. Check the folder: {OUTPUT_FOLDER}/results")

if __name__ == "__main__":
    main()
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = 'runs/detect/yolov8n_streaks_baseline/weights/best.pt'

IMAGE_PATH = 'images_to_predict/img1.jpeg'

OUTPUT_DIR = Path('predictions')

def predict_custom_image():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load model. Check if MODEL_PATH is correct.")
        print(f"Error: {e}")
        return

    print(f"Running prediction for: {IMAGE_PATH}")


    try:
        results = model.predict(
            source=IMAGE_PATH,
            save=True,
            save_txt=True,
            conf=0.1,  # Show even weak detections
            project=str(OUTPUT_DIR),
            name='custom_prediction_1'
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

        print(f"\nOutput image saved to: {OUTPUT_DIR / 'custom_prediction_1'}")

    except Exception as e:
        print(f"[ERROR] An error occurred during prediction. Check if the image path is correct.")
        print(f"Error: {e}")

if __name__ == "__main__":
    predict_custom_image()

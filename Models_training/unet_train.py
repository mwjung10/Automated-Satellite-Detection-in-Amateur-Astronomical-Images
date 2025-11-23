import torch
from ultralytics import YOLO
import os

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Used device: {device}")
 
    print("Loading YOLOv8-Segment model (U-Net architecture)...")
    model = YOLO('yolov8n-seg.pt')
    model.to(device)
    
    try:
        results = model.train(
            data='asta.yaml', 
            epochs=200,      
            batch=8,        
            optimizer='Adam',
            imgsz=640,
            device=device,
            name='unet_asta_nano_with_flipud',
            project='runs/segment',
	    cache = False,
	    flipud=0.5,
	    mosaic=1.0,
	    hsv_h=0.05,
            hsv_s=0.7,
	    hsv_v=0.4,
	    copy_paste=0.3
            )

        print("\nU-Net training completed successfully.")
        print(f"Results and model weights saved in: {results.save_dir}")
        
        metrics = model.val(split='val', data='asta.yaml')

        print("\n--- Segmentation Model Metrics ---")
        print(f"mAP50-95 (Mask): {metrics.seg.map:.4f}")
        print(f"mAP50 (Mask): {metrics.seg.map50:.4f}")
        print(f"mAP75 (Maska): {metrics.seg.map75:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}")
        print("Make sure the 'asta.yaml' file exists and correctly points")
        print("to the /ASTA_dataset_split/ folder, which was created by split_asta_dataset.py")

if __name__ == '__main__':
    main()

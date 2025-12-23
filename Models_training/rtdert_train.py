from ultralytics import RTDETR
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Used device: {device}")

    model = RTDETR('rtdetr-l.pt')

    print("Starting RT-DETR training...")
    
    try:
        results = model.train(
            data='/net/tscratch/people/plgmjung/p_dyplomowa/Automated-Satellite-Detection-in-Amateur-Astronomical-Images/synthetic.yaml',       
            epochs=100,                 
            batch=16,                  
            optimizer='AdamW', 
            lr0=0.0001,
            cos_lr = True,         
            imgsz=640,                 
            device=device,
            name='rtdetr_l_synthetic_augmented',
            hsv_h=0.005,  # bc my photos are mostly dark
            hsv_s=0.1,
            hsv_v = 0.7, 
            degrees = 0.0,     
            scale = 0.5,        
            flipud = 0.5,       
            fliplr = 0.5,       
            mosaic = 0.5,       
            mixup = 0.1
        )
        
        print("\nTraining completed.")
        
        print("Running validation on the test set...")
        metrics = model.val(split='test')
        
        print("\n--- RT-DETR Metrics ---")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == '__main__':
    main()
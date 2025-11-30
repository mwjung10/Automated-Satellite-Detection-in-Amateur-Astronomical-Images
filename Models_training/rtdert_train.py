from ultralytics import RTDETR
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Used device: {device}")

    model = RTDETR('rtdetr-l.pt')

    print("Starting RT-DETR training...")
    
    try:
        results = model.train(
            data='streaks.yaml',       
            epochs=50,                 
            batch=8,  
            optimizer='Adam',                
            imgsz=640,
            device=device,
            name='rtdetr_streaks_test1'
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
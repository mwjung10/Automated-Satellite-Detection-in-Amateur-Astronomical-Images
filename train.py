from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Used device: {device}")


    model = YOLO('yolov9t.pt')

    print("Starting base training...")
    
    try:
        results = model.train(
            data='streaks.yaml',       #
            epochs=30,                 
            batch=16,                  
            optimizer='Adam',          
            imgsz=640,                 
            device=device,
            name='yolov9t_streaks_baseline' 
        )

        print("\nTraining completed successfully.")
        print(f"Results and model weights saved in: {results.save_dir}")


        print("\nStarting validation on 'test' set...")
        metrics = model.val(split='test')

        print("\n--- Base model metrics ---")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"Precision: {metrics.box.p.mean():.4f}") 
        print(f"Recall: {metrics.box.r.mean():.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}")
        print("Make sure the path in 'streaks.yaml' is correctz!")

if __name__ == '__main__':
    main()

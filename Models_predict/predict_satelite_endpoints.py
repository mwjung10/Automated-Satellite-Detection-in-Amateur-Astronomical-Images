import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = 'runs/detect/yolov8n_streaks_baseline/weights/best.pt' 
TEST_IMAGE = 'StreaksYoloDataset/test/images/img_00012.jpg' 

def get_streak_endpoints(image_path, model_path):
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Błąd: Nie można wczytać obrazu {image_path}")
        return

    original_h, original_w = img.shape[:2]
    
    # 2.  predykcję YOLO
    results = model(image_path)
    
    print(f"\n--- Analiza obrazu: {image_path} ---")
    
    found_any = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            found_any = True
            # Pobierz koordynaty ramki (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Margines bezpieczeństwa
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_w, x2), min(original_h, y2)
            
            # --- POST-PROCESSING ---
            roi = img[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            points = cv2.findNonZero(binary)
            
            if points is not None:
                # [vx, vy, x, y] -> wektor kierunkowy i punkt na linii
                [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # --- NOWOŚĆ: OBLICZANIE KĄTA ---
                # Obliczamy kąt w radianach, a potem zamieniamy na stopnie
                angle = np.degrees(np.arctan2(vy, vx))[0]
                
                # Normalizacja kąta do przedziału 0-180 stopni (bo linia nie ma zwrotu)
                if angle < 0:
                    angle += 180
                
                # --- Obliczanie punktów końcowych (tak jak wcześniej) ---
                rows, cols = gray_roi.shape
                lefty = int((-x * vy/vx) + y)
                righty = int(((cols - x) * vy/vx) + y)
                
                p1_local = (0, lefty)
                p2_local = (cols - 1, righty)
                
                p1_global = (x1 + p1_local[0], y1 + p1_local[1])
                p2_global = (x1 + p2_local[0], y1 + p2_local[1])
                
                print(f"Smuga wykryta (Pewność: {conf:.2f})")
                print(f"  > Kąt nachylenia: {angle:.2f} stopni")
                print(f"  > Początek:       {p1_global}")
                print(f"  > Koniec:         {p2_global}")
                print("-" * 20)
                
                # --- WIZUALIZACJA ---
                # Zielona ramka YOLO
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Czerwona linia środkowa
                cv2.line(img, p1_global, p2_global, (0, 0, 255), 2)
                
                # Niebieskie kropki na końcach
                cv2.circle(img, p1_global, 4, (255, 0, 0), -1)
                cv2.circle(img, p2_global, 4, (255, 0, 0), -1)
                
                # Żółty tekst z kątem nad ramką
                label = f"Angle: {angle:.1f} deg"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if not found_any:
        print("Nie wykryto żadnych smug na tym zdjęciu.")

    # Zapisz wynik
    output_filename = "wynik_z_katem.jpg"
    cv2.imwrite(output_filename, img)
    print(f"\nObraz z narysowanymi liniami i kątami zapisano jako: {output_filename}")



if __name__ == "__main__":
    try:
        get_streak_endpoints(TEST_IMAGE, MODEL_PATH)
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

FOLDER_PATH = 'opticalFlow_sequence'
EXTENSION = 'jpg'

def test_sequence_optical_flow():
    search_path = os.path.join(FOLDER_PATH, f'*.{EXTENSION}')
    files = sorted(glob.glob(search_path))

    if len(files) < 2:
        print("At least two images are required in the folder to compute optical flow.")
        return

    print(f"Found {len(files)} images. Calculating Optical Flow...")

    frame1 = cv2.imread(files[0])
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    h, w = prvs.shape
    
    # H (Hue) = Angle (direction), S (Saturation) = 255, V (Value) = Speed
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    total_flow_rgb = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(len(files) - 1):
        path_next = files[i+1]
        frame2 = cv2.imread(path_next)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

     
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 5, 15, 3, 5, 1.2, 0)
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # Speed determines brightness (normalize for visibility)
        # If nothing is visible, decrease '10' to a smaller number (increases sensitivity)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        total_flow_rgb = cv2.add(total_flow_rgb, rgb.astype(np.float32))
        
        prvs = next

    # Normalization of the final result to 0-255
    total_flow_rgb = cv2.normalize(total_flow_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Displaying results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("First Frame")
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Optical Flow (Color = Direction, Brightness = Speed)")
    plt.imshow(cv2.cvtColor(total_flow_rgb, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    test_sequence_optical_flow()
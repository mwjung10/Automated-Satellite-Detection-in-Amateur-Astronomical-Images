import cv2
import numpy as np
import matplotlib.pyplot as plt


IMG_PATH_1 = '../all_images_to_predict/DSC01805.jpg' 
IMG_PATH_2 = '../all_images_to_predict/DSC01806.jpg'  
# ------------------

def test_motion():
    frame1 = cv2.imread(IMG_PATH_1)
    frame2 = cv2.imread(IMG_PATH_2)

    if frame1 is None or frame2 is None:
        print("Couldn't load one of the images. Check the paths.")
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Simple Frame Differencing
    diff = cv2.absdiff(prvs, next)
    
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Optical Flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 5, 15, 3, 7, 1.5, 0)

    h, w = prvs.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image 1")
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Method 1: Difference (Subtraction + Thresholding)")
    plt.imshow(diff_thresh, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Method 2: Optical Flow")
    plt.imshow(cv2.cvtColor(rgb_flow, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    print("Displaying results...")
    plt.show()

if __name__ == "__main__":
    test_motion()
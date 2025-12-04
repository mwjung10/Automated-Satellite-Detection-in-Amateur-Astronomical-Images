import cv2
import numpy as np
import os

IMG_1 = 'StreaksYoloDataset/test/images/64.jpeg'  
IMG_2 = 'StreaksYoloDataset/test/images/65.jpeg'  

OUTPUT_FILE = 'crossing_images/img.jpg'


def create_crossing_trails(image_path1, image_path2, output_path):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None or img2 is None:
        print("Error: One of the images could not be loaded.")
        return

    print(f"Loaded:\n A: {image_path1}\n B: {image_path2}")

    # Resize (in case they are different)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    blended_image = cv2.max(img1, img2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, blended_image)
    print(f"\nSuccess! Created test image: {output_path}")

if __name__ == "__main__":
    if os.path.exists(IMG_1) and os.path.exists(IMG_2):
        create_crossing_trails(IMG_1, IMG_2, OUTPUT_FILE)
    else:
        print("Error: Input files not found.")
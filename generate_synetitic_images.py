import cv2
import numpy as np
import os
import random
from tqdm import tqdm

OUTPUT_DIR = "Amateur_Synthetic_Dataset"
IMG_W, IMG_H = 640, 640
NUM_IMAGES = 10000

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

def generate_procedural_horizon(h, w):
    """
    Generates a procedural horizon mask with tree-like silhouettes.
    Returns a mask: 0 = sky, 255 = obstacle (trees/buildings).
    """
    horizon_mask = np.zeros((h, w), dtype=np.uint8)
    
    current_height = random.randint(h - int(h*0.4), h - 50)
    
    points = []
    step = 5 
    
    for x in range(0, w, step):
        change = random.randint(-15, 15)
        current_height += change
        current_height = np.clip(current_height, h//2, h-1)
        points.append([x, current_height])
    
    points.append([w, h]) 
    points.append([0, h]) 
    
    pts = np.array(points, np.int32)
    cv2.fillPoly(horizon_mask, [pts], 255)
    
    horizon_mask = cv2.GaussianBlur(horizon_mask, (3, 3), 0)
    
    return horizon_mask

def generate_sky_texture(h, w):
    """
    Creates a non-uniform, "dirty" sky background.
    """
    small_h, small_w = h // 16, w // 16
    low_res_noise = np.random.randint(20, 45, (small_h, small_w)).astype(np.uint8)
    sky = cv2.resize(low_res_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    return sky

def add_clouds(img):
    """
    Adds irregular, misty shapes (clouds/nebulae).
    """
    h, w = img.shape
    
    # 30% chance to add clouds
    if random.random() > 0.7:
        return img

    small_h, small_w = h // 8, w // 8
    noise = np.random.randint(0, 255, (small_h, small_w)).astype(np.uint8)
    
    _, mask = cv2.threshold(noise, 180, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (21, 21), 0) 
    
    cloud_layer = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
    cloud_layer = (cloud_layer * 0.3).astype(np.uint8)
    
    return cv2.add(img, cloud_layer)

def add_vignetting(img):
    """Adds dark corners."""
    rows, cols = img.shape
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignetted = img * mask
    return vignetted.astype(np.uint8)

def add_light_pollution(img):
    """Adds city glow gradient from the bottom."""
    h, w = img.shape
    gradient = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    intensity = random.randint(20, 80)
    glow = (gradient * intensity).astype(np.uint8)
    return cv2.add(img, glow)

def generate_image():
    # 1. BACKGROUND
    img = generate_sky_texture(IMG_H, IMG_W)
    img = add_clouds(img)
    # cv2.imshow("Debug", img)
    # cv2.waitKey(0)
    
    # 2. DECIDE HORIZON (Early decision)
    horizon_mask = None
    inv_horizon_mask = None
    
    # 50% chance to include trees/buildings
    if random.choice([True, False]):
        horizon_mask = generate_procedural_horizon(IMG_H, IMG_W)
        inv_horizon_mask = cv2.bitwise_not(horizon_mask)
        # cv2.imshow("Horizon Mask", horizon_mask)
        # cv2.waitKey(0)
    
    # 3. STARS
    num_stars = random.randint(100, 500)
    for _ in range(num_stars):
        x, y = random.randint(0, IMG_W), random.randint(0, IMG_H)
        star_img = np.zeros_like(img)
        bright = random.randint(50, 200) 
        # Use LINE_AA for smooth stars
        radius = random.randint(0,1)
        cv2.circle(star_img, (x, y), radius, bright, -1, lineType=cv2.LINE_AA)
        img = cv2.add(img, star_img)

    # cv2.imshow("Stars Layer", img)
    # cv2.waitKey(0)
    
    # 4. SATELLITES (With Occlusion Logic)
    labels = []
    num_sats = random.randint(0, 2)
    
    sat_layer = np.zeros_like(img)
    
    for _ in range(num_sats):
        x1, y1 = random.randint(-50, IMG_W+50), random.randint(-50, IMG_H+50)
        x2, y2 = random.randint(-50, IMG_W+50), random.randint(-50, IMG_H+50)
        
        if np.hypot(x2-x1, y2-y1) < 50: continue

        # Lower brightness = less visible (30-160 range)
        brightness = random.randint(30, 160)
        thickness = 1
        
        # Draw the line on the visual layer
        cv2.line(sat_layer, (x1, y1), (x2, y2), brightness, thickness, lineType=cv2.LINE_AA)
        
        #  Create a temp mask for this specific satellite
        single_sat_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        # Draw white line on temp mask (thickness 1 or slightly more to capture pixels)
        cv2.line(single_sat_mask, (x1, y1), (x2, y2), 255, thickness) 
        
        # If trees exist, mask out the hidden parts
        if inv_horizon_mask is not None:
            single_sat_mask = cv2.bitwise_and(single_sat_mask, single_sat_mask, mask=inv_horizon_mask)
        
        # Find visible pixels
        visible_points = cv2.findNonZero(single_sat_mask)
        
        if visible_points is not None:
            # Get bounding box of visible parts
            bx, by, bw, bh = cv2.boundingRect(visible_points)
            
            if bw > 5 or bh > 5:
                cx = (bx + bw / 2) / IMG_W
                cy = (by + bh / 2) / IMG_H
                nw = bw / IMG_W
                nh = bh / IMG_H
                
                if 0 < cx < 1 and 0 < cy < 1:
                    labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        # If visible_points is None, the satellite is completely hidden -> No label added.

    img = cv2.add(img, sat_layer)
    # cv2.imshow("Satellites Layer", img)
    # cv2.waitKey(0)

    # 5. APPLY HORIZON (Visual Occlusion)
    if inv_horizon_mask is not None:
        img = cv2.bitwise_and(img, img, mask=inv_horizon_mask)
        
    # cv2.imshow("After Horizon", img)
    # cv2.waitKey(0)

    # 6. OPTICS
    img = add_light_pollution(img)
    # cv2.imshow("After Light Pollution", img)
    # cv2.waitKey(0)
    img = add_vignetting(img)
    # cv2.imshow("After Vignetting", img)
    # cv2.waitKey(0)

    
    # 7. SENSOR NOISE
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # cv2.imshow("After Sensor Noise", img)
    # cv2.waitKey(0)
    return img, labels

print(f"Generating {NUM_IMAGES} synthetic images...")

for i in tqdm(range(NUM_IMAGES)):
    img, labels = generate_image()
    
    filename = f"amateur_syn_{i:04d}"
    
    cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}.jpg", img)
    
    if labels:
        with open(f"{OUTPUT_DIR}/labels/{filename}.txt", "w") as f:
            for label in labels:
                f.write(label + "\n")

print(f"Done! Data saved in '{OUTPUT_DIR}' folder.")
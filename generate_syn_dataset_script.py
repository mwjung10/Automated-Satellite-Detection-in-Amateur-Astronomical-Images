import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# USER CONFIGURATION
# ==========================================

OUTPUT_DIR = "Amateur_Configurable_Dataset"
IMG_W = 640
IMG_H = 640
NUM_IMAGES = 10

# Simulation of telescope tracking errors
TRACKING_ERROR_CHANCE = 0.5 
MAX_TRACKING_BLUR = 5       
PSF_KERNEL_SIZE = 5

STAR_COUNT_MIN = 100
STAR_COUNT_MAX = 600
STAR_BRIGHTNESS_MIN = 0.3
STAR_BRIGHTNESS_MAX = 1.0

SAT_COUNT_MIN = 0
SAT_COUNT_MAX = 2
SAT_BRIGHTNESS_MIN = 50
SAT_BRIGHTNESS_MAX = 180
SAT_THICKNESS = 1
SAT_DEFOCUS_KERNEL_SIZE = 5


LIGHT_POLLUTION_MIN = 20
LIGHT_POLLUTION_MAX = 80
SKY_NOISE_BASE = 30
SKY_NOISE_VAR = 20
ENABLE_OBSTACLES = True   
OBSTACLE_CHANCE = 0.5      

# ==========================================


os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

class OpticsSimulator:
    """
    Handles physics-based rendering: PSF, Tracking Errors, Sensor Noise.
    """
    @staticmethod
    def apply_motion_blur(image, length, angle_deg):
        """Simulates telescope tracking error (stars become lines)."""
        if length <= 1:
            return image
        
        kernel_size = length
        kernel = np.zeros((kernel_size, kernel_size))
        
        center = kernel_size // 2
        angle_rad = np.deg2rad(angle_deg)
        x_offset = int((length/2) * np.cos(angle_rad))
        y_offset = int((length/2) * np.sin(angle_rad))
        
        cv2.line(kernel, (center - x_offset, center - y_offset), 
                 (center + x_offset, center + y_offset), 1, 1)
        
        kernel /= np.sum(kernel) if np.sum(kernel) > 0 else 1
        
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def add_poisson_noise(img):
        """
        Simulates Photon Shot Noise (Poisson distribution).
        """
        img_float = img.astype(float)
        
       
        noise = np.random.normal(0, np.sqrt(img_float + 1)) 
        noisy = img_float + noise
        
        return np.clip(noisy, 0, 255).astype(np.uint8)

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
    noise = np.random.randn(h, w) * SKY_NOISE_VAR + SKY_NOISE_BASE
    return np.clip(noise, 0, 255).astype(np.uint8)

def add_vignetting(img):
    rows, cols = img.shape
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignetted = img * mask
    return vignetted.astype(np.uint8)

def add_light_pollution(img):
    h, w = img.shape
    gradient = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    intensity = random.randint(LIGHT_POLLUTION_MIN, LIGHT_POLLUTION_MAX)
    glow = (gradient * intensity).astype(np.uint8)
    return cv2.add(img, glow)

def generate_image():
    base_sky = generate_sky_texture(IMG_H, IMG_W)
    base_sky = add_light_pollution(base_sky)
    
    star_layer = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    
    sat_layer = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    num_stars = random.randint(STAR_COUNT_MIN, STAR_COUNT_MAX)
    
    k_size = PSF_KERNEL_SIZE
    sigma = 1.0
    k = cv2.getGaussianKernel(k_size, sigma)
    psf_kernel = k * k.T
    psf_kernel = psf_kernel / psf_kernel.max() * 255 
    
    for _ in range(num_stars):
        x, y = random.randint(0, IMG_W-k_size), random.randint(0, IMG_H-k_size)
        brightness = random.uniform(STAR_BRIGHTNESS_MIN, STAR_BRIGHTNESS_MAX) 
        
        star_stamp = (psf_kernel * brightness).astype(np.uint8)
        
        roi = star_layer[y:y+k_size, x:x+k_size]
        blended = cv2.add(roi, star_stamp)
        star_layer[y:y+k_size, x:x+k_size] = blended

    has_tracking_error = random.random() < TRACKING_ERROR_CHANCE
    if has_tracking_error:
        blur_len = random.randint(2, MAX_TRACKING_BLUR)
        angle = random.randint(0, 180)
        star_layer = OpticsSimulator.apply_motion_blur(star_layer, blur_len, angle)

    labels = []
    num_sats = random.randint(SAT_COUNT_MIN, SAT_COUNT_MAX)
    
    horizon_mask = None
    inv_horizon_mask = None
    
    if ENABLE_OBSTACLES and random.random() < OBSTACLE_CHANCE:
        horizon_mask = generate_procedural_horizon(IMG_H, IMG_W)
        inv_horizon_mask = cv2.bitwise_not(horizon_mask)

    for _ in range(num_sats):
        x1, y1 = random.randint(-50, IMG_W+50), random.randint(-50, IMG_H+50)
        x2, y2 = random.randint(-50, IMG_W+50), random.randint(-50, IMG_H+50)
        
        if np.hypot(x2-x1, y2-y1) < 50: continue

        brightness = random.randint(SAT_BRIGHTNESS_MIN, SAT_BRIGHTNESS_MAX)
        thickness = SAT_THICKNESS
        
        cv2.line(sat_layer, (x1, y1), (x2, y2), brightness, thickness, lineType=cv2.LINE_AA)
        
        single_sat_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        cv2.line(single_sat_mask, (x1, y1), (x2, y2), 255, thickness)
        
        if inv_horizon_mask is not None:
            single_sat_mask = cv2.bitwise_and(single_sat_mask, single_sat_mask, mask=inv_horizon_mask)
        
        visible_points = cv2.findNonZero(single_sat_mask)
        if visible_points is not None:
            bx, by, bw, bh = cv2.boundingRect(visible_points)
            if bw > 5 or bh > 5:
                cx, cy = (bx + bw / 2) / IMG_W, (by + bh / 2) / IMG_H
                nw, nh = bw / IMG_W, bh / IMG_H
                if 0 < cx < 1 and 0 < cy < 1:
                    labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    
    if SAT_DEFOCUS_KERNEL_SIZE > 0 and num_sats > 0:
        sat_layer = cv2.GaussianBlur(sat_layer, (SAT_DEFOCUS_KERNEL_SIZE, SAT_DEFOCUS_KERNEL_SIZE), 0)


    composed = cv2.add(base_sky, star_layer)
    composed = cv2.add(composed, sat_layer)
    
    if inv_horizon_mask is not None:
        composed = cv2.bitwise_and(composed, composed, mask=inv_horizon_mask)

    composed = add_vignetting(composed)
    composed = OpticsSimulator.add_poisson_noise(composed) 

    return composed, labels

if __name__ == "__main__":
    print(f"Generating {NUM_IMAGES} physics-based images...")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Tracking Error Chance: {TRACKING_ERROR_CHANCE * 100}%")
    
    for i in tqdm(range(NUM_IMAGES)):
        img, labels = generate_image()
        
        filename = f"amateur_phys_{i:04d}"
        
        cv2.imwrite(f"{OUTPUT_DIR}/images/{filename}.jpg", img)
        
        if labels:
            with open(f"{OUTPUT_DIR}/labels/{filename}.txt", "w") as f:
                for label in labels:
                    f.write(label + "\n")

    print(f"Done! Data saved in '{OUTPUT_DIR}'")
import rawpy
import cv2
import numpy as np
import os
import glob

TARGET_DIRECTORY = 'images_to_predict'

def convert_arw_to_yolo_input(raw_file_path):
    try:
        with rawpy.imread(raw_file_path) as raw:
            image_8bit_rgb = raw.postprocess()

            return image_8bit_rgb

    except Exception as e:
        print(f"Cannot process file {raw_file_path}. Error: {e}")
        return None

def main():
    print(f"Starts conversion: {TARGET_DIRECTORY}")
    
    if not os.path.isdir(TARGET_DIRECTORY):
        print(f"ERROR: Folder '{TARGET_DIRECTORY}' does not exist. Check the path.")
        return

    # 1. Find all .ARW files in the folder
    search_path = os.path.join(TARGET_DIRECTORY, '*.ARW')
    arw_files = glob.glob(search_path, recursive=False)
    
    if not arw_files:
        print("No .ARW files found in the folder.")
        return
        
    print(f"Found {len(arw_files)} .ARW files to convert.")
    
    converted_count = 0
    failed_count = 0
    
    for arw_path in arw_files:
        print(f"--- Processing: {arw_path}")
        
        # 2. Generate new file name (replace .ARW with .jpg)
        jpg_path = os.path.splitext(arw_path)[0] + '.jpg'
        
        # 3. Call conversion function
        processed_image_rgb = convert_arw_to_yolo_input(arw_path)
        
        if processed_image_rgb is not None:
            try:
                # 4. Save as .jpg
                # Convert from RGB (from rawpy) to BGR (for cv2.imwrite)
                image_bgr_for_saving = cv2.cvtColor(processed_image_rgb, cv2.COLOR_RGB2BGR)
                
                # Save with high quality (95%)
                cv2.imwrite(jpg_path, image_bgr_for_saving, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                print(f"Successfully created: {jpg_path}")
                converted_count += 1
            except Exception as e:
                print(f"ERROR: Could not save file {jpg_path}. Error: {e}")
                failed_count += 1
        else:
            print(f"ERROR: Conversion of file {arw_path} failed.")
            failed_count += 1
    
    print("\n--- Conversion finished ---")
    print(f"Successfully converted: {converted_count}")
    print(f"Failures: {failed_count}")

if __name__ == "__main__":
    main()
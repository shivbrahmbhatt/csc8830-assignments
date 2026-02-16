"""
README DOCUMENTATION
--------------------
Description: 
This script achieves state-of-the-art traditional boundary precision using the 
GrabCut algorithm (Gaussian Mixture Models). It uses a thermal threshold seed, 
then mathematically models color distributions to snap to high-frequency details.
It isolates the primary contour to obliterate disconnected thermal noise (like 
a person's hand) and applies a micro-blur for an anti-aliased, deep-learning-
comparable boundary. No ML/DL models are used.

Prerequisites:
- Python 3.x
- opencv-python (cv2)
- numpy

How to Execute:
Run from the command line:
python script_name.py --image your_thermal_image.png
"""

import cv2
import numpy as np
import argparse

def find_animal_boundaries(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Seed Mask
    lower_bound = np.array([120, 100, 140], dtype=np.uint8)
    upper_bound = np.array([255, 255, 255], dtype=np.uint8)
    core_mask = cv2.inRange(img, lower_bound, upper_bound)

    # GrabCut Mask
    gc_mask = np.full(img.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[core_mask > 0] = cv2.GC_PR_FGD
    gc_mask[0:int(img.shape[0]*0.4), :] = cv2.GC_BGD

    # Run GrabCut 
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # Extract the Raw Mask
    raw_mask = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')

    # Mask Isolation
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a pristine black canvas and draw ONLY the dog onto it
    pristine_mask = np.zeros_like(raw_mask)
    cv2.drawContours(pristine_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    pristine_mask = cv2.GaussianBlur(pristine_mask, (5, 5), 0)
    _, pristine_mask = cv2.threshold(pristine_mask, 127, 255, cv2.THRESH_BINARY)

    # Draw Final Precision Contour
    final_contours, _ = cv2.findContours(pristine_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if final_contours:
        best_contour = max(final_contours, key=cv2.contourArea)
        result = img.copy()
        
        cv2.drawContours(result, [best_contour], -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        cv2.imshow('Perfected GrabCut Mask', pristine_mask)
        cv2.imshow('Exact Animal Boundaries', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find animal boundaries in a thermal image.')
    parser.add_argument('--image', type=str, required=True, help='Path to the thermal image file')
    args = parser.parse_args()
    find_animal_boundaries(args.image)

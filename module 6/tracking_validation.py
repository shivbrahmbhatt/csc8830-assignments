"""
# CSc 8830: Computer Vision - Assignment 6
# Part B: Point Tracking Validation

## Description
This script validates theoretical Lucas-Kanade motion tracking against actual pixel locations.
It extracts two consecutive frames (60 and 61) from a video, detects trackable corners using 
the Shi-Tomasi method, and predicts their new sub-pixel coordinates. It then uses template 
matching to find the 'actual' ground truth location in the second frame and computes the 
Euclidean pixel error.

## Execution Instructions
Ensure your source videos are named 'video1.mp4' and 'video2.mp4' and are located in the 
same directory as this script. Run the script from the terminal:

python3 tracking_validation.py

The script will automatically process both videos and output a formatted validation table 
comparing Actual, Original, and Predicted (x,y) coordinates, along with the calculated error.

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
"""

import cv2
import numpy as np

def validate_tracking(video_path, frame_idx=60):
    cap = cv2.VideoCapture(video_path)
    
    # Extract consecutive frames 
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        print(f"Could not read frames from {video_path}")
        return

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 1. Feature Detection using Shi-Tomasi 
    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=20, qualityLevel=0.01, minDistance=30)
    
    if corners is None:
        return
        
    # 2. Prediction using Lucas-Kanade 
    p1_pred, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None, winSize=(15, 15), maxLevel=2)
    
    print(f"TRACKING VALIDATION - {video_path}")
    print(f"{'#':<4} {'Actual (x,y)':<18} {'Original (x,y)':<18} {'Predicted (x,y)':<18} {'Error (px)':<10}")
    
    errors = []
    
    for i, (p0_curr, p1_curr, status) in enumerate(zip(corners, p1_pred, st)):
        if not status:
            continue
            
        x0, y0 = p0_curr.ravel()
        x_pred, y_pred = p1_curr.ravel()
        
        # 3. Ground Truth Extraction via Template Matching 
        w = 7 # 15x15 template window
        x0_int, y0_int = int(round(x0)), int(round(y0))
        
        # Ensure we don't go out of bounds
        if y0_int-w < 0 or y0_int+w+1 > gray1.shape[0] or x0_int-w < 0 or x0_int+w+1 > gray1.shape[1]:
            continue
            
        template = gray1[y0_int-w:y0_int+w+1, x0_int-w:x0_int+w+1]
        
        search_w = 15
        y_start = max(0, y0_int - search_w)
        y_end = min(gray2.shape[0], y0_int + search_w + 1)
        x_start = max(0, x0_int - search_w)
        x_end = min(gray2.shape[1], x0_int + search_w + 1)
        
        search_area = gray2[y_start:y_end, x_start:x_end]
        
        if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
            continue
            
        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        
        x_actual = x_start + max_loc[0] + w
        y_actual = y_start + max_loc[1] + w
        
        # 4. Validation: Compute Euclidean error 
        error = np.sqrt((x_pred - x_actual)**2 + (y_pred - y_actual)**2)
        errors.append(error)
        
        print(f"{i:<4} ({x_actual:5.1f}, {y_actual:5.1f})      ({x0:5.1f}, {y0:5.1f})      ({x_pred:5.1f}, {y_pred:5.1f})      {error:.2f}")

    if errors:
        print(f"\nMean error: {np.mean(errors):.2f} px")
        print(f"Max error:  {np.max(errors):.2f} px")
        print(f"Min error:  {np.min(errors):.2f} px\n")

if __name__ == "__main__":
    validate_tracking('video1.mp4')
    validate_tracking('video2.mp4')

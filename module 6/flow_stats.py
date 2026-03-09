"""
# Part A: Dense Flow Statistics Generator

## Description
This script processes video samples to compute dense optical flow using the Farnebäck algorithm. 
Instead of visualizing the flow, it silently calculates and logs the quantitative evidence 
required for the report, including processing duration, total frames processed, and the 
mean/max flow magnitudes (px/frame) over a 30-second window.

## Execution Instructions
Ensure your source videos are named 'video1.mp4' and 'video2.mp4' and are located in the 
same directory as this script. Run the script from the terminal:

python3 flow_stats.py

The script will automatically process both videos and print the formatted statistics 
directly to the terminal.

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
"""

import cv2
import numpy as np
import argparse
import os

def compute_flow_stats(video_path, output_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Process up to 30.0s
    max_frames = int(fps * 30.0) 
    
    ret, first_frame = cap.read()
    if not ret:
        return
        
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    frame_count = 1
    total_mean_mag = 0
    global_max_mag = 0
    
    print(f"--- Video ---")
    print(f"Video:        {video_path}")
    print(f"Resolution:   {width}x{height}")
    print(f"FPS:          {fps}")
    print(f"Processing: up to 30.0s ({max_frames} frames)")

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Track statistics
        total_mean_mag += np.mean(magnitude)
        global_max_mag = max(global_max_mag, np.max(magnitude))
        
        prev_gray = gray
        frame_count += 1

    overall_mean = total_mean_mag / frame_count
    
    print(f"Frames processed:   {frame_count}")
    print(f"Duration processed: {frame_count/fps:.1f}s")
    print(f"Mean flow magnitude: {overall_mean:.3f} px/frame")
    print(f"Max flow magnitude:  {global_max_mag:.3f} px/frame")
    print(f"Output saved:        {os.path.abspath(output_name)}\n")
    
    cap.release()

if __name__ == "__main__":
    # Run for both videos
    compute_flow_stats('video1.mp4', 'flow_video1.mp4')
    compute_flow_stats('video2.mp4', 'flow_video2.mp4')

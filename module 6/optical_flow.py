"""
# Part A: Optical Flow Computation and Visualization 

## Description
This script computes and visualizes dense optical flow using the Farneback method. 
It processes a 30-second video sample to track and visualize motion between consecutive frames. The output displays both the original video and a color-coded visualization where hue represents the direction of motion and brightness represents the speed.

## Execution Instructions 
To run this script from the terminal, ensure you are in the same directory as the script and use the following command:

python3 optical_flow.py --video <path_to_video_file>

Example:
python3 optical_flow.py --video video1.mp4

Note: Press 'q' on your keyboard to safely exit the visualization windows during playback.

## Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
"""

import cv2
import numpy as np
import argparse

def compute_optical_flow(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        return
    
    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Create a mask image for drawing purposes (same size as frame)
    mask = np.zeros_like(first_frame)
    # Set image saturation to maximum
    mask[..., 1] = 255

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Dense Optical Flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Set image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Set image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        
        # Show the result
        cv2.imshow('Original Video', frame)
        cv2.imshow('Optical Flow Visualization', rgb)
        
        # Update previous frame
        prev_gray = gray
        
        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Optical Flow for a video.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    args = parser.parse_args()
    
    compute_optical_flow(args.video)

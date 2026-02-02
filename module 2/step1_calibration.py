import cv2
import numpy as np
import json
import time

# CONFIGURATION
CAMERA_INDEX = 2          
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 25          # Size in mm
MIN_CAPTURES = 15         # Goal number of images

# PREPARE OBJECT POINTS
# Set A: Landscape (9x6)
objp_landscape = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
objp_landscape[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp_landscape = objp_landscape * SQUARE_SIZE

# Set B: Portrait (6x9)
objp_portrait = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp_portrait[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)
objp_portrait = objp_portrait * SQUARE_SIZE

objpoints = [] 
imgpoints = [] 

# --- START CAMERA ---
cap = cv2.VideoCapture(CAMERA_INDEX)


if not cap.isOpened():
    print(f"ERROR: Could not open camera {CAMERA_INDEX}.")
    print("Try changing CAMERA_INDEX to 1 or 2.")
    exit()

print(f"REAL-TIME CALIBRATION")
print(f"1. Hold checkerboard in front of camera.")
print(f"2. Keep it steady. Script captures automatically every 2 seconds.")
print(f"3. Press 'q' to finish early.")

count_success = 0
last_capture_time = 0
cooldown = 2.0 # Seconds to wait between captures

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Clone frame for drawing text/lines
    display_frame = frame.copy()
    
    # 1. Try Landscape
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    pattern_type = "Landscape"
    
    # 2. Try Portrait if failed
    if not found:
        found, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[1], CHECKERBOARD[0]), 
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        pattern_type = "Portrait"

    # LOGIC: If found AND cooldown passed
    current_time = time.time()
    
    if found:
        # Draw corners (Visual feedback)
        cv2.drawChessboardCorners(display_frame, 
                                  (CHECKERBOARD[0] if pattern_type == "Landscape" else CHECKERBOARD[1], 
                                   CHECKERBOARD[1] if pattern_type == "Landscape" else CHECKERBOARD[0]), 
                                  corners, found)
        
        if (current_time - last_capture_time) > cooldown:
            # SAVE THIS CAPTURE
            if pattern_type == "Landscape":
                objpoints.append(objp_landscape)
            else:
                objpoints.append(objp_portrait)
                
            imgpoints.append(corners)
            
            count_success += 1
            last_capture_time = current_time
            print(f"Captured {count_success}/{MIN_CAPTURES} [{pattern_type}]")
            
            # Flash Green Screen effect
            cv2.rectangle(display_frame, (0,0), (frame.shape[1], frame.shape[0]), (0,255,0), 20)
        else:
            # Show "Wait" timer
            time_left = cooldown - (current_time - last_capture_time)
            cv2.putText(display_frame, f"Wait: {time_left:.1f}s", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # UI Text
    cv2.putText(display_frame, f"Collected: {count_success}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Iriun Calibration Feed', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if count_success >= MIN_CAPTURES:
        print("Target reached! Calibrating...")
        break

cap.release()
cv2.destroyAllWindows()

# CALIBRATION
if count_success < 5:
    print("Not enough data to calibrate.")
    exit()

print("Calculating Camera Matrix...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\nCamera Matrix (Intrinsics):\n", mtx)
print("\nDistortion Coefficients:\n", dist)

data = { "camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist() }
with open("camera_data.json", "w") as f:
    json.dump(data, f)
    print("\nSUCCESS: Calibration data saved to 'camera_data.json'")
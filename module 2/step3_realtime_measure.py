import cv2
import numpy as np
import json
from step2_measure import calculate_real_distance

# CONFIGURATION
CAMERA_INDEX = 2          
DISTANCE_FROM_OBJECT = 2400

# CALIBRATION
try:
    with open("camera_data.json", "r") as f:
        data = json.load(f)
        mtx = np.array(data["camera_matrix"])
        dist = np.array(data["dist_coeff"])
except FileNotFoundError:
    print("ERROR: camera_data.json not found. Run Step 1 first.")
    exit()

# Global variables for mouse clicks
points = []
frozen_frame = None
is_frozen = False

def mouse_callback(event, x, y, flags, param):
    global points, is_frozen
    
    # Only allow clicking if the screen is frozen
    if event == cv2.EVENT_LBUTTONDOWN and is_frozen:
        if len(points) < 2:
            points.append((x, y))
            print(f"Point selected: {x}, {y}")

# Setup Camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cv2.namedWindow("Measurement Tool")
cv2.setMouseCallback("Measurement Tool", mouse_callback)

print("--- CONTROLS ---")
print("1. Point camera at object.")
print(f"2. Ensure object is exactly {DISTANCE_FROM_OBJECT}mm away.")
print("3. Press 'SPACE' to freeze frame.")
print("4. Click two points to measure.")
print("5. Press 'r' to reset points.")
print("6. Press 'q' to quit.")

while True:
    if not is_frozen:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read camera.")
            break
        
        # Undistort the frame
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        current_display = undistorted_frame.copy()
        
        # UI Text
        cv2.putText(current_display, "Press SPACE to Freeze", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # We are frozen, show the frozen frame
        current_display = frozen_frame.copy()
        cv2.putText(current_display, "FROZEN - Click 2 Points", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # DRAWING LOGIC
    for pt in points:
        cv2.circle(current_display, pt, 5, (0, 255, 0), -1)

    if len(points) == 2:
        # Draw line between points
        cv2.line(current_display, points[0], points[1], (255, 0, 0), 2)
        
        # CALCULATE DISTANCE (Using Step 2 Logic)
        real_dist = calculate_real_distance(points[0], points[1], DISTANCE_FROM_OBJECT)
        
        # Display Result
        midpoint = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
        cv2.putText(current_display, f"{real_dist:.1f} mm", midpoint, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(current_display, "Press 'r' to reset", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Measurement Tool", current_display)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): # Quit
        break
    elif key == ord(' '): # Spacebar to Freeze/Unfreeze
        if not is_frozen:
            is_frozen = True
            frozen_frame = undistorted_frame.copy()
            points = [] # Reset points on new freeze
        else:
            is_frozen = False
            points = []
    elif key == ord('r'): # Reset points
        points = []

cap.release()
cv2.destroyAllWindows()
import numpy as np
import json

def calculate_real_distance(point1, point2, distance_z, camera_file="camera_data.json"):
    """
    Inputs:
    - point1, point2: (x, y) pixel coordinates (tuples)
    - distance_z: The real-world distance from camera to object (mm)
    - camera_file: Path to your calibration JSON
    
    Output:
    - Real world distance between point1 and point2 (in mm)
    """
    
    # Load the calibration data you created in Step 1
    with open(camera_file, "r") as f:
        data = json.load(f)
        mtx = np.array(data["camera_matrix"])
    
    # Extract Camera Intrinsics
    fx = mtx[0, 0] # Focal Length X
    fy = mtx[1, 1] # Focal Length Y
    cx = mtx[0, 2] # Optical Center X
    cy = mtx[1, 2] # Optical Center Y

    # Unpack points
    u1, v1 = point1
    u2, v2 = point2

    # Formula: X = (u - cx) * Z / fx
    # We calculate the real-world (X, Y) coordinates for both pixels
    
    x1_real = (u1 - cx) * distance_z / fx
    y1_real = (v1 - cy) * distance_z / fy

    x2_real = (u2 - cx) * distance_z / fx
    y2_real = (v2 - cy) * distance_z / fy

    # Calculate Euclidean distance in the real world
    real_distance = np.sqrt((x2_real - x1_real)**2 + (y2_real - y1_real)**2)
    
    return real_distance

if __name__ == "__main__":
    print("This script is a library. Run step3_realtime_measure.py instead.")
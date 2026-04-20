"""
Face Analysis & Blink Detection Application
Description:
This script processes laboratory video footage to perform two primary 
computer vision tasks using OpenCV and MediaPipe's Face Mesh:
  (A) Estimate the participant's eye blinking rate (blinks per second) 
      across the total duration of the provided video files.
  (B) Estimate the dimensions of specific facial features (Face height/width, 
      left eye width, nose width, mouth width) by calculating the Euclidean 
      distance between targeted 3D facial landmarks.

Dependencies:
To run this script, the following Python libraries are required:
  - opencv-python (cv2)
  - mediapipe
  - numpy

Setup & Execution (Mac / Apple Silicon environments):
Due to known compatibility issues with MediaPipe on Mac M-series chips, 
it is highly recommended to run this script inside a clean Virtual Environment.
  1. python3 -m venv venv
  2. source venv/bin/activate
  3. pip install opencv-python numpy mediapipe-silicon
  4. python face_analysis.py

Configuration Variables:
  - VIDEO_PATHS: List of strings containing the paths to the video files.
  - EAR_THRESHOLD: The Eye Aspect Ratio (EAR) value below which an eye is 
    considered closed.
  - CONSECUTIVE_FRAMES: Number of sequential frames the EAR must remain 
    below the threshold to register as a valid blink.

"""


import cv2
import mediapipe as mp
import math
import numpy as np

# Configuration
VIDEO_PATHS = [
    "GX011084.MP4", 
    "GX011085.MP4", 
    "GX011086.MP4", 
]

EAR_THRESHOLD = 0.22 
CONSECUTIVE_FRAMES = 2

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
FACE_TOP_BOTTOM = [10, 152] # Top of forehead, bottom of chin
FACE_LEFT_RIGHT = [234, 454] # Left edge, right edge (near ears)
NOSE = [168, 2, 198, 209] # Top, tip, left, right
MOUTH = [78, 308, 13, 14] # Left corner, right corner, top lip, bottom lip

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    coords = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    v1 = euclidean_distance(coords[1], coords[5])
    v2 = euclidean_distance(coords[2], coords[4])
    h = euclidean_distance(coords[0], coords[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

def get_distance(landmarks, idx1, idx2, img_w, img_h):
    p1 = (landmarks[idx1].x * img_w, landmarks[idx1].y * img_h)
    p2 = (landmarks[idx2].x * img_w, landmarks[idx2].y * img_h)
    return euclidean_distance(p1, p2)

def process_videos():
    total_blink_count = 0
    total_duration_seconds = 0.0
    
    # Store dimensions across ALL videos
    dimensions = {"face_height": [], "face_width": [], "nose_width": [], "mouth_width": [], "left_eye_width": []}

    print("Starting video processing. This may take a while for 4 hours of footage...")

    for i, video_path in enumerate(VIDEO_PATHS):
        print(f"Processing Video {i+1}/{len(VIDEO_PATHS)}: {video_path} ...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  -> Error: Could not open {video_path}. Please check the file path.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        total_duration_seconds += video_duration

        frames_closed = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break # Reached the end of this video

            img_h, img_w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    # Task A: Blink Detection
                    left_ear = calculate_ear(landmarks, LEFT_EYE, img_w, img_h)
                    right_ear = calculate_ear(landmarks, RIGHT_EYE, img_w, img_h)
                    avg_ear = (left_ear + right_ear) / 2.0

                    if avg_ear < EAR_THRESHOLD:
                        frames_closed += 1
                    else:
                        if frames_closed >= CONSECUTIVE_FRAMES:
                            total_blink_count += 1
                        frames_closed = 0

                    # Task B: Dimension Estimation (in pixels)
                    dimensions["face_height"].append(get_distance(landmarks, FACE_TOP_BOTTOM[0], FACE_TOP_BOTTOM[1], img_w, img_h))
                    dimensions["face_width"].append(get_distance(landmarks, FACE_LEFT_RIGHT[0], FACE_LEFT_RIGHT[1], img_w, img_h))
                    dimensions["nose_width"].append(get_distance(landmarks, NOSE[2], NOSE[3], img_w, img_h))
                    dimensions["mouth_width"].append(get_distance(landmarks, MOUTH[0], MOUTH[1], img_w, img_h))
                    dimensions["left_eye_width"].append(get_distance(landmarks, LEFT_EYE[0], LEFT_EYE[3], img_w, img_h))

        cap.release()
        print(f"  -> Finished {video_path}")

    # Print Final Aggregated Results
    if total_duration_seconds > 0:
        overall_blink_rate = total_blink_count / total_duration_seconds
        
        print(" (A) OVERALL BLINKING RATE ESTIMATION (All Videos)")
        print(f"Total Combined Duration: {total_duration_seconds:.2f} seconds ({total_duration_seconds/3600:.2f} hours)")
        print(f"Total Blinks Detected:   {total_blink_count}")
        print(f"Overall Blinking Rate:   {overall_blink_rate:.4f} blinks per second")

        print(" (B) AVERAGE FACIAL DIMENSIONS (All Videos)")
        print("Note: Values are in pixels. To convert to cm, a reference object is needed.")
        print(f"Face Height (Head to Chin):  {np.mean(dimensions['face_height']):.2f} px")
        print(f"Face Width (Ear to Ear):     {np.mean(dimensions['face_width']):.2f} px")
        print(f"Eye Width (Left Eye):        {np.mean(dimensions['left_eye_width']):.2f} px")
        print(f"Nose Width:                  {np.mean(dimensions['nose_width']):.2f} px")
        print(f"Mouth Width:                 {np.mean(dimensions['mouth_width']):.2f} px")
    else:
        print("\nNo videos were successfully processed. Please check your file paths.")

if __name__ == "__main__":
    process_videos()

"""
* PROBLEM 1: EYE BLINKING RATE PROTOTYPE (MOVIE VS. DOCUMENT)

Description:
This computer vision application tracks facial landmarks to determine a user's 
Eye Aspect Ratio (EAR). It records the number of blinks over a 60-second 
interval to compare blinking rates while watching a dynamic movie versus 
reading a static document.

Prerequisites:
- pip install opencv-python mediapipe

Instructions for Demo:
1. Run this script: `python3 blink_tracker.py`
2. Press 'S' to start a 60-second test.
3. Perform the task (Watch a movie / Read a document).
4. Record the final "Average Rate" (blinks per second) printed in the terminal.

* RESULTS
Test 1: Watching a Movie (1 minute video)
- Total Blinks: 26
- Average Rate: 0.43 blinks/second

Test 2: Reading a Document (1 page PDF/Word)
- Total Blinks: 21
- Average Rate: 0.35 blinks/second

"""

import cv2
import mediapipe as mp
import time
import math

# Config
EAR_THRESHOLD = 0.21  # If EAR drops below this, it's a blink
CONSECUTIVE_FRAMES = 2 # Number of consecutive frames EAR must be below threshold to count as a blink
TEST_DURATION = 60 # 1 minute data sample

# MediaPipe Eye Landmark Indices
# Order: Outer/Inner corner, Top 1, Top 2, Inner/Outer corner, Bottom 1, Bottom 2
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def calculate_ear(eye_indices, landmarks, img_w, img_h):
    # Get actual coordinates from normalized landmarks
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    
    # Calculate vertical distances
    v1 = euclidean_distance(pts[1], pts[5])
    v2 = euclidean_distance(pts[2], pts[4])
    
    # Calculate horizontal distance
    h = euclidean_distance(pts[0], pts[3])
    
    if h == 0: return 0.0
    
    # EAR Formula
    ear = (v1 + v2) / (2.0 * h)
    return ear

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(1) # Open webcam

    blink_count = 0
    frame_counter = 0
    
    print("Press 'S' to start the 1-minute recording. Press 'Q' to quit.")
    
    recording = False
    start_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Mirror image
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate EAR for both eyes
                left_ear = calculate_ear(LEFT_EYE, face_landmarks.landmark, w, h)
                right_ear = calculate_ear(RIGHT_EYE, face_landmarks.landmark, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if recording:
                    # Blink Detection Logic
                    if avg_ear < EAR_THRESHOLD:
                        frame_counter += 1
                    else:
                        if frame_counter >= CONSECUTIVE_FRAMES:
                            blink_count += 1
                        frame_counter = 0

                # Draw EAR on screen
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Handle UI and Timers
        if recording:
            elapsed_time = time.time() - start_time
            remaining_time = max(0, TEST_DURATION - elapsed_time)
            
            cv2.putText(frame, f"Time Left: {remaining_time:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate blinks per second
            if elapsed_time > 0:
                bps = blink_count / elapsed_time
                cv2.putText(frame, f"Rate: {bps:.2f} blinks/sec", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if elapsed_time >= TEST_DURATION:
                recording = False
                print(f"TEST COMPLETE")
                print(f"Total Blinks: {blink_count}")
                print(f"Average Rate: {blink_count / TEST_DURATION:.2f} blinks per second")

        else:
            cv2.putText(frame, "Press 'S' to Start 1-Min Test", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Blink Detector Prototype", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not recording:
            print("Started recording...")
            recording = True
            start_time = time.time()
            blink_count = 0
            frame_counter = 0
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

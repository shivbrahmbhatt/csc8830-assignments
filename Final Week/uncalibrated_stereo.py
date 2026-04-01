import cv2
import numpy as np

# Load the images
img1 = cv2.imread('left.jpg')
img2 = cv2.imread('right.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Physical measurements (in cm)
baseline = 10.1 
ground_truth_depth = 50.8

# Approximate Intrinsic Matrix (K)
h, w = gray1.shape
focal_length = 3000.0 
center_x, center_y = w / 2, h / 2
K = np.array([[focal_length, 0, center_x],
              [0, focal_length, center_y],
              [0, 0, 1]], dtype=np.float64)

# Feature Detection and Matching (SIFT)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# FLANN based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Stricter Lowe's Ratio Test (Changed from 0.8 to 0.65)
good_matches = []
pts1 = []
pts2 = []
for m, n in matches:
    if m.distance < 0.65 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

if len(pts1) < 8:
    print("Error: Not enough good matches found. Try lowering the Lowe's ratio test value.")
    exit()

# Compute Fundamental Matrix (F) with stricter RANSAC threshold
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]
good_matches = [good_matches[i] for i in range(len(good_matches)) if mask.ravel()[i] == 1]

# Compute Essential Matrix (E)
E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Recover Pose (Rotation R and Translation t)
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

# Estimate Distance via Triangulation
P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(K, np.hstack((R, t)))
points_4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

# Safeguard against division by zero 
# Find valid points where the 4th coordinate is not zero
valid_idx = points_4D[3] != 0
points_3D = np.zeros((3, points_4D.shape[1]))
points_3D[:, valid_idx] = points_4D[:3, valid_idx] / points_4D[3, valid_idx]

# Calculate scale
scale = baseline / np.linalg.norm(t)

# Filter out negative depths (points behind camera)
depths = points_3D[2]
valid_depths = depths[depths > 0] * scale

if len(valid_depths) > 0:
    estimated_depth = np.median(valid_depths)
else:
    estimated_depth = float('nan')
    print("Warning: All triangulated points were invalid or behind the camera.")

# Print outputs
print("Matrices for Report")
print("Fundamental Matrix (F):\n", F)
print("\nEssential Matrix (E):\n", E)
print("\nRotation Matrix (R):\n", R)
print(f"\nEstimated Depth: {estimated_depth:.2f} cm")
print(f"Ground Truth Depth: {ground_truth_depth} cm")

# Annotate Image
annotated_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
text1 = f"Estimated Dist: {estimated_depth:.2f} cm"
text2 = f"Ground Truth: {ground_truth_depth} cm"
cv2.putText(annotated_img, text1, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
cv2.putText(annotated_img, text2, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

cv2.imwrite('annotated_setup.jpg', annotated_img)
print("\nSaved annotated image to 'annotated_setup.jpg'")

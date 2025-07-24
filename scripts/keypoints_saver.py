import json
import cv2
import numpy as np

# === Paths ===
image_path = "/home/stud220/git/textured_smplx/data/obj2/images/00.png"          # Replace with your image path
json_path = "/home/stud220/git/textured_smplx/data/obj2/keypoints/00.json"           # Replace with your JSON path
output_path = "/home/stud220/git/ImageTo3DSegmentedClothes/scripts/output/output_pose.jpg"        # Path to save the result

# === Load image ===
image = cv2.imread(image_path)

# === Load keypoints ===
with open(json_path, "r") as f:
    data = json.load(f)

keypoints = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)  # shape: (N, 3)

# === Draw keypoints and skeleton ===
CONF_THRESH = 0.3

# OpenPose COCO keypoint skeleton connections
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4),
    (5, 6), (6, 7), (1, 8), (8, 9),
    (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 1), (0, 14), (14, 16), (0, 15), (15, 17)
]

# Draw keypoints
for x, y, conf in keypoints:
    if conf > CONF_THRESH:
        cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)

# Draw skeleton
# for a, b in POSE_PAIRS:
#     if keypoints[a][2] > CONF_THRESH and keypoints[b][2] > CONF_THRESH:
#         pt1 = tuple(int(v) for v in keypoints[a][:2])
#         pt2 = tuple(int(v) for v in keypoints[b][:2])
#         cv2.line(image, pt1, pt2, (255, 0, 0), 2)

# === Save output ===
cv2.imwrite(output_path, image)
print(f"Saved pose image to: {output_path}")

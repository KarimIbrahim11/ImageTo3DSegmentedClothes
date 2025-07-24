import argparse
import mediapipe as mp
import cv2
import json
import os

# MediaPipe to OpenPose keypoint mapping
# MediaPipe has 33 keypoints, OpenPose has 25 body keypoints
# This mapping converts MediaPipe indices to OpenPose indices
MEDIAPIPE_TO_OPENPOSE_MAPPING = {
    # OpenPose index: MediaPipe index
    0: 0,   # Nose
    1: 2,   # Neck (approximate using right eye)
    2: 5,   # Right Shoulder
    3: 6,   # Right Elbow  
    4: 8,   # Right Wrist
    5: 1,   # Left Shoulder (using left eye as approximation)
    6: 3,   # Left Elbow (using left ear as approximation)
    7: 7,   # Left Wrist (using right ear)
    8: 12,  # Mid Hip (using right hip)
    9: 11,  # Right Hip 
    10: 13, # Right Knee
    11: 15, # Right Ankle
    12: 12, # Left Hip
    13: 14, # Left Knee  
    14: 16, # Left Ankle
    15: 1,  # Right Eye (using left eye inner)
    16: 4,  # Left Eye (using left eye outer) 
    17: 7,  # Right Ear
    18: 8,  # Left Ear
    19: 19, # Left Big Toe
    20: 20, # Left Small Toe
    21: 21, # Left Heel
    22: 22, # Right Big Toe
    23: 23, # Right Small Toe  
    24: 24, # Right Heel
}

# Better mapping based on actual MediaPipe pose landmarks
MEDIAPIPE_POSE_MAPPING = {
    0: 0,   # Nose -> Nose
    1: None, # Neck (we'll calculate this)
    2: 12,  # Right Shoulder -> Right Shoulder  
    3: 14,  # Right Elbow -> Right Elbow
    4: 16,  # Right Wrist -> Right Wrist
    5: 11,  # Left Shoulder -> Left Shoulder
    6: 13,  # Left Elbow -> Left Elbow
    7: 15,  # Left Wrist -> Left Wrist
    8: None, # Mid Hip (we'll calculate this)
    9: 24,  # Right Hip -> Right Hip
    10: 26, # Right Knee -> Right Knee
    11: 28, # Right Ankle -> Right Ankle
    12: 23, # Left Hip -> Left Hip
    13: 25, # Left Knee -> Left Knee
    14: 27, # Left Ankle -> Left Ankle
    15: 2,  # Right Eye -> Right Eye
    16: 5,  # Left Eye -> Left Eye
    17: 8,  # Right Ear -> Right Ear
    18: 7,  # Left Ear -> Left Ear
    19: 31, # Left Big Toe -> Left Foot Index
    20: 29, # Left Small Toe -> Left Heel
    21: 29, # Left Heel -> Left Heel
    22: 32, # Right Big Toe -> Right Foot Index
    23: 30, # Right Small Toe -> Right Heel
    24: 30, # Right Heel -> Right Heel
}

def convert_mediapipe_to_openpose(keypoints):
    """Convert MediaPipe keypoints to OpenPose format"""
    
    openpose_keypoints = []
    
    # Create array for 25 OpenPose keypoints * 3 values each
    pose_keypoints_2d = [0.0] * 75
    
    for openpose_idx in range(25):
        mediapipe_idx = MEDIAPIPE_POSE_MAPPING.get(openpose_idx)
        
        if mediapipe_idx is None:
            # Special cases: calculate neck and mid hip
            if openpose_idx == 1:  # Neck
                # Calculate neck as midpoint between shoulders
                if len(keypoints) > 12 and len(keypoints) > 11:
                    left_shoulder = keypoints[11]   # MediaPipe left shoulder
                    right_shoulder = keypoints[12]  # MediaPipe right shoulder
                    x = (left_shoulder["x"] + right_shoulder["x"]) / 2
                    y = (left_shoulder["y"] + right_shoulder["y"]) / 2
                    confidence = min(left_shoulder["visibility"], right_shoulder["visibility"])
                else:
                    x, y, confidence = 0, 0, 0
            elif openpose_idx == 8:  # Mid Hip
                # Calculate mid hip as midpoint between hips
                if len(keypoints) > 24 and len(keypoints) > 23:
                    left_hip = keypoints[23]   # MediaPipe left hip
                    right_hip = keypoints[24]  # MediaPipe right hip
                    x = (left_hip["x"] + right_hip["x"]) / 2
                    y = (left_hip["y"] + right_hip["y"]) / 2
                    confidence = min(left_hip["visibility"], right_hip["visibility"])
                else:
                    x, y, confidence = 0, 0, 0
            else:
                x, y, confidence = 0, 0, 0
        else:
            # Direct mapping
            if mediapipe_idx < len(keypoints):
                kpt = keypoints[mediapipe_idx]
                x = kpt["x"]
                y = kpt["y"] 
                confidence = kpt["visibility"]
            else:
                x, y, confidence = 0, 0, 0
        
        # Add to OpenPose format array
        idx = openpose_idx * 3
        pose_keypoints_2d[idx] = x
        pose_keypoints_2d[idx + 1] = y
        pose_keypoints_2d[idx + 2] = confidence
    
    return pose_keypoints_2d

def extract_pose_mediapipe_openpose_format(image_dir, output_dir):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    print(f"Processing images from: {image_dir}")
    print(f"Saving keypoints to: {output_dir}")
    
    processed_count = 0
    
    for image_file in sorted(os.listdir(image_dir)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_file)
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read {image_file}")
                    continue
                    
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_image)
                
                # Create OpenPose format structure
                openpose_data = {
                    "version": 1.3,
                    "people": []
                }
                
                if results.pose_landmarks:
                    # Convert MediaPipe landmarks to our format first
                    keypoints = []
                    for landmark in results.pose_landmarks.landmark:
                        keypoints.append({
                            "x": landmark.x * image.shape[1],
                            "y": landmark.y * image.shape[0],
                            "visibility": landmark.visibility
                        })
                    
                    # Convert to OpenPose format
                    pose_keypoints_2d = convert_mediapipe_to_openpose(keypoints)
                    
                    person_data = {
                        "person_id": [-1],
                        "pose_keypoints_2d": pose_keypoints_2d,
                        "face_keypoints_2d": [],
                        "hand_left_keypoints_2d": [],
                        "hand_right_keypoints_2d": [],
                        "pose_keypoints_3d": [],
                        "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [],
                        "hand_right_keypoints_3d": []
                    }
                    
                    openpose_data["people"].append(person_data)
                
                # Save JSON file
                json_filename = os.path.splitext(image_file)[0] + '.json'
                json_path = os.path.join(output_dir, json_filename)
                
                with open(json_path, 'w') as f:
                    json.dump(openpose_data, f, separators=(',', ':'))
                
                processed_count += 1
                print(f"Processed: {image_file} -> {json_filename}")
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
    
    pose.close()
    print(f"\n✅ Successfully processed {processed_count} images")
    print(f"OpenPose format JSON files saved to: {output_dir}")

# Configuration
input_dir = '/home/stud220/git/textured_smplx/data/obj2/images'
keypoints_dir = '/home/stud220/git/textured_smplx/data/obj2/keypoints'
pose_images_dir = '/home/stud220/git/textured_smplx/data/obj2/pose_images'

os.makedirs(keypoints_dir, exist_ok=True)
os.makedirs(pose_images_dir, exist_ok=True)

# Run the extraction
extract_pose_mediapipe_openpose_format(input_dir, keypoints_dir)
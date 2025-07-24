import cv2
import numpy as np

def estimate_camera_pose_from_two_images(img1, img2, camera_matrix):
    """
    Estimate relative camera pose between two images
    
    Args:
        img1, img2: Input images (grayscale or color)
        camera_matrix: Camera intrinsic matrix (3x3)
    
    Returns:
        rmat: Rotation matrix (3x3)
        tvec: Translation vector (3,)
        matches: Number of good matches found
    """
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # Method 1: Using ORB features (works well in most cases)
    orb = cv2.ORB_create(nfeatures=5000)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        print("Warning: Not enough features detected")
        return np.eye(3), np.array([0., 0., 0.]), 0
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 8:
        print("Warning: Not enough matches for pose estimation")
        return np.eye(3), np.array([0., 0., 0.]), len(matches)
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find essential matrix
    essential_mat, mask = cv2.findEssentialMat(
        pts1, pts2, camera_matrix, 
        method=cv2.RANSAC, 
        prob=0.999, 
        threshold=1.0
    )
    
    if essential_mat is None:
        print("Warning: Could not find essential matrix")
        return np.eye(3), np.array([0., 0., 0.]), len(matches)
    
    # Recover pose from essential matrix
    _, rmat, tvec, mask_pose = cv2.recoverPose(
        essential_mat, pts1, pts2, camera_matrix
    )
    
    good_matches = np.sum(mask_pose > 0)
    print(f"Found {len(matches)} matches, {good_matches} used for pose estimation")
    
    return rmat, tvec.flatten(), good_matches


def estimate_pose_with_sift(img1, img2, camera_matrix):
    """
    Alternative using SIFT features (better for textured scenes)
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return np.eye(3), np.array([0., 0., 0.]), 0
    
    # FLANN matcher for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 8:
        return np.eye(3), np.array([0., 0., 0.]), len(good_matches)
    
    # Extract points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find essential matrix and recover pose
    essential_mat, _ = cv2.findEssentialMat(pts1, pts2, camera_matrix)
    if essential_mat is None:
        return np.eye(3), np.array([0., 0., 0.]), len(good_matches)
    
    _, rmat, tvec, _ = cv2.recoverPose(essential_mat, pts1, pts2, camera_matrix)
    
    return rmat, tvec.flatten(), len(good_matches)


def create_camera_matrix(fx, fy, cx, cy):
    """
    Create camera intrinsic matrix
    
    Args:
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point coordinates
    """
    
    # fx = fy = image_width  # Rough estimate for "normal" field of view
    # cx = image_width / 2   # Principal point at image center
    # cy = image_height / 2
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)


import pickle

def load_and_estimate_camera_pose(pkl_file_path, img1_path, img2_path, 
                                 fx=512, fy=512, save_back=True):
    """
    Load data from pickle file, estimate camera pose, and optionally save back
    
    Args:
        pkl_file_path: Path to pickle file
        img1_path, img2_path: Paths to the two images
        fx, fy: Camera focal lengths (adjust based on your camera)
        save_back: Whether to save the estimated pose back to pickle file
    """
    
    # Load existing data from pickle file
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded data from {pkl_file_path}")
    except FileNotFoundError:
        print(f"Creating new data file: {pkl_file_path}")
        data = {}
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    print("SHAPE: ", img2.shape)
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        return data
    
    # Set up camera matrix
    cx, cy = img1.shape[1]//2, img1.shape[0]//2  # principal point (image center)
    camera_matrix = create_camera_matrix(fx, fy, cx, cy)
    
    # Check if pose already exists
    if 'camera_rotation' in data and 'camera_translation' in data:
        print("Camera pose already exists in data:")
        rmat_existing = np.array(data['camera_rotation']).reshape(3,3)
        tvec_existing = np.array(data['camera_translation'])
        print("Existing rotation matrix:")
        print(rmat_existing)
        print("Existing translation vector:")
        print(tvec_existing)
        
        overwrite = input("Overwrite existing pose? (y/n): ").lower() == 'y'
        if not overwrite:
            return data
    
    # Estimate new pose
    print("Estimating camera pose from images...")
    rmat, tvec, num_matches = estimate_pose_with_sift(
        img1, img2, camera_matrix
    )
    
    if num_matches < 8:
        print("Warning: Too few matches for reliable pose estimation")
        return data
    
    # Save to data dictionary
    data['camera_rotation'] = rmat.flatten().tolist()  # Flatten 3x3 to list of 9 elements
    data['camera_translation'] = tvec.tolist()
    
    # Also save some metadata
    data['pose_estimation_info'] = {
        'num_matches': int(num_matches),
        'images_used': [img1_path, img2_path],
        'camera_matrix': camera_matrix.tolist(),
        'estimation_method': 'ORB_features'
    }
    
    print("Estimated rotation matrix:")
    print(rmat)
    print("\nEstimated translation vector:")
    print(tvec)
    print(f"Number of matches used: {num_matches}")
    
    # Save back to pickle file
    if save_back:
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nSaved updated data to {pkl_file_path}")
    
    return data


def load_camera_pose_from_data(data):
    """
    Load camera pose from data dictionary (your original code pattern)
    """
    if 'camera_rotation' in data:
        rmat = np.array(data['camera_rotation']).reshape(3,3)
        print("Loaded camera rotation matrix:")
        print(rmat)
    else:
        print("Warning: camera_rotation not found, using identity matrix")
        rmat = np.eye(3)
    
    if 'camera_translation' in data:
        tvec = np.array(data['camera_translation'])
        print("Loaded camera translation vector:")
        print(tvec)
    else:
        print("Warning: camera_translation not found, using zeros")
        tvec = np.array([0., 0., 0.])
    
    return rmat, tvec


# Example usage
if __name__ == "__main__":
    
    fx, fy = 5000, 5000
    
    pkl_path = "/home/stud220/git/textured_smplx/data/obj2/smpl/results/00/000.pkl"

    
    # Method 1: Estimate and save pose
    data = load_and_estimate_camera_pose(
        pkl_file_path=pkl_path,
        img1_path="/home/stud220/git/textured_smplx/data/obj2/images/00.png",
        img2_path="/home/stud220/git/textured_smplx/data/obj2/images/01.png",
        fx=fx, fy=fy,  # Adjust these based on your camera
        save_back=True
    )
    
    print("\n" + "="*50)
    print("Testing the loading (your original pattern):")
    
    # Method 2: Load pose from saved data (your original pattern)
    rmat, tvec = load_camera_pose_from_data(data)
    
    
    pkl_path = "/home/stud220/git/textured_smplx/data/obj2/smpl/results/01/000.pkl"
    
    # Method 1: Estimate and save pose
    data = load_and_estimate_camera_pose(
        pkl_file_path=pkl_path,
        img1_path="/home/stud220/git/textured_smplx/data/obj2/images/00.png",
        img2_path="/home/stud220/git/textured_smplx/data/obj2/images/01.png",
        fx=fx, fy=fy,  # Adjust these based on your camera
        save_back=True
    )
    
    print("\n" + "="*50)
    print("Testing the loading (your original pattern):")
    
    # Method 2: Load pose from saved data (your original pattern)
    rmat, tvec = load_camera_pose_from_data(data)
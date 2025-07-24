# Replacement for the problematic section in textured_smplx.py
# Replace lines around line 56 where the error occurs

import numpy as np
import pickle
import cv2


def get_texture_SMPL(f_pkl):
    """Modified version with default camera parameters"""
    
    # Load pickle data
    with open(f_pkl, 'rb') as f:
        data = pickle.load(f)
    
    # Handle missing camera parameters with defaults
    if 'camera_rotation' in data:
        rmat = np.array(data['camera_rotation']).reshape(3,3)
    else:
        print("Warning: camera_rotation not found, using identity matrix")
        rmat = np.eye(3)
    
    if 'camera_translation' in data:
        tvec = np.array(data['camera_translation'])
    else:
        print("Warning: camera_translation not found, using zeros")
        tvec = np.array([0., 0., 0.])
    
    if 'camera_center' in data:
        center = np.array(data['camera_center'])
    else:
        print("Warning: camera_center not found, using default [256, 256]")
        center = np.array([256., 256.])
    
    if 'camera_focal_length' in data:
        focal = np.array(data['camera_focal_length'])
    else:
        print("Warning: camera_focal_length not found, using default [1000, 1000]")
        focal = np.array([1000., 1000.])
    
    # Continue with the rest of the original function...
    # (Replace the original lines that caused the error)
    
get_texture_SMPL("/home/stud220/git/textured_smplx/data/obj2/smpl/results/00/000.pkl")
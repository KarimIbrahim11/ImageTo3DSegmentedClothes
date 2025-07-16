import numpy as np
import pickle

# Load SMPLH model in .npz format
npz_path = '/home/stud220/git/ImageTo3DSegmentedClothes/NICP/support_data/body_models/smplh/neutral/model.npz'
data = np.load(npz_path)

# Optional: convert to plain dict
model_data = {key: data[key] for key in data.files}

# Save as .pkl
pkl_path = '/home/stud220/git/ImageTo3DSegmentedClothes/smplx/models/smplh/SMPLH_NEUTRAL.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(model_data, f)

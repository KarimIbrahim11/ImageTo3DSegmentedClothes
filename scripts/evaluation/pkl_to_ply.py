import pickle
import numpy as np
import trimesh
import torch
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)
    
    
# with open("/home/stud220/git/ImageTo3DSegmentedClothes/output/Close-output/outputs.pkl", "rb") as f:
#     smpl_data = CPU_Unpickler(f).load()
    
    
# # # Load the .pkl
# # with open("", "rb") as f:
# #     obj = pickle.load(f)

# # Assume dict with 'vertices', 'faces', and optional 'vertex_colors'
# mesh = trimesh.Trimesh(vertices=obj['vertices'],
#                        faces=obj['faces'],
#                        vertex_colors=obj.get('vertex_colors', None))

# # Export to .ply
# mesh.export("/home/stud220/git/ImageTo3DSegmentedClothes/output/Close-output/outputs.ply")

with open("/home/stud220/git/ImageTo3DSegmentedClothes/output/Close-output/color_out.pkl", "rb") as f:
    obj = CPU_Unpickler(f).load()
    

# Assume dict with 'vertices', 'faces', and optional 'vertex_colors'
mesh = trimesh.Trimesh(vertices=obj['vertices'],
                       faces=obj['faces'],
                       vertex_colors=obj.get('vertex_colors', None))

# Export to .ply
mesh.export("/home/stud220/git/ImageTo3DSegmentedClothes/output/Close-output/color_out.ply")
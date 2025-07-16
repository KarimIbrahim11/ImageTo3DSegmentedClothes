import trimesh
import numpy as np

colored_mesh = trimesh.load('/home/stud220/git/ImageTo3DSegmentedClothes/NICP/demo/test_scan_044/test_scan_044.ply')

# Load your meshes
smpl_mesh = trimesh.load('/home/stud220/git/ImageTo3DSegmentedClothes/NICP/output/1ljjfnbx/demo/test_scan_044/out_ss_cham_0.ply')          # no color, but with UVs
# colored_mesh = trimesh.load('colored_scan.obj')    # colored mesh

# Step 1: Transfer color by nearest neighbor (approximate)
from scipy.spatial import cKDTree
colored_kdtree = cKDTree(colored_mesh.vertices)
dists, idx = colored_kdtree.query(smpl_mesh.vertices)
vertex_colors = colored_mesh.visual.vertex_colors[idx]

# Step 2: Assign these colors to smpl mesh vertices
smpl_mesh.visual.vertex_colors = vertex_colors

# Step 3: Bake vertex colors to texture image using UVs
# --> This part is complex; you can do it in Blender or use PyTorch3D

# If you want me to help with Blender scripts or PyTorch3D code, just ask!

from plyfile import PlyData
import trimesh
import numpy as np

# SH constant
C0 = 0.28209479177387814

# Conversion functions
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

# Load the PLY file using plyfile
input_path = "/home/stud220/git/ImageTo3DSegmentedClothes/Human3Diffusion/output/02/"
ply = PlyData.read(input_path+'gs.ply')

# Extract vertex data
vertex_data = ply['vertex'].data

# Convert structured array to dict of arrays
vertex_dict = {name: vertex_data[name] for name in vertex_data.dtype.names}

# Stack into (N, 3) numpy array for vertices
vertices = np.vstack([vertex_dict['x'], vertex_dict['y'], vertex_dict['z']]).T

# Extract SH coefficients and convert to RGB
sh_coeffs = np.vstack([
    vertex_dict.get('f_dc_0', np.zeros(len(vertices))),
    vertex_dict.get('f_dc_1', np.zeros(len(vertices))),
    vertex_dict.get('f_dc_2', np.zeros(len(vertices)))
]).T

# Convert to RGB
colors_rgb = SH2RGB(sh_coeffs)
colors_rgb = np.clip(colors_rgb, 0.0, 1.0)  # Clamp to valid RGB range

# Create PointCloud with RGB colors
mesh = trimesh.points.PointCloud(vertices, colors=colors_rgb)

# Export as PLY using trimesh
mesh.export(input_path+'output_rgb.ply')

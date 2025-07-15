import open3d as o3d
import numpy as np
import trimesh

# --- Configurable flag ---
REMOVE_COLORS = False  # Set to False if you want to keep vertex colors


# Load original mesh
# mesh = o3d.io.read_triangle_mesh("/home/stud220/git/ImageTo3DSegmentedClothes/Human3Diffusion/output/05/tsdf-rgbd.ply")


# # --- Optionally remove vertex colors ---
# if REMOVE_COLORS and len(mesh.vertex_colors) > 0:
#     mesh.vertex_colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
#     print("Vertex colors removed.")
# else:
#     print("Vertex colors kept or not present.")

# mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float32))

# # Save in desired format (overwriting header)
# o3d.io.write_triangle_mesh("/home/stud220/git/ImageTo3DSegmentedClothes/Human3Diffusion/output/05/nicp-compatible-tsdf-rgbd.ply", mesh, write_ascii=False)


# Load with Open3D
mesh = o3d.io.read_triangle_mesh("/home/stud220/git/ImageTo3DSegmentedClothes/Human3Diffusion/output/05/tsdf-rgbd.ply")

if REMOVE_COLORS and len(mesh.vertex_colors) > 0:
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))

# Convert to float32
vertices = np.asarray(mesh.vertices, dtype=np.float32)
faces = np.asarray(mesh.triangles, dtype=np.int32)

# Create trimesh object (ensures float + correct face type)
tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

# Export with correct header
tmesh.export("/home/stud220/git/ImageTo3DSegmentedClothes/Human3Diffusion/output/05/nicp-compatible-colored-tsdf-rgbd.ply", file_type='ply')
print("Converted mesh written with float32 vertices and int face indices.")



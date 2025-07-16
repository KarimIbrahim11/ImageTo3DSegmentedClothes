import trimesh

# Load your PLY file
mesh = trimesh.load('/home/stud220/git/ImageTo3DSegmentedClothes/inputs/colored_mesh.ply')

# Export to OBJ
mesh.export('/home/stud220/git/ImageTo3DSegmentedClothes/inputs/colored_mesh.obj')

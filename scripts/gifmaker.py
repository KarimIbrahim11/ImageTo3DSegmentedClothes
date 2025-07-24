import open3d as o3d
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# Force matplotlib to use non-interactive backend
plt.switch_backend('Agg')

# === Settings ===
input_paths = [
    "/home/stud220/git/ImageTo3DSegmentedClothes/output/SMPLH_meshes/tsdf-rgbd/aligned.ply",
    "/home/stud220/git/ImageTo3DSegmentedClothes/output/SMPLH_meshes/moved-meshes/out_ss.ply",
    "/home/stud220/git/ImageTo3DSegmentedClothes/output/Close-output/color_out.ply"
]
output_dirs = [
    "/home/stud220/git/ImageTo3DSegmentedClothes/output/gifs/mesh",
    "/home/stud220/git/ImageTo3DSegmentedClothes/output/gifs/smpl",
    "/home/stud220/git/ImageTo3DSegmentedClothes/output/gifs/segmented"
]
gif_names = ["mesh_spinning.gif", "smpl_spinning.gif", "segmented_spinning.gif"]

num_frames = 50
rotation_axis = [0, 1, 0]  # Y-axis rotation (horizontal spin) - better for human figures
degrees_per_frame = 360 / num_frames
img_size = (512, 512)

for mesh_idx, input_path in enumerate(input_paths):
    print(f"\n=== Processing mesh {mesh_idx + 1}/3: {input_path.split('/')[-1]} ===")
    
    # === Load mesh using Open3D (just for loading, no visualization) ===
    print("Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    if len(mesh.vertices) == 0:
        print(f"❌ Could not load mesh from {input_path}")
        continue
        
    mesh.compute_vertex_normals()

    # Extract vertices and faces
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_colors = None

    if mesh.has_vertex_colors():
        print("Mesh has vertex colors.")
        vertex_colors = np.asarray(mesh.vertex_colors)
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0

    print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")

    # === Create output directory ===
    os.makedirs(output_dirs[mesh_idx], exist_ok=True)
    images = []

    # Calculate mesh center and bounds for consistent camera positioning
    center = np.mean(vertices, axis=0)
    bounds_min = np.min(vertices, axis=0)
    bounds_max = np.max(vertices, axis=0)
    max_range = np.max(bounds_max - bounds_min) / 2.0

    print("Rendering frames...")
    for frame_idx in range(num_frames):
        print(f"Frame {frame_idx+1}/{num_frames}")
        
        # Calculate rotation angle
        angle = np.deg2rad(degrees_per_frame * frame_idx)
        
        # Create rotation matrix around specified axis
        if rotation_axis == [0, 1, 0]:  # Y-axis (vertical, spinning horizontally)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
        elif rotation_axis == [1, 0, 0]:  # X-axis (tumbling forward/backward)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
        else:  # Z-axis (rolling left/right)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
        
        # Apply rotation to vertices
        rotated_vertices = np.dot(vertices - center, R.T) + center
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(img_size[0]/100, img_size[1]/100), dpi=100, facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        # Render as point cloud with colors if available
        if vertex_colors is not None:
            # Sample vertices to avoid too many points (for performance)
            if len(vertices) > 10000:
                indices = np.random.choice(len(vertices), 10000, replace=False)
                plot_vertices = rotated_vertices[indices]
                plot_colors = vertex_colors[indices]
            else:
                plot_vertices = rotated_vertices
                plot_colors = vertex_colors
                
            ax.scatter(plot_vertices[:, 0], plot_vertices[:, 1], plot_vertices[:, 2], 
                      c=plot_colors, s=1.0, alpha=0.8)
        else:
            # Sample vertices for performance
            if len(vertices) > 10000:
                indices = np.random.choice(len(vertices), 10000, replace=False)
                plot_vertices = rotated_vertices[indices]
            else:
                plot_vertices = rotated_vertices
                
            ax.scatter(plot_vertices[:, 0], plot_vertices[:, 1], plot_vertices[:, 2], 
                      c='lightblue', s=1.0, alpha=0.8)
        
        # Set equal aspect ratio and consistent view
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        # Remove axes for cleaner look
        ax.set_axis_off()
        
        # Set viewing angle for frontal view
        # For human figures, these angles typically work well:
        ax.view_init(elev=0, azim=0)  # Frontal view: elev=0 (horizontal), azim=0 (front)
        
        # Alternative angles you can try:
        # ax.view_init(elev=10, azim=0)   # Slightly from above
        # ax.view_init(elev=-10, azim=0)  # Slightly from below
        # ax.view_init(elev=0, azim=45)   # 45-degree angle
        
        # Set background to white and remove grid
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make pane edges invisible
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        
        # Remove grid lines
        ax.grid(False)
        
        # Save frame
        img_path = os.path.join(output_dirs[mesh_idx], f"frame_{frame_idx:03d}.png")
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=100, facecolor='white')
        plt.close()
        
        # Read the saved image
        img_array = imageio.imread(img_path)
        images.append(img_array)

    # === Create GIF ===
    if len(images) > 0:
        print("Creating GIF...")
        full_gif_path = os.path.join(output_dirs[mesh_idx], gif_names[mesh_idx])
        imageio.mimsave(full_gif_path, images, duration=0.05, loop=0)
        print(f"✅ GIF saved as: {full_gif_path}")
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for frame_idx in range(len(images)):
            img_path = os.path.join(output_dirs[mesh_idx], f"frame_{frame_idx:03d}.png")
            if os.path.exists(img_path):
                os.remove(img_path)
        
        print(f"📊 Created {len(images)} frames")
        print(f"🎬 GIF duration: {len(images) * 0.05:.2f} seconds")
    else:
        print("❌ No frames were created!")

print("\n🎉 All meshes processed!")
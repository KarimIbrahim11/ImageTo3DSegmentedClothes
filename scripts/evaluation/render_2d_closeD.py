import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import PolyCollection
from pathlib import Path
import argparse
import time

def load_mesh_data(filepath):
    """Load mesh data from .npz file"""
    print(f"Loading data from {filepath}...")
    start_time = time.time()
    
    try:
        data = np.load(filepath)
        print(f"Available keys: {list(data.keys())}")
        
        points = data['points'] if 'points' in data else None
        faces = data['faces'] if 'faces' in data else None
        colors = data['colors'] if 'colors' in data else None
        labels = data['labels'] if 'labels' in data else None
        
        if points is None or faces is None:
            print("Missing required 'points' or 'faces' data")
            return None, None, None, None
        
        print(f"Loaded: {len(points)} vertices, {len(faces)} faces")
        print(f"Data loaded in {time.time() - start_time:.2f}s")
        
        return points, faces, colors, labels
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None, None

def create_fast_mesh_render(points, faces, colors=None, labels=None, view='front', gamma=1.8):
    """
    Fast mesh rendering using matplotlib's PolyCollection with best quality settings
    """
    print(f"Creating {view} view...")
    start_time = time.time()
    
    # Select coordinates based on view
    if view == 'front':
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
    elif view == 'back':
        x, y, z = -points[:, 0], points[:, 1], -points[:, 2]
    elif view == 'side':
        x, y, z = points[:, 1], points[:, 2], points[:, 0]
    else:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    print(f"Preparing {len(faces)} triangles...")
    
    # Create triangle vertices for PolyCollection
    triangles = []
    triangle_colors = []
    
    # Dynamic batch size: 10% of total faces for optimal progress tracking
    batch_size = max(1000, len(faces) // 10)  # At least 1000, but 10% of faces
    print(f"Using batch size: {batch_size}")
    
    for batch_start in range(0, len(faces), batch_size):
        batch_end = min(batch_start + batch_size, len(faces))
        batch_faces = faces[batch_start:batch_end]
        
        # Vectorized triangle creation for better performance
        face_vertices = np.array([
            [[x[face[0]], y[face[0]]],
             [x[face[1]], y[face[1]]],
             [x[face[2]], y[face[2]]]] for face in batch_faces
        ])
        triangles.extend(face_vertices)
        
        # Vectorized color processing for best quality
        if colors is not None:
            # Get all face colors at once
            face_color_batch = colors[batch_faces]  # Shape: (batch_size, 3, 3)
            
            # Normalize if needed
            if face_color_batch.max() > 1.0:
                face_color_batch = face_color_batch / 255.0
            
            # Average vertex colors per face with higher precision
            avg_colors = np.mean(face_color_batch, axis=1)  # Shape: (batch_size, 3)
            
            # Apply gamma correction for better exposure (per-channel)
            avg_colors = np.power(np.clip(avg_colors, 1e-8, 1.0), gamma)
            
            triangle_colors.extend(avg_colors.tolist())
            
        elif labels is not None:
            # Use first vertex label with better color mapping
            label_batch = labels[batch_faces[:, 0]]
            for label in label_batch:
                # Enhanced color mapping with more vibrant colors
                color_map = plt.cm.Set3(label % 12)  # Better color variety
                triangle_colors.append(color_map[:3])
        else:
            # Use depth for coloring with better contrast
            z_batch = z[batch_faces]  # Get z coords for all faces
            avg_z_batch = np.mean(z_batch, axis=1)
            
            # Normalize depth to [0, 1] for better contrast
            if len(avg_z_batch) > 0:
                z_norm = (avg_z_batch - z.min()) / (z.max() - z.min())
                # Use a better colormap for depth
                depth_colors = plt.cm.viridis(z_norm)[:, :3]  # Remove alpha
                triangle_colors.extend(depth_colors.tolist())
        
        # Progress update every batch
        progress = (batch_end / len(faces)) * 100
        print(f"Processing triangles: {progress:.1f}%")
    
    # Final color enhancement for best quality
    if colors is not None and triangle_colors:
        print("Applying final color enhancements...")
        triangle_colors = np.array(triangle_colors)
        
        # Enhance contrast and saturation for better visual quality
        # Convert to HSV-like enhancement
        brightness_boost = np.mean(triangle_colors, axis=1, keepdims=True)
        triangle_colors = triangle_colors + 0.1 * (triangle_colors - brightness_boost)
        
        # Clamp to valid range
        triangle_colors = np.clip(triangle_colors, 0, 1)
        triangle_colors = triangle_colors.tolist()
    
    print(f"Mesh preparation completed in {time.time() - start_time:.2f}s")
    return triangles, triangle_colors

# def create_fast_mesh_render(points, faces, colors=None, labels=None, view='front'):
#     """
#     Fast mesh rendering using matplotlib's PolyCollection
#     """
#     print(f"Creating {view} view...")
#     start_time = time.time()
    
#     # Select coordinates based on view
#     if view == 'front':
#         x, y, z = points[:, 0], points[:, 1], points[:, 2]
#     elif view == 'back':
#         x, y, z = -points[:, 0], points[:, 1], -points[:, 2]
#     elif view == 'side':
#         x, y, z = points[:, 1], points[:, 2], points[:, 0]
#     else:
#         x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
#     print(f"Preparing {len(faces)} triangles...")
    
#     # Create triangle vertices for PolyCollection
#     triangles = []
#     triangle_colors = []
    
#     # Process faces in batches for progress
#     batch_size = 10000
#     for batch_start in range(0, len(faces), batch_size):
#         batch_end = min(batch_start + batch_size, len(faces))
#         batch_faces = faces[batch_start:batch_end]
        
#         for face in batch_faces:
#             # Get triangle vertices
#             triangle = np.array([[x[face[0]], y[face[0]]],
#                                [x[face[1]], y[face[1]]],
#                                [x[face[2]], y[face[2]]]])
#             triangles.append(triangle)
            
#             # Determine triangle color
#             if colors is not None:
#                 # Average vertex colors
#                 face_colors = colors[face]
#                 if face_colors.max() > 1.0:
#                     face_colors = face_colors / 255.0
#                 avg_color = np.mean(face_colors, axis=0)
#                 triangle_colors.append(avg_color)
#             elif labels is not None:
#                 # Use first vertex label
#                 label = labels[face[0]]
#                 # Simple color mapping
#                 color_map = plt.cm.tab20(label % 20)
#                 triangle_colors.append(color_map[:3])
#             else:
#                 # Use depth for coloring
#                 avg_z = np.mean(z[face])
#                 triangle_colors.append([0.5, 0.5, 0.5])  # Gray default
        
#         # Progress update
#         if batch_start % (batch_size * 5) == 0:
#             progress = (batch_end / len(faces)) * 100
#             print(f"Processing triangles: {progress:.1f}%")
    
#     print(f"Mesh preparation completed in {time.time() - start_time:.2f}s")
#     return triangles, triangle_colors

def render_mesh_fast(scan_filepath, output_path=None, view='front', 
                    render_type='color', figsize=(5.12, 5.12), dpi=100):
    """
    Fast mesh rendering function
    """
    total_start = time.time()
    
    # Load data
    points, faces, colors, labels = load_mesh_data(scan_filepath)
    if points is None or faces is None:
        return False
    
    # Create mesh data
    if render_type == 'color':
        triangles, triangle_colors = create_fast_mesh_render(points, faces, colors=colors, view=view)
    elif render_type == 'labels':
        triangles, triangle_colors = create_fast_mesh_render(points, faces, labels=labels, view=view)
    else:  # depth
        triangles, triangle_colors = create_fast_mesh_render(points, faces, view=view)
    
    # Set up output path
    if output_path is None:
        input_path = Path(scan_filepath)
        output_path = input_path.parent / f"{input_path.stem}_mesh_{view}_{render_type}.png"
    
    print(f"Rendering to image...")
    render_start = time.time()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create PolyCollection for fast rendering
    if triangle_colors:
        poly_collection = PolyCollection(triangles, facecolors=triangle_colors, 
                                       edgecolors='none', linewidths=0)
    else:
        poly_collection = PolyCollection(triangles, facecolors='gray', 
                                       edgecolors='none', linewidths=0)
    
    ax.add_collection(poly_collection)
    
    # Set axis properties
    all_points = np.array(triangles).reshape(-1, 2)
    ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
    ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    ax.set_aspect('equal')
    ax.axis('off')
    # ax.set_title(f'{view.title()} View - {render_type.title()} Rendering', pad=20)
    
    # Save image
    print(f"Saving to {output_path}...")
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    total_time = time.time() - total_start
    render_time = time.time() - render_start
    print(f"Image rendered in {render_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Mesh saved to: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Fast mesh renderer for NPZ files')
    
    parser.add_argument('--input_file', help='Path to input .npz file', default= "/home/stud220/git/ImageTo3DSegmentedClothes/inputs/scans/10005_2069.npz")
    parser.add_argument('-o', '--output', help='Output image path (default: input_name.png)', default="/home/stud220/git/ImageTo3DSegmentedClothes/inputs/images/10005_2069.png" )
   
    parser.add_argument('-v', '--view', choices=['front', 'back', 'side'], 
                       default='front', help='View direction (default: front)')
    parser.add_argument('-t', '--type', choices=['color', 'labels', 'depth'], 
                       default='color', help='Rendering type (default: color)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[5.12, 5.12],
                       help='Figure size in inches (default: 5.12 5.12)')
    parser.add_argument('--dpi', type=int, default=100,
                       help='Output DPI (default: 100)')
    
    args = parser.parse_args()
    
    # Ensure matplotlib uses non-interactive backend
    plt.switch_backend('Agg')
    
    print(f"Starting mesh rendering...")
    print(f"Input: {args.input_file}")
    print(f"View: {args.view}, Type: {args.type}")
    print(f"Output size: {args.figsize[0]}x{args.figsize[1]} inches at {args.dpi} DPI")
    
    success = render_mesh_fast(
        args.input_file,
        args.output,
        args.view,
        args.type,
        tuple(args.figsize),
        args.dpi
    )
    
    if success:
        print("✓ Rendering completed successfully!")
    else:
        print("✗ Rendering failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    exit(main())
    
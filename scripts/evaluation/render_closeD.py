import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import PolyCollection
from pathlib import Path
import argparse
import time
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

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
        
        if points is None:
            print("Missing required 'points' data")
            return None, None, None, None
            
        # If faces are not provided, generate them from point cloud
        if faces is None:
            print("No faces found - generating mesh from point cloud...")
            faces = generate_faces_from_points(points)
        
        print(f"Loaded: {len(points)} vertices, {len(faces)} faces")
        print(f"Data loaded in {time.time() - start_time:.2f}s")
        
        return points, faces, colors, labels
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None, None

def generate_faces_from_points(points, method='delaunay_2d'):
    """
    Generate faces from point cloud using various methods
    """
    print(f"Generating faces using {method} method...")
    start_time = time.time()
    
    if method == 'delaunay_2d':
        # Project to 2D and use Delaunay triangulation
        # Use the two dimensions with highest variance
        coords_var = np.var(points, axis=0)
        dim_indices = np.argsort(coords_var)[-2:]  # Two dimensions with highest variance
        
        points_2d = points[:, dim_indices]
        
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(points_2d)
            faces = tri.simplices
            print(f"Generated {len(faces)} faces in {time.time() - start_time:.2f}s")
            return faces
        except Exception as e:
            print(f"Delaunay triangulation failed: {e}")
            return generate_faces_knn(points)
    
    elif method == 'convex_hull':
        try:
            hull = ConvexHull(points)
            faces = hull.simplices
            print(f"Generated {len(faces)} faces using convex hull in {time.time() - start_time:.2f}s")
            return faces
        except Exception as e:
            print(f"Convex hull failed: {e}")
            return generate_faces_knn(points)
    
    else:  # knn method
        return generate_faces_knn(points)

def generate_faces_knn(points, k=8):
    """
    Generate faces using k-nearest neighbors approach
    """
    print("Generating faces using KNN method...")
    start_time = time.time()
    
    # Use KNN to find local neighborhoods
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    faces = []
    for i, neighbors in enumerate(indices):
        # Create triangles with the point and its neighbors
        for j in range(1, len(neighbors)-1):
            for l in range(j+1, len(neighbors)):
                face = [neighbors[0], neighbors[j], neighbors[l]]  # neighbors[0] is the point itself
                faces.append(face)
    
    faces = np.array(faces)
    print(f"Generated {len(faces)} faces using KNN in {time.time() - start_time:.2f}s")
    return faces

def compute_face_depths(points, faces, view='front'):
    """
    Compute depth (z-coordinate) for each face for proper sorting
    """
    if view == 'front':
        z_coords = points[:, 2]
    elif view == 'back':
        z_coords = -points[:, 2]
    elif view == 'side':
        z_coords = points[:, 0]
    else:
        z_coords = points[:, 2]
    
    # Compute average depth for each face
    face_depths = np.mean(z_coords[faces], axis=1)
    return face_depths

def remove_backfacing_triangles(points, faces, view='front'):
    """
    Remove triangles that are facing away from the camera
    """
    print("Removing backfacing triangles...")
    
    # Define view direction
    if view == 'front':
        view_dir = np.array([0, 0, 1])
    elif view == 'back':
        view_dir = np.array([0, 0, -1])
    elif view == 'side':
        view_dir = np.array([1, 0, 0])
    else:
        view_dir = np.array([0, 0, 1])
    
    visible_faces = []
    
    for face in faces:
        # Get triangle vertices
        v0, v1, v2 = points[face]
        
        # Compute normal vector using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Skip degenerate triangles
        if np.linalg.norm(normal) < 1e-10:
            continue
            
        # Normalize
        normal = normal / np.linalg.norm(normal)
        
        # Check if facing towards camera
        if np.dot(normal, view_dir) > 0:
            visible_faces.append(face)
    
    visible_faces = np.array(visible_faces) if visible_faces else faces
    print(f"Kept {len(visible_faces)} visible faces out of {len(faces)}")
    return visible_faces

def create_depth_sorted_mesh_render(points, faces, colors=None, labels=None, view='front', 
                                  gamma=1.8, remove_backfaces=True):
    """
    Enhanced mesh rendering with proper depth sorting and backface culling
    """
    print(f"Creating enhanced {view} view...")
    start_time = time.time()
    
    # Remove backfacing triangles if requested
    if remove_backfaces and len(faces) > 0:
        faces = remove_backfacing_triangles(points, faces, view)
    
    # Select coordinates based on view
    if view == 'front':
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
    elif view == 'back':
        x, y, z = -points[:, 0], points[:, 1], -points[:, 2]
    elif view == 'side':
        x, y, z = points[:, 1], points[:, 2], points[:, 0]
    else:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    if len(faces) == 0:
        print("No faces to render!")
        return [], []
    
    print(f"Preparing {len(faces)} triangles with depth sorting...")
    
    # Compute face depths for sorting
    face_depths = compute_face_depths(points, faces, view)
    
    # Sort faces by depth (back to front for proper rendering)
    depth_order = np.argsort(face_depths)
    sorted_faces = faces[depth_order]
    
    # Create triangle vertices and colors
    triangles = []
    triangle_colors = []
    
    batch_size = max(1000, len(sorted_faces) // 10)
    print(f"Using batch size: {batch_size}")
    
    for batch_start in range(0, len(sorted_faces), batch_size):
        batch_end = min(batch_start + batch_size, len(sorted_faces))
        batch_faces = sorted_faces[batch_start:batch_end]
        
        # Vectorized triangle creation
        face_vertices = np.array([
            [[x[face[0]], y[face[0]]],
             [x[face[1]], y[face[1]]],
             [x[face[2]], y[face[2]]]] for face in batch_faces
        ])
        triangles.extend(face_vertices)
        
        # Color processing
        if colors is not None:
            face_color_batch = colors[batch_faces]
            
            # Handle different color array shapes
            if len(face_color_batch.shape) == 3:  # Per-vertex colors
                # Normalize if needed
                if face_color_batch.max() > 1.0:
                    face_color_batch = face_color_batch / 255.0
                # Average vertex colors per face
                avg_colors = np.mean(face_color_batch, axis=1)
            else:  # Per-face colors
                avg_colors = face_color_batch
                if avg_colors.max() > 1.0:
                    avg_colors = avg_colors / 255.0
            
            # Apply gamma correction
            avg_colors = np.power(np.clip(avg_colors, 1e-8, 1.0), gamma)
            triangle_colors.extend(avg_colors.tolist())
            
        elif labels is not None:
            label_batch = labels[batch_faces[:, 0]]
            for label in label_batch:
                color_map = plt.cm.Set3(label % 12)
                triangle_colors.append(color_map[:3])
        else:
            # Enhanced depth-based coloring
            batch_depths = face_depths[depth_order[batch_start:batch_end]]
            if len(batch_depths) > 0:
                depth_min, depth_max = face_depths.min(), face_depths.max()
                if depth_max > depth_min:
                    z_norm = (batch_depths - depth_min) / (depth_max - depth_min)
                else:
                    z_norm = np.ones_like(batch_depths) * 0.5
                
                # Use a more sophisticated colormap
                depth_colors = plt.cm.plasma(z_norm)[:, :3]
                triangle_colors.extend(depth_colors.tolist())
        
        # Progress update
        progress = (batch_end / len(sorted_faces)) * 100
        print(f"Processing triangles: {progress:.1f}%")
    
    # Final color enhancement
    if colors is not None and triangle_colors:
        print("Applying final color enhancements...")
        triangle_colors = np.array(triangle_colors)
        
        # Enhance contrast while preserving color relationships
        triangle_colors = np.clip(triangle_colors * 1.1, 0, 1)
        triangle_colors = triangle_colors.tolist()
    
    print(f"Enhanced mesh preparation completed in {time.time() - start_time:.2f}s")
    return triangles, triangle_colors

def render_mesh_enhanced(scan_filepath, output_path=None, view='front', 
                        render_type='color', figsize=(10.24, 10.24), dpi=150,
                        face_generation='delaunay_2d', remove_backfaces=True):
    """
    Enhanced mesh rendering function with better quality and depth handling
    """
    total_start = time.time()
    
    # Load data
    points, faces, colors, labels = load_mesh_data(scan_filepath)
    if points is None:
        return False
    
    # Generate faces if needed
    if faces is None:
        faces = generate_faces_from_points(points, method=face_generation)
    
    # Create enhanced mesh data
    if render_type == 'color':
        triangles, triangle_colors = create_depth_sorted_mesh_render(
            points, faces, colors=colors, view=view, remove_backfaces=remove_backfaces)
    elif render_type == 'labels':
        triangles, triangle_colors = create_depth_sorted_mesh_render(
            points, faces, labels=labels, view=view, remove_backfaces=remove_backfaces)
    else:  # depth
        triangles, triangle_colors = create_depth_sorted_mesh_render(
            points, faces, view=view, remove_backfaces=remove_backfaces)
    
    # Set up output path
    if output_path is None:
        input_path = Path(scan_filepath)
        output_path = input_path.parent / f"{input_path.stem}_enhanced_{view}_{render_type}.png"
    
    print(f"Rendering enhanced image...")
    render_start = time.time()
    
    # Create figure with higher quality settings
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create PolyCollection with enhanced settings
    if triangle_colors and triangles:
        poly_collection = PolyCollection(
            triangles, 
            facecolors=triangle_colors,
            edgecolors='none',
            linewidths=0,
            alpha=0.95,  # Slight transparency to blend overlapping areas
            antialiased=True
        )
    else:
        poly_collection = PolyCollection(
            triangles,
            facecolors='lightgray',
            edgecolors='none',
            linewidths=0,
            antialiased=True
        )
    
    ax.add_collection(poly_collection)
    
    # Set axis properties with better margins
    if triangles:
        all_points = np.array(triangles).reshape(-1, 2)
        x_margin = (all_points[:, 0].max() - all_points[:, 0].min()) * 0.02
        y_margin = (all_points[:, 1].max() - all_points[:, 1].min()) * 0.02
        
        ax.set_xlim(all_points[:, 0].min() - x_margin, all_points[:, 0].max() + x_margin)
        ax.set_ylim(all_points[:, 1].min() - y_margin, all_points[:, 1].max() + y_margin)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save with high quality settings
    print(f"Saving enhanced image to {output_path}...")
    plt.tight_layout(pad=0)
    
    # Determine save parameters based on file extension
    save_kwargs = {
        'dpi': dpi,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none',
        'pad_inches': 0.1
    }
    
    # Add quality parameter only for JPEG files
    if str(output_path).lower().endswith(('.jpg', '.jpeg')):
        save_kwargs['quality'] = 95
    
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)
    
    total_time = time.time() - total_start
    render_time = time.time() - render_start
    print(f"Enhanced image rendered in {render_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Enhanced mesh saved to: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Enhanced mesh renderer for NPZ files')
    
    parser.add_argument('--input_file', 
                       help='Path to input .npz file', 
                       default="/home/stud220/git/ImageTo3DSegmentedClothes/inputs/scans/10005_2069.npz")
    parser.add_argument('-o', '--output', 
                       help='Output image path (default: auto-generated)', 
                       default="/home/stud220/git/ImageTo3DSegmentedClothes/inputs/images/10005_2069.png")
    parser.add_argument('-v', '--view', 
                       choices=['front', 'back', 'side'], 
                       default='front', 
                       help='View direction (default: front)')
    parser.add_argument('-t', '--type', 
                       choices=['color', 'labels', 'depth'], 
                       default='color', 
                       help='Rendering type (default: color)')
    parser.add_argument('--figsize', 
                       type=float, nargs=2, 
                       default=[10.24, 10.24],
                       help='Figure size in inches (default: 10.24 10.24)')
    parser.add_argument('--dpi', 
                       type=int, 
                       default=150,
                       help='Output DPI (default: 150)')
    parser.add_argument('--face_generation', 
                       choices=['delaunay_2d', 'convex_hull', 'knn'],
                       default='delaunay_2d',
                       help='Method for generating faces from point cloud (default: delaunay_2d)')
    parser.add_argument('--no_backface_culling', 
                       action='store_true',
                       help='Disable backface culling')
    
    args = parser.parse_args()
    
    # Ensure matplotlib uses non-interactive backend
    plt.switch_backend('Agg')
    
    print(f"Starting enhanced mesh rendering...")
    print(f"Input: {args.input_file}")
    print(f"View: {args.view}, Type: {args.type}")
    print(f"Output size: {args.figsize[0]}x{args.figsize[1]} inches at {args.dpi} DPI")
    print(f"Face generation: {args.face_generation}")
    print(f"Backface culling: {'disabled' if args.no_backface_culling else 'enabled'}")
    
    success = render_mesh_enhanced(
        args.input_file,
        args.output,
        args.view,
        args.type,
        tuple(args.figsize),
        args.dpi,
        args.face_generation,
        not args.no_backface_culling
    )
    
    if success:
        print("✓ Enhanced rendering completed successfully!")
    else:
        print("✗ Enhanced rendering failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    exit(main())

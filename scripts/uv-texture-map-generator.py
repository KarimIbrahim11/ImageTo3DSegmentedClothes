import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import trimesh
import argparse
from pathlib import Path
import cv2
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class UVTextureGenerator:
    def __init__(self, mesh_path):
        """
        Initialize the UV texture generator with a mesh file.
        
        Args:
            mesh_path (str): Path to PLY or OBJ file
        """
        self.mesh_path = Path(mesh_path)
        self.mesh = None
        self.uv_coordinates = None
        self.texture_size = 1024
        
    def load_mesh(self):
        """Load mesh from PLY or OBJ file"""
        try:
            self.mesh = trimesh.load(self.mesh_path)
            print(f"Loaded mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces")
            return True
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return False
    
    def spherical_projection(self):
        """Generate UV coordinates using spherical projection"""
        vertices = self.mesh.vertices
        
        # Center the mesh
        centroid = np.mean(vertices, axis=0)
        centered_vertices = vertices - centroid
        
        # Convert to spherical coordinates
        x, y, z = centered_vertices[:, 0], centered_vertices[:, 1], centered_vertices[:, 2]
        
        # Calculate spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)  # azimuthal angle
        phi = np.arccos(np.clip(z / (r + 1e-8), -1, 1))  # polar angle
        
        # Convert to UV coordinates (0-1 range)
        u = (theta + np.pi) / (2 * np.pi)
        v = phi / np.pi
        
        return np.column_stack([u, v])
    
    def cylindrical_projection(self):
        """Generate UV coordinates using cylindrical projection"""
        vertices = self.mesh.vertices
        
        # Center the mesh
        centroid = np.mean(vertices, axis=0)
        centered_vertices = vertices - centroid
        
        x, y, z = centered_vertices[:, 0], centered_vertices[:, 1], centered_vertices[:, 2]
        
        # Cylindrical coordinates
        theta = np.arctan2(y, x)
        height = z
        
        # Normalize to UV coordinates
        u = (theta + np.pi) / (2 * np.pi)
        v = (height - np.min(height)) / (np.max(height) - np.min(height))
        
        return np.column_stack([u, v])
    
    def planar_projection(self, axis='z'):
        """Generate UV coordinates using planar projection"""
        vertices = self.mesh.vertices
        
        if axis == 'z':
            coords = vertices[:, [0, 1]]  # x, y
        elif axis == 'y':
            coords = vertices[:, [0, 2]]  # x, z
        else:  # axis == 'x'
            coords = vertices[:, [1, 2]]  # y, z
        
        # Normalize to 0-1 range
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        
        uv = (coords - min_coords) / (max_coords - min_coords)
        
        return uv
    
    def generate_uv_map(self, method='spherical'):
        """
        Generate UV coordinates using specified method.
        
        Args:
            method (str): 'spherical', 'cylindrical', or 'planar'
        """
        if method == 'spherical':
            self.uv_coordinates = self.spherical_projection()
        elif method == 'cylindrical':
            self.uv_coordinates = self.cylindrical_projection()
        elif method == 'planar':
            self.uv_coordinates = self.planar_projection()
        else:
            raise ValueError("Method must be 'spherical', 'cylindrical', or 'planar'")
        
        print(f"Generated UV coordinates using {method} projection")
    
    def create_wireframe_texture(self, line_width=2, background_color=(240, 240, 240), 
                                line_color=(0, 0, 0)):
        """
        Create a wireframe texture map showing the UV layout.
        
        Args:
            line_width (int): Width of wireframe lines
            background_color (tuple): RGB background color
            line_color (tuple): RGB line color
        """
        # Create blank texture
        texture = np.full((self.texture_size, self.texture_size, 3), 
                         background_color, dtype=np.uint8)
        
        # Draw triangles
        for face in self.mesh.faces:
            # Get UV coordinates for this face
            face_uv = self.uv_coordinates[face]
            
            # Convert to pixel coordinates
            pixels = (face_uv * (self.texture_size - 1)).astype(np.int32)
            
            # Draw triangle edges
            for i in range(3):
                start = pixels[i]
                end = pixels[(i + 1) % 3]
                cv2.line(texture, tuple(start), tuple(end), line_color, line_width)
        
        return texture
    
    def create_triangle_texture(self, triangle_density=True):
        """
        Create a texture showing triangle density or individual triangles.
        
        Args:
            triangle_density (bool): If True, show density. If False, show individual triangles.
        """
        texture = np.zeros((self.texture_size, self.texture_size, 3), dtype=np.uint8)
        
        if triangle_density:
            # Create density map
            density_map = np.zeros((self.texture_size, self.texture_size))
            
            for face in self.mesh.faces:
                face_uv = self.uv_coordinates[face]
                pixels = (face_uv * (self.texture_size - 1)).astype(np.int32)
                
                # Fill triangle
                triangle = np.array([pixels], dtype=np.int32)
                cv2.fillPoly(density_map, triangle, 1)
            
            # Convert density to color
            density_map = density_map / np.max(density_map) if np.max(density_map) > 0 else density_map
            texture = plt.cm.viridis(density_map)[:, :, :3] * 255
            texture = texture.astype(np.uint8)
        else:
            # Color each triangle differently
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.mesh.faces)))
            
            for i, face in enumerate(self.mesh.faces):
                face_uv = self.uv_coordinates[face]
                pixels = (face_uv * (self.texture_size - 1)).astype(np.int32)
                
                color = (colors[i][:3] * 255).astype(np.uint8)
                triangle = np.array([pixels], dtype=np.int32)
                cv2.fillPoly(texture, triangle, color.tolist())
        
        return texture
    
    def visualize_uv_layout(self, show_vertices=True, show_edges=True, figsize=(12, 8)):
        """Visualize the UV layout using matplotlib"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 3D mesh
        ax1 = axes[0]
        if hasattr(self.mesh.visual, 'vertex_colors'):
            colors = self.mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            colors = 'lightblue'
        
        # Simple 3D visualization (projection to 2D)
        vertices_2d = self.mesh.vertices[:, [0, 1]]  # Project to XY plane
        
        if show_edges:
            for face in self.mesh.faces:
                face_verts = vertices_2d[face]
                polygon = Polygon(face_verts, fill=False, edgecolor='black', linewidth=0.5)
                ax1.add_patch(polygon)
        
        if show_vertices:
            ax1.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c=colors, s=1, alpha=0.6)
        
        ax1.set_title('3D Mesh (XY Projection)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Plot UV layout
        ax2 = axes[1]
        
        if show_edges:
            for face in self.mesh.faces:
                face_uv = self.uv_coordinates[face]
                polygon = Polygon(face_uv, fill=False, edgecolor='black', linewidth=0.5)
                ax2.add_patch(polygon)
        
        if show_vertices:
            ax2.scatter(self.uv_coordinates[:, 0], self.uv_coordinates[:, 1], 
                       c=colors, s=1, alpha=0.6)
        
        ax2.set_title('UV Layout')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('U')
        ax2.set_ylabel('V')
        
        plt.tight_layout()
        return fig
    
    def save_uv_coordinates(self, output_path):
        """Save UV coordinates to a file"""
        np.savetxt(output_path, self.uv_coordinates, fmt='%.6f', 
                  header='U V coordinates', delimiter=' ')
        print(f"UV coordinates saved to {output_path}")
    
    def save_texture(self, texture, output_path):
        """Save texture to image file"""
        # Convert RGB to BGR for OpenCV
        texture_bgr = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), texture_bgr)
        print(f"Texture saved to {output_path}")
    
    def export_obj_with_uv(self, output_path):
        """Export mesh as OBJ file with UV coordinates"""
        with open(output_path, 'w') as f:
            # Write vertices
            for vertex in self.mesh.vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write UV coordinates
            for uv in self.uv_coordinates:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            
            # Write faces with UV references
            for face in self.mesh.faces:
                f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
        
        print(f"OBJ with UV coordinates saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate UV texture maps for 3D meshes')
    parser.add_argument('mesh_file', help='Path to PLY or OBJ file')
    parser.add_argument('--method', choices=['spherical', 'cylindrical', 'planar'], 
                       default='spherical', help='UV projection method')
    parser.add_argument('--output_dir', default='./uv_output', help='Output directory')
    parser.add_argument('--texture_size', type=int, default=1024, help='Texture resolution')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = UVTextureGenerator(args.mesh_file)
    generator.texture_size = args.texture_size
    
    # Load mesh
    if not generator.load_mesh():
        return
    
    # Generate UV coordinates
    generator.generate_uv_map(method=args.method)
    
    # Create visualizations
    fig = generator.visualize_uv_layout()
    fig.savefig(output_dir / 'uv_layout_visualization.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Create wireframe texture
    wireframe_texture = generator.create_wireframe_texture()
    generator.save_texture(wireframe_texture, output_dir / 'wireframe_texture.png')
    
    # Create triangle density texture
    density_texture = generator.create_triangle_texture(triangle_density=True)
    generator.save_texture(density_texture, output_dir / 'density_texture.png')
    
    # Create colored triangles texture
    triangle_texture = generator.create_triangle_texture(triangle_density=False)
    generator.save_texture(triangle_texture, output_dir / 'triangle_texture.png')
    
    # Save UV coordinates
    generator.save_uv_coordinates(output_dir / 'uv_coordinates.txt')
    
    # Export OBJ with UV coordinates
    generator.export_obj_with_uv(output_dir / 'mesh_with_uv.obj')
    
    print(f"\nUV texture generation complete! Check the '{output_dir}' directory for outputs.")


if __name__ == "__main__":
    # Example usage if running directly
    # if len(sys.argv) == 1:
    #     print("Example usage:")
    #     print("python uv_generator.py mesh.ply --method spherical --output_dir ./output")
    #     print("\nOr use as a library:")
    #     print("generator = UVTextureGenerator('mesh.ply')")
    #     print("generator.load_mesh()")
    #     print("generator.generate_uv_map('spherical')")
    #     print("texture = generator.create_wireframe_texture()")
    # else:
    main()
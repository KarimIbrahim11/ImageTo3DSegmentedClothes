import trimesh
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def generate_uv_texture_map(mesh_path, output_obj_path, texture_output_path, texture_size=512):
    """
    Generate a UV texture map from a mesh file.
    
    Args:
        mesh_path: Path to input mesh file
        output_obj_path: Path to save the mesh with UV coordinates
        texture_output_path: Path to save the texture map (should end with .jpg)
        texture_size: Size of the texture map (default 512x512)
    """
    
    # Load the mesh
    mesh = trimesh.load(mesh_path)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    
    # Method 1: Simple planar projection (basic approach)
    def planar_uv_unwrap(vertices):
        """Simple planar projection to UV coordinates"""
        # Project onto XY plane and normalize to [0,1]
        min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
        min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
        
        u = (vertices[:, 0] - min_x) / (max_x - min_x) if max_x != min_x else np.zeros(len(vertices))
        v = (vertices[:, 1] - min_y) / (max_y - min_y) if max_y != min_y else np.zeros(len(vertices))
        
        return np.column_stack([u, v])
    
    # Method 2: Cylindrical projection (better for human-like objects)
    def cylindrical_uv_unwrap(vertices):
        """Cylindrical projection for UV coordinates"""
        # Convert to cylindrical coordinates
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        
        # U coordinate from angle around Y axis
        u = (np.arctan2(x, z) + np.pi) / (2 * np.pi)
        
        # V coordinate from height (Y axis)
        min_y, max_y = y.min(), y.max()
        v = (y - min_y) / (max_y - min_y) if max_y != min_y else np.zeros(len(vertices))
        
        return np.column_stack([u, v])
    
    # Generate UV coordinates using cylindrical projection (better for human models)
    uv_coords = cylindrical_uv_unwrap(mesh.vertices)
    
    # Create texture map based on vertex properties
    texture_map = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    
    # Option 1: Color based on vertex normals (if available)
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        # Use normals to create a colorful texture
        for i, (u, v) in enumerate(uv_coords):
            tex_x = int(u * (texture_size - 1))
            tex_y = int(v * (texture_size - 1))
            
            # Convert normal to color (normalize from [-1,1] to [0,255])
            normal = mesh.vertex_normals[i]
            color = ((normal + 1) * 127.5).astype(np.uint8)
            texture_map[tex_y, tex_x] = color
    
    # Option 2: Color based on vertex position (fallback)
    else:
        for i, (u, v) in enumerate(uv_coords):
            tex_x = int(u * (texture_size - 1))
            tex_y = int(v * (texture_size - 1))
            
            # Use vertex position to generate color
            vertex = mesh.vertices[i]
            # Normalize coordinates to [0,255] range
            color = ((vertex - vertex.min()) / (vertex.max() - vertex.min()) * 255).astype(np.uint8)
            texture_map[tex_y, tex_x] = color[:3]  # Take only RGB
    
    # Fill empty pixels with interpolation
    from scipy.ndimage import gaussian_filter
    for channel in range(3):
        mask = texture_map[:, :, channel] > 0
        if np.any(mask):
            # Apply Gaussian blur to fill gaps
            texture_map[:, :, channel] = gaussian_filter(texture_map[:, :, channel].astype(float), sigma=1)
    
    # Save texture map
    texture_image = Image.fromarray(texture_map)
    texture_image.save(texture_output_path, 'JPEG', quality=95)
    print(f"Texture map saved to: {texture_output_path}")
    
    # Create new mesh with UV coordinates
    # Note: trimesh doesn't directly support UV coordinates, so we'll export as OBJ
    export_obj_with_uv(mesh, uv_coords, output_obj_path, texture_output_path)
    
    return texture_map, uv_coords

def export_obj_with_uv(mesh, uv_coords, obj_path, texture_path):
    """Export mesh as OBJ file with UV coordinates and material file"""
    
    # Exporting RGBMESG
    mesh.export(output_rgb_obj_path)

    # Create material file (.mtl)
    mtl_path = obj_path.replace('.obj', '.mtl')
    texture_filename = texture_path.split('/')[-1]  # Get just the filename
    
    with open(mtl_path, 'w') as f:
        f.write("# Material file\n")
        f.write("newmtl material_0\n")
        f.write("Ka 1.0 1.0 1.0\n")  # Ambient color
        f.write("Kd 1.0 1.0 1.0\n")  # Diffuse color
        f.write("Ks 0.0 0.0 0.0\n")  # Specular color
        f.write(f"map_Kd {texture_filename}\n")  # Texture map
    
    # Export OBJ file with UV coordinates
    with open(obj_path, 'w') as f:
        f.write("# OBJ file with UV coordinates\n")
        f.write(f"mtllib {mtl_path.split('/')[-1]}\n")
        
        # Write vertices
        for vertex in mesh.vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write UV coordinates
        for uv in uv_coords:
            f.write(f"vt {uv[0]} {uv[1]}\n")
        
        # Write faces with UV references
        f.write("usemtl material_0\n")
        for face in mesh.faces:
            # OBJ uses 1-based indexing
            f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
    print(f"OBJ with RGB coordinates saved to: {output_rgb_obj_path}")
    print(f"OBJ with UV coordinates saved to: {obj_path}")
    print(f"Material file saved to: {mtl_path}")

# Usage example
if __name__ == "__main__":
    # Your mesh path
    mesh_path = '/home/stud220/git/ImageTo3DSegmentedClothes/Human3Diffusion/output/05/tsdf-rgbd.ply'
    
    # Output paths
    
    output_rgb_obj_path = 'output/exportedobjectmesh_with_rgb.obj'
    output_obj_path = 'output/exportedobjectmesh_with_uv.obj'
    texture_output_path = 'output/texture_map.jpg'
    
    # Generate UV texture map
    texture_map, uv_coords = generate_uv_texture_map(
        mesh_path, 
        output_obj_path, 
        texture_output_path, 
        texture_size=512
    )
    
    # Optional: Display the texture map
    plt.figure(figsize=(8, 8))
    plt.imshow(texture_map)
    plt.title('Generated UV Texture Map')
    plt.axis('off')
    plt.show()
    
    print("UV texture map generation complete!")
    print(f"Files created:")
    
    print(f"  - Mesh with RGB: {output_rgb_obj_path}")
    print(f"  - Mesh with UV: {output_obj_path}")
    print(f"  - Texture map: {texture_output_path}")
    print(f"  - Material file: {output_obj_path.replace('.obj', '.mtl')}")
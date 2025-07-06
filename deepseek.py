import trimesh
import xatlas
import numpy as np
from PIL import Image

def ply_to_uv_texture_no_uvs(ply_path, output_jpeg_path, texture_size=1024):
    """
    Convert a PLY scan mesh with vertex colors to a UV texture map (JPEG)
    by first performing UV unwrapping using xatlas.
    
    Args:
        ply_path: Path to input PLY file
        output_jpeg_path: Path to save output JPEG texture
        texture_size: Size of the output texture map (square)
    """
    # Load the mesh
    mesh = trimesh.load(ply_path)
    
    # Verify we have vertex colors
    if not hasattr(mesh.visual, 'vertex_colors'):
        raise ValueError("Input mesh does not have vertex colors")
    
    # Convert vertex colors to 0-255 RGB if they're in 0-1 range
    vertex_colors = mesh.visual.vertex_colors
    if vertex_colors.max() <= 1.0:
        vertex_colors = (vertex_colors * 255).astype(np.uint8)
    
    # Perform UV unwrapping using xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = texture_size
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, indices, uvs = atlas.get_mesh(0)
    
    # Create new UV coordinates for the mesh
    mesh.visual.uv = uvs
    
    # Create blank texture
    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    
    # Rasterize each triangle to the texture
    for face_idx in range(len(mesh.faces)):
        # Get the UV coordinates for this face
        uv0 = uvs[indices[face_idx][0]]
        uv1 = uvs[indices[face_idx][1]]
        uv2 = uvs[indices[face_idx][2]]
        
        # Get vertex colors for this face
        color0 = vertex_colors[mesh.faces[face_idx][0]]
        color1 = vertex_colors[mesh.faces[face_idx][1]]
        color2 = vertex_colors[mesh.faces[face_idx][2]]
        
        # Convert UVs to pixel coordinates (flip Y axis)
        x0, y0 = int(uv0[0] * (texture_size-1)), int((1-uv0[1]) * (texture_size-1))
        x1, y1 = int(uv1[0] * (texture_size-1)), int((1-uv1[1]) * (texture_size-1))
        x2, y2 = int(uv2[0] * (texture_size-1)), int((1-uv2[1]) * (texture_size-1))
        
        # Simple triangle rasterization (replace with proper implementation)
        texture[y0:y0+1, x0:x0+1] = color0[:3]  # Only use RGB if there's alpha
        texture[y1:y1+1, x1:x1+1] = color1[:3]
        texture[y2:y2+1, x2:x2+1] = color2[:3]
    
    # Save the texture
    Image.fromarray(texture).save(output_jpeg_path, quality=95)
    
# Example usage
ply_to_uv_texture_no_uvs("/home/stud220/git/ImageTo3DSegmentedClothes/Human3Diffusion/output/05/tsdf-rgbd.ply", "output_texture.jpg", texture_size=2048)
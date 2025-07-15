import xatlas
from PIL import Image, ImageDraw
import numpy as np
import trimesh

mesh = trimesh.load('/home/stud220/git/ImageTo3DSegmentedClothes/Human3Diffusion/output/05/nicp-compatible-tsdf-rgbd.ply')
mesh.export('/home/stud220/git/ImageTo3DSegmentedClothes/output/nicp-compatible-tsdf-rgbd.obj')

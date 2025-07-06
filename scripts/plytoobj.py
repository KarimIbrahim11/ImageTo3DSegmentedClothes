import xatlas
from PIL import Image, ImageDraw
import numpy as np
import trimesh

mesh = trimesh.load('/home/stud220/git/ImageTo3DSegmentedClothes/Human3Diffusion/output/05/tsdf-rgbd.ply')
mesh.export('output/exportedobjectmesh.obj')

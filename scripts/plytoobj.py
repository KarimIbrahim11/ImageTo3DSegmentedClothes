import xatlas
from PIL import Image, ImageDraw
import numpy as np
import trimesh

mesh = trimesh.load('/home/stud220/git/ImageTo3DSegmentedClothes/output/SMPLH_meshes/tsdf-rgbd/aligned.ply')
mesh.export('/home/stud220/git/ImageTo3DSegmentedClothes/output/SMPL_meshes/tsdf-rgbd.obj')

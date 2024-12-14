#import open3d as o3d
from .yacs import CfgNode as CN
import numpy as np

cfg = CN()

# data
cfg.body_sample_ratio = 0.5
cfg.face_sample_ratio = 0.
cfg.rot_ratio = 0.
cfg.rot_range = np.pi / 32

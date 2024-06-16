import torch
from termcolor import colored
import torch.utils.data as data
#from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from plyfile import PlyData

import pickle

import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw

from uvm_lib.engine.thutil.io.prints import *
from uvm_lib.engine.thutil.animation.free_view import gen_path
from uvm_lib.engine.lib.data.base_dataset import BaseDataset
from uvm_lib.engine.thutil.distributed.sampler import data_sampler

class Dataset(BaseDataset):
 
    def initialize(self, opt, phase, data_dict):
        self.opt = opt
        self.initSupp()
        
        self.init(self.opt.dataroot, phase, data_dict)
        
    def init(self, data_root, split, data_dict):
 
        data_flag = split
        short_dataset_name = data_dict["resname"]        
        root_dir = '../data/zju_mocap'
        
        subdir = data_dict["dirname"]        
        self.data_root = os.path.join(root_dir, subdir)
        
        self.dataset_id = data_dict["id"]        
        data_root = self.data_root
        
        self.gender = data_dict["gender"]
        
        self.uv_dir = "uv256"

        self.split = split
        self.short_dataset_name = short_dataset_name
    
        self.human = data_dict.dirname
            
        ann_file = os.path.join(data_root, "annots.npy")        
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        self.load_meta_data(split)


        num_cams = len(self.cams['K'])
        self.full_num_cams = num_cams
        
        views = self.opt.multiview_ids
        view_id = [int(v) for v in views]
        self.view_id = view_id


        i_intv = data_dict.dataset_step

        if self.opt.motion_mode and split == 'train':
            i_intv = self.opt.posenet_setup.velocity
            
        self.data_flag=data_flag
        
        self.is_train = False

        if data_flag == "evaluate":
            
            known_id = data_dict["train_begin"][0]
            novel_id = int(data_dict["test_begin"][0] + data_dict["test_end"][0])
            novel_id = int(novel_id/2)

            self.eva_id = np.array([known_id, novel_id]).astype(int)
            
            self.ims = np.array([
                np.array(ims_data['ims'])[view_id[0]]
                for ims_data in np.array(annots['ims'])[self.eva_id]
            ]).ravel()
            self.cam_inds = np.array([
                np.arange(len(ims_data['ims']))[view_id[0]]
                for ims_data in np.array(annots['ims'])[self.eva_id]
            ]).ravel()
            
            self.num_cams = len(view_id)
            self.is_train = True
            return 
        
        if split == 'train':
                  
            begin = data_dict["train_begin"]
            end = data_dict["train_end"]
                
            self.is_train = True
                
        elif split == 'test':

            if self.opt.test_novel_pose:
                if self.opt.make_demo:
                    begin = data_dict["novel_pose_begin"]
                    end = data_dict["novel_pose_end"]
                else:
                    begin = data_dict["novel_pose_begin"]
                    end = data_dict["novel_pose_end"]
            else:
                begin = data_dict["test_begin"]
                end = data_dict["test_end"]
           
            if self.opt.debug_1pose_all_views: 
                end = [begin[0] + 1]

            i_intv = self.opt.test_step_size

        if self.opt.demo_all:
            begin = []
            end = []            
            frames = self.opt.demo_frame.split(" ")
            for f in frames:
                begin.append(int(f))
                end.append(int(f) + 1)
                
        ims = []
        cam_inds = []

        full_ims = []
        full_cam_inds = []

        for i in range(len(begin)):
            b = begin[i]
            e = end[i]
                    
            ims.append(np.array([
                np.array(ims_data['ims'])[view_id]
                for ims_data in annots['ims'][b:e][::i_intv]
            ]).ravel())
            cam_inds.append(np.array([
                np.arange(len(ims_data['ims']))[view_id]
                for ims_data in annots['ims'][b:e][::i_intv]
            ]).ravel()
            )

            full_ims.append(np.array([
                np.array(ims_data['ims'])
                for ims_data in annots['ims'][b:e][::i_intv]
            ]).ravel())
            full_cam_inds.append(np.array([
                np.arange(len(ims_data['ims']))
                for ims_data in annots['ims'][b:e][::i_intv]
            ]).ravel()
            )
        self.ims = np.array(ims)[0]
        self.cam_inds = np.array(cam_inds)[0]

        self.full_ims = np.array(full_ims)[0]
        self.full_cam_inds = np.array(full_cam_inds)[0]

        self.num_cams = len(view_id)

        if self.opt.make_demo:
            self.demo_setup()


    def demo_setup(self):
        if self.opt.motion.motion_chain:    
            motion_steps = self.opt.motion.motion_steps.split(" ")
            
            self.ims = np.repeat(self.ims, len(motion_steps))
            self.cam_inds = np.repeat(self.cam_inds, len(motion_steps))

            self.full_ims = np.repeat(self.full_ims, len(motion_steps))
            self.full_cam_inds = np.repeat(self.full_cam_inds, len(motion_steps))

    def get_train_sampler(self):
        print("base class data sampler")
        self.train_sampler = data_sampler(self, shuffle=True, distributed = self.opt.training.distributed)
        return self.train_sampler

    def get_mask(self, index, is_load_training = False):
        
        msk_path = os.path.join(self.data_root, 'mask',
                                self.ims[index])[:-4] + '.png'
        msk_path_chip = os.path.join(self.data_root, 'mask_cihp',
                    self.ims[index])[:-4] + '.png'
            
        msk = imageio.imread(msk_path)
        msk = (msk != 0).astype(np.uint8)

        msk_cihp = imageio.imread(msk_path_chip)
        msk_cihp = (msk_cihp != 0).astype(np.uint8)

        msk = (msk | msk_cihp).astype(np.uint8)

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk
    
    def prepare_input(self, i):

        frame_index = i                        
        vertices_path = os.path.join(self.data_root, "new_vertices",
                                     '{}.npy'.format(i))
        if not os.path.isfile(vertices_path): return None

        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)
        world_xyz = np.copy(xyz)
        
                
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        params_path = os.path.join(self.data_root, "new_params",
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        betas = params['shapes'].astype(np.float32)
        poses = params['poses'].astype(np.float32)

        # transformation augmentation
        center = np.array([0, 0, 0]).astype(np.float32)
        rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
        rot = rot.astype(np.float32)
        trans = np.array([0, 0, 0]).astype(np.float32)
                
        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        nxyz = nxyz.astype(np.float32)
        world_xyz = world_xyz.astype(np.float32)
        feature = np.concatenate([cxyz, world_xyz], axis=1).astype(np.float32)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(self.opt.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, betas, poses

    def get_item_packed(self, index, read_pre = False):
                
        img_path = os.path.join(self.data_root, self.ims[index])

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
        else:
            i = int(os.path.basename(img_path)[:-4])

        frame_index = i

        img = cv2.imread(img_path).astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = cv2.resize(img, (1024, 1024))
        msk = self.get_mask(index)

        cam_ind = self.cam_inds[index]
        Cam_K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, Cam_K, D)

        msk = cv2.undistort(msk, Cam_K, D)

        Cam_R = np.array(self.cams['R'][cam_ind])
        Cam_T = np.array(self.cams['T'][cam_ind]) / 1000.

        msk[msk>0] = 1
        img[msk == 0] = 0


        ret = {}

        H_gen, W_gen = int(img.shape[0] * self.opt.gen_ratio), int(img.shape[1] * self.opt.gen_ratio)
        img_gen = cv2.resize(img, (W_gen, H_gen)) 
        msk_gen = cv2.resize(msk, (W_gen, H_gen)) 
        
        msk_3c = np.array(msk)[...,None].repeat(3, axis=2)
                    
        img_gen = img_gen * msk_gen[...,None].repeat(3, axis=2)
                    
        K = Cam_K
        Cam_K_gen = np.copy(K)
        Cam_K_gen[:2] = Cam_K_gen[:2] * self.opt.gen_ratio

        H_nerf, W_nerf = int(img.shape[0] * self.opt.nerf_ratio), int(img.shape[1] * self.opt.nerf_ratio)
        
        img = img * msk_3c
        
        if self.opt.not_blur_nerf:
            img_nerf = cv2.resize(img, (W_nerf, H_nerf))#
            msk_nerf = cv2.resize(msk, (W_nerf, H_nerf))
        else:
            img = cv2.GaussianBlur(img,(9,9),cv2.BORDER_DEFAULT)            
            img_nerf = cv2.resize(img, (W_nerf, H_nerf))#, interpolation=cv2.INTER_AREA                
            msk_nerf = (img_nerf != 0)[...,0]
        
        
                    
        Cam_K_nerf = K.copy().astype(np.float32)
        Cam_K_nerf[:2] = Cam_K_nerf[:2] * self.opt.nerf_ratio
        
        _meta = {                
            'img_gen': img_gen * 2 - 1.0,
            'mask_gen': msk_gen,
            'Cam_K_gen': Cam_K_gen,
            'img_nerf': img_nerf,
            'mask_nerf': msk_nerf,
            'Cam_K_nerf': Cam_K_nerf
        }
        ret.update(_meta)
        
        index = frame_index
        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, betas, poses = self.prepare_input(
            i)

        if self.data_flag == "train":
            assert  len(self.ims) % len(self.view_id) == 0            
            frame_each_cam = int(len(self.ims) / len(self.view_id))
            time_lat_index = frame_index % frame_each_cam + 1
        else:
            time_lat_index = 0

        img_name = img_path.split('/')
        img_name = "%s_%s" % (img_name[-2], img_name[-1])

  
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        i = index // self.num_cams
 
   
        
        ret.update({
            "can_bounds": can_bounds,
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'frame_index': frame_index,
            'betas': betas, 
            'poses': poses,            
            'img_name': img_name,
            'dataset': self.short_dataset_name,
            'cam_ind': cam_ind,
            "phase": self.data_flag,
            "time_lat_index" : time_lat_index,
            "rgbd5_full_pose" : 0,

            "lr_depth": self.meta_dict["lr_depth"][frame_index][cam_ind],
            "full_uv": self.meta_dict["full_uv"][frame_index][cam_ind],
            "normal_3d": self.meta_dict["normal_3d"][frame_index]
        })
        
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': index,
            'index': 0,
            'Cam_R': Cam_R,
            'Cam_T': Cam_T,
            'smpl_trans': Th,
            'dataset_id': self.dataset_id
            
        }

        ret.update(meta)
        return ret

    def get_smpl_vts(self, index):

        if index >= 1: 
            return self.prepare_input(index)[0][:, :3,...]
        else: 
            return None

    def arah_get_mask(self, mask_in):
        mask = (mask_in != 0).astype(np.uint8)

        border = 5
        kernel = np.ones((border, border), np.uint8)
        mask_erode = cv2.erode(mask.copy(), kernel)
        mask_dilate = cv2.dilate(mask.copy(), kernel)
        mask[(mask_dilate - mask_erode) == 1] = 100

        return mask
    
    def __getitem__(self, index):
        ret = self.get_item_packed(index)
        current_smpl_pts = ret["feature"][:,:3,...]
        current_smpl_pts_global = ret["feature"][:,3:,...]

        frame_index = ret["frame_index"]

        if self.opt.motion_mode:

            velocity = self.opt.posenet_setup.velocity

            f_b = 1 #motion forward or backward

            if self.opt.motion.motion_point != 0:
                assert velocity == 1
                frame_index += self.opt.motion.motion_point

            elif self.opt.motion.motion_chain:
                motion_steps = self.opt.motion.motion_steps.split(" ")
                pre_step = int(motion_steps[index])
                ret.update({"pre_step": pre_step})
                frame_index += pre_step
                if pre_step > 0: f_b = -1

            next_smpl_pts = self.get_smpl_vts(frame_index + velocity)
            zero_motion = 0 * current_smpl_pts

            if self.opt.posenet_setup.c_velo:
                motion_seq = []
                for i in range(1, self.opt.posenet_setup.size_motion_window, 1):                    
                    motion_seq.append(self.get_smpl_vts(frame_index - f_b * i * velocity))

                if self.opt.posenet_setup.new_dynamics:
                    ret["feature"] = np.concatenate([current_smpl_pts] + [next_smpl_pts if next_smpl_pts is not None else zero_motion] + [m if m is not None else zero_motion for m in motion_seq]   + [ret["normal_3d"]] + [current_smpl_pts_global], 1)
                    return ret
                
        return ret
        
    def load_meta_data(self, data_root, split):
        meta_file = os.path.join(data_root, "meta.pkl")        
        with open(meta_file, 'rb') as f:
            self.meta_dict = pickle.load(f)

    def __len__(self):        
        return len(self.ims)
  
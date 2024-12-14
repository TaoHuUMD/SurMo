from pickle import TRUE
import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from Engine.th_utils.io.prints import *

is_debug_visualization=False

#generator, discriminator.
from Engine.th_utils.my_pytorch3d.smpl_util import SMPL_Util
from Engine.th_utils.animation.uv_generator import Index_UV_Generator
from Engine.th_utils.networks import networks

import cv2

class PosFeatureNet(nn.Module):
    def __init__(self, opt):
        
        super(PosFeatureNet, self).__init__()
        
        self.opt = opt

        self.uv_reso = opt.posenet_setup.uv_reso
        
        self.render_posmap = Index_UV_Generator(self.uv_reso, self.uv_reso, uv_type=opt.uv_type, data_dir="./data/asset/data/uv_sampler")
        
        self.vts_uv = self.render_posmap.get_vts_uv().cuda().permute(0,2,1)
        self.vts_uv.requires_grad = False
                
        self.smpl_util = SMPL_Util(gender = opt.gender, faces_uvs=self.render_posmap.faces_uvs, verts_uvs = self.render_posmap.verts_uvs, smpl_uv_vts = self.vts_uv)
        
        self.use_posmap = self.opt.use_posmap


        self.posenet_setup = opt.posenet_setup
        for key in self.posenet_setup:
            setattr(self, key, self.posenet_setup[key])


        if 'local_rank' in self.opt:            
            self.gpu_ids = [self.opt.local_rank]
        elif 'gpu_ids' in self.opt:
            self.gpu_ids = self.opt.gpu_ids
    
        self.criterionL2 = lambda x, y : torch.mean((x - y) ** 2)
        self.criterionL1 = torch.nn.L1Loss()
    
        self.add_net()
    
    def add_net(self):
        opt = self.opt
        self.uvdim = opt.posenet_setup.tex_latent_dim

        if self.use_posmap:

            bound_dim = opt.ngf

            self.netPos_input_nc = 3
            if self.c_velo:
                self.netPos_input_nc += 3

            if self.c_acce:
                self.netPos_input_nc += 3

            if self.c_traj:
                self.netPos_input_nc += 3

            if self.combine_pose_style:
                self.netPos_input_nc += self.uvdim

            netPos_output_nc = self.posenet_outdim

            if self.ab_c_v10 and self.ab_c_norm:
                self.netPos_input_nc = (1 + 10) * 3

            self.posNet = networks.define_PoseNet(self.netPos_input_nc, netPos_output_nc, bound_dim, opt.netG, self.posemap_down, self.posemap_resnet, opt.n_local_enhancers,
            opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
    
        if self.use_style and self.smooth_latent:
            self.smooth_latent_net = networks.define_uvlatent_smooth_net(input_nc = self.uvdim, output_nc = self.uvdim, ngf = self.uvdim)
                                
        if self.pred_normal_uv:
            self.conv_norm_net = networks.define_G(netPos_output_nc, 3, 64, opt.netG, 1, 2, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        

        if self.pred_pose_uv:
            outc = 3
            if self.pred_velocity_uv: outc = 6

            self.net_pred_pose = networks.define_G(netPos_output_nc, outc, 64, opt.netG, 1, 2, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        
        if self.pred_offset_uv:
            input_dim = netPos_output_nc
            if not self.combine_pose_style: #style 
                input_dim += self.uvdim
                    
            self.conv_offset_net = networks.define_G(input_dim, 3, 64, opt.netG, 1, 2, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)            
    
    def update_weight_loss(self):
        self.weights = {
            'posmap_feat': self.w_posmap_feat,

            'pred_pose_uv': self.w_pred_pose_uv,
            'pred_velocity_uv': self.w_pred_velocity_uv,
            'pred_normal_uv': self.w_pred_normal_uv,

            "pred_normal_uv_rot": self.w_pred_normal_uv * 0.5,
            "pred_pose_uv_rot": self.w_pred_pose_uv * 0.5,
            "pred_velocity_uv_rot": self.w_pred_velocity_uv * 0.5,

            'rot_normal': self.w_rot_normal,
            'rot_offset': self.w_rot_offset,            
                        
            'offset_normal_smoothes': self.w_offset_normal_smoothes,
            'offset_reg': self.w_offset_reg,

            'posmap_img': self.w_img_encoder_loss,
            'posmap_img_rot': self.w_img_encoder_loss_rot

        }

        self.losses = {
            'posmap_feat': 0,
        
            'rot_normal': 0,
            'rot_offset': 0,

            'pred_pose_uv': 0,
            'pred_velocity_uv': 0,
            'pred_normal_uv': 0,

            "pred_normal_uv_rot": 0,
            "pred_pose_uv_rot": 0,
            "pred_velocity_uv_rot": 0,
                        
            'offset_normal_smoothes': 0,
            'offset_reg': 0,

            'posmap_img': 0,
            'posmap_img_rot': 0
        }  
    
    def index_posmap_by_vts(self, feat, uvs):
        return self.render_posmap.index_posmap_by_vts(feat, uvs)
    

    def extrat_dynamics_from_motion_seq(self, motion_seq_):        
        b, n, c = motion_seq_.shape

        motion_seq = motion_seq_.split(3, 2)

        current_smpl_pts = motion_seq[0]
        next_smpl_pts = motion_seq[1]
        zero_motion = 0 * current_smpl_pts

        pre_vts_1 = motion_seq[2]
        pre_vts_2 = motion_seq[3]

        cur_vel = zero_motion if (not pre_vts_1.any()) else current_smpl_pts - pre_vts_1
        cur_acc = zero_motion if ((not pre_vts_2.any()) or (not pre_vts_1.any())) else current_smpl_pts + pre_vts_2 - 2 * pre_vts_1

        if not next_smpl_pts.any():
            next_pose = zero_motion
            next_vel = zero_motion
        else:
            next_pose = next_smpl_pts
            next_vel = next_pose - current_smpl_pts
        
        velocity = self.opt.posenet_setup.velocity

        assert self.opt.posenet_setup.size_motion_window - 1 == len(motion_seq[2:])
        seq_size = self.opt.posenet_setup.size_motion_window

        if self.opt.posenet_setup.c_traj:
            traj_vts = None
            cnt = 0
            for i, vts in enumerate(motion_seq[2:]):
                
                if vts is None: break

                vts = vts * (seq_size - i) * velocity
                cnt = cnt + (seq_size - i) * velocity
                traj_vts = traj_vts + vts if traj_vts is not None else vts
            if cnt: traj_vts /= cnt
            else: traj_vts = 0 * current_smpl_pts

            if self.opt.phase == "test":
                cur_vel /= self.opt.motion.infer_velocity
                cur_acc /= self.opt.motion.infer_velocity

            if self.opt.free_view_rot_smpl:
                cur_vel =  cur_acc = traj_vts = next_pose = next_vel = cur_vel * 0

            return torch.cat([current_smpl_pts, cur_vel, cur_acc, traj_vts, next_pose, next_vel], -1)


    def pred_normal_func(self, posnet_output, smpl_vertices, vts_uv):

        pred_normal_uv_map = self.conv_norm_net(posnet_output, vts_uv)
        
        pred_normal_uv_vts = self.render_posmap.index_posmap_by_vts(pred_normal_uv_map, vts_uv)        
        gt_normal_vts = self.smpl_util.get_normal(smpl_vertices[...,:3])

        return pred_normal_uv_map, self.criterionL2(gt_normal_vts.permute(0,2,1), pred_normal_uv_vts) #


    def pred_pose_func(self, posnet_output, pos_map_data, vts_uv):
        pred_next_step_uv = self.net_pred_pose(posnet_output)
        if self.ab_pred_pose_by_velocity:
            assert not self.pred_velocity_uv
            next_vel = pred_next_step_uv[:,:3,...]
            #print("**  ", next_vel.shape, pos_map_data["cur_pose"].shape)
            next_pose_uv = next_vel + pos_map_data["cur_pose"]
        else:
            next_pose_uv = pred_next_step_uv[:,:3,...]

        pose_loss, velo_loss = 0, 0

        if self.ab_sparse_pose:
            pose_in_3d = self.render_posmap.index_posmap_by_vts(torch.cat((pos_map_data["gt_next_pose"], next_pose_uv), 1), vts_uv)
            pose_loss = self.criterionL2(pose_in_3d[:,:3,...], pose_in_3d[:,3:6,...])
        else:
            if self.pred_current_state:
                pose_loss = self.criterionL2(pos_map_data["cur_pose"], pred_next_step_uv[:,:3,...])
            else:
                pose_loss = self.criterionL2(pos_map_data["gt_next_pose"], pred_next_step_uv[:,:3,...])

        if self.pred_velocity_uv:
            velo_loss = self.criterionL2(pos_map_data["gt_next_velocity"], pred_next_step_uv[:,3:6,...])

        if not pos_map_data["gt_next_pose"][0][0].any():
            pose_loss *= 0
            velo_loss *= 0
            
        if True: #modify visualization
            next_pose_uv = pred_next_step_uv[:,:3,...]

        return next_pose_uv, pose_loss, velo_loss
        pos_map_data.update({"pred_pose_uv": next_pose_uv})

    def get_posenet_input(self, pos_map, uvlatent, pose_latent):
        posnet_input = pos_map[:,:self.netPos_input_nc,...]
        if self.combine_pose_style:                        
            posnet_input = torch.cat((posnet_input, uvlatent), 1)
        elif pose_latent is not None:
            posnet_input = torch.cat((posnet_input, pose_latent), 1)
        return posnet_input

    def augment_dynamics(self, smpl_vertices_, uvlatent, pose_latent, pos_map_data, vts_uv):

        if self.rot_posmap or self.pred_normal_uv_rot or self.pred_pose_uv_rot:
            assert self.new_dynamics
            device = smpl_vertices_.device

            b, n, c = smpl_vertices_.shape
            
            num = smpl_vertices_.shape[-1] // 3
            R_list=[]
            for i in range(num):
                Rh = (np.random.rand(1,3) *2 - 1.0) * np.pi * self.posmap_rot_scale
                R = cv2.Rodrigues(Rh)[0].astype(np.float32)
                R = torch.from_numpy(R).float().to(device)[None, ...]
                R_list.append(R)
            
            if self.rot_all_same:
                allmat = smpl_vertices_.view(b, n, -1, 1, 3).matmul(R_list[0]).reshape(b, n, -1)
                allmat[...,3:6] =  smpl_vertices_[..., 3:6]
            elif self.only_rot_pose:
                poserot = smpl_vertices_[...,:3].matmul(R_list[0]).reshape(b, n, -1)
                allmat = torch.cat([poserot, smpl_vertices_[..., 3:]], -1)
            else:
                if num > 1: R_list[1] = torch.eye(3)[None,...].to(device)
                R_mat = torch.cat(R_list, 0).unsqueeze(0).unsqueeze(0).expand(b, n, num, 3, 3)
                allmat = smpl_vertices_.reshape(b, n, -1, 1, 3).matmul(R_mat).reshape(b, n, -1)
            
            rot_smpl_vertices = allmat
            rot_smpl_vertices = self.extrat_dynamics_from_motion_seq(rot_smpl_vertices)

            rot_pos_map = self.render_posmap(rot_smpl_vertices).permute(0,3,1,2)

            rot_posnet_input = self.get_posenet_input(rot_pos_map, uvlatent, pose_latent)
            rot_posnet_output = self.posNet(rot_posnet_input)

            if self.pred_pose_uv_rot:
                _, self.losses['pred_pose_uv_rot'], self.losses['pred_velocity_uv_rot'] = self.pred_pose_func(rot_posnet_output, pos_map_data, vts_uv)

        else: pass

    def forward(self, smpl_vertices_, poses, shapes, uvlatent, pose_latent = None, other_inputs = None):
        
        if smpl_vertices_.ndimension() < 3:
            smpl_vertices_ = smpl_vertices_[None,...]

        if self.new_dynamics: #
            smpl_vertices = self.extrat_dynamics_from_motion_seq(smpl_vertices_)
        else:
            smpl_vertices = smpl_vertices_

        batch_size = smpl_vertices.shape[0]
        device = smpl_vertices.device
        pred_offset_map = None
        
        self.update_weight_loss()
        if self.use_posmap:
            pos_map = self.render_posmap(smpl_vertices).permute(0,3,1,2)
            pos_map_data = {"cur_pose": pos_map[:,:3,...]}

            if self.opt.motion_mode:
                ofs = 3
                for item in ["c_velo", "c_acce", "c_traj"]:
                    if getattr(self.opt.posenet_setup, item):
                        pos_map_data.update({item: pos_map[:, ofs : ofs + 3,...]})
                        ofs += 3
                pos_map_data.update({
                    "gt_next_pose": pos_map[:, ofs : ofs + 3,...],
                    "gt_next_velocity": pos_map[:, ofs + 3: ofs + 6,...]
                })
                
            posnet_input = self.get_posenet_input(pos_map, uvlatent, pose_latent)

            posnet_output = self.posNet(posnet_input)

            vts_uv = self.vts_uv.detach().expand(batch_size,-1,-1).to(device)
                        
            if self.pred_normal_uv:
                pred_normal_uv_map, self.losses['pred_normal_uv'] = self.pred_normal_func(posnet_output, smpl_vertices, vts_uv)
                pos_map_data.update({"pred_normal_uv": pred_normal_uv_map if self.pred_normal_uv else None})
                 
            if self.pred_pose_uv:
                next_pose_uv, self.losses['pred_pose_uv'], self.losses['pred_velocity_uv'] = self.pred_pose_func(posnet_output, pos_map_data, vts_uv)

                v_ = next_pose_uv - pos_map_data["cur_pose"]
                pos_map_data.update({"pred_pose_uv": (v_) * 10})
     
            if self.pred_offset_uv: 
                pred_offset_map = None
                pred_offset_vts = torch.zeros(smpl_vertices.shape).to(smpl_vertices)                                
                offset = pred_offset_vts
                smpl_vertices_offseted = smpl_vertices + offset
                self.losses['offset_reg'] = 0
                            
            self.augment_dynamics(smpl_vertices_, uvlatent, pose_latent, pos_map_data, vts_uv)

        for item in self.losses.keys():
            self.losses[item] *= self.weights[item]


        visualize_result={"cur_pose": pos_map_data["cur_pose"]}

        if self.opt.motion_mode:

            visout_list = ["ab_c_v10", "c_velo", "c_acce", "c_traj", "pred_pose_uv", "ab_c_norm", "pred_normal_uv"]

            for item in visout_list:
                if getattr(self.opt.posenet_setup, item):
                    if item == "c_velo" or item == "c_acce":
                        visualize_result.update({item: pos_map_data[item] * 10})
                    else:
                        visualize_result.update({item: pos_map_data[item]})

        if self.combine_pose_style:
            return posnet_output, visualize_result, self.losses
        
        if uvlatent is None:
            return posnet_output, visualize_result, self.losses
        elif self.use_style:
            style_output = uvlatent
            if self.smooth_latent:
                style_output = self.smooth_latent_net(uvlatent)

            if self.use_posmap:
                if not self.ab_no_tex_latent:
                    output = torch.cat((posnet_output, style_output), 1)
                else: output = posnet_output

                if self.cat_pos_geo_tex:
                    output = torch.cat((pos_map, pose_latent, style_output), 1)

                return output, visualize_result, self.losses
            else:
                return style_output, visualize_result, None
                            
        else:
            return uvlatent, visualize_result, None
        

        
    
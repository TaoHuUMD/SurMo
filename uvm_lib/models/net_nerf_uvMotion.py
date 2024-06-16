from pickle import TRUE
from torch._C import device
from torch.autograd import grad
import torch.nn as nn
#from ..depend import spconv
import torch.nn.functional as F
import torch

import numpy as np

from uvm_lib.engine.thutil.networks import embedder

#generator, discriminator.
from uvm_lib.engine.thutil.my_pytorch3d.smpl_util import SMPL_Util

from uvm_lib.engine.thutil.animation.uv_generator import Index_UV_Generator

from uvm_lib.engine.thutil.networks.embedder import pose_embedder,pose_dim

is_debug = False

class HumanUVNerfMotion(nn.Module):
    def __init__(self, opt):
        
        super(HumanUVNerfMotion, self).__init__()
        
        self.opt = opt

        nerf_inputdim = opt.posenet_setup.posenet_outdim if self.opt.uv_2dplane else opt.posenet_setup.posenet_outdim // 3

        if self.opt.combine_pose_style:
            self.uvh_feat_dim = nerf_inputdim
        else:
            self.uvh_feat_dim = opt.posenet_setup.tex_latent_dim + nerf_inputdim
            if self.opt.posenet_setup.pred_texture_uv:
                self.uvh_feat_dim += self.opt.posenet_setup.pred_texture_dim

        if self.opt.debug_only_enc:
            self.uvh_feat_dim = 0

        self.output_rgb_dim = opt.motion.nerf_dim
        
        self.add_layer_density_color = opt.add_layer_density_color
        self.add_layer_geometry = opt.add_layer_geometry

        self.use_pose_cond = not opt.not_pose_cond
        
        self.input_dim = 0        
        self.actvn = nn.ReLU()
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)

        self.voxel_size = 0.005
        
        self.h_bound = [-1.2, 1.1]
        
        if 'local_rank' in self.opt:            
            self.gpu_ids = [self.opt.local_rank]
        elif 'gpu_ids' in self.opt:
            self.gpu_ids = self.opt.gpu_ids
        
        
        self.render_posmap = Index_UV_Generator(self.opt.posenet_setup.uv_reso, self.opt.posenet_setup.uv_reso, uv_type=self.opt.posenet_setup.uv_type, data_dir="../asset/data/uv_sampler")
        self.vts_uv = self.render_posmap.get_vts_uv().cuda().permute(0,2,1)
        self.vts_uv.requires_grad = False
                
        self.smpl_util = SMPL_Util(gpu_ids = self.gpu_ids, gender = self.opt.gender, faces_uvs=self.render_posmap.faces_uvs, verts_uvs = self.render_posmap.verts_uvs, smpl_uv_vts = self.vts_uv)
                        
        self.uvdim = self.opt.posenet_setup.tex_latent_dim
        self.uv_reso = self.opt.posenet_setup.uv_reso
                        
        self.add_nerf()
    
    def add_nerf(self):

        if self.opt.plus_uvh_enc: #yes.
            self.geo_fc_0 = nn.Conv1d(self.uvh_feat_dim + embedder.uvh_dim, 256, 1)
        else:
            self.geo_fc_0 = nn.Conv1d(self.uvh_feat_dim, 256, 1)

        self.geo_fc_1 = nn.Conv1d(256, 256, 1)

        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.view_fc_0 = nn.Conv1d(256 + self.uvh_feat_dim + embedder.view_dim, 256, 1)
        self.view_fc_1 = nn.Conv1d(256, 128, 1)
        self.rgb_fc = nn.Conv1d(128, self.output_rgb_dim, 1)

        #torch.nn.Softplus()
        
    def trans_to_uv(self, smpl_vertices, sampled_pts_smpl_space):    
        k = 1
        
        debug = False
        if debug:
            sampled_pts_smpl_space = (sampled_pts_smpl_space[0][0])[None,None,...]
       
        near_uv, h = self.smpl_util.get_nearest_pts_in_mesh_torch(smpl_vertices, sampled_pts_smpl_space, k=k, num_samples = self.opt.uvVol_smpl_pts)
        
        return near_uv, h
                        
    def extract_uvstyle_features(self):
        t = 0
    
    def get_posmap_loss(self):
        return self.posmap_loss
    
    def gen_UVlatent(self, batch, input_latent):
        
        sampled_pts_smpl_space = batch["sampled_pts_smpl"]
        smpl_vertices = batch["smpl_vertices"]
        
        device = sampled_pts_smpl_space.device
        
        near_uv, h_pred = self.trans_to_uv(smpl_vertices, sampled_pts_smpl_space)
                    
        batch_size = near_uv.shape[0]
        
        b,c,h,w = input_latent.shape

        if self.opt.combine_pose_style:
            uvh_plane = input_latent
        else:#style
            uvh_plane = input_latent[:,:self.opt.posenet_setup.posenet_outdim, ...]
            style_uv = input_latent[:, self.opt.posenet_setup.posenet_outdim: , ...]
            c -= self.opt.posenet_setup.tex_latent_dim

        if self.opt.uv_2dplane:
            pts_huv = torch.cat((h_pred, near_uv), 2)
            
            input = uvh_plane
            if not self.opt.combine_pose_style:
                input = torch.cat((input, style_uv), 1)

            fused_feat = self.render_posmap.index_posmap_by_vts(input, near_uv)

            huvlat = embedder.uvh_embedder(pts_huv).permute(0,2,1)
            return fused_feat, huvlat, near_uv

        else:
            
            self.opt.motion.ab_uvh_plane_c = self.opt.posenet_setup.posenet_outdim // 3
            c = self.opt.posenet_setup.posenet_outdim
            uh_dim = self.opt.motion.ab_uvh_plane_c
            vh_dim = uh_dim
            uv_dim = c - uh_dim - vh_dim

            uv_plane = uvh_plane[:, :uv_dim, ...]
            uh_plane = uvh_plane[:, uv_dim: uv_dim + uh_dim, ...]
            hv_plane = uvh_plane[:, uv_dim + uh_dim:, ...]

            self.depth_bound = [-0.1, 0.1]

            pts_huv = torch.cat((h_pred, near_uv), 2)
            huvlat = embedder.uvh_embedder(pts_huv).permute(0,2,1)


            h_pred *= 1/self.depth_bound[1]#-1,1
            h_pred = (h_pred + 1)/2 #[0, 1]
            if is_debug: print('***   ', near_uv.shape, near_uv[:,[0],...].shape, h_pred.shape)

            ##B, N, C
            uh = torch.cat((near_uv[..., [0]], h_pred), -1)
            hv = torch.cat((h_pred, near_uv[..., [1]]), -1)

            uv_feature = self.render_posmap.index_posmap_by_vts(uv_plane, near_uv)
            uh_feature = self.render_posmap.index_posmap_by_vts(uh_plane, uh)
            hv_feature = self.render_posmap.index_posmap_by_vts(hv_plane, hv)

            if self.opt.combine_pose_style:
                fused_feat = torch.cat([uv_feature.unsqueeze(1), uh_feature.unsqueeze(1), hv_feature.unsqueeze(1)], 1).mean(1)
            else:
                fused_feat = torch.cat([uv_feature.unsqueeze(1), uh_feature.unsqueeze(1), hv_feature.unsqueeze(1)], 1).mean(1)
                style_uv_feat = self.render_posmap.index_posmap_by_vts(style_uv, near_uv)
                fused_feat = torch.cat((style_uv_feat, fused_feat), 1) 

            return fused_feat, huvlat, near_uv

                 
    def forward(self, batch, uv_latent, only_density=False):

        viewdir = batch['view_dir']

        if self.opt.uv_2dplane:
            uv_feat, uvh_encoding, uv_coord = self.gen_UVlatent(batch, uv_latent)
            nerf_input = uv_feat
            if self.opt.plus_uvh_enc:
                nerf_input = torch.cat((nerf_input, uvh_encoding), 1)
            uvh_feat = uv_feat
        else:
            uvh_feat, uvh_encoding, uv_coord= self.gen_UVlatent(batch, uv_latent)            
            nerf_input = uvh_feat

            if self.opt.plus_uvh_enc: #yes.
                nerf_input = torch.cat((uvh_feat, uvh_encoding), 1) 

        if self.opt.debug_only_enc:
            uvh_feat = None
            nerf_input = uvh_encoding

        net = self.geo_fc_0(nerf_input)
        net = self.actvn(net)

        net = self.geo_fc_1(net)
        net = self.actvn(net)

        alpha = self.alpha_fc(net)

        if self.opt.vrnr_mesh_demo:
            return alpha.transpose(1, 2)
        

        viewdir = viewdir.transpose(1, 2)

        feat_view = torch.cat((net, viewdir), 1)
        if uvh_feat is not None:
            feat_view = torch.cat((feat_view, uvh_feat), 1)
            
        net = self.view_fc_0(feat_view)
        net = self.actvn(net)

        net = self.view_fc_1(net)
        net = self.actvn(net)
        
        rgb = self.rgb_fc(net)

        raw = torch.cat((alpha, rgb), dim=1)
        raw = raw.transpose(1, 2)

        if self.opt.learn_uv:
            return torch.cat((raw, uv_coord), dim=-1)        
        else:
            return raw
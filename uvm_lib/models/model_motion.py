import argparse
import numpy as np
import torch
import os

from .base_model import BaseModel

from Engine.th_utils import util

import sys
sys.path.append("..")
from Engine.th_utils.networks import networks
from Engine.th_utils.networks import losses

from Engine.th_utils.files import get_croped_size
from Engine.th_utils.util import split_dict_batch

from Engine.th_utils.networks.util.image_pool import ImagePool

from Engine.th_utils.io.prints import *

import torch.nn as nn

from Engine.th_utils.io.prints import *

from Engine.th_utils.animation.util import map_normalized_dp_to_tex_pytorch

from Engine.th_utils.networks.net_utils import load_model, save_model

from Engine.th_utils.animation.uv_generator import Index_UV_Generator

import cv2

from Engine.th_utils.num import mask_4d_img

import torchvision.transforms 
import torchvision.transforms as T

from .net_smooth import GeomConvLayers

from .net_nerf_uvMotion import HumanUVNerfMotion as netNeRF
from .net_PosFeature import PosFeatureNet
from .nerf_render import NerfRender

class Model(BaseModel):
    def name(self):
        return 'Model'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, tex):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake, tex), flags) if f]

        return loss_filter

    def initialize(self, opt):
        
        BaseModel.initialize(self, opt)
         
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        
        self.visual_names = []
        self.visual_names += ['nr_uv', 'real_image_t', 'fake_image']
        if not self.opt.uv2pix and not self.opt.no_encoder:
            self.visual_names += ['rendered_viz']
                
        if opt.debug_out:
            self.visual_names += ['nr_depth', 'nerf_depth', 'nerf_dia_depth']
                            
        self.model_names = []

        self.G_model_names, self.D_model_names = [], []
        
        self.uv_reso = self.opt.posenet_setup.uv_reso
        self.uvdim = self.opt.posenet_setup.tex_latent_dim
        self.uv_type = self.opt.posenet_setup.uv_type

        self.posFeatNet = PosFeatureNet(self.opt)

        self.render_posmap = Index_UV_Generator(self.uv_reso, self.uv_reso, uv_type=self.uv_type, data_dir="./data/asset/data/uv_sampler")
        self.vts_uv = self.render_posmap.get_vts_uv().cuda().permute(0,2,1)
        self.vts_uv.requires_grad = False
                        
        if self.opt.use_nerf:
            self.nerfRender = NerfRender(self.opt, None, opt.phase, netNerf = netNeRF(self.opt))
            self.nerfRender.apply(networks.weights_init)


        self.uv_lat, self.pose_lat, self.pose_lat_2d, self.uv_lat_2d, self.posed_uv_lat = None, None, None, None, None
        
        self.build_nr_net()
           
        if self.isTrain:
            self.optimization_setup()
                            
        self.actvn = nn.ReLU()    
        
        self.first_round = int(0)
        self.static_count = 0

        self.loss_contrastive = 0

        self.loss_names = []
        self.loss_names += ['G_GAN', 'D_real', 'D_fake', 'Face']

        if opt.use_posmap:
            self.posmap_loss_out = []
            self.posmap_vis = ["cur_pose"]
            
            for item in ["ab_c_v10", "c_velo", "c_acce", "c_traj", "pred_pose_uv", "pred_texture_uv", "ab_c_norm", "pred_normal_uv"]:
                if getattr(self.opt.posenet_setup, item):
                    self.posmap_vis += [item]
        
            self.loss_names += ['posmap_feat']
            if self.opt.posenet_setup.pred_normal_uv:
                self.posmap_loss_out += ['pred_normal_uv', 'rot_normal']

            for item in ["pred_pose_uv", "pred_texture_uv", "pred_normal_uv_rot", "pred_pose_uv_rot", "pred_velocity_uv_rot"]:
                if getattr(self.opt.posenet_setup, item):
                    self.posmap_loss_out += [item]

            self.loss_names += self.posmap_loss_out
            self.visual_names += ["cur_pose", "pred_normal_uv"]

        self.loss_names += ['G_GAN', 'D_real', 'D_fake', 'Face', 'nerf_rec', 'G_L1', 'G_VGG', 'G_GAN_Feat']

        self.visual_names += ['real_image_t', 'fake_image']
        if not self.isTrain:
            if self.opt.demo_all:
                self.visual_names += ['rendered_viz']

        if self.opt.motion.ab_nerf_rec:
            self.loss_names += ['nerf_rec']
            self.visual_names += ['nerf_rec']
            
        if self.opt.pred_nerf_depth:
            self.visual_names += ['nr_nerf_depth']
                                
        if self.opt.nr_pred_mask:
            self.loss_names += ['G_Mask']
            self.visual_names += ['G_Mask']

        if self.opt.vrnr_mesh_demo:
            self.visual_names = []
            
        self.loss_lr = self.opt.lr
        
        self.criterionL2 = lambda x, y : torch.mean((x - y) ** 2)
        self.criterionL1 = torch.nn.L1Loss()
        
        self.nr_weight = 1.0
        if self.opt.no_encoder:
            self.nr_weight = 0
                        
        self.is_evaluate = False
        self.is_vis_lat = False

        self.visual_names += ['visOutput']

        if self.opt.is_inference:
            self.test_visual_names = self.visual_names
            self.visual_names = ['visOutput']

        #1, h, w
        self.template_uv_mask = self.render_posmap.get_uv_mask()[None, ...].detach()

    def build_nr_net(self):
         
        opt = self.opt

        self.img_gen_h, self.img_gen_w, self.img_nerf_h, self.img_nerf_w = get_croped_size(self.opt)

        params = []

        self.net_G_list = []
        self.net_D_list = []

        uv_latent_code = nn.Embedding(self.uv_reso*self.uv_reso, self.uvdim)
        self.uv_latent_code = networks.net_to_GPU(uv_latent_code, self.gpu_ids)
        self.G_model_names.append('uv_latent_code')
        params += list(self.uv_latent_code.parameters())

        self.pose_latent_code = nn.Embedding(self.uv_reso*self.uv_reso, self.uvdim).cuda()
        self.G_model_names.append('pose_latent_code')
        params += list(self.pose_latent_code.parameters())

        self.G_model_names.append('netLatSmooth')
        self.netLatSmooth = GeomConvLayers(input_nc=self.uvdim, output_nc=self.uvdim)
        self.netLatSmooth = networks.net_to_GPU(self.netLatSmooth, self.gpu_ids)
        params += list(self.netLatSmooth.parameters())

        #input dim for supp. network
        input_dim = self.opt.motion.nerf_dim
        
        input_dim += self.opt.posenet_setup.tex_latent_dim        
        tex_lat_dim = self.opt.posenet_setup.tex_latent_dim

        #resize style latent
        if self.opt.is_pad_img: 
            self.netResize2DStyle = networks.define_ResizeNet(tex_lat_dim, tex_lat_dim, org_size = (self.img_gen_h, self.img_gen_h),  tgt_size = (self.img_nerf_h, self.img_nerf_h))
        else:
            self.netResize2DStyle = networks.define_ResizeNet(tex_lat_dim, tex_lat_dim, org_size = (self.img_gen_h, self.img_gen_w),  tgt_size = (self.img_nerf_h, self.img_nerf_w))

        self.model_names.append('netResize2DStyle')
        params += list(self.netResize2DStyle.parameters())

        if self.opt.learn_uv:
            input_dim += 2

        if self.opt.nr_insert_smpl_uv: input_dim += 2

        input_dim += 1 #nerf depth included

        if self.opt.superreso=="LightSup":
            printb('upsampling factor ', self.opt.gen_ratio // self.opt.nerf_ratio)
            out_list = ["rgb"]
            output_nc = 3
            if self.opt.nr_pred_mask:
                output_nc += 1
                out_list.append("mask")
            self.netSup = networks.define_supreso_lightweight(input_nc = input_dim, output_nc = output_nc, factor = int(self.opt.gen_ratio // self.opt.nerf_ratio), out_list=out_list)  
        else:
            raise NotImplementedError
        
        self.netSup = networks.net_to_GPU(self.netSup, self.gpu_ids)
        self.G_model_names.append('netSup')
        params += list(self.netSup.parameters())

        D_params = []
        
        indim = 3
        if self.opt.motion.ab_Ddual: indim += 3 
        if self.opt.motion.ab_D_pose: indim += 2

        if self.opt.motion.use_org_discrim:
            use_sigmoid = self.opt.no_lsgan if self.isTrain else False
            self.netD_img = networks.define_D(indim, self.opt.ndf, self.opt.n_layers_D, self.opt.norm, use_sigmoid, self.opt.num_D, not self.opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        else:
            raise NotImplementedError()
        
        self.netD_img = networks.net_to_GPU(self.netD_img, self.gpu_ids)
        self.model_names.append('D_img')
        D_params += list(self.netD_img.parameters())

        
        self.G_model_names.append('posFeatNet')
        self.G_model_names.append('nerfRender')

        params += list(self.posFeatNet.parameters())                    
        if opt.use_nerf: params += list(self.nerfRender.parameters())
        
        self.old_lr = opt.lr
        self.old_lr_D = opt.D_lr

        self.nerf_weight = self.opt.hypara.w_nerf_rec 

        # parameters of TexG belongs to net G
        self.optimizer_G = torch.optim.Adam(params, lr = self.old_lr, betas=(opt.beta1, 0.999))
                
        # optimizer D
        if len(D_params):
            self.optimizer_D = torch.optim.Adam(D_params, lr = self.old_lr_D, betas=(opt.beta1, 0.999))

        if self.opt.verbose:
            print('---------- Networks initialized -------------')
    
    def requires_grad_D(self, flag):
        for netname in self.D_model_names:        
            networks.requires_grad(getattr(self, netname), flag)
            
    def check_net(self, net):
        printb(net)
        networks.check_trainable_params(net)

    def requires_grad_G(self, flag):        
        for netname in self.G_model_names:        
            networks.requires_grad(getattr(self, netname), flag)


    def optimization_setup(self):
        opt = self.opt
                
        if opt.niter_fix_global > 0:
            import sys
            if sys.version_info >= (3, 0):
                finetune_list = set()
            else:
                from sets import Set
                finetune_list = Set()

            params_dict = dict(self.netG.named_parameters())
            params = []
            for key, value in params_dict.items():
                if key.startswith('model' + str(opt.n_local_enhancers)):
                    params += [value]
                    finetune_list.add(key.split('.')[0])
            print(
                '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
            print('The layers that are finetuned are ', sorted(finetune_list))

    def define_loss(self):

        if not self.opt.no_vgg_loss:
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionVGG.to(self.opt.gpu_ids[0])
            
        self.fake_pool = ImagePool(self.opt.pool_size)

        self.criterionGAN = networks.GANLoss(use_lsgan = not self.opt.no_lsgan, tensor=self.Tensor)            
        self.criterionContrastive = losses.ContrastiveLoss()

        self.loss_filter = self.init_loss_filter(not self.opt.no_ganFeat_loss, not self.opt.no_vgg_loss)
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionTexture = torch.nn.L1Loss()

        self.criterionTV = losses.TotalVariation()

        self.criterionMask = nn.BCEWithLogitsLoss()

    
    def set_dataset_id(self, batch):
        is_evaluate = (batch['state']=='evaluate')
        self.dataset_id = self.opt.dataset_id[0][batch['dataset'][0]] 
        if is_evaluate and self.opt.nerf.swap_tex:
            self.dataset_id = self.opt.dataset_id_swap[0][batch['dataset'][0]]

    def input_tocuda(self, batch):
        for k in batch:
            if not isinstance(batch[k][0], str) and (not isinstance(batch[k], np.int)):
                batch[k] = batch[k].cuda()
    
    def pts_to_world(self, pts, batch):

        if "is_global_pose" in batch.keys() and batch["is_global_pose"]==1: return pts

        """transform pts from the smpl coordinate to world coordinate"""
        Th = batch['Th'][:, None]
        R = batch['R']

        sh = pts.shape        
        
        iv_R = torch.inverse(R)
        world_pts = torch.matmul(pts.view(sh[0], -1, 3), iv_R)

        if self.opt.use_smpl_scaling:
            world_pts *= batch["scaling"]

        world_pts += Th.squeeze(2)

        world_pts = world_pts.view(*sh)
        return world_pts
  

    def update_weights(self, epoch):
        if self.opt.hypara.adjust_gamma:
            if self.opt.hypara.w_D_grad >= self.opt.hypara.gamma_lb and epoch % 100 == 0 and epoch != 0:
                self.opt.hypara.w_D_grad = self.opt.hypara.w_D_grad // 2

        if self.opt.nerf_decay:
            if epoch < 30: return
            if epoch <= 60: 
                self.nerf_weight = self.opt.w_nerf_rec / 2
                return
            
            decay = self.opt.w_nerf_rec * 0.25 / (self.opt.niter + self.opt.niter_decay)
            self.nerf_weight = self.opt.w_nerf_rec * 0.25 - decay * (epoch - 60)
        
        return       

    def warp_img_to_uv(self, gen_img, posed_uv):
        
        gen_uv_img, gen_uv_mask = map_normalized_dp_to_tex_pytorch(gen_img[0].permute(1,2,0), posed_uv[0].permute(1,2,0), self.uv_reso, fillconst=0)
        gen_uv_img = gen_uv_img.clone().permute(2,0,1)
        gen_uv_img = T.functional.vflip(gen_uv_img)[None,...]
        gen_uv_mask = T.functional.vflip(gen_uv_mask)[None,...]

        return gen_uv_img, gen_uv_mask
    
    def pad_img_w(self, input, padding=0):
        #h>w        
        device = input.device
        b,c,h,w = input.shape
        if h==w: return input
        tmp = torch.zeros((b, c, h, h)).to(device).requires_grad_(input.requires_grad) + padding
        tmp[:,:,:,:w] = input
        return tmp
    
    def forward_G(self, batch, infer=False):
        
        if not hasattr(self, "criterionContrastive"):
            self.define_loss()

        self.input_tocuda(batch)

        batch_size = batch["img_gen"].shape[0]

        poses = batch["poses"]
        shapes = batch["betas"]

        device = batch["img_gen"].device

        real_image_t = batch["img_gen"]
        real_mask = batch["mask_gen"]

        batch["img_gen"] = batch["img_gen"].permute(0,3,1,2)
        batch["mask_gen"] = batch["mask_gen"].float()[...,None].permute(0,3,1,2)
        
        self.loss_G_L1 = 0

        self.dataset_id = 0      
        self.frame_index = batch["img_name"]

        pose_latent = None

        if self.opt.motion_mode:
            smpl_vertices = batch["feature"][...,:-3].clone()
        else:
            smpl_vertices = batch["feature"][...,:3].clone()


        assert smpl_vertices.ndimension() == 3
        
        is_mask = False 
        is_normal = False
        is_depth = False
        
        posed_uv = batch["posed_uv"]

        if is_depth:
            full_depth = posed_depth.clone()
            posed_depth = posed_depth[...,0].unsqueeze(-1).permute(0,3,1,2)

        posed_uv[:,[0,1],...] = posed_uv[:,[1,0],...]
                
        if (not infer) and self.opt.aug_nr and np.random.rand() > 0.6:
            if self.opt.small_rot: scale = 20
            else: scale = 180

            r = np.random.rand() * 2 - 1.0
            transforms = T.Compose([
                torchvision.transforms.RandomRotation(degrees=(r*scale, r*scale), fill = 0)
                ]
            )
            transforms_img = T.Compose([
                torchvision.transforms.RandomRotation(degrees=(r*scale, r*scale), fill = 1 if self.opt.white_bg else -1)
                ]
            )
        else: 
            transforms = T.Compose([])
            transforms_img = T.Compose([])
        
        uvlatent = self.uv_latent_code(torch.arange(0, self.uv_reso*self.uv_reso).to(device)).view(1, self.uv_reso, self.uv_reso, -1).permute(0,3,1,2)
        uvlatent = uvlatent.expand(batch_size, -1, -1, -1)

        uvlatent = self.netLatSmooth(uvlatent)

        if not self.opt.motion.ab_cond_uv_latent: uvlatent *= 0
        
        uvmask = self.template_uv_mask.unsqueeze(1).repeat(1, uvlatent.shape[1], 1, 1).to(device)
        uvlatent = uvlatent * uvmask

        pose_latent = None
        posnet_output, inter_posenet_result, posmap_loss = self.posFeatNet(smpl_vertices, poses, shapes, None, pose_latent, other_inputs = {"full_uv_img": batch["full_uv_img"]} if self.opt.posenet_setup.pred_texture_uv else None)
        posnet_output = torch.cat((posnet_output, uvlatent), 1)

        for key in self.posmap_vis:
            setattr(self, key, inter_posenet_result[key].detach())
        
        if self.opt.use_posmap:
            self.posmap_loss = 0
            for key in self.posmap_loss_out:
                setattr(self, "loss_%s" % key, posmap_loss[key])
                self.posmap_loss += posmap_loss[key]

        real_image_t = transforms_img(batch["img_gen"])
        real_mask = transforms(batch["mask_gen"])
        posed_uv = transforms(posed_uv)

        if self.opt.is_crop:
            #b,c,h,w
            b, c, h, w = posed_uv.shape
                            
            posed_uv = posed_uv[:,:,self.opt.y0_crop : h - self.opt.y1_crop, self.opt.x0_crop : w - self.opt.x1_crop]
            if is_depth: posed_depth = posed_depth[:,:,self.opt.y0_crop : h - self.opt.y1_crop, self.opt.x0_crop : w - self.opt.x1_crop]
            
            if posed_normal is not None: posed_normal = posed_normal[:,:,self.opt.y0_crop : h - self.opt.y1_crop, self.opt.x0_crop : w - self.opt.x1_crop]
        
            if posed_mask is not None: posed_mask = posed_mask[:,:,self.opt.y0_crop : h - self.opt.y1_crop, self.opt.x0_crop : w - self.opt.x1_crop]
        
            b, c, h, w = real_image_t.shape                
            real_image_t = real_image_t[:,:,self.opt.y0_crop : h - self.opt.y1_crop, self.opt.x0_crop : w - self.opt.x1_crop]
            real_mask = real_mask[:,:,self.opt.y0_crop : h - self.opt.y1_crop, self.opt.x0_crop : w - self.opt.x1_crop]
        
        if self.opt.is_pad_img:            
            posed_uv = self.pad_img_w(posed_uv, padding=0)
            real_image_t = self.pad_img_w(real_image_t, padding=1)
                
        if self.opt.vrnr:

            nerf_input_latent = posnet_output
                
            if self.opt.vrnr_mesh_demo:                    
                mesh_dict = self.nerfRender.render_nerf(batch, nerf_input_latent)
                self.vrnr_mesh = mesh_dict['mesh']
                return 

            if self.opt.use_sdf_render:
                self.loss_names += ["g_eikonal", "g_minimal_surface"]

            self.loss_g_eikonal = 0
            self.loss_g_minimal_surface = 0

            nerf_img = []
            nerf_depth = []
            batch_id = 0

            for one_batch in split_dict_batch(batch, 1):
                nerf_img_, nerf_depth_, sdf_, eikonal_term_ = self.nerfRender.render_nerf(one_batch, nerf_input_latent[[batch_id]])
                batch_id += 1
                 
                if nerf_img_ is None: return -1

                nerf_img.append(nerf_img_[None,...])
                nerf_depth.append(nerf_depth_[None,...])
            
            self.loss_g_eikonal /= batch_size
            self.loss_g_minimal_surface /= batch_size
            nerf_img = torch.cat((nerf_img), 0)
            nerf_depth= torch.cat((nerf_depth), 0)
        
            batch["nerf_pred_img"] = nerf_img.detach()
            batch["nerf_pred_depth"] = nerf_depth.detach()
            self.nerf_gt = batch["img_nerf"].detach()
                
            nerf_pred_img = (nerf_img.permute(0,3,1,2) * 2 - 1.0)
            nerf_latent = nerf_pred_img

            nerf_pred_depth = nerf_depth.permute(0,3,1,2)

#           
            nerf_latent = torch.cat((nerf_latent, nerf_pred_depth), 1)

            nerf_latent = transforms_img(nerf_latent)            
                        
            posed_uv_latent = None

            posed_uv_latent = self.render_posmap.index_posmap_by_uvmap(uvlatent, posed_uv)

            smpl_mask = torch.unsqueeze(torch.any(posed_uv > 0, 1), 1)            
            posed_uv = mask_4d_img(posed_uv, smpl_mask)

            if self.opt.is_crop: #aist crop
                b, c, h, w = real_image_t.shape    
                
                _, _, hn, wn = nerf_latent.shape

                factor = self.img_gen_h // self.img_nerf_h

                nerf_latent = nerf_latent[:,:, int(self.opt.y0_crop // factor) : hn - (self.opt.y1_crop // factor), (self.opt.x0_crop // factor) : wn - (self.opt.x1_crop // factor)]

            if self.opt.is_pad_img:
                nerf_latent = self.pad_img_w(nerf_latent, padding=1)
 
            if posed_uv_latent is not None:
                tgt_h, tgt_w = self.img_nerf_h, self.img_nerf_w
                if self.opt.is_pad_img:
                    tgt_w = tgt_h
                
                if hasattr(self, "netResize2DStyle"):
                    posed_uv_latent = self.netResize2DStyle(posed_uv_latent)
                else:
                    posed_uv_latent = torch.nn.functional.interpolate(posed_uv_latent, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)#, antialias=True

                self.visual_names += ["tex_feat"]
            
            if self.opt.nr_insert_smpl_uv:
                 uvlow = torch.nn.functional.interpolate(posed_uv, size=(self.img_nerf_h, self.img_nerf_w), mode='bilinear', align_corners=False)#, antialias=True
                 nerf_latent = torch.cat((nerf_latent, uvlow), 1)

            nerf_latent = torch.cat((nerf_latent, posed_uv_latent), 1)
    
            if self.opt.superreso=="LightSup":
                fake_image = self.netSup(nerf_latent)

            else:
                raise NotImplementedError

            if self.opt.is_pad_img:
                fake_image = self.pad_img_w(fake_image, padding=1)

        if self.is_vis_lat:
            self.uv_lat = uvlatent.detach()
            self.pose_lat = pose_latent.detach()
                    
        self.rendered_viz = None
        
        self.fake_image = fake_image[:,:3,:, :].detach()
        self.real_image_t = real_image_t[:,:3,:, :].detach()
        self.nr_uv = posed_uv.detach()
        self.nerf_rec = nerf_latent[:,:3,...].detach()

        if posed_uv_latent is not None:
            self.tex_feat = posed_uv_latent[:,:3,...].detach()
        else: self.tex_feat = None

        if self.opt.nr_pred_mask:
            self.G_Mask = fake_image[:, 3:4, ...].detach()

        if infer: return 1

        #*********************
        #calculate loss
        #masked generated image to calculate loss
        if self.opt.masked_loss:
            mask_full = real_mask.expand(-1, 3, -1, -1)
            mask_down = torch.nn.functional.interpolate(real_mask, size=(self.img_nerf_h, self.img_nerf_w), mode='bilinear', align_corners=False).expand(-1, 3, -1, -1)#, antialias=True
        else:
            mask_full, mask_down = None, None

        fake_img3 = fake_image[:,:3,...]
        real_img3 = real_image_t[:,:3,...]
        nerf_img3 = nerf_latent[:,:3,...]

        #
        self.loss_tex=0
        self.loss_Rerender = 0
        self.loss_nr_pred_mask = 0

        if infer: return 1

        # Face loss
        self.loss_Face = 0

        self.loss_G_L1 = self.criterionL1(fake_image[:,:3,...] * (mask_full if mask_full is not None else 1), real_image_t[:,:3,...] * (mask_full if mask_full is not None else 1))
    
        gen_h, gen_w = self.img_gen_h, self.img_gen_w
        nerf_h, nerf_w = self.img_nerf_h, self.img_nerf_w


        real_downsample = torch.nn.functional.interpolate(real_img3, size=(nerf_h, nerf_w), mode='bilinear', align_corners=False)#, antialias=True

        #nerf_rec
        self.loss_nerf_rec = self.criterionL1(real_downsample * (mask_down if mask_down is not None else 1), nerf_img3 * (mask_down if mask_down is not None else 1))


        if self.opt.motion.ab_Ddual: #updample NeRF
            
            nerf_upsample = torch.nn.functional.interpolate(nerf_img3, size=(gen_h, gen_w), mode='bilinear', align_corners=False)#, antialias=True

            D_input_img_fake = torch.cat((fake_img3, nerf_upsample), 1)

            real_uped = torch.nn.functional.interpolate(real_downsample, size=(gen_h, gen_w), mode='bilinear', align_corners=False)#, antialias=True
            D_input_img_real = torch.cat((real_img3, real_uped), 1)

            if self.opt.motion.ab_D_pose:
                D_input_img_real = torch.cat((D_input_img_real, posed_uv), 1)
                D_input_img_fake = torch.cat((D_input_img_fake, posed_uv), 1)
        else:
            D_input_img_fake = fake_img3
            D_input_img_real = real_img3
            if self.opt.motion.ab_D_pose:
                D_input_img_real = torch.cat((D_input_img_real, posed_uv), 1)
                D_input_img_fake = torch.cat((D_input_img_fake, posed_uv), 1)
        
        pred_fake_0 = self.netD_img(D_input_img_fake)
        if self.opt.motion.use_org_gan_loss:
            ggan = self.criterionGAN(pred_fake_0, True)
        else:    
            ggan = networks.g_nonsaturating_loss(pred_fake_0) if not infer else 0
        self.loss_G_GAN = ggan
    
        # GAN feature matching loss
        self.loss_G_GAN_Feat = 0

        if not self.opt.no_ganFeat_loss:
            pred_real = self.netD_img(D_input_img_real)
        
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_0[i]) - 1):
                    self.loss_G_GAN_Feat += D_weights * feat_weights * \
                                            self.criterionFeat(pred_fake_0[i][j],
                                                               pred_real[i][j].detach())

        # VGG feature matching loss
        self.loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            self.loss_G_VGG = self.criterionVGG(fake_image[:,:3,...], real_image_t[:,:3,...] )


        #D_img, real, fake
        self.D_input_img_fake_detach = D_input_img_fake.detach()
        self.D_input_img_real_detach = D_input_img_real.detach()

        return 1
    
    def init_setup(self):
        self.D_input_img_fake_detach = []
        self.D_input_img_real_detach = []
        self.D_input_uv_fake_detach = []
        self.D_input_uv_real_detach = []

    def forward_D(self):
        
        self.loss_D_real = 0
        self.loss_D_fake = 0
        self.loss_D_grad_penalty = 0

        D_input_img_fake = (self.D_input_img_fake_detach)
        D_input_img_real = (self.D_input_img_real_detach)

        D_input_img_fake_tmp = D_input_img_fake
        D_input_img_real_tmp = D_input_img_real.requires_grad_(True)

        fake_query = self.fake_pool.query(D_input_img_fake_tmp)
        pred_fake = self.netD_img(fake_query)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real Detection and Loss
        pred_real = self.netD_img(D_input_img_real_tmp)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_grad = 0
            
    def discriminate(self, net, input, use_pool=False):

        if use_pool:
            fake_query = self.fake_pool.query(input)
            return net.forward(fake_query)
        else:
            return net.forward(input)

    def evaluate(self, data):
        self.is_evaluate = True
        return self.forward_G(data, True)

    def forward_calc_loss(self, data, is_opt_G = False, is_opt_D = False, is_opt_Dr1 = False):
 
        if is_opt_G:

            loss_weight = {
           
                "G_GAN": self.opt.hypara.w_G_GAN,
                "Face": self.opt.hypara.w_Face,

                "G_L1": self.opt.hypara.w_G_L1,
                "G_Mask": self.opt.hypara.w_G_Mask,
                "G_GAN_Feat": self.opt.hypara.w_G_feat,
                "G_VGG": self.opt.hypara.w_G_feat,

                "nerf_rec": self.nerf_weight,
                "posmap": self.opt.hypara.w_posmap
            }
        
            for l in self.loss_names:
                setattr(self, "loss_%s" % l, 0)

            self.forward_G(data)

            loss_G = 0
            self.loss_G_GAN *= loss_weight["G_GAN"]
            self.loss_Face *= loss_weight["Face"]

            self.loss_G_L1 *= loss_weight["G_L1"]
            self.loss_G_GAN_Feat *= loss_weight["G_GAN_Feat"]
            self.loss_G_VGG *= loss_weight["G_VGG"]

            loss_G += self.loss_G_GAN + self.loss_Face + self.loss_G_L1 + self.loss_G_GAN_Feat + self.loss_G_VGG

            if self.opt.nr_pred_mask:
                self.loss_G_Mask *= loss_weight["G_Mask"]

            if self.opt.use_posmap: 
                self.posmap_loss *= loss_weight["posmap"]
                loss_G += self.posmap_loss

            loss_G += self.loss_g_minimal_surface + self.loss_g_eikonal

            if self.opt.motion.ab_nerf_rec:
                self.loss_nerf_rec *= loss_weight["nerf_rec"]
                loss_G += self.loss_nerf_rec

            return loss_G
        
        else:

            loss_weight = {
                "D": self.opt.hypara.w_D,
                "D_grad": self.opt.hypara.w_D_grad,            
            }
            
            assert is_opt_D or is_opt_Dr1

            self.forward_D()

            loss_D = 0
            self.loss_D_fake *= loss_weight["D"]
            self.loss_D_real *= loss_weight["D"]
            loss_D += self.loss_D_real + self.loss_D_fake

            if is_opt_Dr1:
                self.loss_D_grad_penalty *= loss_weight["D_grad"]
                loss_D += self.loss_D_grad_penalty
            
            return loss_D


    def forward(self, data, is_opt_G = False, is_opt_D = False, is_opt_Dr1 = False):
        return self.forward_calc_loss(data, is_opt_G = is_opt_G, is_opt_D = is_opt_D, is_opt_Dr1 = is_opt_Dr1)


    def compute_visuals(self, epoch):

        ind=0
        row1, row2 = [], []

        if self.opt.vrnr_mesh_demo and not self.isTrain:
            mesh_dir = os.path.join(self.save_dir, "web/mesh")
            os.makedirs(mesh_dir, exist_ok=True)
            
            mesh_dir = os.path.join(self.save_dir, "web/mesh/%d" % self.opt.vrnr_voxel_factor)
            os.makedirs(mesh_dir, exist_ok=True)

            save_filename = '%s.obj' % (self.frame_index)
            save_path = os.path.join(mesh_dir, save_filename)    

            import trimesh
            self.vrnr_mesh = trimesh.smoothing.filter_humphrey(self.vrnr_mesh, beta=0.2, iterations=5)            

            self.vrnr_mesh.export(save_path)
            print("mesh saved", save_path)
            return
        

        self.uv_lat, self.uv_lat_2d, self.pose_lat, self.pose_lat_2d, self.posed_uv_lat = None, None, None, None, None
        
        self.fake_image = util.tensor2im(self.fake_image[ind])
        
        self.real_image_t = util.tensor2im(self.real_image_t[ind])
        
        self.nr_uv = self.nr_uv[ind].permute(1,2,0).cpu().numpy()
        im0 = np.zeros((self.nr_uv.shape[0], self.nr_uv.shape[1], 1))
        self.nr_uv = np.concatenate((im0, self.nr_uv), 2) * 255
      
        row2 = []
        r2list = ["tex_feat", "nerf_rec"]


        r2list = self.posmap_vis + r2list

        for s in r2list:    
            if s in self.visual_names or s in self.test_visual_names:
                tmp = getattr(self, s)
                tmp = (tmp[0].permute(1,2,0)[:,:,:3].cpu().numpy() + 1)/2 * 255
                setattr(self, s, tmp)
                row2.append(tmp)

        h, w, _ = self.fake_image.shape
        if self.opt.is_pad_img:            
            if self.opt.is_inference:
                w >>= 1

        row1.append(self.fake_image[:,:w,:])
        row1.append(self.real_image_t[:,:w,:])
        
        if "nr_uv" in self.visual_names:
            row1.append(self.nr_uv[:,:w,:])

        valid_r2 = []
        r2_h, r2_w = 0, 0
        for i in row2:
            if i is None: continue
            r2_h = max(r2_h, i.shape[0])
            r2_w = max(r2_w, i.shape[1])
    
        def fc_pad_img(img, tgt_h, tgt_w):
            im0 = np.ones((tgt_h, tgt_w, 3)) * 255
            if img is not None:
                h, w, _ = img.shape
                im0[:h, :w, :] = img
            return im0

        valid_r2 = []
        for i in row2:
            if i is None: continue
            valid_r2.append(fc_pad_img(i, r2_h, i.shape[1]))

        valid_r1 = row1
        row1 = np.concatenate(valid_r1, 1)
        if len(valid_r2) == 0:
            self.visOutput = row1
            return

        row2 = np.concatenate(valid_r2, 1)
        r1_h, r1_w, _  = row1.shape
        r2_h, r2_w, _ = row2.shape

        if r1_w < r2_w:
            im = valid_r2[0]
            valid_r1.append(fc_pad_img(im, r1_h, im.shape[1]))
            valid_r2 = valid_r2[1:]
            
            row1 = np.concatenate(valid_r1, 1)
            row2 = np.concatenate(valid_r2, 1)
            r1_h, r1_w, _  = row1.shape
            r2_h, r2_w, _ = row2.shape


        if r1_w > r2_w:
            row2 = np.concatenate((row2, fc_pad_img(None, r2_h, r1_w - r2_w)), 1)
        elif r1_w < r2_w:
            row1 = np.concatenate((row1, fc_pad_img(None, r1_h, r2_w - r1_w)), 1)
        
        self.visOutput = np.concatenate((row1, row2), 0)

        self.is_evaluate = False

    def normalize_tensor(self, input):
        max_d = input.max()
        min_d = input.min()        
        return (input - min_d) / (max_d - min_d)
         
    def update_fixed_params(self):
        return
    
    def update_learning_rate(self):
        self.old_lr_D -=  self.opt.D_lr / self.opt.niter_decay
        self.old_lr -=  self.opt.lr / self.opt.niter_decay

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = self.old_lr_D
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = self.old_lr
        if self.opt.verbose:
            print('update learning rate D: %f -> %f, %f -> %f' % (self.opt.lr_d, self.old_lr_D, self.opt.lr, self.old_lr))
    
    def load_all(self, epoch = "", resume = True):
                
        print(self.save_dir, epoch)
        begin_epoch, epoch_iter, lr = load_model(self.save_dir, self,
                       None, None,
                       scheduler = None, recorder =  None,
                       resume = resume, epoch = epoch)
        return begin_epoch, epoch_iter
     
    def save_all(self, label = "", epoch = -1, iter = 0):
        
        save_model(self.save_dir, self, None, None, 
                        label = label, epoch = epoch, iter = iter, lr = self.old_lr)
        
        return 
    
    def inference(self, batch, infer=False):
        return self.forward_G(batch, True)
        
class InferenceModel(Model):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)


from numpy.core.fromnumeric import argmax
import torch



#from lib.datasets.data_util import get_nb_dataset_shape

import cv2

from Engine.th_utils.pointcloud import render_human_pointcloud, vis_pointcloud, scale_pointcloud, get_normalized_grid_points, save_pointcloud, vis_pcd_vts_color, o3d_save_captured_image

from Engine.th_utils.load_smpl_tmp import *
from Engine.th_utils.my_pytorch3d.smpl_renderer import render_pointcloud, render_mesh
from Engine.th_utils.my_pytorch3d.vis import *

from Engine.th_utils.num import index_2d_UVMap, index_2d_dimension, array_info, index_2d_img, normalize_to_img
from Engine.th_utils.io.prints import *

from Engine.th_utils.animation.smpl_can_pose import SMPL_CanPosed

from Engine.th_utils.my_pytorch3d.smpl_uv import SMPL_UV_Shader

from Engine.th_utils.my_pytorch3d.textures import get_smpl_uv, get_smpl_face_uv

from Engine.th_utils.num import index_2d_UVMap, index_2d_UVList
from Engine.th_utils.networks import embedder

from termcolor import colored

import mcubes
import trimesh
import numpy as np

#from .human_nerf import HumanNerf

from Engine.th_utils.networks.nerf_util.nerf_net_utils import *
from Engine.th_utils.networks.nerf_util import nerf_data_util as if_nerf_dutils

is_debug = False

class NerfRender(nn.Module):
    
    def __init__(self, opt=None, smpl_render=None, phase = "train", netNerf = None):
        
        super(NerfRender, self).__init__()

        self.opt = opt
        
        if self.opt.df and self.opt.use_sdf_render:
            from Engine.th_utils.networks.nerf_util.sdf_render import SDFRenderer
            self.sdfRender = SDFRenderer(opt)

        if netNerf is not None:
            self.netNerf = netNerf 
        else:
            if self.opt.org_NB:
                from Engine.th_utils.networks.nerf_net.nbnet import OrgNB
                from Engine.th_utils.networks.nerf_net.nbnet_test import OrgNB_Test
                if self.opt.res_lighting:
                    self.netNerf = OrgNB_Test(self.opt)
                else:    
                    self.netNerf = OrgNB(self.opt)

            elif self.opt.Hash_NeRF:
                from Engine.th_utils.networks.nerf_net.human_HashNeRF import HumanHashNeRF
                self.netNerf = HumanHashNeRF(self.opt)

            elif self.opt.use_pixel_aligned_img:
                from Engine.th_utils.networks.nerf_net.human_uvNerf_pixel_aligned import HumanUVNerfPix
                self.netNerf = HumanUVNerfPix(self.opt)

            elif self.opt.patch_nerf:
                from Engine.th_utils.networks.nerf_net.human_uvPatchNerf import HumanUVPatchNerf
                self.netNerf = HumanUVPatchNerf(self.opt)
            elif self.opt.uv_mvp:
                from Engine.th_utils.networks.nerf_net.human_uvMVP import HumanUVMVP
                self.netNerf = HumanUVMVP(self.opt)
            elif self.opt.uv_3d_nerf:
                from Engine.th_utils.networks.nerf_net.human_uv3dNerf import HumanUV3DNerf
                self.netNerf = HumanUV3DNerf(self.opt)

            elif self.opt.model=="P_Fashion":
                from Engine.th_utils.networks.nerf_net.human_uvNerfFashion import HumanUVNerfFashion
                self.netNerf = HumanUVNerfFashion(self.opt)
            else:
                assert self.opt.vrnr
                from Engine.th_utils.networks.nerf_net.human_uvNerf import HumanUVNerf 
                self.netNerf = HumanUVNerf(self.opt)
        
        self.isTrain = False
        self.phase = phase
        if phase=="train":
            self.isTrain = True
        #split = "train" if self.isTrain else "test"
        
        self.image_size = (int(self.opt.org_img_reso * self.opt.nerf_ratio), int(self.opt.org_img_reso * self.opt.nerf_ratio))
        
        if self.opt.train_dress: 
            self.image_size = (int(self.opt.img_H * self.opt.nerf_ratio), int(self.opt.img_W * self.opt.nerf_ratio))
        
        elif self.opt.model =="P_Fashion":
            self.image_size = (int(self.opt.img_H * self.opt.nerf_ratio), int(self.opt.img_W * self.opt.nerf_ratio))
        
        self.head_bbox = None

        self.smpl_render = smpl_render
    
    def get_sub_loss(self):
        return self.netNerf.get_sub_loss()
      
    def get_sampling_points_depth(self, batch):
                
        ray_o = batch["ray_o"]
        ray_d = batch["ray_d"]
        ray_coord = batch["ray_coord"]

        depth_multi, silh_mask, cur_world_human, cur_smpl_human = self.project_depth(batch)
     
        ray_o_list = []
        ray_d_list = []
        full_idx_list = []
        depth_list=[]

        for channel in range(depth_multi.shape[-1]):
            if is_debug:
                printg(depth_multi[..., channel, None].shape, ray_coord.shape)
                exit()

            depth = index_2d_dimension(depth_multi[..., channel, None], ray_coord)            
            if self.opt.local_only_body :
                batch_size = depth.shape[0]

                full_idx = torch.cat([(depth[i]>0).unsqueeze(0) for i in range(batch_size) ], dim = 0 )
                org_valid_idx = torch.cat( [(depth[i]>0).nonzero().unsqueeze(0) for i in range(batch_size) ], dim = 0 )
                valid_idx = org_valid_idx

                depth_tmp = torch.cat([ depth[i, valid_idx[i, :, 0], :].unsqueeze(0) for i in range(batch_size) ], dim=0)
                
                ray_o_tmp = torch.cat([ ray_o[i, valid_idx[i, :, 0], :].unsqueeze(0) for i in range(batch_size) ], dim=0)
                ray_d_tmp = torch.cat([ ray_d[i, valid_idx[i, :, 0], :].unsqueeze(0) for i in range(batch_size) ], dim=0)
                full_idx_tmp = full_idx
                #mask = torch.cat([ mask[i, valid_idx[i, :, 0]].unsqueeze(0) for i in range(batch_size) ], dim=0)

            depth_list.append(depth_tmp)
            ray_o_list.append(ray_o_tmp)
            ray_d_list.append(ray_d_tmp)
            full_idx_list = full_idx_tmp

        ray_o = ray_o_list[0]#torch.cat(ray_o_list[0],1)
        ray_d = ray_d_list[0]#torch.cat(ray_d_list[0],1)
        #depth = torch.cat(depth_list, 1)
    
        interval = torch.tensor(self.opt.max_ray_interval).to(depth)        
        
        if self.opt.use_small_dilation: # for 394
            near = (depth_list[0] - 0.6 * interval)
            far =  (depth_list[1] + 0.6 * interval)
        elif self.opt.sample_all_pixels:
            near = (depth_list[0] - 0.6 * interval)
            far =  (depth_list[1] + 0.6 * interval)
        else:
            near = (depth - interval)
            far =  (depth + interval)

        #samples num
        
        ray_sample = self.opt.N_samples
        if self.opt.not_even:

            p1, p2, p3, p4 = 0.35, 0.65, 0.9, 1.0

            n1 = int(ray_sample*p1)
            n2 = int(ray_sample*p2) - n1
            n3 = int(ray_sample*p3) - n2 - n1
            n4 = ray_sample - n3 - n2 - n1

            t1 = torch.linspace(0., p1, steps=n1).to(depth)
            t2 = torch.linspace(p1, p2, steps=n2).to(depth)
            t3 = torch.linspace(p2, p3, steps=n3).to(depth)
            t4 = torch.linspace(p3, p4, steps=n4).to(depth)

            t_vals = torch.cat([t1, t2, t3, t4], dim=-1)
            #t_vals = torch.linspace(0., 1., steps=ray_sample).to(depth)
        else:
            t_vals = torch.linspace(0., 1., steps=ray_sample).to(depth)

        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals
        
        if True and self.opt.perturb > 0. and self.isTrain:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand
                
        z_vals = z_vals.squeeze(2)

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals, ray_o, ray_d, full_idx_list, silh_mask, cur_world_human, cur_smpl_human, None

    def get_sampling_points(self, ray_o, ray_d, near, far):

        # calculate the steps for each ray
        ray_sample = self.opt.N_samples

        if self.opt.sample_all_pixels:
            ray_sample = int(459000 / ray_o.shape[1] - 1)

        t_vals = torch.linspace(0., 1., steps=ray_sample).to(near)
        #t_vals = torch.linspace(0., 1., steps=1).to(near)
        
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if True and self.opt.perturb > 0. and self.isTrain:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def pts_to_world(self, pts, batch):
        
        if "is_global_pose" in batch.keys() and batch["is_global_pose"]==1: return pts

        """transform pts from the smpl coordinate to world coordinate"""
        Th = batch['Th'][:, None]
        R = batch['R']

        sh = pts.shape        
        
        iv_R = torch.inverse(R)
        world_pts = torch.matmul(pts.view(sh[0], -1, 3), iv_R)

        world_pts += Th.squeeze(2)

        world_pts = world_pts.view(*sh)
        return world_pts

    def pts_to_can_pts(self, pts, batch):

        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch['Th'][:, None]
        pts = pts - Th
        R = batch['R']
        sh = pts.shape
        #printy(sh,  R.shape)
        #pts = torch.matmul(pts.view(sh[0], -1, sh[3]), R)
        pts = torch.matmul(pts.view(sh[0], -1, 3), R)
        pts = pts.view(*sh)

        return pts
    
    def get_headbox(self):
        if self.head_bbox is not None:
            return self.head_bbox        
    
    def get_dilated_depth(self):
        return self.dila_depth
    
    def project_depth(self, batch, world_grids=None):
        
        image_size = self.image_size

        betas = batch["betas"]
        poses = batch["poses"]

        Cam_K = batch["Cam_K_nerf"].float()#.reshape(-1,4,4)
        Cam_R = batch["Cam_R"].float()#.reshape(-1,4,4)
        Cam_T = batch["Cam_T"].reshape(-1,3)

        if is_debug:
            printg("im ", image_size)
            exit()

        smpl_human = batch['feature'][..., :3]
        #print(smpl_human.shape)

        smpl_human = self.smpl_render.get_dilated_smpl(smpl_human, self.opt.dila_dist)

        self.smpl_render.set_camera(Cam_R = Cam_R, Cam_T = Cam_T, Cam_K = Cam_K, is_opencv_intrinsic = True, image_size=image_size)
        
        dilated_world_human = self.pts_to_world(smpl_human, batch)
                
        depth_tmp, uv_res, norm_res, mask, head_bbox = self.smpl_render.render_depth_uv_mask_rgb(vertices = dilated_world_human, is_depth = True, is_uv = False, is_head_bbox = False,  is_mask = False, image_size=image_size)
            
        if False and self.opt.use_face:
            
            head_idx, head_faces = self.smpl_render.get_head_vts()
            head_vertices = dilated_world_human[0][head_idx].unsqueeze(0)

            head_depth,_,_ = render_pointcloud(head_vertices, None, Cam_T = Cam_T, \
            Cam_R = Cam_R, Cam_K = Cam_K, is_depth = True, is_silhouette = False, \
                is_opencv_intrinsic = True, image_size=image_size)
                            
            self.head_bbox = self.smpl_render.getFaceBB(head_depth)
                                
        def test_depth_map(depth_map):
            
            depth = depth_map[0][...,0]
            os.makedirs("../data/tmp/org/tmp", exist_ok=True)

            cv2.imwrite("../data/tmp/org/tmp/depth_%d.jpg" % batch["frame_index"][0], depth.cpu().numpy()*255)
            
        if False:
            test_depth_map(depth_tmp)
            exit()
        
        if mask is not None:
            mask = mask[..., 3]

        if self.opt.use_dilate_model:
            far = torch.max(depth_tmp, dim=-1).values[...,None]            
            depth = torch.cat([depth_tmp[...,[0]], far], dim = -1)            
        else:
            depth = depth_tmp[...,0].unsqueeze(-1)
        
        self.dila_depth = depth
        
        return depth, mask, dilated_world_human, smpl_human

    def world_pts_to_screen(self, pts, batch):
        if 'Ks' not in batch:
            __import__('ipdb').set_trace()
            return raw

        sh = pts.shape
        pts = pts.view(sh[0], -1, sh[3])

        insides = []
                
        for nv in range(batch['Ks'].size(1)):
            # project pts to image space
            R = batch['RT'][:, nv, :3, :3]
            T = batch['RT'][:, nv, :3, 3]
            pts_ = torch.matmul(pts, R.transpose(2, 1)) + T[:, None]
            pts_ = torch.matmul(pts_, batch['Ks'][:, nv].transpose(2, 1))
            pts2d = pts_[..., :2] / pts_[..., 2:]

            # ensure that pts2d is inside the image
            pts2d = pts2d.round().long()
            H, W = int(self.opt.org_img_reso * self.opt.ratio), int(self.opt.org_img_reso * self.opt.ratio)
            pts2d[..., 0] = torch.clamp(pts2d[..., 0], 0, W - 1)
            pts2d[..., 1] = torch.clamp(pts2d[..., 1], 0, H - 1)
            
        print(pts2d.shape)
    
    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # feature: [N, f_channels]
        sh = batch['feature'].shape
        batch['feature'] = batch['feature'].view(-1, sh[-1])

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        #printb("sh", sh)
        idx = [torch.full([sh[1]], i, dtype=torch.long) for i in range(sh[0])]

        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        #printg(coord.shape, idx[:, None].shape, sh)

        if batch['coord'].shape[-1] == 3:
            batch['coord'] = torch.cat([idx[:, None], coord], dim=1)

            out_sh, _ = torch.max(batch['out_sh'], dim=0)
            batch['out_sh'] = out_sh.tolist()
            batch['batch_size'] = sh[0]

    def prepare_sp_input_old(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # feature: [N, f_channels]
        sh = batch['feature'].shape
        sp_input['feature'] = batch['feature'].view(-1, sh[-1])

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i, dtype=torch.long) for i in range(sh[0])]

        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        sp_input['i'] = batch['i']
        sp_input['frame_index'] = batch['frame_index']
        
        if 'betas' in batch.keys():
            sp_input['betas'] = batch['betas']
        if 'poses' in batch.keys():
            sp_input['poses'] = batch['poses']
        
        sp_input['bounds'] = batch['bounds']
        
        sp_input['state'] = batch['state']
        sp_input['dataset'] = batch['dataset']

        if self.opt.use_pixel_aligned_img:
            sp_input['src_train_images'] = batch['src_train_images']
            sp_input['src_train_intrinsic'] = batch['src_train_intrinsic']
            sp_input['src_train_extrinsic'] = batch['src_train_extrinsic']
            
        if self.opt.use_img_feature:
            sp_input['train_views'] = batch['train_views']
            sp_input['train_masks'] = batch['train_masks']
            sp_input['train_intrinsic'] = batch['train_intrinsic']
            sp_input['train_extrinsic'] = batch['train_extrinsic']

        if self.opt.vrnr:
            #sp_input['uv2d'] = batch['uv2d']
            sp_input['vrnr_img_gen'] = batch['vrnr_img_gen']
            sp_input['vrnr_mask_gen'] = batch['vrnr_mask_gen']
            
            sp_input['in_img'] = batch['in_img']
            sp_input['gt_img'] = batch['gt_img']

        if False and self.opt.In_Canonical:
            sp_input['C_out_sh'] = batch['C_out_sh']
            sp_input['C_bounds'] = batch['C_bounds']

            sp_input['C_coord_xyz'] = batch['C_coord_xyz']
            #sp_input['C_coord']  = batch['C_coord']

            sh = batch['C_coord_voxel'].shape
            idx = [torch.full([sh[1]], i) for i in range(sh[0])]
            idx = torch.cat(idx).to(batch['C_coord_voxel'])
            coord = batch['C_coord_voxel'].view(-1, sh[-1])
            sp_input['C_coord_voxel'] = torch.cat([idx[:, None], coord], dim=1)
        return sp_input

    def get_grid_coords(self, pts, sp_input, batch):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None, None]

        dhw = dhw / torch.tensor(self.opt.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    # def batchify_rays(self, rays_flat, chunk=1024 * 32, net_c=None):
    def batchify_rays(self,
                      sp_input,
                      grid_coords,
                      viewdir,
                      light_pts,
                      chunk=1024 * 32,
                      net_c=None,
                      only_density = False):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = []
        part_loss_list = []
        
        for i in range(0, grid_coords.shape[1], chunk):
            # ret = self.render_rays(rays_flat[i:i + chunk], net_c)
            ret, part_loss = self.net(sp_input, grid_coords[:, i:i + chunk],
                           viewdir[:, i:i + chunk] if viewdir is not None else None, light_pts[:, i:i + chunk], only_density)
            # for k in ret:
            #     if k not in all_ret:
            #         all_ret[k] = []
            #     all_ret[k].append(ret[k])
            all_ret.append(ret)
            part_loss_list.append(part_loss)
        # all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        all_ret = torch.cat(all_ret, 1)

        return all_ret, part_loss_list

    def get_render_mesh_pts(self, vertices):

        # obtain the original bounds for point sampling
        xyz = vertices.detach().cpu().numpy()[0]
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        
        interval = self.opt.max_ray_interval #self.opt.min_ray_interval
        
        min_xyz -= interval
        max_xyz += interval
        
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        if isinstance(self.opt.voxel_size, list):
            voxel_size = self.opt.voxel_size
        else:
            voxel_size = [self.opt.voxel_size, self.opt.voxel_size, self.opt.voxel_size]
            
        if self.opt.vrnr_mesh_demo:
            for i in range(len(voxel_size)):
                voxel_size[i] *= self.opt.vrnr_voxel_factor
            
        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        pts = torch.from_numpy(pts).to(vertices)[None,...]

        #inside = self.prepare_inside_pts_mesh(pts, i)
        inside = 0
        return pts, inside


    def get_render_mesh_pts_sample(self, vertices):

        # obtain the original bounds for point sampling
        #xyz = vertices.detach().cpu().numpy()[0]

        xyz = vertices.clone()

        near = xyz - self.interval
        far = xyz + self.interval

        t_vals = torch.linspace(0., 1., steps=32).to(near)
        #t_vals = torch.linspace(0., 1., steps=1).to(near)
        
        pts = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        inside = 0
        return pts.permute(0,1,3,2), inside

    def sample_pts_numpy(self, batch):
        split = "train" if self.isTrain else "test"
                
        img_label="img_nerf"
        msk_label="mask_nerf"
        cam_k_label="Cam_K_nerf"
        
        batch_size = batch[img_label].shape[0]
        device = batch[img_label].device
        
        rgb_list = [] 
        ray_o_list = [] 
        ray_d_list = [] 
        near_list = [] 
        far_list = [] 
        coord_list = []
        mask_at_box_list = []
        for i in range(batch_size):                 
            rgb_i, ray_o_i, ray_d_i, near_i, far_i, coord_i, mask_at_box_i = if_nerf_dutils.sample_ray(batch[img_label][i].cpu().numpy(), batch[msk_label][i].cpu().numpy(), batch[cam_k_label][i].cpu().numpy(), batch["Cam_R"][i].cpu().numpy(), batch["Cam_T"][i].cpu().numpy(), batch["can_bounds"][i].cpu().numpy(), self.opt.nrays, split, self.opt)

            rgb_list.append(rgb_i)
            ray_o_list.append(ray_o_i)
            ray_d_list.append(ray_d_i)
            near_list.append(near_i)
            far_list.append(far_i)
            coord_list.append(coord_i)
            mask_at_box_list.append(mask_at_box_i)

        rgb = torch.from_numpy(np.concatenate(rgb_list, 1)).float().to(device)
        ray_o  = torch.from_numpy(np.concatenate(ray_o_list, 1)).float().to(device)
        ray_d = torch.from_numpy(np.concatenate(ray_d_list, 1)).float().to(device)
        near = torch.from_numpy(np.concatenate( near_list, 1)).float().to(device)
        far = torch.from_numpy(np.concatenate(far_list, 1)).float().to(device)
        coord_ = torch.from_numpy(np.concatenate(coord_list, 1)).float().to(device)
        mask_at_box = torch.from_numpy(np.concatenate(mask_at_box_list, 1)).float().to(device)
        
        # rgb = np.concatenate(rgb_list, 1)
        # ray_o  = torch.cat(ray_o_list, 1)
        # ray_d = torch.cat(ray_d_list)
        # near = torch.cat( near_list)
        # far = torch.cat(far_list)
        # coord_ = torch.cat(coord_list)
        # mask_at_box = torch.cat(mask_at_box_list)
        
        batch["coord_"] = coord_
        batch["rgb"] = rgb
        batch["mask_at_box"] = mask_at_box
        
        batch["ray_o"] = ray_o
        batch["ray_d"] = ray_d
        batch["near"] = near
        batch["far"] = far

    def sample_pts(self, batch, is_high = False):
        
        split = batch["phase"]
               
        img_label="img_nerf"
        msk_label="mask_nerf"
        cam_k_label="Cam_K_nerf"
        
        if is_high:
            img_label = "h_img_nerf"
            msk_label = "h_mask_nerf"
            cam_k_label = "h_Cam_K_nerf"
            
        batch_size = batch[img_label].shape[0]
        device = batch[img_label].device
        
        rgb_list = [] 
        ray_o_list = [] 
        ray_d_list = [] 
        near_list = [] 
        far_list = [] 
        coord_list = []
        mask_at_box_list = []
        for i in range(batch_size):

            if self.opt.dataset != "h36":
                rgb_i, ray_o_i, ray_d_i, near_i, far_i, coord_i, mask_at_box_i = if_nerf_dutils.sample_ray(batch[img_label][i].cpu().numpy(), batch[msk_label][i].cpu().numpy(), batch[cam_k_label][i].cpu().numpy(), batch["Cam_R"][i].cpu().numpy(), batch["Cam_T"][i].cpu().numpy(), batch["can_bounds"][i].cpu().numpy(), self.opt.nrays, split[i], self.opt, is_high)
            else:
                rgb_i, ray_o_i, ray_d_i, near_i, far_i, coord_i, mask_at_box_i = if_nerf_dutils.sample_ray_h36m(batch[img_label][i].cpu().numpy(), batch[msk_label][i].cpu().numpy(), batch[cam_k_label][i].cpu().numpy(), batch["Cam_R"][i].cpu().numpy(), batch["Cam_T"][i].cpu().numpy(), batch["can_bounds"][i].cpu().numpy(), self.opt.nrays, split[i], self.opt)
            #printb('********', batch[img_label][i].cpu().numpy().shape, batch[msk_label][i].cpu().numpy().shape, batch[cam_k_label][i].cpu().numpy().shape, batch["Cam_R"][i].cpu().numpy().shape, batch["Cam_T"][i].cpu().numpy().shape, batch["can_bounds"][i].cpu().numpy().shape)

            rgb_list.append(rgb_i[None,...])
            ray_o_list.append(ray_o_i[None,...])
            ray_d_list.append(ray_d_i[None,...])
            near_list.append(near_i[None,...])
            far_list.append(far_i[None,...])
            coord_list.append(coord_i[None,...])
            mask_at_box_list.append(mask_at_box_i[None,...])            

        #print(len(coord_list), coord_list[0].shape)

        rgb = torch.from_numpy(np.concatenate(rgb_list, 0)).float().to(device)
        ray_o  = torch.from_numpy(np.concatenate(ray_o_list, 0)).float().to(device)
        ray_d = torch.from_numpy(np.concatenate(ray_d_list, 0)).float().to(device)
        near = torch.from_numpy(np.concatenate(near_list, 0)).float().to(device)
        far = torch.from_numpy(np.concatenate(far_list, 0)).float().to(device)
        coord_ = torch.from_numpy(np.concatenate(coord_list, 0)).long().to(device)
        mask_at_box = torch.from_numpy(np.concatenate(mask_at_box_list, 0)).bool().to(device)

        # rgb = torch.from_numpy(np.concatenate(rgb_list, 1)).float().to(device)
        # ray_o  = torch.from_numpy(np.concatenate(ray_o_list, 1)).float().to(device)
        # ray_d = torch.from_numpy(np.concatenate(ray_d_list, 1)).float().to(device)
        # near = torch.from_numpy(np.concatenate(near_list, 1)).float().to(device)
        # far = torch.from_numpy(np.concatenate(far_list, 1)).float().to(device)
        # coord_ = torch.from_numpy(np.concatenate(coord_list, 1)).float().to(device)
        # mask_at_box = torch.from_numpy(np.concatenate(mask_at_box_list, 1)).float().to(device)
        
        # rgb = np.concatenate(rgb_list, 1)
        # ray_o  = torch.cat(ray_o_list, 1)
        # ray_d = torch.cat(ray_d_list)
        # near = torch.cat( near_list)
        # far = torch.cat(far_list)
        # coord_ = torch.cat(coord_list)
        # mask_at_box = torch.cat(mask_at_box_list)
        
        #if "ray_coord" in batch.keys():
        #    print('123', is_high, batch["ray_coord"].shape, coord_.long().shape)
            #exit()

        batch["ray_coord"] = coord_.long()
        batch["rgb"] = rgb
        batch["mask_at_box"] = mask_at_box
        
        batch["ray_o"] = ray_o
        batch["ray_d"] = ray_d
        batch["near"] = near
        batch["far"] = far
                
    def render_nerf(self, batch, input_latent, is_multi_scale = False):
        
        self.opt = self.opt        
        is_evaluate = not self.isTrain
        is_output_mesh = is_evaluate #is_evaluate and (batch['eva_counter'] == 0)
        
        self.sample_pts(batch, is_multi_scale)
        
        is_check_dilation = False

        head_bbox = 0        
        if not is_multi_scale and (not self.opt.no_local_nerf):
            pts, z_vals, ray_o, ray_d, full_idx, silh_mask, cur_world_human, cur_smpl_human, head_bbox  = \
                self.get_sampling_points_depth(batch)
            sh = ray_o.shape

        else:

            if not is_multi_scale and self.opt.sample_all_pixels:
                pts, z_vals, ray_o, ray_d, full_idx, silh_mask, cur_world_human, cur_smpl_human,_  = \
                self.get_sampling_points_depth(batch)
                sh = ray_o.shape
            else:
                ray_o = batch['ray_o']
                ray_d = batch['ray_d']
                near = batch['near']
                far = batch['far']
                sh = ray_o.shape
                pts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)
        posed_smpl_vertices = batch['feature'][..., :3]
        


        #pts_full, z_vals_full = self.get_sampling_points(batch['ray_o'], batch['ray_d'], batch['near'], batch['far'])
        
        batch_size = sh[0]
         
        check_mesh = (self.opt.check_mesh and is_output_mesh)
        
        if check_mesh:
            
            smpl_vertices = batch['feature'][...,:3]
            smpl_world_pts = self.pts_to_world(smpl_vertices, batch)
            pts,_ = self.get_render_mesh_pts(smpl_world_pts)

            p_sh_3d = pts.shape            
            pts = pts.reshape(batch_size, -1 ,3)
            if False:
                pcd = render_human_pointcloud(pts)
                smpl_world_pcd = render_human_pointcloud(smpl_world_pts, color = 0)
                vis_pointcloud([pcd, smpl_world_pcd])            
                exit()
            
        elif is_output_mesh and self.opt.check_can_mesh:
            pts,_ = self.get_render_mesh_pts(self.net.get_current_can_vertices()[None,...])

            base_path = '/home/th/projects/neural_body/data/result/if_nerf/summary'
            f = self.net.get_smpl_faces().detach().cpu().numpy()

            p_sh_3d = pts.shape
            pts = pts.reshape(batch_size, -1 ,3)

        world_pts = pts.clone()

        if self.opt.uv_mvp:
            if True:
                ray_o_smpl = self.pts_to_can_pts(ray_o, batch)

                R = batch['R']
                sh = ray_d.shape
                ray_d_smpl = torch.matmul(ray_d.view(sh[0], -1, 3), R)
                ray_d_smpl = ray_d_smpl.view(*sh)
                #ray_d_smpl = self.pts_to_can_pts(ray_d, batch)
            #ray_o_smpl = ray_o
            #ray_d_smpl = ray_d
            ray_d_smpl = ray_d_smpl / torch.norm(ray_d_smpl, dim=2, keepdim=True)
            #printg(torch.norm(ray_d_smpl[0][0]), torch.norm(ray_d_smpl[0][100]))
            #exit()
            smpl_vertices = batch['feature'][...,:3]
            smpl_world_pts = self.pts_to_world(smpl_vertices, batch)
            
        pts = self.pts_to_can_pts(pts, batch)
        grid_p = pts
            
        verify_sampling = False
        if self.opt.debug_df: verify_sampling = True

        if verify_sampling:
            printg("**************very sampling")
            smpl_vertices = posed_smpl_vertices
            
            world_smpl_pcd = render_human_pointcloud(batch['feature'][..., 3:])
            world_pts_pcd = render_human_pointcloud(world_pts, 1)

            smpl_pts_pcd = render_human_pointcloud(pts, 1)
            smpl_smpl_pcd = render_human_pointcloud(batch['feature'][..., :3])

            img_path = batch["frame_index"][0]
            if torch.is_tensor(img_path): 
                img_path = img_path.item() 

            outdir="../data/tmp/df"  #self.opt.checkpoints_dir
            save_pointcloud(world_smpl_pcd + world_pts_pcd, os.path.join(outdir, "%s_world.pcd" % img_path))
            save_pointcloud(smpl_smpl_pcd + smpl_pts_pcd, os.path.join(outdir, "%s.pcd" % img_path))

            printg(self.isTrain, 'render ', img_path, batch["poses"])

            #save_pointcloud(smpl_sampled_pcd + pcd_smpl, os.path.join(outdir, "%s.pcd" % img_path))

            #o3d_save_captured_image([smpl_sampled_pcd, pcd_smpl], os.path.join(outdir, "%s.jpg" % img_path))

            ##
            exit()

            from Engine.th_utils.animation.canonical import Inverse_Skinning_NN
            self.smpl_model = Inverse_Skinning_NN(gender = self.opt.gender)
            p3 = self.smpl_model.get_smpl_vertices(batch['poses'][0] * 0, batch['betas'][0])
            
            p3 = render_human_pointcloud(p3, 1)
            smpl_vts_pcd = render_human_pointcloud(batch['feature'][..., :3], 2)
            
            #pts_full = self.pts_to_can_pts(pts_full, batch)
            #pts_full_pcd = render_human_pointcloud(pts_full)
            
            #cur_smpl_human_pcd = render_human_pointcloud(cur_smpl_human, 1)
            print(pts.shape)
            #vis_pointcloud([smpl_sampled_pcd, smpl_vts_pcd, pcd_smpl, p3]) #vts_smpl_space
            vis_pointcloud([pcd_smpl, smpl_sampled_pcd]) #vts_smpl_space
            exit()
                    
        self.prepare_sp_input(batch)
        sp_input = batch
                        
        if check_mesh:
            light_pts = None
        else:
            light_pts = embedder.xyz_embedder(world_pts)
            light_pts = light_pts.view(sh[0], -1, embedder.xyz_dim)
    
        ray_d0 = ray_d.clone()  #batch['ray_d']

        if check_mesh:
            viewdir = None
        else:
            if self.opt.uv3dNR:
                viewdir = None
            else:            
                viewdir = ray_d0 / torch.norm(ray_d0, dim=2, keepdim=True)
                if self.opt.Hash_NeRF:
                    raw_viewdir = viewdir[:, :, None].repeat(1, 1, pts.size(2), 1).contiguous()
                    raw_viewdir = raw_viewdir.view(sh[0], -1, 3)
                    #printb(pts.shape, ray_d0.shape, raw_viewdir.shape)
                if self.opt.patch_nerf:
                    viewdir = viewdir[:, :, None].repeat(1, 1, pts.size(2), 1).contiguous()
                    viewdir = viewdir.view(sh[0], -1, 3)
                else:
                    viewdir = embedder.view_embedder(viewdir)
                    viewdir = viewdir[:, :, None].repeat(1, 1, pts.size(2), 1).contiguous()
                    viewdir = viewdir.view(sh[0], -1, embedder.view_dim)
            

        grid_coords = grid_p.view(sh[0], -1, 3)
            
        smpl_verts = batch['feature'][..., :3]
        
        sp_input['smpl_vertices'] = smpl_verts
        
        batch_size = world_pts.shape[0]
        world_pts = world_pts.view(batch_size, -1, 3)
        grid_p = grid_p.view(batch_size, -1, 3)

        check_mesh = check_mesh or self.opt.vrnr_mesh_demo

        return_eikonal = True if (self.isTrain and self.opt.use_sdf_render) else False
        if return_eikonal:
            grid_p.requires_grad = True

        if ((self.opt.uv3dNR and self.opt.sample_all_pixels) or (ray_o.size(1) <= 2048 and not check_mesh)):                           
            sp_input['sampled_pts_world'] = world_pts
            sp_input['sampled_pts_smpl'] = grid_p
            sp_input['view_dir'] = viewdir

            if self.opt.Hash_NeRF:
                sp_input['raw_viewdir'] = raw_viewdir

            if self.opt.org_NB:
                sp_input['light_pts'] = light_pts

        
            if self.opt.uv_mvp:
                sp_input['ray_o_smpl'] = ray_o_smpl
                sp_input['ray_d_smpl'] = ray_d_smpl
                sp_input['tminmax'] = torch.cat((near, far), 0).permute(1,0)[None,None,...]

            if ray_o.size(1) < 500:
                #projection errors. too sparse points
                if self.opt.df or self.opt.vrnr:
                    flag = sp_input["frame_index"].item if torch.is_tensor(sp_input["frame_index"]) else sp_input["frame_index"]
                    print("!!!!!!!!!!!!!!! wrong projections %s" % flag, self.opt.uv3dNR, self.opt.sample_all_pixels, ray_o.size(),ray_o.size(1), check_mesh, grid_p.shape)
                    if self.isTrain:
                        wrong_file = os.path.join(self.opt.wrong_input_dir, "train.txt")
                    else:
                        wrong_file = os.path.join(self.opt.wrong_input_dir, "test.txt")

                    np.savetxt(wrong_file, (flag), delimiter=',', fmt='%s')

                    if self.opt.use_sdf_render:
                        return None, None, None, None
                    else: return None, None

            raw = self.netNerf(sp_input, input_latent, check_mesh)

        else:            
            """Render rays in smaller minibatches to avoid OOM.
            """
            chunk = 1024 * 32
            if self.opt.patch_nerf:
                printy("here")
                chunk = 1024 * 8
            
            all_ret = []

            #if self.opt.uv_mvp:
            #printg(ray_o.shape, ray_d.shape, near.shape, far.shape)
            #exit()
            
            for i in range(0, grid_p.shape[1], chunk):

                sp_input['sampled_pts_world'] = world_pts[:, i:i + chunk]
                sp_input['sampled_pts_smpl'] = grid_p[:, i:i + chunk]
                sp_input['view_dir'] = viewdir[:, i:i + chunk] if viewdir is not None else None
                
                if self.opt.Hash_NeRF:
                    sp_input['raw_viewdir'] = raw_viewdir[:, i:i + chunk] if raw_viewdir is not None else None

                if self.opt.org_NB:
                    sp_input['light_pts'] = light_pts[:, i:i + chunk] if light_pts is not None else None

                if self.opt.uv_mvp:
                    printg(ray_o_smpl.shape, world_pts.shape)
                    sp_input['ray_o_smpl'] = ray_o_smpl[:, i:i + chunk]
                    sp_input['ray_d_smpl'] = ray_d_smpl[:, i:i + chunk]
                    sp_input['tminmax'] = torch.cat((near, far), 0).permute(1,0)[None,None,...]
                    exit()

                ret = self.netNerf(sp_input, input_latent, check_mesh)
                all_ret.append(ret)

            raw = torch.cat(all_ret, 1)


        if check_mesh or self.opt.vrnr_mesh_demo or self.opt.nerf_mesh_demo:

            alpha = raw[0, :, 0].detach().cpu().numpy()
            
            #p_pts = batch['pts']
            #inside = batch['inside']
            inside = 0
            #p_sh_3d = pts.shape

            cube = np.zeros(p_sh_3d[1:-1])

            if inside == 0:
                cube = alpha.reshape(p_sh_3d[1:-1])
            else:
                inside = inside.detach().cpu().numpy()
                cube[inside == 1] = alpha

            #cube[full_idx] = alpha
            #cube = np.pad(cube, 10, mode='constant')
            cube = np.pad(cube, 10, mode='constant')
            vertices, triangles = mcubes.marching_cubes(cube, self.opt.meth_th) #self.opt.mesh_th
            mesh = trimesh.Trimesh(vertices, triangles)

            return {'cube': cube, 'mesh': mesh}

        # reshape to [num_rays, num_samples along ray, 4]
        raw = raw.reshape(-1, z_vals.size(2), raw.shape[-1])

        pts_each_ray = z_vals.shape[-1]

        z_vals = z_vals.view(-1, z_vals.size(2))
        ray_d = ray_d.view(-1, 3)
        


        if self.opt.use_sdf_render:
            #printg("here   ", return_eikonal, self.isTrain)
            rgb_map, acc_map, depth_map, sdf, eikonal_term = self.sdfRender.sdf_raw2outputs(raw, z_vals, ray_d, grid_p, white_bkgd = self.opt.white_bg, return_eikonal = return_eikonal)
        else:
            if False and self.opt.uv3dNR:
                vol_dim = self.opt.N_samples * self.opt.uvvol3dDim / self.opt.uvDepth
                pts_num = raw.shape[0]
                
                rgb_map = raw.reshape(pts_num, int(vol_dim))
                with torch.no_grad():
                    acc_map = rgb_map[:,0] * 0
                    depth_map = rgb_map[:,0] * 0
            else:

                if self.opt.uv_mvp:
                    raw = raw.reshape(1, -1, 4)
                    rgb_map, acc_map, depth_map = raw[..., :3], raw[..., 3], raw[..., 3]
                else:
                    rgb_map, disp_map, acc_map, weights, depth_map, cum = raw2outputs(
                        raw, z_vals, ray_d, self.opt.raw_noise_std, self.opt.white_bg, self.opt)
                
            #self.opt.nb_lighting:

            rgb_map = rgb_map.view(*sh[:-1], -1)
            acc_map = acc_map.view(*sh[:-1])
            depth_map = depth_map.view(*sh[:-1])

        batch_size, all_ray_num, _ = batch['ray_o'].shape

        if (not is_multi_scale) and (not self.opt.no_local_nerf or self.opt.sample_all_pixels):
            if len(full_idx) == 1:
                full_idx_0 = full_idx[0].unsqueeze(0).squeeze(-1)

                if self.opt.white_bg:
                    ret_full_rgb = torch.ones(batch_size, all_ray_num, rgb_map.shape[-1]).to(raw)
                else:
                    ret_full_rgb = torch.zeros(batch_size, all_ray_num, rgb_map.shape[-1]).to(raw)
                ret_full_acc = torch.zeros(batch_size, all_ray_num).to(raw)
                ret_full_depth = torch.zeros(batch_size, all_ray_num).to(raw)

                ret_full_rgb[full_idx_0] = rgb_map
                ret_full_acc[full_idx_0] = acc_map
                ret_full_depth[full_idx_0] = depth_map
        else:
            ret_full_rgb = rgb_map
            ret_full_acc = acc_map
            ret_full_depth = depth_map
              
        ret={}
        
        if check_mesh:
            ret.update(mesh_ret)

        if (not is_multi_scale) and (self.opt.vrnr or self.opt.uv3dNR):
            mask = batch['mask_at_box']
                    
            H, W = int(self.opt.org_img_reso * self.opt.nerf_ratio), int(self.opt.org_img_reso * self.opt.nerf_ratio)
            
            if self.opt.img_H != 0:
                H, W = self.opt.img_H * self.opt.nerf_ratio, self.opt.img_W * self.opt.nerf_ratio
                H, W = int(H), int(W)
                #H, W = int(np.sqrt(mask.shape[1])), int(np.sqrt(mask.shape[1]))
            
            mask_at_box = mask.reshape(H, W)

            device = ret_full_rgb.device

            pred_c = ret_full_rgb[0].shape[-1]
            if self.opt.white_bg:
                nerf_img_pred = torch.ones((H, W, pred_c), device=device)
            else:
                nerf_img_pred = torch.zeros((H, W, pred_c), device=device)
                            
            nerf_depth_pred = torch.zeros((H, W, 1), device=device)
            nerf_img_pred[mask_at_box] = ret_full_rgb[0]#[...,:3]
            
            nerf_depth_pred[mask_at_box] = ret_full_depth[...,None]

            if self.opt.use_sdf_render:
                return nerf_img_pred, nerf_depth_pred, sdf, eikonal_term

            return nerf_img_pred, nerf_depth_pred
        
        else:
            
            if (not self.opt.no_local_nerf) or self.opt.sample_all_pixels:
                if self.opt.use_sdf_render:
                    return ret_full_rgb, ret_full_depth, sdf, eikonal_term
        
                return ret_full_rgb, ret_full_depth            
            else:
                
                if self.opt.use_sdf_render:
                    return rgb_map, depth_map, sdf, eikonal_term

                return rgb_map, depth_map

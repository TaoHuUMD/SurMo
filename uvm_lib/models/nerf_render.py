import torch

from Engine.th_utils.load_smpl_tmp import *
from Engine.th_utils.my_pytorch3d.vis import *

from Engine.th_utils.num import index_2d_dimension
from Engine.th_utils.io.prints import *

from Engine.th_utils.networks import embedder

import numpy as np
from Engine.th_utils.networks.nerf_util.nerf_net_utils import *
from Engine.th_utils.networks.nerf_util import nerf_data_util as if_nerf_dutils

is_debug = False

class NerfRender(nn.Module):
    
    def __init__(self, opt=None, smpl_render=None, phase = "train", netNerf = None):
        
        super(NerfRender, self).__init__()

        self.opt = opt
        
        self.netNerf = netNerf     
        self.isTrain = False
        self.phase = phase
        if phase=="train":
            self.isTrain = True

        self.image_size = (int(self.opt.img_H * self.opt.nerf_ratio), int(self.opt.img_W * self.opt.nerf_ratio))
        
        self.head_bbox = None

        self.smpl_render = smpl_render
    
    def get_sub_loss(self):
        return self.netNerf.get_sub_loss()
      
    def get_sampling_points_depth(self, batch):
                
        ray_o = batch["ray_o"]
        ray_d = batch["ray_d"]
        ray_coord = batch["ray_coord"]

        depth_multi, silh_mask, cur_world_human, cur_smpl_human = batch["lr_depth"], None, None, None
     
        ray_o_list = []
        ray_d_list = []
        full_idx_list = []
        depth_list=[]

        for channel in range(depth_multi.shape[-1]):
            
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

            depth_list.append(depth_tmp)
            ray_o_list.append(ray_o_tmp)
            ray_d_list.append(ray_d_tmp)
            full_idx_list = full_idx_tmp

        ray_o = ray_o_list[0]
        ray_d = ray_d_list[0]
    
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
        else:
            t_vals = torch.linspace(0., 1., steps=ray_sample).to(depth)

        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals
        
        if True and self.opt.perturb > 0. and self.isTrain:
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

        if self.opt.use_smpl_scaling:
            world_pts *= batch["scaling"]

        world_pts += Th.squeeze(2)

        world_pts = world_pts.view(*sh)

        
        return world_pts

    def pts_to_can_pts(self, pts, batch):

        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch['Th'][:, None]
        pts = pts - Th
        R = batch['R']
        sh = pts.shape

        #scaling
        if self.opt.use_smpl_scaling:
            pts /= batch["scaling"]

        pts = torch.matmul(pts.view(sh[0], -1, 3), R)
        pts = pts.view(*sh)

        return pts
   
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
            ret, part_loss = self.net(sp_input, grid_coords[:, i:i + chunk],
                           viewdir[:, i:i + chunk] if viewdir is not None else None, light_pts[:, i:i + chunk], only_density)
            all_ret.append(ret)
            part_loss_list.append(part_loss)
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

        inside = 0
        return pts, inside


    def get_render_mesh_pts_sample(self, vertices):

        xyz = vertices.clone()

        near = xyz - self.interval
        far = xyz + self.interval

        t_vals = torch.linspace(0., 1., steps=32).to(near)
        
        pts = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        inside = 0
        return pts.permute(0,1,3,2), inside

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

            rgb_list.append(rgb_i[None,...])
            ray_o_list.append(ray_o_i[None,...])
            ray_d_list.append(ray_d_i[None,...])
            near_list.append(near_i[None,...])
            far_list.append(far_i[None,...])
            coord_list.append(coord_i[None,...])
            mask_at_box_list.append(mask_at_box_i[None,...])

        rgb = torch.from_numpy(np.concatenate(rgb_list, 0)).float().to(device)
        ray_o  = torch.from_numpy(np.concatenate(ray_o_list, 0)).float().to(device)
        ray_d = torch.from_numpy(np.concatenate(ray_d_list, 0)).float().to(device)
        near = torch.from_numpy(np.concatenate(near_list, 0)).float().to(device)
        far = torch.from_numpy(np.concatenate(far_list, 0)).float().to(device)
        coord_ = torch.from_numpy(np.concatenate(coord_list, 0)).long().to(device)
        mask_at_box = torch.from_numpy(np.concatenate(mask_at_box_list, 0)).bool().to(device)

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
        
        batch_size = sh[0]
         
        check_mesh = (self.opt.check_mesh and is_output_mesh)
        
      

        world_pts = pts.clone()

        pts = self.pts_to_can_pts(pts, batch)
        grid_p = pts
            
        verify_sampling = False
                    
        self.prepare_sp_input(batch)
        sp_input = batch
                        
        if check_mesh:
            light_pts = None
        else:
            light_pts = embedder.xyz_embedder(world_pts)
            light_pts = light_pts.view(sh[0], -1, embedder.xyz_dim)
    
        ray_d0 = ray_d.clone()  #batch['ray_d']

        viewdir = ray_d0 / torch.norm(ray_d0, dim=2, keepdim=True)            
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

        if True:            
            """Render rays in smaller minibatches to avoid OOM.
            """
            chunk = 1024 * 32
            if self.opt.patch_nerf:
                printy("here")
                chunk = 1024 * 8
            
            all_ret = []

            for i in range(0, grid_p.shape[1], chunk):
                sp_input['sampled_pts_world'] = world_pts[:, i:i + chunk]
                sp_input['sampled_pts_smpl'] = grid_p[:, i:i + chunk]
                sp_input['view_dir'] = viewdir[:, i:i + chunk] if viewdir is not None else None
            
                ret = self.netNerf(sp_input, input_latent, check_mesh)
                all_ret.append(ret)

            raw = torch.cat(all_ret, 1)

        raw = raw.reshape(-1, z_vals.size(2), raw.shape[-1])

        pts_each_ray = z_vals.shape[-1]

        z_vals = z_vals.view(-1, z_vals.size(2))
        ray_d = ray_d.view(-1, 3)
        

        rgb_map, disp_map, acc_map, weights, depth_map, cum = raw2outputs(
            raw, z_vals, ray_d, self.opt.raw_noise_std, self.opt.white_bg, self.opt)

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
        
      

        if (not is_multi_scale) and (self.opt.vrnr or self.opt.uv3dNR):
            mask = batch['mask_at_box']
                    
            H, W = int(self.opt.org_img_reso * self.opt.nerf_ratio), int(self.opt.org_img_reso * self.opt.nerf_ratio)
            
            if self.opt.img_H != 0:
                H, W = self.opt.img_H * self.opt.nerf_ratio, self.opt.img_W * self.opt.nerf_ratio
                H, W = int(H), int(W)
            
            mask_at_box = mask.reshape(H, W)

            device = ret_full_rgb.device

            debug_bk = False

            pred_c = ret_full_rgb[0].shape[-1]
            if self.opt.white_bg:
                nerf_img_pred = torch.ones((H, W, pred_c), device=device)
            else:
                nerf_img_pred = torch.zeros((H, W, pred_c), device=device)
                            
            nerf_depth_pred = torch.zeros((H, W, 1), device=device)

            if not debug_bk:
                nerf_img_pred[mask_at_box] = ret_full_rgb[0]#[...,:3]
            
            nerf_depth_pred[mask_at_box] = ret_full_depth[...,None]

            return nerf_img_pred, nerf_depth_pred, None, None
        
        else:
            
            if (not self.opt.no_local_nerf) or self.opt.sample_all_pixels:               
                return ret_full_rgb, ret_full_depth, None, None
            else:                
                return rgb_map, depth_map, None, None

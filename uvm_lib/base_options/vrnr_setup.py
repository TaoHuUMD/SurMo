import argparse
import os

def vrnr_init(parser):
    
    parser.add_argument('--only_nerf', action='store_true')
    parser.add_argument('--only_gen', action='store_true')
    parser.add_argument('--use_nerf', action='store_true')
    parser.add_argument('--use_gen', action='store_true')
    
    parser.add_argument('--uv2pix', action='store_true', help="pix2pix")
        
    parser.add_argument('--id_num_total', type=int, default=6)  # 3 channels
    
    parser.add_argument('--share_3D', action='store_true')

    parser.add_argument('--nerf_light', action='store_true', help="lighting in nerf")
    parser.add_argument('--not_latent_uv', action='store_true', help="3d vts latetnt in nerf")
    #parser.add_argument('--uv_normal', action='store_true', help="unknown")                

    parser.add_argument('--local_only_body', type=bool, default=True, help="sample points only on body local nerf")     
    parser.add_argument('--use_dilate_model', type=bool, default=True, help="sample points only on body local nerf")
    parser.add_argument('--is_opt_mask', action='store_true', help="whether opt nerf mask")          
    
    parser.add_argument('--white_vgg', action='store_true', help="vgg image mask")          

    parser.add_argument('--swap_tex', action='store_true', help="vgg image mask")          

    parser.add_argument('--not_even', type=bool, default=True, help="sample points only on body local nerf")     
    parser.add_argument('--N_samples', type=int, default=24, help="") 

    #parser.add_argument('--dataset_name', type=str, help="dataset name")
    
    parser.add_argument('--nerf_reso', type=int, default=64, help="") 
    parser.add_argument('--gen_reso', type=int, default=256, help="")
    parser.add_argument('--nerf_ratio', type=float, default=0, help="") 
    parser.add_argument('--gen_ratio', type=float, default=0, help="") 
         
    parser.add_argument('--is_smooth_nerf_latent', action='store_true', help="whether smooth nerf latent")
         
    parser.add_argument('--In_Canonical', action='store_true', help="Canonical nerf")
    parser.add_argument('--check_mesh', action='store_true', help="check can mesh")
    parser.add_argument('--check_can_mesh', action='store_true', help="check can mesh")
    
    parser.add_argument('--no_local_nerf', action='store_true', help="global nerf sampling")
    parser.add_argument('--use_img_feature', action='store_true', help="global nerf sampling")
    parser.add_argument('--sample_all_pixels', action='store_true', help="global nerf sampling")
            
    #nerf
    parser.add_argument('--not_use_conv3d', action='store_true', help="conv3d in nerf")
    parser.add_argument('--not_pose_cond', action='store_true', help="not pose condit")
    
    #generator neural rendering
    parser.add_argument('--no_encoder', action='store_true', help="encoder in nr")
    parser.add_argument('--pred_depth', action='store_true', help="nr predits depth")
    #parser.add_argument('--pred_normal_uv', action='store_true', help="nr predits normal")
        
    parser.add_argument('--is_cat_max_mean', action='store_true', help="fuse nr and nerf features")
            
    parser.add_argument('--begin_i', type=int, default = 10, help="vrnr")
    parser.add_argument('--ni', type=int, default = 200, help="vrnr")
    parser.add_argument('--N_rand', type=int, default = 64, help="vrnr")
    parser.add_argument('--voxel_size', type=float, default=0.005, help="weight") 
    
    parser.add_argument('--nrays', type=int, default = 1024, help = "number of rays.. not used in vrnr")
                         
    parser.add_argument('--w_nerf', type=float, default=3.0, help="weight") 

    parser.add_argument('--use_small_dilation', type=bool, default=True, help="whether multiple identities")
    
    parser.add_argument('--max_ray_interval', type=float, default=0.25*0.2, help="local sampling interval") 
    parser.add_argument('--perturb', type=int, default=1, help="perturb training")     
        
    parser.add_argument('--white_bkgd', action='store_true', help="train snapshot")
    parser.add_argument('--raw_noise_std', type=float, default=0, help="weight") 
        
    parser.add_argument('--uvVol_smpl_pts', type=float, default=5, help="train snapshot")
    
    #dataset
    parser.add_argument('--dataset_step', type=int, default=1, help="split dataset")     
    parser.add_argument('--org_img_reso', type=int, default=1024, help="")
    
    parser.add_argument('--train_rgbd', action='store_true', help="train snapshot")
    
    parser.add_argument('--train_snap', action='store_true', help="train snapshot")
    parser.add_argument('--multi_id', type=bool, default=True, help="whether multiple identities")
    parser.add_argument('--id_num', type=int, default=1, help="identi number in training")
    parser.add_argument('--uvdim', type=int, default=16, help="uv latent dim")         
    
    parser.add_argument('--debug_1pose_all_views', action='store_true', help="debug 1 frames")
    parser.add_argument('--vrnr_swap2', action='store_true', help="swap textures")
    
    parser.add_argument('--use_density_th', action='store_true', help="local nerf new dist")
    
    parser.add_argument('--use_new_dist', action='store_true', help="local nerf new dist")
    parser.add_argument('--par_density_norm', type=float, default=1000, help="uv latent dim")         
    parser.add_argument('--density_th', type=float, default=0.5, help="uv latent dim")         

def dataset_setup(cfg):
    
    cfg.data_list = cfg.data_list.split(" ")
     
    dataset_id = {}    
    for i in range(len(cfg.data_list)):        
        dataset_id.update({cfg.data_list[i]: i})
    
    cfg.dataset_id = [dataset_id]
    

            
def vrnr_parse(cfg):
        
    cfg.use_nerf = True
    cfg.use_gen = True
    cfg.use_face = True

    if cfg.gen_ratio > 1: 
        cfg.gen_ratio = 1 / cfg.gen_ratio
    
    if cfg.nerf_ratio > 1: 
        cfg.nerf_ratio = 1 / cfg.nerf_ratio

    if cfg.gen_ratio == 0: cfg.gen_ratio = cfg.gen_reso / cfg.org_img_reso
    if cfg.nerf_ratio == 0: cfg.nerf_ratio = cfg.nerf_reso / cfg.org_img_reso
        
def make_dataset_id(cfg):
          
    cfg.dataset_id = [{
            "xyzc_394": 0,
            "xyzc_377": 1,
            "xyzc_313": 2,
            "xyzc_386": 3,
            "xyzc_392": 4,
            "xyzc_311": 5
        }]
    
    cfg.dataset_id_swap = cfg.dataset_id_swap1

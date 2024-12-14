
import numpy as np
import os
import pickle

def save_dict(d, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    print("keys saved ", len(d.keys()))

def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_croped_size(opt, mode="all"):
    
    ratio = opt.gen_ratio if not opt.only_nerf else opt.nerf_ratio

    h, w = opt.img_H * ratio, opt.img_W * ratio

    img_gen_h, img_gen_w = int(h), int(w)

    if opt.is_crop:
        img_gen_h = img_gen_h - opt.y0_crop - opt.y1_crop
        img_gen_w = img_gen_w - opt.x0_crop - opt.x1_crop

    if mode == "all" and not opt.only_nerf:
        img_nerf_h, img_nerf_w = int(img_gen_h * opt.nerf_ratio / opt.gen_ratio), int(img_gen_w * opt.nerf_ratio / opt.gen_ratio)
        return (img_gen_h, img_gen_w, img_nerf_h, img_nerf_w)
    else:
        return (img_gen_h, img_gen_w)


    dir="/home/th/projects/neural_body/neuralbody/configs"
    #rename_files(dir)

    #dir1="/fs/vulcan-projects/egocentric_video/neural_body/data/result/nr/result/wxd/wxd_12345_12345/wxd_12345_12345/smplpix_dnr_720_noaug_wxd_12345_12345/test_3_100/pose_dense/d0_gt.mp4"
    #dir2="/fs/vulcan-projects/egocentric_video/neural_body/data/result/nr/result/wxd/wxd_12345_12345/wxd_12345_12345/gt3.mp4"
    
    dir1="/fs/vulcan-projects/egocentric_video/neural_body/data/result/nr/result/bw/bw_12345_12345/bw_12345_12345/smplpix_dnr_720_noaug_bw_12345_12345/test_5_102/pose_dense/d0_gt.mp4"
    dir2="/fs/vulcan-projects/egocentric_video/neural_body/data/result/nr/result/bw/gt5.mp4"

    trans_video_format(dir1, dir2)
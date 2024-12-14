#!/bin/bash

id=$1
gpu="${id}"
 
method="--use_style --learn_3d --use_posmap --uvVol_2d --nr_pose_dep_uv"

config="--config zju/motion_387_fv.yml --project_config uvm.yml --method_config motion.yml"

render="--superreso LightSup --vrnr"
setup="--motion_mode --uv_type SMPL --uv_reso 256 --batchSize 1"
net="--posenet_outdim 96 --tex_latent_dim 16 --ab_uvh_plane_c 32 --nerf_dim 32 --style_dim 256"
  
ab="--c_velo  --c_acce --c_traj --velocity 1 --pred_pose_uv --pred_normal_uv --ab_pred_pose_by_velocity --new_dynamics --pred_pose_uv_rot --rot_all_same"

aug="--small_rot --aug_nr"

modelname="train_387_Mp3dS"
debug=""

epoch="--niter 200 --niter_decay 0"
reso="--gen_ratio 0.5 --nerf_ratio 0.25"

improve="--ab_cond_uv_latent --ab_D_pose"

basic="--N_samples 28 --uvVol_smpl_pts 8 --distributed --use_org_discrim --use_org_gan_loss --learn_uv --ab_uvh_plane --plus_uvh_enc --ab_nerf_rec --load_tmp_rendering"

w="--w_D 1.0 --w_G_GAN 1.0 --w_Face 5 --w_G_L1 0.5 --w_G_feat 10 --w_nerf_rec 15 --w_pred_normal_uv 1.0 --w_rot_normal 1.0 --w_posmap_feat 0 --w_posmap 1"

cmd="${basic} ${setup} ${net} ${ab} ${w} ${debug} ${epoch} ${reso} ${method} ${aug} ${render} ${improve} ${config} --name $modelname"

if [ -z "$gpu" ]
  then
    gpu=$CUDA_VISIBLE_DEVICES
fi

export NVIDIA_VISIBLE_DEVICES=${gpu}
export CUDA_VISIBLE_DEVICES=${gpu}

MASTER_PORT=$((12000 + $RANDOM % 20000))
gpures="${gpu//[^,]}"      
NUM_GPU="${#gpures}"
NUM_GPU=$((NUM_GPU+1))
extra_opt="${extra_opt} --gpu_ids ${gpu}"

python -W ignore -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=${MASTER_PORT} train_dist.py ${cmd}

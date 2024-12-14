
import time
import os
import numpy as np
import torch
os.environ["NCCL_DEBUG"] = "INFO"
import copy

import sys

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import random

os.environ['TORCH_DISTRIBUTED_DEBUG']="DETAIL"

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from torch.nn.modules import module

import sys
sys.path.append("..")


from Engine.th_utils.networks import networks 

from Engine.th_utils.io.prints import *

from Engine.th_utils.distributed.distributed import synchronize

from Engine.th_utils.networks.networks import accumulate

from Engine.th_utils.distributed.sampler import data_sampler

from uvm_lib.options.train_option import ProjectOptions


from uvm_lib.data.data_loader import CreateDataLoader, CreateDataLoaderDistributed, CreateDataset
from uvm_lib.models.models import create_model
from uvm_lib.util.visualizer import Visualizer

import warnings
warnings.filterwarnings("ignore")


def setup_training(opt, datasize):
    opt.max_iters_epoch = 500
    total_epoch = opt.niter + opt.niter_decay
    opt.save_latest_freq = 2
    opt.eva_epoch_freq = 5
    opt.save_epoch_freq = int(total_epoch / 5)

    opt.print_freq = 20
    opt.display_freq = int(opt.max_iters_epoch / 2)


def dist_init_new(opt):
    strgpu=""
    for g in opt.gpu_ids[:-1]:
        strgpu += "%s," % g
    strgpu+= "%s" % opt.gpu_ids[-1]

    os.environ['CUDA_VISIBLE_DEVICES'] = strgpu
    opt.world_size = torch.cuda.device_count()    
    assert opt.world_size ==  len(opt.gpu_ids)

    print(strgpu, opt.world_size, opt.local_rank)

    if not opt.training.distributed:
        opt.training.distributed = len(opt.gpu_ids) > 1
    if opt.training.distributed:
        if opt.local_rank != -1:
            opt.rank = opt.local_rank
            opt.gpu = opt.local_rank 
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            opt.rank = int(os.environ['SLURM_PROCID'])
            opt.gpu = opt.rank % torch.cuda.device_count()
        
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=opt.world_size, rank=opt.rank)#
        synchronize()
    else:
        opt.rank = opt.gpu_ids[0]
        opt.gpu = opt.gpu_ids[0]

    opt.gpu_num = len(opt.gpu_ids)

    if opt.training.distributed:
        opt.gpu_ids = [opt.gpu] 
    torch.cuda.set_device(opt.gpu)

def split_dict(d, n):
    keys = list(d.keys())
    for i in range(0, len(keys), n):
        yield {k: d[k] for k in keys[i: i + n]}

def split_dict_batch(d, n=1):
    keys = list(d.keys())
    batch_size = len(d[keys[0]])
    if batch_size==1: yield d
    for i in range(0, batch_size, n):
        yield {k: d[k][i:i+n] for k in keys}

def make_uv_noise(opt):    
    uvnoise = torch.randn(1, 4, opt.texlat_init_size, opt.texlat_init_size, device=opt.gpu)
    if opt.texlat_init_size != opt.posenet_setup.uv_reso:
        uvnoise = torch.nn.functional.interpolate(uvnoise, size=(opt.posenet_setup.uv_reso, opt.posenet_setup.uv_reso), mode='bilinear', align_corners=False)#, antialias=True
    return uvnoise

if __name__ == "__main__":
    
    print('train')

    opt = ProjectOptions().parse()
    
    dist_init_new(opt)
    
    opt.continue_train = True
    opt.phase = "train"
    train_dataset = CreateDataset(opt, "train")

    train_sampler = data_sampler(train_dataset, shuffle=True, distributed = opt.training.distributed, world_size = opt.world_size, rank = opt.rank)

    train_data_loader  = CreateDataLoaderDistributed(opt, train_dataset, train_sampler, "train")
    dataset = train_data_loader.load_data()
    dataset_size = len(train_data_loader)

    setup_training(opt, dataset_size)

    eva_data_loader = CreateDataLoader(opt, "evaluate")
    eva_dataset = eva_data_loader.load_data()

    is_eva = True
    model = create_model(opt)

    eva_opt = copy.deepcopy(opt)
    eva_opt.phase = "evaluate"
    eva_opt.batchSize = 1
    model_ema = create_model(eva_opt) #estimate ema = True
    model.apply(networks.weights_init)
    
    #load dataset
    if opt.rank == 0:            
        print('#training images = %d' % dataset_size)
        print("##batch size ", opt.batchSize)
        print('#evaluation images = %d' % len(eva_data_loader))

    if opt.training.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[opt.gpu],
            output_device=opt.gpu,
            broadcast_buffers=False
        )

    model_ema_no_dpp = model_ema.module
    model_no_dpp = model.module

    if opt.rank == 0:
        print("load model:")
    
    # start_epoch = 1
    # epoch_iter = 0
    start_epoch, epoch_iter = model_no_dpp.load_all(opt.which_epoch, True)
    if start_epoch == -1:
       start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d / %d at iteration %d' % (start_epoch, opt.niter + opt.niter_decay, epoch_iter))
    model_ema.eval()
    accumulate(model_ema_no_dpp, model_no_dpp, 0)
    _,_ = model_ema_no_dpp.load_all("ema_latest", True)
    
    model.train()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    if start_epoch >= opt.niter + opt.niter_decay: sys.exit() #correct exit

    visualizer = Visualizer(opt)

    total_steps = (start_epoch - 1) * opt.max_iters_epoch + epoch_iter
    cur_total_steps = 0

    accum = 0.5 ** (32 / (10 * 1000))

    actual_batch_size = opt.gpu_num * opt.batchSize

    print("actual batch size ", actual_batch_size)

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        np.random.seed(epoch)
        random.seed(epoch)
        if opt.training.distributed:
            dataset.sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = 0
        
        last_time = time.time()
        data_time_sum = 0

        
        if epoch > start_epoch + 2:
            opt.print_freq = 20
            opt.display_freq = int(opt.max_iters_epoch * 5)
        else:
            opt.print_freq = 10

        for i, data in enumerate(dataset, start=epoch_iter):

            if i > opt.max_iters_epoch: break
            if epoch_iter >= dataset_size or epoch_iter >= opt.max_iters_epoch:
                break

            if opt.rank == 0: 
                data_time = time.time() - last_time
                if cur_total_steps % opt.print_freq <= actual_batch_size:
                    iter_start_time = time.time()

            cur_total_steps += actual_batch_size
            total_steps += actual_batch_size

            epoch_iter += actual_batch_size
            
            save_fake = False
            if cur_total_steps % opt.display_freq + actual_batch_size >= opt.display_freq: save_fake = True

            model.module.init_setup()
            model.module.requires_grad_D(False)
            model.module.requires_grad_G(True)
            

            with torch.no_grad():
                uvnoise = make_uv_noise(opt)
                data["uvnoise"] = uvnoise

            model.module.optimizer_G.zero_grad()
            loss_G = model(data, is_opt_G = True)
            loss_G.backward()#
            model.module.optimizer_G.step()


            is_opt_D = True 
            is_opt_Dr1 = True 
            if is_opt_D or is_opt_Dr1:
                model.module.requires_grad_D(True)
                model.module.requires_grad_G(False)  
                model.module.optimizer_D.zero_grad()
                
                loss_D = model(data, is_opt_D = is_opt_D, is_opt_Dr1 = is_opt_Dr1)
                loss_D.backward()#
                model.module.optimizer_D.step()
                        
            accumulate(model_ema_no_dpp, model_no_dpp, accum)
          
            if opt.rank == 0: 
                last_time = time.time()
                data_time_sum += data_time
                if cur_total_steps % opt.print_freq + actual_batch_size >= opt.print_freq:
                    losses = model.module.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.print_freq
                    visualizer.print_current_errors(epoch, epoch_iter, losses, [data_time_sum / opt.print_freq, t])
                    visualizer.plot_current_errors(losses, total_steps)
                    data_time_sum = 0
                    
                if save_fake: #
                    model.module.compute_visuals(epoch)
                    
                    tcid = data["cam_ind"][0] #.item() + 1
                    tvid = data["frame_index"][0]
                    tcid = tcid.item() if torch.is_tensor(tcid) else tcid
                    tvid = tvid.item() if torch.is_tensor(tvid) else tvid
                    visualizer.display_current_results(model.module.get_current_visuals(),
                                                    epoch, "%s_%s" % (tcid, tvid))
                
                
        model.module.update_weights(epoch)
        
        if opt.rank == 0: 
            # end of epoch
            if dataset_size > 100:
                print('End of epoch %d / %d \t Time Taken: %d sec' %
                    (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

            if epoch % opt.save_epoch_freq == 0:
                model.module.save_all(label = epoch, epoch = epoch, iter = epoch_iter)
                model_ema_no_dpp.save_all(label = "ema_%s" % epoch, epoch = epoch, iter = epoch_iter)

                np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

            ### save model for this epoch
            if epoch % opt.save_latest_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))            
                model.module.save_all(label = 'latest', epoch = epoch, iter = epoch_iter)
                model_ema_no_dpp.save_all(label = 'ema_latest', epoch = epoch, iter = epoch_iter)
                np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

            if is_eva and (epoch % opt.eva_epoch_freq  == 0):
                print("evaluate")
                for e_i, e_data in enumerate(eva_dataset, start=0):
                    print("start evaluating epoch")
                    with torch.no_grad():
                        uvnoise = make_uv_noise(opt)
                        e_data["uvnoise"] = uvnoise
                        exp_out = model_ema_no_dpp.evaluate(e_data)
                        assert isinstance(exp_out,int)
                        if exp_out == -1: #error in this batch
                            print("errror in this batch")
                            continue
                        model_ema_no_dpp.compute_visuals("evaluate")

                        tcid = e_data["cam_ind"][0] #.item() + 1
                        tvid = e_data["frame_index"][0]

                        tcid = tcid.item() if torch.is_tensor(tcid) else tcid
                        tvid = tvid.item() if torch.is_tensor(tvid) else tvid

                        visualizer.display_current_results(model_ema_no_dpp.get_current_visuals(),
                                                        epoch, "eva_%s_%s" % (tcid, tvid))
                
                print("finish")  
                continue
                    

        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

        if epoch > opt.niter:
            model.module.update_learning_rate()

    sys.exit()
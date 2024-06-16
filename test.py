import numpy

import os

import sys
sys.path.append("..")

import torch

from uvm_lib.engine.lib.data.data_loader import CreateDataLoader
from uvm_lib.engine.lib.models.models import create_model
from uvm_lib.engine.lib.util.visualizer import Visualizer
from uvm_lib.engine.lib.util import html
from uvm_lib.engine.thutil.io.prints import *
from uvm_lib.engine.thutil.dirs import *

from uvm_lib.options.test_option import ProjectOptions

if __name__ == "__main__":

    opt = ProjectOptions().parse(save=False)
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.is_inference = True

    printb("test  ", opt.gpu_ids)

    torch.cuda.set_device(opt.gpu_ids[0])

    opt.phase = "test"
    if opt.no_label:
        opt.render_with_dp_label = False

    data_loader = CreateDataLoader(opt, opt.phase)
    dataset = data_loader.load_data()
    
    visualizer = Visualizer(opt)

    # create website
    save_name = opt.name
    if opt.save_name != '':
        save_name = opt.save_name

    which_epoch = opt.which_epoch
    
    model = create_model(opt).cuda().module #opt.gpu_ids[0]
    
    if opt.test_eval:
        model_name = "ema_latest" if opt.which_epoch == "-1" else "ema_%s" % opt.which_epoch
    else: model_name = opt.which_epoch

    test_epoch, epoch_iter = model.load_all(model_name, True)
    opt.which_epoch = test_epoch #- 1
    
    #start_epoch, epoch_iter = 90, 0
    if test_epoch == -1:
        test_epoch, epoch_iter = 1, 0
        printy("test model not trained")
        exit()
        
    printb(test_epoch)
    which_epoch = test_epoch #- 1

    view_num = len(opt.multiview_ids)

    cnt = 0

    if opt.free_view_fly_camera or opt.free_view_rot_smpl:
        opt.multiview_ids = opt.multiview_ids[:1]

    for view_id in opt.multiview_ids:
          
        cnt += 1
        if cnt > 1 and opt.vrnr_mesh_demo:
            sys.exit()#correct exit 
    
        web_dir = get_web_dir(save_name, which_epoch, view_id, opt)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (save_name, opt.phase, which_epoch))

        result_dir = get_img_dir(save_name, which_epoch, view_id, opt)
        result_image_num = len(os.listdir(result_dir))
        test_image_num = len(dataset)

        printy("test images: ", test_image_num, "already generated images", result_image_num)
        if (result_image_num >= test_image_num  / view_num ) and not opt.inference_time:        
            if not (opt.check_mesh): #opt.use_nerf and 
                printy("skip test")
                continue

        # test
        if not opt.engine and not opt.onnx:            
            if opt.data_type == 16:
                model.half()
            elif opt.data_type == 8:
                model.type(torch.uint8)

            if opt.verbose:
                print(model)
        else:
            t = 0

        model.eval()

        for i, data in enumerate(dataset):

            if i >= opt.how_many: break

            data_vid = data["cam_ind"].cpu().numpy()[0]
            if opt.dataset == "rgbd5": #or opt.dataset == "aistdata":
                data_vid += 1
              
            if data_vid != int(view_id): continue        
              
            minibatch = 1 
   
            with torch.no_grad(): 

                generated = model.inference(data)
  
            model.compute_visuals(which_epoch)
            img_idx = data['frame_index'][0].cpu().numpy()
            dataset_id = model.dataset_id if isinstance(model.dataset_id, int) else model.dataset_id[0].cpu().numpy()
 
            if opt.make_demo and opt.motion.motion_chain:
                motion_step_idx = data['pre_step'][0].cpu().numpy()
                img_idx = "m%03d_%04d" % (motion_step_idx + 100, img_idx) 
            else:                
                img_idx = "d%s_%04d" % (dataset_id, img_idx)

            print('process image... %s' % img_idx)
            visualizer.save_images(webpage, model.get_current_visuals(), img_idx, "test")
                    
            if opt.make_overview:
                break
                
        webpage.save()

        printb("test finished")

    sys.exit()#correct exit
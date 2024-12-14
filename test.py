
import os

import sys

import torch

from uvm_lib.data.data_loader import CreateDataLoader
from uvm_lib.models.models import create_model
from uvm_lib.util.visualizer import Visualizer
from uvm_lib.util import html

from uvm_lib.options.test_option import ProjectOptions


if __name__ == "__main__":

    opt = ProjectOptions().parse(save=False)
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.is_inference = True

    print("test  ", opt.gpu_ids)

    torch.cuda.set_device(opt.gpu_ids[0])
    
    opt.phase = "test"

    if opt.no_label:
        opt.render_with_dp_label = False

    data_loader = CreateDataLoader(opt, opt.phase)
    dataset = data_loader.load_data()
    
    visualizer = Visualizer(opt)

    exp_name = opt.name
    
    which_epoch = opt.which_epoch
    
    model = create_model(opt).cuda().module
    
    if opt.test_eval:
        model_name = "ema_latest" if opt.which_epoch == "-1" else "ema_%s" % opt.which_epoch
    else: model_name = opt.which_epoch

    test_epoch, epoch_iter = model.load_all(model_name, True)
    opt.which_epoch = test_epoch #- 1
    
    if test_epoch == -1:
        test_epoch, epoch_iter = 1, 0
        print("test model not trained")

        if not opt.save_tmp_rendering: 
            exit()
        
    print(test_epoch)
    which_epoch = test_epoch #- 1

    view_num = len(opt.multiview_ids)

    #to remove
    opt.multiview_ids = opt.multiview_ids


    for view_id in opt.multiview_ids:
              
        web_dir = os.path.join(opt.results_dir, exp_name)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (exp_name, opt.phase, which_epoch))

        result_dir = os.path.join(opt.results_dir, exp_name, "images")
        result_image_num = len(os.listdir(result_dir))
        test_image_num = len(dataset)
        
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
    
            minibatch = 1 

            data_vid = data["cam_ind"].cpu().numpy()[0]
            if data_vid != int(view_id): continue    
   
            with torch.no_grad(): 
                generated = model.inference(data)
            
            model.compute_visuals(which_epoch)
            img_idx = data['frame_index'][0].cpu().numpy()
            dataset_id = model.dataset_id if isinstance(model.dataset_id, int) else model.dataset_id[0].cpu().numpy()
 
            img_idx = "d%s_%04d" % (dataset_id, img_idx)

            print('process image... %s' % img_idx)
            visualizer.save_images(webpage, model.get_current_visuals(), img_idx, "test")

        print("test finished")
                 
    webpage.save()

    sys.exit()
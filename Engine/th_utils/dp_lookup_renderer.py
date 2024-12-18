import torch
import torch.nn as nn
import torch.nn.functional as F
#from PIL import Image

import matplotlib.pylab as plt
import numpy as np
import torch.nn.functional as F


def getDPTexListTensor(combinedtex_b):
    r, c = 4, 6
    combinedtex_b = combinedtex_b.permute(0, 2, 3, 1)
    psize = int(combinedtex_b.shape[1] / r)

    dptexlist = torch.zeros(combinedtex_b.shape[0], 24, psize, psize, combinedtex_b.shape[-1]).cuda()
    count = 0
    for i in range(r):
        for j in range(c):
            dptexlist[:, count] = combinedtex_b[:, i * psize:i * psize + psize, j * psize:j * psize + psize, : ]
            count += 1
    return dptexlist


class DPLookupRenderer(nn.Module):

    '''Given an IUV denspose image (iuvimage) and densepose texture(dptex), propogates the texture image by differentiable sampling (lookup_mode supports bilinear and nearest)'''
    def __init__(self, lookup_mode = 'bilinear'):
        super(DPLookupRenderer, self).__init__()
        self.lookup_mode = lookup_mode


    def forward(self, dp_comb, iuv_image_b):
        #input dp_tex, iuv_image

        dp_tex = getDPTexListTensor(dp_comb)

        iuv_image_b = iuv_image_b.permute(0, 2, 3, 1)

        nbatch = iuv_image_b.shape[0]
        rendered = torch.zeros(nbatch, dp_tex.shape[-1], iuv_image_b.shape[1], iuv_image_b.shape[2]).cuda()
        flowzero = torch.ones(nbatch, iuv_image_b.shape[1], iuv_image_b.shape[2], 2).cuda() * 5 #5 is random invalid number

        # for all 24 texmap
        for i in range(dp_tex.shape[1]):
            flow = torch.where(torch.unsqueeze(iuv_image_b[:, :, :, 0] == (i + 1), -1), iuv_image_b[:, :, :, 1:],
                               flowzero)

            input_t = dp_tex[:,i, ...].permute(0, 3, 2, 1)
            out_t = grid_sample_fix.grid_sample(input_t, flow, mode=self.lookup_mode, align_corners=True)
            #print(out_t.max(), dp_tex[i].max(), flow.max(), flow.min(), input_t.shape, flow.shape, out_t.shape)
            rendered += out_t

        return rendered

    def forward_single(self, dp_tex, iuv_image_b):
        # input dp_tex, iuv_image

        iuv_image_b = iuv_image_b.permute(0, 2, 3, 1)

        nbatch = iuv_image_b.shape[0]
        rendered = torch.zeros(nbatch, dp_tex.shape[-1], iuv_image_b.shape[1], iuv_image_b.shape[2]).cuda()
        flowzero = torch.ones(nbatch, iuv_image_b.shape[1], iuv_image_b.shape[2],
                              2).cuda() * 5  # 5 is random invalid number

        # for all 24 texmap
        for i in range(dp_tex.shape[0]):
            flow = torch.where(torch.unsqueeze(iuv_image_b[:, :, :, 0] == (i + 1), -1), iuv_image_b[:, :, :, 1:],
                               flowzero)

            input_t = dp_tex[i].unsqueeze(0).repeat(nbatch, 1, 1, 1).permute(0, 3, 2, 1)
            out_t = grid_sample_fix.grid_sample(input_t, flow, mode=self.lookup_mode, align_corners=True)
            # print(out_t.max(), dp_tex[i].max(), flow.max(), flow.min(), input_t.shape, flow.shape, out_t.shape)
            rendered += out_t

        return rendered

class DPLookupRendererNormal(nn.Module):

    '''Given an IUV denspose image (iuvimage) and densepose texture(dptex), propogates the texture image by differentiable sampling (lookup_mode supports bilinear and nearest)'''
    def __init__(self, lookup_mode = 'bilinear'):
        super(DPLookupRendererNormal, self).__init__()
        self.lookup_mode = lookup_mode


    def forward(self, normal_tex, normal_flow_b):
        nbatch = normal_flow_b.shape[0]
        normal_flow_b = normal_flow_b.permute(0, 2, 3, 1)
        flowzero = torch.ones(nbatch, normal_flow_b.shape[1], normal_flow_b.shape[2], 2).float().cuda() * 5


        
        flow = torch.where(torch.unsqueeze(normal_flow_b[:, :, :, 0] == 1, -1), normal_flow_b[:, :, :, 1:], flowzero)
        input_t = normal_tex

        #print('flow,', flow.shape)
        out_t = grid_sample_fix.grid_sample(input_t, flow, mode=self.lookup_mode, align_corners=True)

        return out_t

    def forward_single(self, normal_tex, normal_flow_b):
        nbatch = normal_flow_b.shape[0]
        flowzero = torch.ones(nbatch, normal_flow_b.shape[1], normal_flow_b.shape[2], 2) * 5

        flow = torch.where(torch.unsqueeze(normal_flow_b[:, :, :, 0] == 1, -1), normal_flow_b[:, :, :, 1:], flowzero)
        input_t = normal_tex.unsqueeze(0).repeat(nbatch, 1, 1, 1).permute(0, 3, 2, 1)
        out_t = grid_sample_fix.grid_sample(input_t, flow, mode=self.lookup_mode, align_corners=True)

        return out_t


if __name__ == '__main__':
    import sys


    sys.path.extend(['/HPS/impl_deep_volume/static00/detectron2/projects/DensePose/',
                     '/HPS/impl_deep_volume/static00/detectron2/projects/DensePose/densepose',
                     '/HPS/impl_deep_volume/static00/tex2shape/lib'])

    from structures import DensePoseResult
    # from maps import map_densepose_to_tex, normalize

    from detectron2.structures.boxes import BoxMode
    from util.densepose_utils import uvTransform, getDPImg, showDPtex, uvTransformDP, renderDP
    import pickle

    import cv2
    import six



    f = open('/HPS/impl_deep_volume3/static00/exp_data/marc_densepose/densepose.pkl', 'rb')
    dposer = pickle.load(f)
    dp = dposer[1]



    img = cv2.imread(dp['file_name'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    dp_image = getDPImg((256, 256, 3), dp)
    dp_tex = uvTransformDP(img, dp, 64)
    dp_tex = np.random.random((24, 64, 64, 128))


    #plt.imshow(dp_image)
    #plt.show()

    #preprocessing for dataloader, get_i function
    iuv_image = dp_image.copy().astype(dtype=np.float32)
    iuv_image[:, :, 1:] = (iuv_image[:, :, 1:] / 255.0)
    iuv_image[:, :, 1:] = (iuv_image[:, :, 1:] - 0.5) * 2

    #batch formation and to tensor
    iuv_image_b = torch.from_numpy(iuv_image)
    iuv_image_b = torch.unsqueeze(iuv_image_b, dim=0)

    dp_tex = torch.from_numpy(dp_tex).float()


    rendered = model(dp_tex, iuv_image_b)
    rendered_viz = rendered.permute(0, 2, 3, 1).numpy()
    plt.imshow(rendered_viz[0].sum(axis = -1))
    plt.show()



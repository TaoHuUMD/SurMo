import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .facenet import Sphere20a, senet50
import torch.nn.functional as F

#import util.util as util


###########################CUSTOM LOSSES################################
class FaceLoss(nn.Module):
    def __init__(self, pretrained_path='asset/spretrains/sphere20a_20171020.pth'):
        super(FaceLoss, self).__init__()
        if 'senet' in pretrained_path:
            self.net = senet50(include_top=False)
            self.load_senet_model(pretrained_path)
            self.height, self.width = 224, 224
        else:
            self.net = Sphere20a()
            self.load_sphere_model(pretrained_path)
            self.height, self.width = 112, 96

        self.net.eval()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        # from utils.demo_visualizer import MotionImitationVisualizer
        # self._visualizer = MotionImitationVisualizer('debug', ip='http://10.10.10.100', port=31102)

    def forward(self, imgs1, imgs2, kps1=None, kps2=None, bbox1=None, bbox2=None):
        """
        Args:
            imgs1:
            imgs2:
            kps1:
            kps2:
            bbox1:
            bbox2:

        Returns:

        """
        # filtering images
        imgs_n1, imgs_n2, bbox_n1, bbox_n2 = [], [], [], []
        for i in range(bbox1.shape[0]):
            if not torch.all(torch.eq(bbox1[i], torch.zeros(4))):
                imgs_n1.append(imgs1[i])
                imgs_n2.append((imgs2[i]))
                bbox_n1.append(bbox1[i])
                bbox_n2.append(bbox2[i])
        if len(imgs_n1) == 0:
            return 0.0

        [imgs_n1, imgs_n2, bbox_n1, bbox_n2] = map(torch.stack, [imgs_n1, imgs_n2, bbox_n1, bbox_n2])

        if kps1 is not None:
            head_imgs1 = self.crop_head_kps(imgs1, kps1)
        elif bbox1 is not None:
            head_imgs1 = self.crop_head_bbox(imgs_n1, bbox_n1)
        elif self.check_need_resize(imgs1):
            head_imgs1 = F.interpolate(imgs1, size=(self.height, self.width), mode='bilinear', align_corners=True)
        else:
            head_imgs1 = imgs1

        if kps2 is not None:
            head_imgs2 = self.crop_head_kps(imgs2, kps2)
        elif bbox2 is not None:
            head_imgs2 = self.crop_head_bbox(imgs_n2, bbox_n2)
        elif self.check_need_resize(imgs2):
            head_imgs2 = F.interpolate(imgs1, size=(self.height, self.width), mode='bilinear', align_corners=True)
        else:
            head_imgs2 = imgs2

        loss = self.compute_loss(head_imgs1, head_imgs2)

        # import cv2
        # facereal = util.tensor2im(head_imgs2[1])
        # cv2.imwrite('/HPS/impl_deep_volume3/static00/exp_data/tex_debug/face/facereal.jpg', facereal)

        return loss

    def compute_loss(self, img1, img2):
        """
        :param img1: (n, 3, 112, 96), [-1, 1]
        :param img2: (n, 3, 112, 96), [-1, 1], if it is used in training,
                     img2 is reference image (GT), use detach() to stop backpropagation.
        :return:
        """
        f1, f2 = self.net(img1), self.net(img2)

        loss = 0.0
        for i in range(len(f1)):
            loss += self.criterion(f1[i], f2[i].detach())

        return loss

    def check_need_resize(self, img):
        return img.shape[2] != self.height or img.shape[3] != self.width

    def crop_head_bbox(self, imgs, bboxs):
        """
        Args:
            bboxs: (N, 4), 4 = [lt_x, lt_y, rt_x, rt_y]
            #wrong you mf!, code says xx yy, and documents says xy xy, you slimy f

        Returns:
            resize_image:
        """
        bs, _, ori_h, ori_w = imgs.shape

        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = bboxs[i]
            head = imgs[i:i + 1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
            head = F.interpolate(head, size=(self.height, self.width), mode='bilinear', align_corners=True)
            head_imgs.append(head)

        head_imgs = torch.cat(head_imgs, dim=0)

        return head_imgs

    def crop_head_kps(self, imgs, kps):
        """
        :param imgs: (N, C, H, W)
        :param kps: (N, 19, 2)
        :return:
        """
        bs, _, ori_h, ori_w = imgs.shape

        rects = self.find_head_rect(kps, ori_h, ori_w)
        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = rects[i]
            head = imgs[i:i + 1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
            head = F.interpolate(head, size=(self.height, self.width), mode='bilinear', align_corners=True)
            head_imgs.append(head)

        head_imgs = torch.cat(head_imgs, dim=0)

        return head_imgs

    @staticmethod
    @torch.no_grad()
    def find_head_rect(kps, height, width):
        NECK_IDS = 12

        kps = (kps + 1) / 2.0

        necks = kps[:, NECK_IDS, 0]
        zeros = torch.zeros_like(necks)
        ones = torch.ones_like(necks)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_x, _ = torch.min(kps[:, NECK_IDS:, 0] - 0.05, dim=1)
        min_x = torch.max(min_x, zeros)

        max_x, _ = torch.max(kps[:, NECK_IDS:, 0] + 0.05, dim=1)
        max_x = torch.min(max_x, ones)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_y, _ = torch.min(kps[:, NECK_IDS:, 1] - 0.05, dim=1)
        min_y = torch.max(min_y, zeros)

        max_y, _ = torch.max(kps[:, NECK_IDS:, 1], dim=1)
        max_y = torch.min(max_y, ones)

        min_x = (min_x * width).long()  # (T, 1)
        max_x = (max_x * width).long()  # (T, 1)
        min_y = (min_y * height).long()  # (T, 1)
        max_y = (max_y * height).long()  # (T, 1)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        rects = torch.stack((min_x, max_x, min_y, max_y), dim=1)

        # import ipdb
        # ipdb.set_trace()

        return rects

    def load_senet_model(self, pretrain_model):
        # saved_data = torch.load(pretrain_model, encoding='latin1')
        from utils.util import load_pickle_file
        saved_data = load_pickle_file(pretrain_model)
        save_weights_dict = dict()

        for key, val in saved_data.items():
            if key.startswith('fc'):
                continue
            save_weights_dict[key] = torch.from_numpy(val)

        self.net.load_state_dict(save_weights_dict)

        print('load face model from {}'.format(pretrain_model))

    def load_sphere_model(self, pretrain_model):
        saved_data = torch.load(pretrain_model)
        save_weights_dict = dict()

        for key, val in saved_data.items():
            if key.startswith('fc6'):
                continue
            save_weights_dict[key] = val

        self.net.load_state_dict(save_weights_dict)

        print('load face model from {}'.format(pretrain_model))


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    #by default it treats as if it is from the same identity
    def forward(self, output1, output2, target = 1, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target * distances +
                        (1 + -1 * target) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
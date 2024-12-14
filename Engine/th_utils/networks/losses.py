import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

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

class TotalVariation(nn.Module):
    r"""Computes the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N,)` or scalar.

    Examples:
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([0., 0.])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    
    def __init__(self, direction="hw"):
        super(TotalVariation, self).__init__()
        self.direction = direction

    def forward(self, img) -> torch.Tensor:
        if self.direction == "hw":
            return self.total_variation_hw(img)
        elif self.direction == "h":
            return self.total_variation_h(img)
        elif self.direction == "w":
            return self.total_variation_w(img)
    
    def total_variation_h(self, img: torch.Tensor) -> torch.Tensor:
        r"""Function that computes Total Variation according to [1].

        Args:
            img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

        Return:
            a scalar with the computer loss.

        Examples:
            >>> total_variation(torch.ones(3, 4, 4))
            tensor(0.)

        Reference:
            [1] https://en.wikipedia.org/wiki/Total_variation
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

        if len(img.shape) < 3 or len(img.shape) > 4:
            raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

        pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]

        reduce_axes = (-3, -2, -1)
        res1 = pixel_dif1.abs().sum(dim=reduce_axes)

        return res1
    
    def total_variation_w(self, img: torch.Tensor) -> torch.Tensor:
        r"""Function that computes Total Variation according to [1].

        Args:
            img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

        Return:
            a scalar with the computer loss.

        Examples:
            >>> total_variation(torch.ones(3, 4, 4))
            tensor(0.)

        Reference:
            [1] https://en.wikipedia.org/wiki/Total_variation
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

        if len(img.shape) < 3 or len(img.shape) > 4:
            raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

        pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

        reduce_axes = (-3, -2, -1)
        res2 = pixel_dif2.abs().sum(dim=reduce_axes)

        return res2
    
    def total_variation_hw(self, img: torch.Tensor) -> torch.Tensor:
        r"""Function that computes Total Variation according to [1].

        Args:
            img: the input image with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

        Return:
            a scalar with the computer loss.

        Examples:
            >>> total_variation(torch.ones(3, 4, 4))
            tensor(0.)

        Reference:
            [1] https://en.wikipedia.org/wiki/Total_variation
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

        if len(img.shape) < 3 or len(img.shape) > 4:
            raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}.")

        pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
        pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

        reduce_axes = (-3, -2, -1)
        res1 = pixel_dif1.abs().sum(dim=reduce_axes)
        res2 = pixel_dif2.abs().sum(dim=reduce_axes)

        return res1 + res2

def Total_variation_loss(y):
    tv = (
        torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
        torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    )
    
    tv /= (2*y.nelement())
    
    return tv

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids=0):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda(gpu_ids)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

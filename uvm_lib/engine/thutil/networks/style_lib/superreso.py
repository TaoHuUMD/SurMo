import torch
import torch.nn as nn

from .stylegan_v2.common.decoderv2 import Decoderv2Block
from .stylegan_v2.op import FusedLeakyReLU
from .stylegan_v2.modules import EqualConv2d

class SuperresoNet(torch.nn.Module):

    def __init__(
            self,
            in_channel,
            out_channel,
            upsample_factor=4, #2, 4
            norm_layer=None,
            same=False
    ):
        super().__init__()

        
        self.up1 = Decoderv2Block(in_channel, 0, in_channel)
        self.up2 = Decoderv2Block(in_channel, 0, in_channel)
        

        n_out = in_channel

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.rgb_head = nn.Sequential(
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, 3, 3, 1, 1, bias=True),
            nn.Tanh())


    def forward(self, input):
        out = self.up1(input)
        out = self.up2(out)
        return self.rgb_head(out)


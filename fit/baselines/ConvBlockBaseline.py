import torch
from fast_transformers.builders import TransformerDecoderBuilder, TransformerEncoderBuilder

from fit.transformers.PositionalEncoding2D import PositionalEncoding2D
from fit.utils import convert2FC, convert_to_dft
from torch.nn import functional as F


class ConvBlockBaseline(torch.nn.Module):
    def __init__(self,
                 d_query=32,):
        super(ConvBlockBaseline, self).__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, d_query, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(d_query),
            torch.nn.Conv2d(d_query, d_query, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(d_query),
            torch.nn.Conv2d(d_query, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        img_post = self.conv_block(x)
        img_post += x

        return img_post

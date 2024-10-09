from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from .kan_convolutional.KANConv import KAN_Convolutional_Layer
from .pos_enc import ImgPosEnc


class KanConv(nn.Module):
    def __init__(self):
        super(KanConv, self).__init__()
        self.kan_conv = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3)),
            nn.BatchNorm2d(2),
            torch.nn.AvgPool2d(kernel_size=(3, 3)),

            KAN_Convolutional_Layer(n_convs=4, kernel_size=(3, 3)),
            nn.BatchNorm2d(2*4),
            torch.nn.AvgPool2d(kernel_size=(2, 2)),

            KAN_Convolutional_Layer(n_convs=6, kernel_size=(3, 3)),
            nn.BatchNorm2d(2*4*6),
            torch.nn.AvgPool2d(kernel_size=(2, 2)),

            KAN_Convolutional_Layer(n_convs=8, kernel_size=(3, 3)),
            nn.BatchNorm2d(2*4*6*8),
            torch.nn.AvgPool2d(kernel_size=(2, 2)),
        )

    def forward(self, x, mask):
        return self.kan_conv(x), mask[:, 0::4, 0::4][:, 0::2, 0::2][:, 0::2, 0::2][:, 0::2, 0::2]


class KanConvEncoder(pl.LightningModule):
    def __init__(self, d_model: int):
        super(KanConvEncoder, self).__init__()
        self.kan_conv = KanConv()

        self.feature_proj = nn.Conv2d(
            2*4*6*8, d_model, kernel_size=1)

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, img, img_mask) -> Any:
        feature, mask = self.kan_conv(img, img_mask)
        feature = self.feature_proj(feature)

        feature = rearrange(feature, "b d h w -> b h w d")

        feature = self.pos_enc_2d(feature, mask)
        feature = self.norm(feature)

        return feature, mask

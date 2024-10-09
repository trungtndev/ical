from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from .kan_convolutional.KANConv import KAN_Convolutional_Layer
from .pos_enc import ImgPosEnc


class KanConv(nn.Module):
    def __init__(self, num_layers=[2, 4, 6]):
        super(KanConv, self).__init__()

        n1, n2, n3 = num_layers

        self.kan_conv = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=n1, kernel_size=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n1),
            torch.nn.AvgPool2d(kernel_size=(2, 2)),

            KAN_Convolutional_Layer(n_convs=n2, kernel_size=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n1 * n2),
            torch.nn.AvgPool2d(kernel_size=(2, 2)),

            KAN_Convolutional_Layer(n_convs=n3, kernel_size=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n1 * n2 * n3),
            torch.nn.AvgPool2d(kernel_size=(2, 2)),

            # KAN_Convolutional_Layer(n_convs=5, kernel_size=(2, 2), padding=(1, 1)),
            # nn.BatchNorm2d(2 * 3 * 4 * 5),
            # torch.nn.AvgPool2d(kernel_size=(2, 2)),
        )

    def forward(self, x, mask):
        return self.kan_conv(x), mask[:, 0::2, 0::2][:, 0::2, 0::2][:, 0::2, 0::2]


class KanConvEncoder(pl.LightningModule):
    def __init__(self, d_model: int, num_layers=[2, 4, 6]):
        super(KanConvEncoder, self).__init__()
        n1, n2, n3 = num_layers
        self.kan_conv = KanConv(num_layers)

        self.feature_proj = nn.Conv2d(
            n1 * n2 * n3, d_model, kernel_size=1)

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, img, img_mask) -> Any:
        feature, mask = self.kan_conv(img, img_mask)
        feature = self.feature_proj(feature)

        feature = rearrange(feature, "b d h w -> b h w d")

        feature = self.pos_enc_2d(feature, mask)
        feature = self.norm(feature)

        return feature, mask

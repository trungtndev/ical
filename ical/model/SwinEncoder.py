import pytorch_lightning as pl
import torch
import timm
from timm.models.swin_transformer import SwinTransformer


class SwinV1Encoder(pl.LightningModule):
    def __init__(self,
                 d_model: int,
                 requires_grad=True,
                 drop_rate=0.1,
                 proj_drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 ):
        super().__init__()
        self.swinv1 = SwinTransformer(
            img_size=224,
            patch_size=8,
            in_chans=1,
            embed_dim=64,
            depths=(2, 8, 2),
            num_heads=(3, 8, 12),
            window_size=(12, 6, 4),
            mlp_ratio=4,

            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # add output layer
        self.swinv1.head = torch.nn.Sequential(
            torch.nn.Linear(256, d_model),
            torch.nn.LayerNorm(d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(drop_rate),
        )

    #
    def forward(self, img, img_mask):
        x = self.swinv1(img)
        img_mask = img_mask[:, 0::4, 0::4][:, 0::2, 0::2][:, 0::2, 0::2][:, 0::2, 0::2]
        return x, img_mask

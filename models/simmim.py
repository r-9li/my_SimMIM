# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .MiT_transformer import MiT


class MitForSimMIM(MiT):
    def __init__(self, mask_patch_size, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(3, mask_patch_size, mask_patch_size))
        self.mask_patch_size = mask_patch_size
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def process_x(self, x, mask):
        assert mask is not None
        B, _, H, W = x.shape

        mask_tokens = self.mask_token.repeat(B, 1, int(H / self.mask_patch_size), int(W / self.mask_patch_size))
        w = mask.type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w
        return x

    def forward(self, x, mask):
        x = self.process_x(x, mask)
        _, _, _, x = self._forward(x)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride, encoder_output_dim, in_chans):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=encoder_output_dim,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = in_chans

    def forward(self, x, mask):
        mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config):
    model_type = config.MODEL.TYPE
    if model_type == 'mit':
        encoder = MitForSimMIM(model_name=config.MODEL.MIT.SIZE, mask_patch_size=config.DATA.MASK_PATCH_SIZE)
        out_dim = config.MODEL.MIT.OUTPUT_DIM
        encoder_stride = 32
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride, encoder_output_dim=out_dim, in_chans=3)

    return model

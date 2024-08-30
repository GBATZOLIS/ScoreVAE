import math
from numbers import Number
from typing import NamedTuple, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

# Import necessary items from BeatGANs_choices.py
from .BeatGANs_utils.BeatGANs_choices import ModelType

# Import necessary items from config_base.py
from .BeatGANs_utils.config_base import BaseConfig

# Import necessary items from BeatGANs_blocks.py
from .BeatGANs_utils.BeatGANs_blocks import (
    ResBlock, ResBlockConfig, TimestepEmbedSequential, AttentionBlock, 
    Downsample, Upsample
)

# Import necessary items from BeatGANs_nn.py
from .BeatGANs_utils.BeatGANs_nn import (
    conv_nd, linear, normalization, timestep_embedding, 
    zero_module, SiLU
)

class BeatGANsUNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.conf = model_config

        if self.conf.num_heads_upsample == -1:
            self.num_heads_upsample = self.conf.num_heads

        self.time_emb_channels = self.conf.time_embed_channels or self.conf.model_channels
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, self.conf.embed_channels),
            nn.SiLU(),
            linear(self.conf.embed_channels, self.conf.embed_channels),
        )

        ch = input_ch = int(self.conf.channel_mult[0] * self.conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(self.conf.dims, self.conf.in_channels, ch, 3, padding=1))
        ])

        kwargs = dict(
            use_condition=True,
            two_cond=self.conf.resnet_two_cond,
            use_zero_module=self.conf.resnet_use_zero_module,
            cond_emb_channels=self.conf.resnet_cond_channels,
        )

        self._feature_size = ch
        input_block_chans = [[] for _ in range(len(self.conf.channel_mult))]
        input_block_chans[0].append(ch)

        self.input_num_blocks = [0 for _ in range(len(self.conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(self.conf.channel_mult))]

        ds = 1
        resolution = self.conf.image_size
        for level, mult in enumerate(self.conf.input_channel_mult or self.conf.channel_mult):
            for _ in range(self.conf.num_input_res_blocks or self.conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch, self.conf.embed_channels, self.conf.dropout,
                        out_channels=int(mult * self.conf.model_channels),
                        dims=self.conf.dims, use_checkpoint=self.conf.use_checkpoint, **kwargs,
                    ).make_model()
                ]
                ch = int(mult * self.conf.model_channels)
                if resolution in self.conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=self.conf.use_checkpoint or self.conf.attn_checkpoint,
                            num_heads=self.conf.num_heads, num_head_channels=self.conf.num_head_channels,
                            use_new_attention_order=self.conf.use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1

            if level != len(self.conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch, self.conf.embed_channels, self.conf.dropout,
                            out_channels=out_ch, dims=self.conf.dims,
                            use_checkpoint=self.conf.use_checkpoint, down=True, **kwargs,
                        ).make_model() if self.conf.resblock_updown else Downsample(
                            ch, self.conf.conv_resample, dims=self.conf.dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch, self.conf.embed_channels, self.conf.dropout,
                dims=self.conf.dims, use_checkpoint=self.conf.use_checkpoint, **kwargs,
            ).make_model(),
            AttentionBlock(
                ch, use_checkpoint=self.conf.use_checkpoint or self.conf.attn_checkpoint,
                num_heads=self.conf.num_heads, num_head_channels=self.conf.num_head_channels,
                use_new_attention_order=self.conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch, self.conf.embed_channels, self.conf.dropout,
                dims=self.conf.dims, use_checkpoint=self.conf.use_checkpoint, **kwargs,
            ).make_model(),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.conf.channel_mult))[::-1]:
            for i in range(self.conf.num_res_blocks + 1):
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    ich = 0
                layers = [
                    ResBlockConfig(
                        channels=ch + ich, emb_channels=self.conf.embed_channels, dropout=self.conf.dropout,
                        out_channels=int(self.conf.model_channels * mult), dims=self.conf.dims,
                        use_checkpoint=self.conf.use_checkpoint, has_lateral=True if ich > 0 else False,
                        lateral_channels=None, **kwargs,
                    ).make_model()
                ]
                ch = int(self.conf.model_channels * mult)
                if resolution in self.conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=self.conf.use_checkpoint or self.conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample, num_head_channels=self.conf.num_head_channels,
                            use_new_attention_order=self.conf.use_new_attention_order,
                        ))
                if level and i == self.conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch, self.conf.embed_channels, self.conf.dropout,
                            out_channels=out_ch, dims=self.conf.dims,
                            use_checkpoint=self.conf.use_checkpoint, up=True, **kwargs,
                        ).make_model() if self.conf.resblock_updown else Upsample(
                            ch, self.conf.conv_resample, dims=self.conf.dims, out_channels=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        if self.conf.resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(
                    conv_nd(self.conf.dims, input_ch, self.conf.out_channels, 3, padding=1)
                ),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(self.conf.dims, input_ch, self.conf.out_channels, 3, padding=1),
            )

    def forward(self, x, y, t):
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        h = x  # No need to change type here
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        h = self.middle_block(h, emb=emb)
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                try:
                    lateral = hs[-i - 1].pop()
                except IndexError:
                    lateral = None
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        pred = self.out(h)
        return pred
    
    def get_score_fn(self, sde, train):
        """
        Returns a function that computes the score.
        
        Args:
            sde: The SDE object that provides the marginal probability.
            train: Boolean flag indicating whether in training mode.
            
        Returns:
            score_fn: A function that computes the score based on the diffusion model's noise prediction.
        """
        def score_fn(x, y, t):
            noise_prediction = self.forward(x, y, t)
            _, std = sde.marginal_prob(x, t)
            std = std.view(std.shape[0], *[1 for _ in range(len(x.shape) - 1)])  # Expand std to match the shape of noise_prediction
            score = -noise_prediction / std
            return score
        
        return score_fn

    def print_model_summary(self):
        """
        Prints the number of trainable and non-trainable parameters in the diffusion model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_non_trainable_params = total_params - total_trainable_params

        print(f"Total number of parameters: {total_params}")
        print(f"Total number of trainable parameters: {total_trainable_params}")
        print(f"Total number of non-trainable parameters: {total_non_trainable_params}")


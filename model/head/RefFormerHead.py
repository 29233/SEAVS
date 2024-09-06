import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.ops import Conv2d, SpatialReducedAttention, ChannelAttention

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

import fvcore.nn.weight_init as weight_init

BN_MOMENTUM = 0.1


# Fusion Block presented in ReferFormer
class FusionBlock(nn.Module):
    def __init__(self, scale_factor, d_model, nhead, pos_emb, dropout=0.1):
        super().__init__()
        self.spatial_attn = SpatialReducedAttention(scale_factor, d_model, nhead, pos_emb, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, audio_feat, visual_feat):
        bt, c, h, w = visual_feat.shape
        visual_feat = self.spatial_attn(visual_feat)  # output: bst, hw, c
        k, v = self.kv_proj(audio_feat).chunk(2, dim=-1)
        tgt = self.cross_attn(visual_feat, k, v, attn_mask=None, key_padding_mask=None)[0]
        tgt = tgt + self.dropout(visual_feat)
        tgt1 = self.norm1(tgt)
        tgt2 = self.ffn(tgt1)
        tgt3 = self.norm2(tgt2)  # bst, hw, c
        return tgt3.permute(0, 2, 1).reshape(bt, c, h, w)


class FusionBlock_CHA(nn.Module):
    def __init__(self, scale_factor, d_model, nhead, pos_emb, dropout=0.1):
        super().__init__()
        self.spatial_attn = SpatialReducedAttention(scale_factor, d_model, nhead, pos_emb, dropout)
        self.channel_attn = ChannelAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, audio_feat, visual_feat):
        bt, c, h, w = visual_feat.shape
        visual_feat = self.spatial_attn(visual_feat)  # output: bst, hw, c
        tgt = self.channel_attn(audio_feat, visual_feat)
        tgt1 = self.norm1(tgt)
        tgt2 = self.ffn(tgt1)
        tgt3 = self.norm2(tgt2)  # bst, hw, c
        return tgt3.permute(0, 2, 1).reshape(bt, c, h, w)



class RefFormerHead(nn.Module):
    def __init__(self, scale_factors=[8, 4, 2, 1], d_models=[64, 128, 320, 512], nhead=8, pos_emb=None, dropout=0.1,
                 conv_dim=256, mask_dim=1, use_bias=False, interpolate_scale=4, *args, **kwargs):
        super().__init__()
        lateral_convs = []
        output_convs = []
        fusion_blocks = []
        self.interpolate_scale = interpolate_scale
        for idx, (scale_factor, d_model) in enumerate(zip(scale_factors, d_models)):
            fusion_block = FusionBlock(scale_factor, d_model, nhead, pos_emb, dropout)
            self.add_module("fusion_block_{}".format(idx), fusion_block)
            fusion_blocks.append(fusion_block)

            # in_channels: 4x -> 32x
            lateral_norm = None
            output_norm = None

            lateral_conv = Conv2d(  # 降维卷积 1 x 1
                d_model, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(  # 输出卷积 3 x 3
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)  # 权重初始化
            weight_init.c2_xavier_fill(output_conv)
            stage = idx + 1
            self.add_module("adapter_{}".format(stage), lateral_conv)
            self.add_module("layer_{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]  # res5 -> res2
        self.output_convs = output_convs[::-1]
        self.fusion_blocks = fusion_blocks[::-1]
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    def forward(self, audio_feats, vis_feats):
        '''
        vis_feat: list[bs * t, dim, h, w]      res2 -> res5
        '''
        for idx, (audio_feat, vis_feat) in enumerate(zip(audio_feats[::-1], vis_feats[::-1])):  # res5 -> res2
            fusion_block = self.fusion_blocks[idx]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            vis_feat = fusion_block(audio_feat, vis_feat)
            cur_fpn = lateral_conv(vis_feat)

            if idx == 0:
                output = output_conv(cur_fpn)
            else:
                output = cur_fpn + F.interpolate(output, cur_fpn.shape[-2:], mode='nearest')
                output = output_conv(output)
        output = F.interpolate(output, scale_factor=self.interpolate_scale, mode='bilinear')
        return self.mask_features(output)


class RefFormerHead_res50(nn.Module):
    def __init__(self, scale_factors=[8, 4, 2, 1], d_models=[64, 128, 320, 512], nhead=8, pos_emb=None, dropout=0.1,
                 conv_dim=256, mask_dim=1, use_bias=False, interpolate_scale=4, *args, **kwargs):
        super().__init__()
        lateral_convs = []
        output_convs = []
        audio_projs = []
        fusion_blocks = []
        self.interpolate_scale = interpolate_scale
        for idx, (scale_factor, d_model) in enumerate(zip(scale_factors, d_models)):
            fusion_block = FusionBlock(scale_factor, conv_dim, nhead, pos_emb, dropout)
            self.add_module("fusion_block_{}".format(idx), fusion_block)
            fusion_blocks.append(fusion_block)

            # in_channels: 4x -> 32x
            lateral_norm = None
            output_norm = None

            audio_proj = nn.Linear(d_model, conv_dim)
            lateral_conv = Conv2d(  # 降维卷积 1 x 1
                d_model, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(  # 输出卷积 3 x 3
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)  # 权重初始化
            weight_init.c2_xavier_fill(output_conv)
            stage = idx + 1
            self.add_module("adapter_{}".format(stage), lateral_conv)
            self.add_module("layer_{}".format(stage), output_conv)
            self.add_module("audio_proj_{}".format(stage), audio_proj)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
            audio_projs.append(audio_proj)
        self.lateral_convs = lateral_convs[::-1]  # res5 -> res2
        self.output_convs = output_convs[::-1]
        self.fusion_blocks = fusion_blocks[::-1]
        self.audio_projs = audio_projs[::-1]
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    def forward(self, audio_feats, vis_feats):
        '''
        vis_feat: list[bs * t, dim, h, w]      res2 -> res5
        '''
        for idx, (audio_feat, vis_feat) in enumerate(zip(audio_feats[::-1], vis_feats[::-1])):  # res5 -> res2
            fusion_block = self.fusion_blocks[idx]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            audio_feat = self.audio_projs[idx](audio_feat)
            vis_feat = lateral_conv(vis_feat)
            cur_fpn = fusion_block(audio_feat, vis_feat)

            if idx == 0:
                output = output_conv(cur_fpn)
            else:
                output = cur_fpn + F.interpolate(output, cur_fpn.shape[-2:], mode='nearest')
                output = output_conv(output)
        output = F.interpolate(output, scale_factor=self.interpolate_scale, mode='bilinear')
        return self.mask_features(output)


class RefFormerHead_CHA(nn.Module):
    def __init__(self, scale_factors=[8, 4, 2, 1], d_models=[64, 128, 320, 512], nhead=8, pos_emb=None, dropout=0.1,
                 conv_dim=256, mask_dim=1, use_bias=False, interpolate_scale=4, *args, **kwargs):
        super().__init__()
        lateral_convs = []
        output_convs = []
        fusion_blocks = []
        self.interpolate_scale = interpolate_scale
        for idx, (scale_factor, d_model) in enumerate(zip(scale_factors, d_models)):
            fusion_block = FusionBlock_CHA(scale_factor, d_model, nhead, pos_emb, dropout)
            self.add_module("fusion_block_{}".format(idx), fusion_block)
            fusion_blocks.append(fusion_block)

            # in_channels: 4x -> 32x
            lateral_norm = None
            output_norm = None

            lateral_conv = Conv2d(  # 降维卷积 1 x 1
                d_model, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(  # 输出卷积 3 x 3
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)  # 权重初始化
            weight_init.c2_xavier_fill(output_conv)
            stage = idx + 1
            self.add_module("adapter_{}".format(stage), lateral_conv)
            self.add_module("layer_{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]  # res5 -> res2
        self.output_convs = output_convs[::-1]
        self.fusion_blocks = fusion_blocks[::-1]
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    def forward(self, audio_feats, vis_feats):
        '''
        vis_feat: list[bs * t, dim, h, w]      res2 -> res5
        '''
        for idx, (audio_feat, vis_feat) in enumerate(zip(audio_feats[::-1], vis_feats[::-1])):  # res5 -> res2
            fusion_block = self.fusion_blocks[idx]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            vis_feat = fusion_block(audio_feat, vis_feat)
            cur_fpn = lateral_conv(vis_feat)

            if idx == 0:
                output = output_conv(cur_fpn)
            else:
                output = cur_fpn + F.interpolate(output, cur_fpn.shape[-2:], mode='nearest')
                output = output_conv(output)
        output = F.interpolate(output, scale_factor=self.interpolate_scale, mode='bilinear')
        return self.mask_features(output)


class RefFormerHead_res50_CHA(nn.Module):
    def __init__(self, scale_factors=[8, 4, 2, 1], d_models=[64, 128, 320, 512], nhead=8, pos_emb=None, dropout=0.1,
                 conv_dim=256, mask_dim=1, use_bias=False, interpolate_scale=4, *args, **kwargs):
        super().__init__()
        lateral_convs = []
        output_convs = []
        audio_projs = []
        fusion_blocks = []
        self.interpolate_scale = interpolate_scale
        for idx, (scale_factor, d_model) in enumerate(zip(scale_factors, d_models)):
            fusion_block = FusionBlock_CHA(scale_factor, conv_dim, nhead, pos_emb, dropout)
            self.add_module("fusion_block_{}".format(idx), fusion_block)
            fusion_blocks.append(fusion_block)

            # in_channels: 4x -> 32x
            lateral_norm = None
            output_norm = None

            audio_proj = nn.Linear(d_model, conv_dim)
            lateral_conv = Conv2d(  # 降维卷积 1 x 1
                d_model, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(  # 输出卷积 3 x 3
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)  # 权重初始化
            weight_init.c2_xavier_fill(output_conv)
            stage = idx + 1
            self.add_module("adapter_{}".format(stage), lateral_conv)
            self.add_module("layer_{}".format(stage), output_conv)
            self.add_module("audio_proj_{}".format(stage), audio_proj)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
            audio_projs.append(audio_proj)
        self.lateral_convs = lateral_convs[::-1]  # res5 -> res2
        self.output_convs = output_convs[::-1]
        self.fusion_blocks = fusion_blocks[::-1]
        self.audio_projs = audio_projs[::-1]
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    def forward(self, audio_feats, vis_feats):
        '''
        vis_feat: list[bs * t, dim, h, w]      res2 -> res5
        '''
        for idx, (audio_feat, vis_feat) in enumerate(zip(audio_feats[::-1], vis_feats[::-1])):  # res5 -> res2
            fusion_block = self.fusion_blocks[idx]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            audio_feat = self.audio_projs[idx](audio_feat)
            vis_feat = lateral_conv(vis_feat)
            cur_fpn = fusion_block(audio_feat, vis_feat)

            if idx == 0:
                output = output_conv(cur_fpn)
            else:
                output = cur_fpn + F.interpolate(output, cur_fpn.shape[-2:], mode='nearest')
                output = output_conv(output)
        output = F.interpolate(output, scale_factor=self.interpolate_scale, mode='bilinear')
        return self.mask_features(output)


if __name__ == '__main__':
    cha = ChannelAttention(8, 64, 0.1).cuda()
    visual_Feat = torch.randn(20, 3136, 64).cuda()
    audio_feat = torch.randn(20, 1, 64).cuda()
    cha(audio_feat, visual_Feat)
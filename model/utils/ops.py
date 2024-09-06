from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.utils.positional_encoding import SinePositionalEncoding


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv2d(n, k, kernel_size=1, stride=1, padding=0)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SpatialReducedAttention(nn.Module):
    def __init__(self, scale_factor, d_model, nhead, pos_emb, drop_out=0.1):
        super().__init__()
        self.scale_factor = scale_factor  # decoder中使用列表循环生成
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=drop_out, batch_first=True)
        self.dropout = nn.Dropout(drop_out)
        self.norm = nn.LayerNorm(d_model)
        # self.cross = cross
        if pos_emb is not None:
            self.pos_emb = SinePositionalEncoding(d_model // 2)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x: Tensor, y: Optional[Tensor] = None):  # [bt, c, h, w]
        if y is not None:
            assert x.shape == y.shape, "The shape of x and y should be the same"
            bt, c, h, w = x.shape
            mask = torch.zeros((bt, h, w), device=x.device, dtype=torch.bool)
            pos_emb = self.pos_emb(mask) if hasattr(self, 'pos_emb') else None
            x_ = self.with_pos_embed(x, pos_emb)
            y_ = self.with_pos_embed(y, pos_emb)
            q = x_.view(bt, c, -1).permute(0, 2, 1)
            if self.scale_factor > 1:
                new_h = int(h * 1. / self.scale_factor)
                new_w = int(w * 1. / self.scale_factor)
                size = (new_h, new_w)
                _y = F.interpolate(y_, size=size, mode='nearest')
            else:
                _y = y_
            _y = _y.view(bt, c, -1).permute(0, 2, 1)
            k, v = self.kv_proj(_y).chunk(2, dim=-1)
            tgt = self.self_attn(q, k, v, attn_mask=None, key_padding_mask=None)[0]
            res = x.view(bt, c, -1).permute(0, 2, 1) + self.dropout(tgt)
            res = self.norm(res)
            return res
        else:
            bt, c, h, w = x.shape
            mask = torch.zeros((bt, h, w), device=x.device, dtype=torch.bool)
            pos_emb = self.pos_emb(mask) if hasattr(self, 'pos_emb') else None
            x_ = self.with_pos_embed(x, pos_emb)
            q = x_.view(bt, c, -1).permute(0, 2, 1)
            if self.scale_factor > 1:
                new_h = int(h * 1. / self.scale_factor)
                new_w = int(w * 1. / self.scale_factor)
                size = (new_h, new_w)
                _x = F.interpolate(x_, size=size, mode='nearest')
            else:
                _x = x_
            _x = _x.view(bt, c, -1).permute(0, 2, 1)
            k, v = self.kv_proj(_x).chunk(2, dim=-1)
            tgt = self.self_attn(q, k, v, attn_mask=None, key_padding_mask=None)[0]
            res = x.view(bt, c, -1).permute(0, 2, 1) + self.dropout(tgt)
            res = self.norm(res)
            return res  # b, n, c


class ChannelAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, batch_first=True):
        super(ChannelAttention, self).__init__()
        self.atten = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_feat, visual_feat):
        weights = self.atten(audio_feat, visual_feat, visual_feat)[0]
        tgt = weights * visual_feat
        return self.dropout(tgt) + visual_feat



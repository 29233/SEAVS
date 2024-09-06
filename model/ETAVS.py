import torch
from torch import nn

from .backbone import build_backbone
from .vggish import VGGish
from .head import build_head
from .adapters import build_bridger



class AVS_Model(nn.Module):
    def __init__(self,
                 backbone,
                 vggish,
                 bridger,
                 head,
                 freeze,
                 *args,
                 **kwargs):

        super().__init__()

        self.visual_encoder = build_backbone(**backbone)
        self.audio_encoder = VGGish(**vggish)

        self.bridger = build_bridger(**bridger)
        self.decoder = build_head(**head)

        self.freeze(**freeze)

    def freeze(self, audio_backbone, visual_backbone):
        if audio_backbone:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
        if visual_backbone:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False


    def tune_backbone(self):
        for p in self.visual_encoder.parameters():
            p.requires_grad = True

    def mul_temporal_mask(self, feats, vid_temporal_mask_flag=None):
        if vid_temporal_mask_flag is None:
            return feats
        else:
            if isinstance(feats, list):
                out = []
                for x in feats:
                    out.append(x * vid_temporal_mask_flag)
            elif isinstance(feats, torch.Tensor):
                out = feats * vid_temporal_mask_flag

            return out

    def forward(self, audio_feat, vis_feat,
                vid_temporal_mask_flag=None):  # 这里不确定是将vis_feat与audio_feat混合较好，还是分别采用self-attn处理较好，故首先采用第一种方式来处理
        '''
        vis_feat: bs * t, dim, h, w
        audio_feat: bs * t, dim, a, f
        '''
        if vid_temporal_mask_flag is not None:
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1)
        with torch.no_grad():
            audio_feat = self.audio_encoder(audio_feat)
        audio_feat = audio_feat.unsqueeze(1)  # 40, 1, 128

        audio_feats, vis_feats = self.bridger(audio_feat, vis_feat, self.visual_encoder)
        vis_feats = [self.mul_temporal_mask(vis_feat, vid_temporal_mask_flag) for vis_feat in vis_feats]
        output = self.decoder(audio_feats, vis_feats)
        output = self.mul_temporal_mask(output, vid_temporal_mask_flag)

        return output
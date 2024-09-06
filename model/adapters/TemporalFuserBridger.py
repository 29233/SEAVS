import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from utils.compute_flops import count_flops


class TemporalFusionBridger(nn.Module):  # fusion by intermediate queries
    def __init__(self, d_model, nheads, dropout, num_queries, t=5, hidden_dim=128):
        super().__init__()
        self.nheads = nheads
        self.num_queries = num_queries
        head_dim = hidden_dim // nheads
        self.scale = head_dim ** (-0.5)
        self.hidden_dim = hidden_dim
        self.t = t

        # intermediate queries
        self.intermediate_queries = nn.Embedding(num_queries, hidden_dim)  # 0.05M

        self.intermediate_query_proj = nn.Linear(hidden_dim, 2 * hidden_dim)  # 0.53M

        self.audio_feat_proj = nn.Linear(128, 2 * hidden_dim)  # 0.53M
        self.vis_feat_proj = nn.Linear(d_model, 2 * hidden_dim)  # 0.53M

        # self.audio_disentangler = nn.Linear(128, 5 * 128)
        self.dis_drop = nn.Dropout(dropout)

        self.audio_query_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.vis_query_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.audio_q_proj = nn.Linear(128, hidden_dim)
        self.visual_q_proj = nn.Linear(d_model, hidden_dim)

        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)  # 1.05M

        self.vis_norm = nn.LayerNorm(2 * hidden_dim)
        self.audio_norm = nn.LayerNorm(2 * hidden_dim)
        #
        self.vis_out = nn.Linear(hidden_dim, d_model)  # 0.26M
        self.vis_drop = nn.Dropout(dropout)
        self.audio_out = nn.Linear(hidden_dim, d_model)  # 0.26M
        self.audio_drop = nn.Dropout(dropout)

        # self.query_proj = nn.Linear(hidden_dim, d_model)

        nn.init.xavier_uniform_(self.intermediate_queries.weight)

    def forward(self, audio_feat, vis_feat, latent_query=None):
        """
        audio_feat: bs, (t h w), dim
        vis_feat: bs, (t h w), dim
        """
        bt, hw, _c = vis_feat.shape
        c = self.hidden_dim
        bs = bt // self.t
        audio_queries, vis_queries = self.intermediate_query_proj(self.intermediate_queries.weight).chunk(2, dim=-1)
        audio_queries = audio_queries.repeat(bt, 1, 1)
        vis_queries = vis_queries.repeat(bt, 1, 1)

        # with 1d conv
        # audio_feat = self.audio_disentangler(audio_feat)
        # audio_feat = self.dis_drop(audio_feat)
        # audio_feat = audio_feat.view(bt, 5, -1)

        audio_ki, audio_vi = self.audio_norm(self.audio_feat_proj(audio_feat)).chunk(2,
                                                                                     dim=-1)  # character i in audio_ki represents intermediate
        vis_ki, vis_vi = self.vis_norm(self.vis_feat_proj(vis_feat)).chunk(2, dim=-1)

        audio_queries = audio_queries.reshape(bt, self.num_queries, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        vis_queries = vis_queries.reshape(bt, self.num_queries, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        audio_ki = audio_ki.reshape(bt, -1, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        audio_vi = audio_vi.reshape(bt, -1, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        vis_ki = vis_ki.reshape(bt, hw, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        vis_vi = vis_vi.reshape(bt, hw, self.nheads, c // self.nheads).permute(0, 2, 1, 3)

        attn_map_a = torch.matmul(audio_queries, audio_ki.transpose(-2, -1)) * self.scale
        attn_map_a = F.softmax(attn_map_a, dim=-1)
        audio_intermediate_queries = (torch.matmul(attn_map_a, audio_vi) + audio_queries).transpose(1, 2).reshape(bt,
                                                                                                                  self.num_queries,
                                                                                                                  c)

        attn_map_v = torch.matmul(vis_queries, vis_ki.transpose(-2, -1)) * self.scale
        attn_map_v = F.softmax(attn_map_v, dim=-1)
        vis_intermediate_queries = (torch.matmul(attn_map_v, vis_vi) + vis_queries).transpose(1, 2).reshape(bt,
                                                                                                            self.num_queries,
                                                                                                            c)

        queries = torch.cat((audio_intermediate_queries, vis_intermediate_queries), dim=-2)
        queries = queries.reshape(bs, self.t, -1, c).reshape(bs, -1, c)
        queries = self.self_attn(queries, queries, queries)[0]
        queries = queries.reshape(bs, self.t, -1, c).reshape(bt, -1, c)
        audio_intermediate_queries, vis_intermediate_queries = queries.chunk(2, dim=-2)

        audio_kf, audio_vf = self.audio_query_proj(audio_intermediate_queries).chunk(2,
                                                                                     dim=-1)  # character f in audio_kf represents fuse
        vis_kf, vis_vf = self.vis_query_proj(vis_intermediate_queries).chunk(2, dim=-1)

        _audio_feat = self.audio_q_proj(audio_feat).reshape(bt, -1, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        _vis_feat = self.visual_q_proj(vis_feat).reshape(bt, -1, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        audio_kf = audio_kf.reshape(bt, self.num_queries, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        audio_vf = audio_vf.reshape(bt, self.num_queries, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        vis_kf = vis_kf.reshape(bt, self.num_queries, self.nheads, c // self.nheads).permute(0, 2, 1, 3)
        vis_vf = vis_vf.reshape(bt, self.num_queries, self.nheads, c // self.nheads).permute(0, 2, 1, 3)

        attn_map_av = torch.matmul(_audio_feat, vis_kf.transpose(-2, -1)) * self.scale
        attn_map_av = F.softmax(attn_map_av, dim=-1)
        audio_feat_ = torch.matmul(attn_map_av, vis_vf).transpose(1, 2).reshape(bt, -1, c)
        audio_feat_ = self.audio_drop(self.audio_out(audio_feat_ + audio_feat))

        attn_map_va = torch.matmul(_vis_feat, audio_kf.transpose(-2, -1)) * self.scale
        attn_map_va = F.softmax(attn_map_va, dim=-1)
        vis_feat_ = torch.matmul(attn_map_va, audio_vf).transpose(1, 2).reshape(bt, hw, c)
        vis_feat_ = self.vis_drop(self.vis_out(vis_feat_) + vis_feat)

        return audio_feat_, vis_feat_, audio_intermediate_queries


class TemporalFusers(nn.Module):
    def __init__(self, num_queries, nhead=8, dropout=0.1, feature_channel=[64, 128, 320, 512], *args, **kwargs):
        super().__init__()
        Fusers = []
        for i, channel in enumerate(feature_channel):
            Fusers.append(
                TemporalFusionBridger(d_model=channel, nheads=nhead, dropout=dropout, num_queries=num_queries))
        self.Fusers = nn.ModuleList(Fusers)

    def forward(self, audio_feat, vis_feat, visual_enc):

        vis_collect_feats = []
        audio_collect_feats = []
        idx = 0
        for i in range(visual_enc.num_stages):
            patch_embed_v = getattr(visual_enc, f"patch_embed{i + 1}")
            block_v = getattr(visual_enc, f"block{i + 1}")
            norm_v = getattr(visual_enc, f"norm{i + 1}")
            vis_feat, H, W = patch_embed_v(vis_feat)  # bs * t, dim, h * w
            for blk_v in block_v:
                vis_feat_ = blk_v.attn(blk_v.norm1(vis_feat), H, W)
                vis_feat = vis_feat + blk_v.drop_path(vis_feat_)

                vis_feat = vis_feat + blk_v.drop_path(blk_v.mlp(blk_v.norm2(vis_feat), H, W))
                idx += 1

            vis_feat = norm_v(vis_feat)
            # audio_feat bt, 1, c   20, 1, 128
            # vis_feat bt, hw, c    20, 3136, 64
            audio_feat_, vis_feat, query = self.Fusers[i](audio_feat, vis_feat)
            vis_feat = rearrange(vis_feat, 'bt (h w) d -> bt d h w', h=H, w=W)
            vis_collect_feats.append(vis_feat)
            audio_collect_feats.append(audio_feat_)

        return audio_collect_feats, vis_collect_feats


class TemporalFusers_res50(nn.Module):
    def __init__(self, num_queries, nhead=8, dropout=0.1, feature_channel=[64, 128, 320, 512], *args, **kwargs):
        super().__init__()
        Fusers = []
        for i, channel in enumerate(feature_channel):
            Fusers.append(
                TemporalFusionBridger(d_model=channel, nheads=nhead, dropout=dropout, num_queries=num_queries))
        self.Fusers = nn.ModuleList(Fusers)

    def forward(self, audio_feat, vis_feat, visual_enc):

        vis_collect_feats = []
        audio_collect_feats = []
        vis_feat = visual_enc.conv1(vis_feat)
        vis_feat = visual_enc.bn1(vis_feat)
        vis_feat = visual_enc.relu(vis_feat)
        vis_feat = visual_enc.maxpool(vis_feat)
        for i in range(visual_enc.num_stages):
            if i <= 1:
                layer_v = getattr(visual_enc, f"layer{i + 1}")
            else:
                layer_v = getattr(visual_enc, f"layer{i + 1}_1")

            vis_feat = layer_v(vis_feat)
            B, C, H, W = vis_feat.shape
            vis_feat = rearrange(vis_feat, 'b c h w -> b (h w) c')
            # audio_feat bt, 1, c   20, 1, 128
            # vis_feat bt, hw, c    20, 3136, 64
            audio_feat_, vis_feat, query = self.Fusers[i](audio_feat, vis_feat)
            vis_feat = rearrange(vis_feat, 'bt (h w) d -> bt d h w', h=H, w=W)
            vis_collect_feats.append(vis_feat)
            audio_collect_feats.append(audio_feat_)

        return audio_collect_feats, vis_collect_feats


class TemporalFusers_res50_w_query(nn.Module):
    def __init__(self, nhead=8, dropout=0.1, feature_channel=[64, 128, 320, 512], hidden_dim=128, *args, **kwargs):
        super().__init__()
        Fusers = []
        for i, channel in enumerate(feature_channel):
            Fusers.append(
                TemporalFusionBridger(d_model=channel, nheads=nhead, dropout=dropout, hidden_dim=hidden_dim))
        self.Fusers = nn.ModuleList(Fusers)

    def forward(self, audio_feat, vis_feat, visual_enc):

        vis_collect_feats = []
        audio_collect_feats = []
        query_collect = []
        vis_feat = visual_enc.conv1(vis_feat)
        vis_feat = visual_enc.bn1(vis_feat)
        vis_feat = visual_enc.relu(vis_feat)
        vis_feat = visual_enc.maxpool(vis_feat)
        for i in range(visual_enc.num_stages):
            if i <= 1:
                layer_v = getattr(visual_enc, f"layer{i + 1}")
            else:
                layer_v = getattr(visual_enc, f"layer{i + 1}_1")

            vis_feat = layer_v(vis_feat)
            B, C, H, W = vis_feat.shape
            vis_feat = rearrange(vis_feat, 'b c h w -> b (h w) c')
            # audio_feat bt, 1, c   20, 1, 128
            # vis_feat bt, hw, c    20, 3136, 64
            audio_feat_, vis_feat, query = self.Fusers[i](audio_feat, vis_feat)
            vis_feat = rearrange(vis_feat, 'bt (h w) d -> bt d h w', h=H, w=W)
            vis_collect_feats.append(vis_feat)
            audio_collect_feats.append(audio_feat_)
            # query_collect.append(query)

        return audio_collect_feats, vis_collect_feats


if __name__ == '__main__':
    # model = TemporalFusionBridger(d_model=64, nheads=8, dropout=0.1)
    model = TemporalFusionBridger(d_model=64, nheads=8, dropout=0.1).cuda()
    audio_feat = torch.randn(5, 1, 128).cuda()
    vis_feat = torch.randn(5, 3136, 64).cuda()

    print('%.2fGFLOPS' % count_flops(model, inputs=(audio_feat, vis_feat)))

    model2 = nn.MultiheadAttention(128, 8, dropout=0.1, batch_first=True)  # 1.05M
    queries = torch.randn(1, 50, 128).cuda()
    model2 = model2.cuda()

    print('%.2fGFLOPS' % count_flops(model2, inputs=(queries, queries, queries)))
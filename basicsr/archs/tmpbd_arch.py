"""
TMPBD: network architecture for the official BD (blur-down) model.

Structure:
    conv_first(3->64) -> 3x ResidualBlockNoBN(64)
    -> Align (key/value/convs 224->96, reconstruction 15x RB@96)
    -> upconv1(96->48) -> PixelShuffle(4) -> + bilinear(base x4)

Forward semantics follow the released TMPAlign:
    online frame-by-frame propagation with hidden states
    (offsets, key, (value_feat, rec_feat));
    offsets are estimated by the CUDA extension TemporalMotionPropagation
    (tmpalign_util.tmp_forward);
    warping = grid_sample(nearest, border, align_corners=True);
    fusion weight = exp(-||key - warped(last_key)||^2);
    hidden (value, rec) are initialized to zeros at the first frame.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer
from .gradlayer_util import GradLayer
from .tmpalign_util import tmp_forward


class TMPAlignBD(nn.Module):
    """Alignment + reconstruction module of the BD model (96-ch reconstruction)."""

    def __init__(self, num_feat=64, recon_feat=96, num_reconstruct_block=15):
        super().__init__()
        self.recon_feat = recon_feat
        self.convs = nn.Sequential(
            nn.Conv2d(num_feat * 2 + recon_feat, num_feat * 2 + recon_feat, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat * 2 + recon_feat, recon_feat, 3, padding=1))
        self.key = nn.Sequential(
            GradLayer(0.1), make_layer(ResidualBlockNoBN, 1, num_feat=num_feat),
            nn.Conv2d(num_feat, num_feat // 2, 3, padding=1),
            make_layer(ResidualBlockNoBN, 1, num_feat=num_feat // 2),
            nn.Conv2d(num_feat // 2, num_feat // 4, 3, padding=1), GradLayer(10))
        self.value = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, padding=1))
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=recon_feat)

    def forward(self, x, lqs, hidden_states=None):
        b, t, c, h, w = x.shape
        if hidden_states is None:
            hidden_offsets, hidden_key, hidden_feats = None, None, (None, None)
        else:
            hidden_offsets, hidden_key, hidden_feats = hidden_states
        inp_key = self.key(x.view(-1, c, h, w)).view(b, t, -1, h, w)
        last_key = hidden_key if hidden_key is not None else inp_key[:, 0]
        with torch.no_grad():
            offsets = tmp_forward(inp_key, hidden_offsets, last_key)
        gy, gx = torch.meshgrid(torch.arange(0, h).to(torch.int32), torch.arange(0, w).to(torch.int32), indexing='ij')
        pos = torch.stack((gx, gy), 2).to(x.device).view(1, 1, h, w, 2)
        ap = pos + offsets
        apx = 2 * ap[:, :, :, :, 0:1].to(torch.float) / max(w - 1, 1) - 1.0
        apy = 2 * ap[:, :, :, :, 1:2].to(torch.float) / max(h - 1, 1) - 1.0
        ap = torch.cat([apx, apy], dim=4)
        inp_value = self.value(x.view(-1, c, h, w)).view(b, t, -1, h, w)
        rec_feats = []
        for fi in range(t):
            cap, ck = ap[:, fi], inp_key[:, fi]
            if fi == 0 and hidden_feats[0] is None:
                wf_l = torch.zeros(b, inp_value.size(2), h, w, dtype=torch.float, device=x.device)
                wf_h = torch.zeros(b, self.recon_feat, h, w, dtype=torch.float, device=x.device)
            else:
                wf_l = F.grid_sample(hidden_feats[0], cap, mode='nearest', padding_mode='border', align_corners=True)
                wf_h = F.grid_sample(hidden_feats[1], cap, mode='nearest', padding_mode='border', align_corners=True)
                wl_key = F.grid_sample(last_key, cap, mode='nearest', padding_mode='border', align_corners=True)
                wgt = torch.exp(-torch.sum((ck - wl_key)**2, dim=1, keepdims=True))
                wf_l, wf_h = wf_l * wgt, wf_h * wgt
            rec = self.reconstruction(self.convs(torch.cat([wf_l, wf_h, inp_value[:, fi]], dim=1)))
            rec_feats.append(rec)
            hidden_feats = (inp_value[:, fi], rec)
            hidden_key = last_key = ck
        return torch.stack(rec_feats, dim=1), (offsets[:, -1], hidden_key, hidden_feats)


@ARCH_REGISTRY.register()
class TMPBD(nn.Module):
    """TMP network for the BD weights (yml: network_g.type: TMPBD)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_extract_block=3,
                 recon_feat=96, num_reconstruct_block=15, upscale=4, hr_in=False, **kwargs):
        super().__init__()
        self.upscale = upscale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.align = TMPAlignBD(num_feat=num_feat, recon_feat=recon_feat,
                                num_reconstruct_block=num_reconstruct_block)
        self.upconv1 = nn.Conv2d(recon_feat, recon_feat // 2, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, hidden_states=None, return_hs=False):
        b, t, c, h, w = x.size()
        fo = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        fl = self.feature_extraction(fo)
        feat, hidden_states = self.align(fl.view(b, t, -1, h, w), x, hidden_states)
        feat = feat.view(b * t, -1, h, w)
        out = self.pixel_shuffle(self.upconv1(feat)).view(b, t, c, self.upscale * h, self.upscale * w)
        base = F.interpolate(x.view(-1, c, h, w), scale_factor=self.upscale, mode='bilinear',
                             align_corners=False).view(b, t, c, self.upscale * h, self.upscale * w)
        out = out + base
        return (out, hidden_states) if return_hs else out

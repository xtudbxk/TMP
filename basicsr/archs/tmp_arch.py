'''
 Copyright 2023 xtudbxk
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer
from .tmpalign_util import TMPAlign


@ARCH_REGISTRY.register()
class TMP(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 hr_in=False):

        super().__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        self.hr_in = hr_in

        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # align
        self.align = TMPAlign(num_feat=num_feat)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, 48, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(4)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, hidden_states=None, return_hs=False):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        # extract features for each frame
        feat_origin = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_origin)

        # align
        feat, hidden_states = self.align(feat_l1.view(b, t, -1, h, w), x, hidden_states)
        feat = feat.view(b*t, -1, h, w)

        # upsample
        out = feat
        out = self.pixel_shuffle(self.upconv1(out)).view(b, t, c, 4*h, 4*w)
        if self.hr_in:
            base = x
        else:
            base = F.interpolate(x.view(-1, c, h, w), scale_factor=4, mode='bilinear', align_corners=False).view(b, t, c, 4*h, 4*w)
        out += base

        if return_hs is True:
            return out, hidden_states
        else:
            return out

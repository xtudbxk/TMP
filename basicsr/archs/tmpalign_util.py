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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tmp.tmpwrapper import tmp
from .arch_util import ResidualBlockNoBN, make_layer
from .gradlayer_util import GradLayer

def tmp_forward(keys, last_offsets=None, last_key=None, additional_jump=2, iters_s=1, iters_t=8, sigma=30):
   b, t, c, h, w = keys.shape

   # 
   if last_offsets is None:
       last_offsets = torch.zeros(b, h, w, 2, dtype=torch.int32, device=keys.device)

   # 
   offsets = []
   distance = torch.zeros(h, w, dtype=torch.float,device=keys.device)
   for bi in range(b):
       offsets_single_batch = []
       last_offsets_s = last_offsets[bi].contiguous()

       # for the first frame
       if last_key is None:
           offsets_single_batch.append(last_offsets_s)
           last_key_s = keys[bi, 0].contiguous()
       else:
           last_key_s = last_key[bi,0].contiguous()
           current_key_s = keys[bi,0].contiguous()
           current_offsets = torch.zeros_like(last_offsets_s).contiguous()
           tmp(current_key_s, last_key_s, current_offsets, last_offsets_s, distance, additional_jump=additional_jump, iters_s=iters_s, iters_t=iters_t, sigma=sigma)

           offsets_single_batch.append(current_offsets)
           last_key_s = current_key_s
           last_offsets_s = current_offsets
           
       # for the left frame
       for fi in range(1, t):
           current_key_s = keys[bi,fi].contiguous()
           current_offsets = torch.zeros_like(last_offsets_s).contiguous()
           tmp(current_key_s, last_key_s, current_offsets, last_offsets_s, distance, additional_jump=additional_jump, iters_s=iters_s, iters_t=iters_t, sigma=sigma)

           offsets_single_batch.append(current_offsets)
           last_key_s = current_key_s
           last_offsets_s = current_offsets

       offsets.append( torch.stack(offsets_single_batch, dim=0) )

   del distance
   offsets = torch.stack(offsets, dim=0) # (b, t, h, w, 2)
   return offsets
                    

class TMPAlign(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.convs = nn.Sequential(
                         nn.Conv2d(num_feat*3, num_feat*3, 3, padding=1),
                         nn.LeakyReLU(negative_slope=0.1, inplace=True),
                         nn.Conv2d(num_feat*3, num_feat, 3, padding=1))

        self.key = nn.Sequential(
                     GradLayer(0.1),
                     make_layer(ResidualBlockNoBN, 1, num_feat=num_feat),
                     nn.Conv2d(num_feat, num_feat//2, 3, padding=1),
                     make_layer(ResidualBlockNoBN, 1, num_feat=num_feat//2),
                     nn.Conv2d(num_feat//2, num_feat//4, 3, padding=1),
                     GradLayer(10))

        self.value = nn.Sequential(
                     nn.Conv2d(num_feat, num_feat, 3, padding=1))
                   
        self.reconstruction = make_layer(ResidualBlockNoBN, 30, num_feat=num_feat)

    def forward(self, x, lqs, hidden_states=None):
        b, t, c, h, w = x.shape

        # hidden state
        if hidden_states is None:
            hidden_offsets = None
            hidden_key = None
            hidden_feats = (None, None)
        else:
            hidden_offsets, hidden_key, hidden_feats = hidden_states

        # generating keys
        inp_key = self.key(x.view(-1, c, h, w)).view(b, t, -1, h, w)
        if hidden_key is not None:
            last_key = hidden_key
        else:
            last_key = inp_key[:, 0]
 
        # estimate offsets
        with torch.no_grad():
            offsets = tmp_forward(inp_key, hidden_offsets, last_key) # [b, t, h, w, 2]

        # change to grid position to match the requirement of F.grid_sample function
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h).to(torch.int32), torch.arange(0, w).to(torch.int32), indexing='ij')
        positions = torch.stack((grid_x, grid_y), 2).to(x.device).view(1, 1, h, w, 2)  # (1, 1, h, w, 2), (x, y)
        aligned_positions = positions + offsets
        aligned_positions_x = 2*aligned_positions[:, :, :, :, 0:1].to(torch.float)/max(w-1, 1) - 1.0
        aligned_positions_y = 2*aligned_positions[:, :, :, :, 1:2].to(torch.float)/max(h-1, 1) - 1.0
        aligned_positions = torch.cat([aligned_positions_x, aligned_positions_y], dim=4)

        # warp
        inp_value = self.value(x.view(-1, c, h, w)).view(b, t, -1, h, w)
        reconstruction_feats = []
        for fi in range(t):

            current_aligned_position = aligned_positions[:, fi]
            current_key = inp_key[:, fi]

            # align the feats
            if fi==0 and hidden_feats[0] is None:
                warped_feats_l = torch.zeros(b, c, h, w, dtype=torch.float, device=x.device)
                warped_feats_h = torch.zeros(b, c, h, w, dtype=torch.float, device=x.device)
            else:
                warped_feats_l = F.grid_sample(hidden_feats[0], current_aligned_position , mode='nearest', padding_mode='border', align_corners=True)
                warped_feats_h = F.grid_sample(hidden_feats[1], current_aligned_position , mode='nearest', padding_mode='border', align_corners=True)

                warped_last_key = F.grid_sample(last_key, current_aligned_position, mode='nearest', padding_mode='border', align_corners=True)
                 
                key_distance = torch.sum((current_key - warped_last_key)**2, dim=1, keepdims=True)
                weight = torch.exp(-key_distance)
                warped_feats_l = warped_feats_l*weight
                warped_feats_h = warped_feats_h*weight

            aligned_feat = self.convs(torch.cat([warped_feats_l, warped_feats_h, inp_value[:, fi]], dim=1))
            reconstruction_feats.append( self.reconstruction(aligned_feat) )

            hidden_feats = (inp_value[:, fi], reconstruction_feats[-1])
            hidden_key = current_key
            last_key = current_key

        reconstruction_feats = torch.stack(reconstruction_feats, dim=1)

        # hidden offsets
        hidden_offsets = offsets[:, -1]

        return reconstruction_feats, (hidden_offsets, hidden_key, hidden_feats)

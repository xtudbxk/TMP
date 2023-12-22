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
import torch.nn as nn


class GradLayer(nn.Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p
        self.register_full_backward_hook(self.backward_hook)

    def forward(self, x):
        return x

    @staticmethod
    def backward_hook(module, grad_input, grad_output):
        return tuple(module.p*tmp for tmp in grad_output)


if __name__ == '__main__':
    import numpy as np

    model = nn.Conv2d(8, 16, 3, padding=1)
    grad_layer = GradLayer(0.1)
    inp = torch.from_numpy( np.random.random( (1, 8, 64, 64) ).astype(np.float32) )
    

    inp.requires_grad = True
    out = model(inp)
    loss = torch.sum(out)
    loss.backward()
    print(torch.sum(inp.grad))

    inp.grad = None
    out = model(inp)
    loss = torch.sum(out)
    loss.backward()
    print(torch.sum(inp.grad))

    inp.grad = None
    out = model(inp)
    out = grad_layer(out)
    loss = torch.sum(out)
    loss.backward()
    print(torch.sum(inp.grad))

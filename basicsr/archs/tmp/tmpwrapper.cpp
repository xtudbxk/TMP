/*
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
*/


#include <torch/extension.h>
#include "tmp.h"

void temporal_motion_propagation(
        torch::Tensor &feat,
        torch::Tensor &feat_pre,
        torch::Tensor &offsets,
        torch::Tensor &offsets_pre,
        torch::Tensor &distance,
        const int height, 
        const int width, 
        const int channels, 
        const int iters_t,
        const int sigma,
        const int iters_s,
        const int additional_jump){

    _temporal_motion_propagation(
        (const float *)feat.data_ptr(),
        (const float *)feat_pre.data_ptr(),
        (int *)offsets.data_ptr(),
        (const int *)offsets_pre.data_ptr(),
        (float *)distance.data_ptr(),
        height, 
        width, 
        channels, 
        iters_t,
        sigma,
        iters_s,
	additional_jump);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("TemporalMotionPropagation",
          &temporal_motion_propagation,
          "cuda version wrapper");
}

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


void _temporal_motion_propagation(
                const float *feat, 
                const float *feat_pre, 
                int *offsets,
                const int *offsets_pre,
                float *distance,
                const int height, 
                const int width, 
                const int channel, 
                const int iters_t,
                const int sigma,
                const int iters_s,
                const int additional_jump);

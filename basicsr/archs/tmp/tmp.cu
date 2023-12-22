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

#include <stdio.h>
#include <cmath>
#include <curand_kernel.h>

__host__ __device__ unsigned int XY_TO_INT(int x, int y) {//r represent the number of 10 degree, x,y - 11 bits, max = 2047, r - max = 36, 6 bits
        return (((y) << 11) | (x));
}
__host__ __device__ int INT_TO_X(unsigned int v) {
        return (v)&((1 << 11) - 1);
}
__host__ __device__ int INT_TO_Y(unsigned int v) {
        return (v >> 11)&((1 << 11) - 1);
}

__device__ int get_max(int x,int y){

        if(x>y)
                return x;
        return y;
}

__device__ int get_min(int x,int y){

        if(x<y)
                return x;
        return y;
}


__device__ float compute_distance(
                const float *feat, 
                const float *feat_pre, 
                const int height,
                const int width,
                const int channel,
                const int x1,
                const int y1,
                const int x2,
                const int y2){

	float distance = 0.0;

	int p1 = y1*width+x1;
	int p2 = y2*width+x2;
	int pixels_count = height*width;
	for(int c=0; c<channel; c++){
	    distance += (feat[p1]-feat_pre[p2])*(feat[p1]-feat_pre[p2]);
	    p1 += pixels_count;
	    p2 += pixels_count;
	}
        return distance;
}

__device__ void update(
                const float *feat, 
                const float *feat_pre, 
                int *offsets , 
                float *distance , 
                const int height, 
                const int width, 
                const int channel, 
                int pixel_x, 
                int pixel_y,
                int test_x,
                int test_y) {

        float test_dist = compute_distance(feat,feat_pre,height,width,channel,pixel_x,pixel_y,test_x,test_y); 
        
	int p = pixel_y*width + pixel_x;
        if(distance[p] > test_dist){
	    distance[p] = test_dist;
	    offsets[2*p+0] = test_x-pixel_x;
	    offsets[2*p+1] = test_y-pixel_y;
        }
}


__device__ float get_rand(curandState *state){
        return curand_uniform(state);
}

__device__ float get_rand_normal(curandState *state){
        return curand_normal(state);
}

__device__ void InitcuRand(curandState *state) {//random number in cuda, between 0 and 1

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        curand_init(i, j, 0, state);

}
        extern "C" 
__global__ void _temporal_motion_propagation_cuda(
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
                const int additional_jump) {

        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        if(row >= height || col >= width)
            return;

        curandState state;
        InitcuRand(&state);

        // initalize the distance memory
        int p = row*width+col;
        int p2 = 2*p;

        int init_x  = (int)(get_rand(&state)*(width)) % width;
        int init_y  = (int)(get_rand(&state)*(height)) % height;
        offsets[p2+0] = init_x-col;
	offsets[p2+1] = init_y-row;
        distance[p] = compute_distance(feat,feat_pre,height,width,channel,col,row,init_x,init_y); 

	// temporal propagation
	// temporal propagation - global motion for static object
	int pixel_x = offsets_pre[p2+0] + col; // the position of pixel in the previous frame which moves to (row, col) of current frame
	int pixel_y = offsets_pre[p2+1] + row; 
	update(feat,feat_pre,offsets,distance,height,width,channel,col,row,pixel_x,pixel_y);

	// pick some pixels using the gaussian probability model
	int test_x, test_y;
        int xmin, xmax, ymin, ymax;
        for(int j =0; j < iters_t; j++){
            test_x = pixel_x + (int)(get_rand_normal(&state)*sigma); // pick from N(0, sigma^2)
	    if(test_x < 0 || test_x >= width){ // in case (test_x, test_y) is out of current frame
                xmin = get_max(pixel_x-sigma, 0), xmax = get_min(pixel_x+sigma+1, width-1);
		test_x = xmin + (int)(get_rand(&state)*(xmax-xmin)) % (xmax-xmin);
            }
	    test_y = pixel_y + (int)(get_rand_normal(&state)*sigma); // pick from N(0, sigma^2)
            if(test_y < 0 || test_y >= height){
                ymin = get_max(pixel_y-sigma, 0), ymax = get_min(pixel_y+sigma+1, height-1);
		test_y = ymin + (int)(get_rand(&state)*(ymax-ymin)) % (ymax-ymin);
	    }
	    update(feat,feat_pre,offsets,distance,height,width,channel,col,row,test_x,test_y);
	}
        __syncthreads();

	// temporal propagation - the object keeps same motion
	int cpo_x = col - offsets_pre[p2 + 0]; // the position of object in current frame moving from (row, col) of last frame
        cpo_x = get_min(get_max(cpo_x, 0), width-1);
	int cpo_y = row - offsets_pre[p2 + 1];
        cpo_y = get_min(get_max(cpo_y, 0), height-1);
	update(feat,feat_pre,offsets,distance,height,width,channel,cpo_x,cpo_y,col,row);

	// pick some pixels using the gaussian probability model
	for(int j =0; j < iters_t; j++){
	    test_x = cpo_x + (int)(get_rand_normal(&state)*sigma); // pick from N(0, sigma^2)
            if(test_x < 0 || test_x >= width){
                xmin = get_max(cpo_x-sigma, 0), xmax = get_min(cpo_x+sigma+1, width-1);
		test_x = xmin + (int)(get_rand(&state)*(xmax-xmin)) % (xmax-xmin);
	    }
	    test_y = cpo_y + (int)(get_rand_normal(&state)*sigma); // pick from N(0, sigma^2)
	    if(test_y < 0 || test_y >= height){
                ymin = get_max(cpo_y-sigma, 0), ymax = get_min(cpo_y+sigma+1, height-1);
		test_y = ymin + (int)(get_rand(&state)*(ymax-ymin)) % (ymax-ymin);
	    }
	    update(feat,feat_pre,offsets,distance,height,width,channel,test_x,test_y,col,row);
        }           
        __syncthreads();

	// spatial propagation using JFA
	for(int i=0; i < iters_s; i++){
            for(int jump=(int)(get_max(width, height)/2.0), jc=additional_jump; (jump > 0) || (jc > 0); jump /= 2){
                if(jump<=0){ // start to count the additional jump
                    jc--;
                }
    
        	// propagate to the 4-neighbors of each pixel
        	int neighbors[4][2] = {{-jump, 0}, {jump, 0}, {0, -jump}, {0, jump}};
        
        	for(int ofi=0; ofi<4; ofi++){
        	    int neighbor_x = col + neighbors[ofi][0];
                    int neighbor_y = row + neighbors[ofi][1];
    
        	    if( 0 > neighbor_x || neighbor_x >= width || 0 > neighbor_y || neighbor_y >= height ){
        		continue;
                     }
        
        	    int tmp = neighbor_y*width + neighbor_x;
                    int test_x = col + offsets[2*tmp+0];
    		    int test_y = row + offsets[2*tmp+1];
    		    if( 0 <= test_x && test_x < width && 0 <= test_y && test_y < height){
    	                update(feat,feat_pre,offsets,distance,height,width,channel,col,row,test_x,test_y);
    		    }
                }

            __syncthreads();
        }
    }
}


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
                const int additional_jump
                ) {
    int threads = 16;
    dim3 grid(floor(height/threads)+1, floor(width/threads)+1);
    dim3 block(threads, threads);
    _temporal_motion_propagation_cuda<<<grid, block>>>(feat, feat_pre, offsets, offsets_pre, distance, height, width, channel, iters_t, sigma, iters_s, additional_jump);
}

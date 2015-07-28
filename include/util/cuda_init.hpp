#pragma once

template <typename T>
__global__
void init(T* v, unsigned N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        v[i] = i+1;
    }
}
#pragma once

#include <cmath>
#include <util/cuda_error.hpp>
#include <util/cuda_device.hpp>
#include <util/range.hpp>

namespace cuda { namespace util {

unsigned ceil(float arg) {
    return static_cast<unsigned>(std::ceil(arg));
}

dim3 getGridDimensions(unsigned  sx, unsigned  sy, unsigned  sz,
                       unsigned& bx, unsigned& by, unsigned& bz) {
    error err;
    cuda::device gpu;
    cudaDeviceProp prop;

    err = cudaGetDeviceProperties(&prop, gpu.get_device_id());

    //  32 <= bx <= 1024
    //   1 <= by <= 1024
    //   1 <= bz <=   64
    if (bx < 32)   bx = 32;
    if (bx > prop.maxThreadsDim[0]) bx = prop.maxThreadsDim[0];
    if (by > prop.maxThreadsDim[1]) by = prop.maxThreadsDim[1];
    if (bz > prop.maxThreadsDim[2]) bz = prop.maxThreadsDim[2];

    while (bx * by * bz > prop.maxThreadsPerBlock) {
        if (bz > 1)      bz /= 2;
        else if (by > 1) by /= 2;
        else             bx /= 2;
    }

    unsigned grid_size_x = std::min( ceil(sx/bx), static_cast<unsigned>(65535)); //prop.maxGridSize[0]));
    unsigned grid_size_y = std::min( ceil(sy/by), static_cast<unsigned>(prop.maxGridSize[1]));
    unsigned grid_size_z = std::min( ceil(sz/bz), static_cast<unsigned>(prop.maxGridSize[2]));

    return {grid_size_x, grid_size_y, grid_size_z};
}

}}

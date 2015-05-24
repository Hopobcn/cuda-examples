#include <util/cuda_grid_config.hpp>
#include <cmath>
#include <util/cuda_error.hpp>
#include <util/cuda_device.hpp>
#include <util/range.hpp>

namespace cuda { namespace util {

dim3 getGridDimensions(unsigned  sx, unsigned  sy, unsigned  sz,
                       unsigned& bx, unsigned& by, unsigned& bz) {
    error err;
    cuda::device gpu;
    cudaDeviceProp prop;

    err = cudaGetDeviceProperties(&prop, gpu.get_device_id());

    //  32 <= bx <= 1024
    //   1 <= by <= 1024
    //   1 <= bz <=   64
    unsigned warpSize       = static_cast<unsigned>(prop.warpSize);
    unsigned maxThreadDim_x = static_cast<unsigned>(prop.maxThreadsDim[0]);
    unsigned maxThreadDim_y = static_cast<unsigned>(prop.maxThreadsDim[1]);
    unsigned maxThreadDim_z = static_cast<unsigned>(prop.maxThreadsDim[2]);
    unsigned maxThreadsPerBlock = static_cast<unsigned>(prop.maxThreadsPerMultiProcessor);

    if (bx < warpSize)   bx = warpSize;
    if (bx > maxThreadDim_x) bx = maxThreadDim_x;
    if (by > maxThreadDim_y) by = maxThreadDim_y;
    if (bz > maxThreadDim_z) bz = maxThreadDim_z;

    while (bx * by * bz > maxThreadsPerBlock) {
        if (bz > 1)      bz /= 2;
        else if (by > 1) by /= 2;
        else             bx /= 2;
    }

    unsigned grid_size_x = std::min( static_cast<unsigned>(std::ceil(sx/bx)), static_cast<unsigned>(65535)); //prop.maxGridSize[0]));
    unsigned grid_size_y = std::min( static_cast<unsigned>(std::ceil(sy/by)), static_cast<unsigned>(prop.maxGridSize[1]));
    unsigned grid_size_z = std::min( static_cast<unsigned>(std::ceil(sz/bz)), static_cast<unsigned>(prop.maxGridSize[2]));

    return {grid_size_x, grid_size_y, grid_size_z};
}

}}
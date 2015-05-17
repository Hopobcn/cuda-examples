#pragma once

#include <cuda.h>
#include <util/device_callable.hpp>
#include <util/range.hpp>

namespace cuda {

using namespace util::lang;


// type alias to simplify typing...
template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template<typename T>
DEVICE_CALLABLE
step_range<T> grid_stride_range(T begin, T end) {
    begin += blockDim.x * blockIdx.x + threadIdx.x;
    return range(begin, end).step(gridDim.x * blockDim.x);
}

template<typename T>
DEVICE_CALLABLE
step_range<T> grid_stride_range(T begin, T end, unsigned n) {
    begin += n * blockDim.x * blockIdx.x + threadIdx.x;
    return range(begin, end - n * blockDim.x * gridDim.x).step(n * gridDim.x * blockDim.x);
}

}
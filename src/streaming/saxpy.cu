#include <iostream>
#include <cmath>
#include <cstdio>
#include <streaming/saxpy.hpp>
#include <util/grid_stride.hpp>
#include <util/cuda_error.hpp>

using cuda::grid_stride_range;
using cuda::util::lang::range;

template <typename T>
void saxpy_cpu(const cuda::vector<T>& x, const cuda::vector<T>& y, cuda::vector<T>& z, unsigned N, T alpha) {
    for (unsigned i = 0; i < N; i++) {
        z[i] += alpha * x[i] + y[i];
    }
}

template <typename T>
__global__
void saxpy_gpu_c_array(const T* x, const T* y, T* z, unsigned N, T alpha) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < N;
             i += blockDim.x * gridDim.x) {
        z[i] = alpha * x[i] + y[i];
    }
}

template <typename T>
__global__
void saxpy_gpu_cpp_array(const T* x, const T* y, T* z, unsigned N, T alpha) {
    for (unsigned i : grid_stride_range<T>(0, N) ) {
        z[i] = alpha * x[i] + y[i];
    }
}

template <typename T>
__global__
void saxpy_gpu_c_vector(const cuda::vector<T>& x, const cuda::vector<T>& y, cuda::vector<T>& z, unsigned N, T alpha) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        z[i] = alpha * x[i] + y[i];
    }
}

template <typename T>
__global__
void saxpy_gpu_cpp_vector(const cuda::vector<T>& x, const cuda::vector<T>& y, cuda::vector<T>& z, unsigned N, T alpha) {
    for (auto i : grid_stride_range<T>(0, N) ) {
        z[i] = alpha * x[i] + y[i];
    }
}

template <typename T, const int unroll>
__global__
void saxpy_gpu_unrolled(cuda::vector<T>& out, const cuda::vector<T>& x, const cuda::vector<T>& y, unsigned N, T alpha) {
    T x_reg[unroll], y_reg[unroll];
    auto i_last = 0;
    for (auto i : grid_stride_range(0, N, unroll) ) {
        for (auto j : range(unroll)) {
            unsigned gindex = j * blockDim.x + i;
            x_reg[i] = x[gindex];
            y_reg[i] = y[gindex];
        }
        for (auto j : range(unroll)) {
            unsigned gindex = j * blockDim.x + i;
            out[index] = alpha * x[i] + y[i];
        }
        i_last = i;
    }
    // to avoid the (index<N) conditional in the inner loop,
    // we left off some work at the end
    for (auto j : range(unroll)) {
        for (auto j : range(unroll)) {
            unsigned gindex = j * blockDim.x + i_last;
            if (gindex < N) {
                x_reg[j] = x[gindex];
                y_reg[j] = y[gindex];
            }
        }
        for (auto j : range(unroll) ) {
            unsigned gindex = j * blockDim.x + i_last;
            if (gindex < N)
                out[gindex] = alpha * x_reg[j] + y_reg[j];
        }
    }
}

void run_saxpy(const cuda::vector<float>& x,
               const cuda::vector<float>& y,
                     cuda::vector<float>& z,
               const float* px,
               const float* py,
                     float* pz,
               unsigned N,
               float alpha) {
    using T = float;

    const unsigned block_size = 64;
    dim3 dimGrid( std::ceil(N/block_size) );
    dim3 dimBlock( block_size );
    std::cout << "Launching saxpy kernels" << std::endl;

    saxpy_gpu_c_array   <T><<<dimGrid, dimBlock>>>(px, py, pz, N, alpha);
    saxpy_gpu_c_vector  <T><<<dimGrid, dimBlock>>>( x,  y,  z, N, alpha);
    saxpy_gpu_cpp_array <T><<<dimGrid, dimBlock>>>(px, py, pz, N, alpha);
    saxpy_gpu_cpp_vector<T><<<dimGrid, dimBlock>>>( x,  y,  z, N, alpha);

    cudaDeviceSynchronize();
}

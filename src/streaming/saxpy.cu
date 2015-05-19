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
    for (auto i : grid_stride_range<unsigned>(0, N) ) {
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
    for (auto i : grid_stride_range<unsigned>(0, N) ) {
        z[i] = alpha * x[i] + y[i];
    }
}


template <typename T, const int unroll>
__global__
void saxpy_gpu_c_vector_unroll(const T* x, const T* y, T* z, unsigned N, T alpha) {
    T x_reg[unroll], y_reg[unroll];
    unsigned i;
    for ( i = unroll * blockIdx.x * blockDim.x + threadIdx.x;
          i < N - unroll * blockDim.x * gridDim.x;
          i +=    unroll * blockDim.x * gridDim.x ) {
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
            unsigned gindex = j * blockDim.x + i;
            x_reg[j] = x[gindex];
            y_reg[j] = y[gindex];
        }
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
            unsigned gindex = j * blockDim.x + i;
            z[gindex] = alpha * x_reg[j] + y_reg[j];
        }
    }
    // to avoid the (index<N) conditional in the inner loop,
    // we left off some work at the end
    for (int j = 0; j < unroll; j++) {
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
            unsigned gindex = j * blockDim.x + i;
            if (gindex < N) {
                x_reg[j] = x[gindex];
                y_reg[j] = y[gindex];
            }
        }
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
            unsigned gindex = j * blockDim.x + i;
            if (gindex < N)
                z[gindex] = alpha * x_reg[j] + y_reg[j];
        }
    }
}

template <typename T, const int unroll>
__global__
void saxpy_gpu_cpp_vector_unroll(const cuda::vector<T>& x, const cuda::vector<T>& y, cuda::vector<T>& z, unsigned N, T alpha) {
    T x_reg[unroll], y_reg[unroll];
    auto i_last = 0;
    for (auto i : grid_stride_range<unsigned>(0, N, unroll) ) {
        for (auto j : range<unsigned>(0, unroll)) {
            unsigned gindex = j * blockDim.x + i;
            x_reg[j] = x[gindex];
            y_reg[j] = y[gindex];
        }
        for (auto j : range(0, unroll)) {
            unsigned gindex = j * blockDim.x + i;
            z[gindex] = alpha * x_reg[j] + y_reg[j];
        }
        i_last = i;
    }
    // to avoid the (index<N) conditional in the inner loop,
    // we left off some work at the end
    for (auto j : range<unsigned>(0, unroll)) {
        for (auto j : range<unsigned>(0, unroll)) {
            unsigned gindex = j * blockDim.x + i_last;
            if (gindex < N) {
                x_reg[j] = x[gindex];
                y_reg[j] = y[gindex];
            }
        }
        for (auto j : range<unsigned>(0, unroll) ) {
            unsigned gindex = j * blockDim.x + i_last;
            if (gindex < N)
                z[gindex] = alpha * x_reg[j] + y_reg[j];
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

    const unsigned unroll = 2;
    dim3 dimGrid2( std::ceil(N/block_size)/unroll );
    dim3 dimBlock2( block_size );

    std::cout << "Launching saxpy kernels" << std::endl;

    for (int i = 0; i < 100; i++) {
        saxpy_gpu_c_array<T> << < dimGrid, dimBlock >> > (px, py, pz, N, alpha);
        saxpy_gpu_c_vector<T> << < dimGrid, dimBlock >> > (x, y, z, N, alpha);
        saxpy_gpu_cpp_array<T> << < dimGrid, dimBlock >> > (px, py, pz, N, alpha);
        saxpy_gpu_cpp_vector<T> << < dimGrid, dimBlock >> > (x, y, z, N, alpha);

        saxpy_gpu_c_vector_unroll<T, unroll> << < dimGrid2, dimBlock2 >> > (px, py, pz, N, alpha);
        saxpy_gpu_cpp_vector_unroll<T, unroll> << < dimGrid2, dimBlock2 >> > (x, y, z, N, alpha);
    }

    cudaDeviceSynchronize();
}

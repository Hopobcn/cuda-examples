#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <streaming/saxpy.hpp>
#include <util/grid_stride.hpp>
#include <util/cuda_grid_config.hpp>
#include <util/cuda_error.hpp>

using cuda::grid_stride_range;
using cuda::util::lang::range;


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
               float alpha,
               unsigned repetitions) {
    using T = float;

    cuda::error err;
    unsigned block_size_x = 64;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = cuda::util::getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );

    unsigned block_size2_x = 64;
    unsigned block_size2_y = 1;
    unsigned block_size2_z = 1;
    const unsigned unroll2 = 2;
    dim3 dimGrid2  = cuda::util::getGridDimensions(N/unroll2, 1, 1, block_size2_x, block_size2_y, block_size2_z);
    dim3 dimBlock2( block_size2_x, block_size2_y, block_size2_z );

    unsigned block_size4_x = 64;
    unsigned block_size4_y = 1;
    unsigned block_size4_z = 1;
    const unsigned unroll4 = 4;
    dim3 dimGrid4 = cuda::util::getGridDimensions(N/unroll4, 1, 1, block_size4_x, block_size4_y, block_size4_z);
    dim3 dimBlock4( block_size4_x, block_size4_y, block_size4_z );

    std::cout << "Launching saxpy kernels" << std::endl;
    std::cout << "Grid 1 [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Grid 2 [" << dimGrid2.x << "," << dimGrid2.y << "," << dimGrid2.z << "]" << std::endl;
    std::cout << "Grid 4 [" << dimGrid4.x << "," << dimGrid4.y << "," << dimGrid4.z << "]" << std::endl;

    std::cout << "Block 1 [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "Block 2 [" << dimBlock2.x << "," << dimBlock2.y << "," << dimBlock2.z << "]" << std::endl;
    std::cout << "Block 4 [" << dimBlock4.x << "," << dimBlock4.y << "," << dimBlock4.z << "]" << std::endl;

    for (int i = 0; i < repetitions; i++) {
        saxpy_gpu_c_vector<T><<<dimGrid.x, dimBlock.x>>>(x, y, z, N, alpha);
        err = cudaGetLastError();

        saxpy_gpu_cpp_vector<T><<<dimGrid.x, dimBlock>>>(x, y, z, N, alpha);
        err = cudaGetLastError();

        saxpy_gpu_cpp_array<T><<<dimGrid.x, dimBlock>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();

        saxpy_gpu_c_array<T><<<dimGrid.x, dimBlock>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();


        saxpy_gpu_cpp_vector_unroll<T, unroll2><<<dimGrid2.x, dimBlock2>>>(x, y, z, N, alpha);
        err = cudaGetLastError();
        saxpy_gpu_cpp_vector_unroll<T, unroll4><<<dimGrid4.x, dimBlock4>>>(x, y, z, N, alpha);
        err = cudaGetLastError();


        saxpy_gpu_c_vector_unroll<T, unroll2><<<dimGrid2.x, dimBlock2>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();
        saxpy_gpu_c_vector_unroll<T, unroll4><<<dimGrid4.x, dimBlock4>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();
    }

    err = cudaDeviceSynchronize();
}

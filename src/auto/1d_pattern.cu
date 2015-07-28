#include <iostream>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <auto/auto.hpp>
#include <util/grid_stride.hpp>
#include <util/cuda_grid_config.hpp>
#include <util/cuda_error.hpp>
#include <util/cuda_init.hpp>
#include <cub/cub/cub.cuh>

using cuda::grid_stride_range;
using cuda::util::getGridDimensions;
using cuda::util::lang::range;

    
template<typename T>
__global__ 
void memset_kernel(T * x, T value, int count) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
             i < count;
             i += blockDim.x * gridDim.x) {
        x[i] = value;
    }
}

template <typename T, typename Function>
__global__
void streaming_lambda_by_ref(const T* x, const T* y, T* z, unsigned N, T alpha, Function& lambda) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
             i < count;
             i += blockDim.x * gridDim.x) {
        z[i] = lambda(x[i],y[i],alpha);
    }
}

template <typename T>
__global__
void streaming_local_lambda(const T* x, const T* y, T* z, unsigned N, T alpha) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        auto local_lambda = [&](T x, T y, T alpha) { return alpha * x + y; };

        z[i] = local_lambda(x[i],y[i],alpha);

        if (i == 0)
        printf("z[%d] = alpha * x[%d] + y[%d] \t %f = %f * %f + %f\n",
           i, i, i, z[i], alpha, x[i], y[i]);
    }
}

template <typename T>
void lambda_test(cuda::device& gpu, unsigned N) {
    try {
        cuda::device gpu;
        cuda::error err;

        cout << "--- testing lambda ---" << endl;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        T *px, *py, *pz;
        err = cudaMalloc((void**)&px, N * sizeof(T));
        err = cudaMalloc((void**)&py, N * sizeof(T));
        err = cudaMalloc((void**)&pz, N * sizeof(T));
        T alpha = 0.8;

        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t est_program_alloc = 3 * N * sizeof(T);
        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Difference "                       << factor << endl;

        unsigned block_size_x = 128;
        unsigned block_size_y = 1;
        unsigned block_size_z = 1;
        dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
        dim3 dimBlock( block_size_x, block_size_y, block_size_z );

        std::cout << "Launching streaming test" << std::endl;
        std::cout << "Grid [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;

        memset_kernel<<<dimGrid, dimBlock>>>(px, (T)3.0, N);
        memset_kernel<<<dimGrid, dimBlock>>>(py, (T)2.0, N);

        streaming_local_lambda<<<dimGrid, dimBlock>>>(px, py, pz, N, alpha);

        auto pass_by_ref_lambda = [] __device__ (T x, T y, T alpha) { return alpha * x + y; };
        auto pass_by_val_lambda = [=] (T x, T y, T alpha) { return alpha * x + y; };

        streaming_lambda_by_ref<<<dimGrid, dimBlock>>>(px, py, pz, N, alpha, pass_by_ref_lambda);


        err = cudaFree(px);
        err = cudaFree(py);
        err = cudaFree(pz);

    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

void cuda_cpp11_testing(cuda::device& gpu, unsigned N)
{
    lambda_test<float>(gpu, N);

}

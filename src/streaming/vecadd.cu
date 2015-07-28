#include <iostream>
#include <cmath>
#include <cassert>
#include <streaming/vecadd.hpp>
#include <util/cuda_vector.hpp>
#include <util/grid_stride.hpp>
#include <util/cuda_grid_config.hpp>
#include <util/cuda_error.hpp>
#include <cublas_v2.h>
#include <cub/cub/cub.cuh>


using cuda::grid_stride_range;
using cuda::util::getGridDimensions;
using cuda::util::lang::range;


template <typename T>
__global__
void vecadd_gpu_naive(const T* x,
                      const T* y,
                      T* z,
                      unsigned N)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        z[i] = x[i] + y[i];
    }
}


template <typename T>
__global__
void vecadd_gpu_naive_rangebasedloop(const T* x,
                                     const T* y,
                                     T* z,
                                     unsigned N)
{
    for (auto i : grid_stride_range<unsigned>(0, N)) {
        z[i] = x[i] + y[i];
    }
}

template <typename T, const int unroll>
__global__
void vecadd_gpu_unroll(const T *x,
                       const T *y,
                       T *z,
                       unsigned N) {
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
            z[gindex] = x_reg[j] + y_reg[j];
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
                z[gindex] = x_reg[j] + y_reg[j];
        }
    }
}

template <typename T, const unsigned blockDimx, const int unroll>
__global__
void vecadd_gpu_cub(const T* x,
                    const T* y,
                    T* z,
                    unsigned N) {
    using BlockLoad  = cub::BlockLoad<const T*, blockDimx, unroll, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<T*, blockDimx, unroll, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    __shared__ union
    {
        typename BlockLoad::TempStorage  load_x;
        typename BlockLoad::TempStorage  load_y;
        typename BlockStore::TempStorage store;
    } storage_smem;

    T x_reg[unroll], y_reg[unroll], z_reg[unroll];
    BlockLoad(storage_smem.load_x).Load(x, x_reg);
    BlockLoad(storage_smem.load_y).Load(y, y_reg);

    __syncthreads();

    for (int i = 0; i < unroll; i++)
        z_reg[i] = x_reg[i] + y_reg[i];

    BlockStore(storage_smem.store).Store(z, z_reg);
};





template <typename T>
void run_vecadd_c(const T* px,
                  const T* py,
                  T* pz,
                  unsigned N,
                  unsigned repetitions)
{
    cuda::error err;
    unsigned block_size_x = 128;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );

    unsigned block_size2_x = 128;
    unsigned block_size2_y = 1;
    unsigned block_size2_z = 1;
    const unsigned unroll2 = 2;
    dim3 dimGrid2  = getGridDimensions(N/unroll2, 1, 1, block_size2_x, block_size2_y, block_size2_z);
    dim3 dimBlock2( block_size2_x, block_size2_y, block_size2_z );

    unsigned block_size4_x = 128;
    unsigned block_size4_y = 1;
    unsigned block_size4_z = 1;
    const unsigned unroll4 = 4;
    dim3 dimGrid4 = getGridDimensions(N/unroll4, 1, 1, block_size4_x, block_size4_y, block_size4_z);
    dim3 dimBlock4( block_size4_x, block_size4_y, block_size4_z );

    std::cout << "Launching vecadd C kernels" << std::endl;
    std::cout << "Grid 1 [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Grid 2 [" << dimGrid2.x << "," << dimGrid2.y << "," << dimGrid2.z << "]" << std::endl;
    std::cout << "Grid 4 [" << dimGrid4.x << "," << dimGrid4.y << "," << dimGrid4.z << "]" << std::endl;

    std::cout << "Block 1 [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "Block 2 [" << dimBlock2.x << "," << dimBlock2.y << "," << dimBlock2.z << "]" << std::endl;
    std::cout << "Block 4 [" << dimBlock4.x << "," << dimBlock4.y << "," << dimBlock4.z << "]" << std::endl;

    for (int i = 0; i < repetitions; i++) {
        vecadd_gpu_naive<T><<<dimGrid, dimBlock>>>(px, py, pz, N);
        err = cudaGetLastError();
        vecadd_gpu_naive_rangebasedloop<T><<<dimGrid, dimBlock>>>(px, py, pz, N);
        err = cudaGetLastError();
        vecadd_gpu_unroll<T, unroll2><<<dimGrid2, dimBlock2>>>(px, py, pz, N);
        err = cudaGetLastError();
        vecadd_gpu_unroll<T, unroll4><<<dimGrid4, dimBlock4>>>(px, py, pz, N);
        err = cudaGetLastError();
    }
    std::cout << std::endl;

    err = cudaDeviceSynchronize();
}

template <typename T>
void run_vecadd_cublas(const T* px,
                       T* py,
                       unsigned N,
                       unsigned repetitions);


template <>
void run_vecadd_cublas(const float* px,
                       float* py,
                       unsigned N,
                       unsigned repetitions) {
    cublasStatus_t status;
    cublasHandle_t handle;

    std::cout << "Launching vecadd CUBLAS" << std::endl;

    status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);

    for (int i = 0; i < repetitions; i++) {
        //status = cublasSasum(handle, N, px, 1, py);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }
    std::cout << std::endl;
}

template <>
void run_vecadd_cublas(const double* px,
                       double* py,
                       unsigned N,
                       unsigned repetitions) {
    cublasStatus_t status;
    cublasHandle_t handle;

    std::cout << "Launching vecadd CUBLAS" << std::endl;

    status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);

    for (int i = 0; i < repetitions; i++) {
        //status = cublasDasum(handle, N, px, 1, py);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }
    std::cout << std::endl;
}

template <typename T>
void run_vecadd_cub(const T* px,
                   const T* py,
                   T* pz,
                   unsigned N,
                   unsigned repetitions) {
    cuda::error err;
    unsigned block_size_x = 128;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );

    unsigned block_size2_x = 128;
    unsigned block_size2_y = 1;
    unsigned block_size2_z = 1;
    const unsigned unroll2 = 2;
    dim3 dimGrid2  = getGridDimensions(N/unroll2, 1, 1, block_size2_x, block_size2_y, block_size2_z);
    dim3 dimBlock2( block_size2_x, block_size2_y, block_size2_z );

    unsigned block_size4_x = 128;
    unsigned block_size4_y = 1;
    unsigned block_size4_z = 1;
    const unsigned unroll4 = 4;
    dim3 dimGrid4 = getGridDimensions(N/unroll4, 1, 1, block_size4_x, block_size4_y, block_size4_z);
    dim3 dimBlock4( block_size4_x, block_size4_y, block_size4_z );

    std::cout << "Launching vecadd CUB kernels" << std::endl;
    std::cout << "Grid 1 [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Grid 2 [" << dimGrid2.x << "," << dimGrid2.y << "," << dimGrid2.z << "]" << std::endl;
    std::cout << "Grid 4 [" << dimGrid4.x << "," << dimGrid4.y << "," << dimGrid4.z << "]" << std::endl;

    std::cout << "Block 1 [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "Block 2 [" << dimBlock2.x << "," << dimBlock2.y << "," << dimBlock2.z << "]" << std::endl;
    std::cout << "Block 4 [" << dimBlock4.x << "," << dimBlock4.y << "," << dimBlock4.z << "]" << std::endl;

    for (int i = 0; i < repetitions; i++) {
        vecadd_gpu_cub<T, 128,       1><<<dimGrid, dimBlock>>>(px, py, pz, N);
        err = cudaGetLastError();
        vecadd_gpu_cub<T, 128, unroll2><<<dimGrid2, dimBlock2>>>(px, py, pz, N);
        err = cudaGetLastError();
        vecadd_gpu_cub<T, 128, unroll4><<<dimGrid4, dimBlock4>>>(px, py, pz, N);
        err = cudaGetLastError();
    }
    std::cout << std::endl;

    err = cudaDeviceSynchronize();
}







template <typename T>
void vecadd_c(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::device gpu;
        cuda::error err;

        cout << "--- VECADD C ---" << endl;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        T *px, *py, *pz;
        err = cudaMalloc((void**)&px, N * sizeof(T));
        err = cudaMalloc((void**)&py, N * sizeof(T));
        err = cudaMalloc((void**)&pz, N * sizeof(T));

        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t est_program_alloc = 3 * N * sizeof(T);
        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Difference "                       << factor << endl;


        run_vecadd_c(px, py, pz, N, repetitions);

        err = cudaFree(px);
        err = cudaFree(py);
        err = cudaFree(pz);

    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}


template <typename T>
void vecadd_cublas(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::device gpu;
        cuda::error err;

        cout << "--- VECADD Cublas ---" << endl;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        T *px, *py;
        err = cudaMalloc((void**)&px, N * sizeof(T));
        err = cudaMalloc((void**)&py, N * sizeof(T));

        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t est_program_alloc = 2 * N * sizeof(T);
        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Difference "                       << factor << endl;


        run_vecadd_cublas(px, py, N, repetitions);

        err = cudaFree(px);
        err = cudaFree(py);
    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

template <typename T>
void vecadd_cub(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::device gpu;
        cuda::error err;

        cout << "--- VECADD CUB ---" << endl;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        T *px, *py, *pz;
        err = cudaMalloc((void**)&px, N * sizeof(T));
        err = cudaMalloc((void**)&py, N * sizeof(T));
        err = cudaMalloc((void**)&pz, N * sizeof(T));

        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t est_program_alloc = 3 * N * sizeof(T);
        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Difference "                       << factor << endl;


        run_vecadd_cub(px, py, pz, N, repetitions);

        err = cudaFree(px);
        err = cudaFree(py);
        err = cudaFree(pz);

    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

void launch_vecadd(cuda::device& gpu, unsigned N, unsigned repetitions)
{
    vecadd_c<float>(gpu, N, repetitions);
    vecadd_cublas<float>(gpu, N, repetitions);
    vecadd_cub<float>(gpu, N, repetitions);


    vecadd_c<double>(gpu, N, repetitions);
    vecadd_cublas<double>(gpu, N, repetitions);
    vecadd_cub<double>(gpu, N, repetitions);
}
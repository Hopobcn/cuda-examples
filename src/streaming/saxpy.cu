#include <iostream>
#include <cmath>
#include <cassert>
#include <streaming/saxpy.hpp>
#include <util/cuda_vector.hpp>
#include <util/grid_stride.hpp>
#include <util/cuda_grid_config.hpp>
#include <util/cuda_error.hpp>
#include <util/cuda_init.hpp>
#include <cublas_v2.h>
#include <cub/cub/cub.cuh>


using cuda::grid_stride_range;
using cuda::util::getGridDimensions;
using cuda::util::lang::range;

template <typename T>
__global__
void saxpy_gpu_naive(const T *x,
                     const T *y,
                     T *z,
                     unsigned N, T alpha) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < N;
             i += blockDim.x * gridDim.x) {
        z[i] = alpha * x[i] + y[i];
    }
}

template <typename T>
__global__
void saxpy_gpu_naive_rangebasedloop(const T *x,
                                    const T *y,
                                    T *z,
                                    unsigned N, T alpha) {
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
void saxpy_gpu_unroll(const T *x,
                      const T *y,
                      T *z,
                      unsigned N, T alpha) {
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

template <typename T, const unsigned blockDimx, const int unroll>
__global__
void saxpy_gpu_cub(const T* x,
                   const T* y,
                   T* z,
                   unsigned N, T alpha) {
    using BlockLoad  = cub::BlockLoad<const T*, blockDimx, unroll, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<T*, blockDimx, unroll, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    __shared__ union
    {
        typename BlockLoad::TempStorage  load_x;
        typename BlockLoad::TempStorage  load_y;
        typename BlockStore::TempStorage store;
    } storage_smem;

    T x_reg[unroll], y_reg[unroll], z_reg[unroll];
    BlockLoad(storage_smem.load_x).Load(x, x_reg, N);
    BlockLoad(storage_smem.load_y).Load(y, y_reg, N);

    __syncthreads();

    for (int i = 0; i < unroll; i++)
        z_reg[i] = alpha * x_reg[i] + y_reg[i];

    BlockStore(storage_smem.store).Store(z, z_reg, N);
};

/*
template <typename T, const unsigned blockDimx, const unsigned blockDimy, const int unroll>
__global__
void test_cub_2d(const T* x, const T* y, T* z, unsigned N, T alpha) {
    using BlockLoad  = cub::BlockLoad<const T*, blockDimx, unroll, cub::BLOCK_LOAD_WARP_TRANSPOSE, blockDimy>;
    using BlockStore = cub::BlockStore<T*, blockDimx, unroll, cub::BLOCK_STORE_WARP_TRANSPOSE, blockDimy>;

    __shared__ union
    {
        typename BlockLoad::TempStorage  load_x;
        typename BlockLoad::TempStorage  load_y;
        typename BlockStore::TempStorage store;
    } storage_smem;

    T x_reg[unroll*unroll], y_reg[unroll*unroll], z_reg[unroll*unroll];
    BlockLoad(storage_smem.load_x).Load(x, x_reg);
    BlockLoad(storage_smem.load_y).Load(y, y_reg);

    __syncthreads();

    for (int i = 0; i < unroll; i++) {
        for (int j = 0; j < unroll; j++) {
            const unsigned index = i * unroll + j;
            z_reg[index] = alpha * x_reg[index] + y_reg[index];
        }
    }

    BlockStore(storage_smem.store).Store(z, z_reg);
};
 */

////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

template <typename T>
void run_saxpy_c(const T* px,
                 const T* py,
                       T* pz,
                 unsigned N,
                 T alpha,
                 unsigned repetitions) {
    cuda::error err;

    T* aux;
    err = cudaMallocManaged((void**)&aux, N * sizeof(T));


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

    std::cout << "Launching saxpy C kernels" << std::endl;
    std::cout << "Grid 1 [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Grid 2 [" << dimGrid2.x << "," << dimGrid2.y << "," << dimGrid2.z << "]" << std::endl;
    std::cout << "Grid 4 [" << dimGrid4.x << "," << dimGrid4.y << "," << dimGrid4.z << "]" << std::endl;

    std::cout << "Block 1 [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "Block 2 [" << dimBlock2.x << "," << dimBlock2.y << "," << dimBlock2.z << "]" << std::endl;
    std::cout << "Block 4 [" << dimBlock4.x << "," << dimBlock4.y << "," << dimBlock4.z << "]" << std::endl;

    for (int i = 0; i < repetitions; i++) {
        saxpy_gpu_naive<T><<<dimGrid, dimBlock>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();
        saxpy_gpu_naive_rangebasedloop<T><<<dimGrid, dimBlock>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();
        saxpy_gpu_unroll<T, unroll2><<<dimGrid2, dimBlock2>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();
        saxpy_gpu_unroll<T, unroll4><<<dimGrid4, dimBlock4>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();
    }
    std::cout << std::endl;

    err = cudaDeviceSynchronize();
    err = cudaFree(aux);
}

template <typename T>
void run_saxpy_cpp(const cuda::vector<T>& x,
                   const cuda::vector<T>& y,
                         cuda::vector<T>& z,
                   unsigned N,
                   T alpha,
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

    std::cout << "Launching saxpy CPP kernels" << std::endl;
    std::cout << "Grid 1 [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Grid 2 [" << dimGrid2.x << "," << dimGrid2.y << "," << dimGrid2.z << "]" << std::endl;
    std::cout << "Grid 4 [" << dimGrid4.x << "," << dimGrid4.y << "," << dimGrid4.z << "]" << std::endl;

    std::cout << "Block 1 [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "Block 2 [" << dimBlock2.x << "," << dimBlock2.y << "," << dimBlock2.z << "]" << std::endl;
    std::cout << "Block 4 [" << dimBlock4.x << "," << dimBlock4.y << "," << dimBlock4.z << "]" << std::endl;

    for (int i = 0; i < repetitions; i++) {
        saxpy_gpu_cpp_vector<T><<<dimGrid, dimBlock>>>(x, y, z, N, alpha);
        err = cudaGetLastError();
        saxpy_gpu_cpp_vector_unroll<T, unroll2><<<dimGrid2, dimBlock2>>>(x, y, z, N, alpha);
        err = cudaGetLastError();
        saxpy_gpu_cpp_vector_unroll<T, unroll4><<<dimGrid4, dimBlock4>>>(x, y, z, N, alpha);
        err = cudaGetLastError();
    }
    std::cout << std::endl;

    err = cudaDeviceSynchronize();
}

template <typename T>
void run_saxpy_cublas(const T* px,
                            T* py,
                      unsigned N,
                      const T alpha,
                      unsigned repetitions);


template <>
void run_saxpy_cublas(const float* px,
                            float* py,
                      unsigned N,
                      const float alpha,
                      unsigned repetitions) {
    cublasStatus_t status;
    cublasHandle_t handle;

    std::cout << "Launching saxpy CUBLAS" << std::endl;

    status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);

    for (int i = 0; i < repetitions; i++) {
        status = cublasSaxpy(handle, N, &alpha, px, 1, py, 1);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }
    std::cout << std::endl;
}

template <>
void run_saxpy_cublas(const double* px,
                            double* py,
                      unsigned N,
                      const double alpha,
                      unsigned repetitions) {
    cublasStatus_t status;
    cublasHandle_t handle;

    std::cout << "Launching saxpy CUBLAS" << std::endl;

    status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);

    for (int i = 0; i < repetitions; i++) {
        status = cublasDaxpy(handle, N, &alpha, px, 1, py, 1);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }
    std::cout << std::endl;
}

template <typename T>
void run_saxpy_cub(const T* px,
                   const T* py,
                         T* pz,
                   unsigned N,
                   T alpha,
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

    std::cout << "Launching saxpy CUB kernels" << std::endl;
    std::cout << "Grid 1 [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Grid 2 [" << dimGrid2.x << "," << dimGrid2.y << "," << dimGrid2.z << "]" << std::endl;
    std::cout << "Grid 4 [" << dimGrid4.x << "," << dimGrid4.y << "," << dimGrid4.z << "]" << std::endl;

    std::cout << "Block 1 [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "Block 2 [" << dimBlock2.x << "," << dimBlock2.y << "," << dimBlock2.z << "]" << std::endl;
    std::cout << "Block 4 [" << dimBlock4.x << "," << dimBlock4.y << "," << dimBlock4.z << "]" << std::endl;

    for (int i = 0; i < repetitions; i++) {
        saxpy_gpu_cub<T, 128,       1><<<dimGrid, dimBlock>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();
        saxpy_gpu_cub<T, 128, unroll2><<<dimGrid2, dimBlock2>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();
        saxpy_gpu_cub<T, 128, unroll4><<<dimGrid4, dimBlock4>>>(px, py, pz, N, alpha);
        err = cudaGetLastError();
    }
    std::cout << std::endl;

    err = cudaDeviceSynchronize();
}


template <typename T>
void saxpy_c(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::device gpu;
        cuda::error err;

        cout << "--- SAXPY C ---" << endl;

        size_t free, total;
        gpu.getMemInfo(free, total);

        size_t allocated_by_os = total - free;
        size_t est_program_alloc = 3 * N * sizeof(T);

        T *px, *py, *pz;
        err = cudaMalloc((void**)&px, N * sizeof(T));
        err = cudaMalloc((void**)&py, N * sizeof(T));
        err = cudaMalloc((void**)&pz, N * sizeof(T));
        T alpha = 0.8;

        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Difference "                       << factor << endl;


        run_saxpy_c(px, py, pz, N, alpha, repetitions);

        err = cudaFree(px);
        err = cudaFree(py);
        err = cudaFree(pz);

    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

template <typename T>
void saxpy_cpp(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::error err;

        cout << "--- SAXPY Cpp ---" << endl;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        cuda::vector<T> *a = new cuda::vector<T>(N);
        cuda::vector<T> *b = new cuda::vector<T>(N);
        cuda::vector<T> *c = new cuda::vector<T>(N);
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

        run_saxpy_cpp(*a, *b, *c, N, alpha, repetitions);

        delete(a);
        delete(b);
        delete(c);
    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

template <typename T>
void saxpy_cublas(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::device gpu;
        cuda::error err;

        cout << "--- SAXPY Cublas ---" << endl;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        T *px, *py;
        err = cudaMalloc((void**)&px, N * sizeof(T));
        err = cudaMalloc((void**)&py, N * sizeof(T));
        const T alpha = 0.8;

        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t est_program_alloc = 2 * N * sizeof(T);
        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Difference "                       << factor << endl;


        run_saxpy_cublas(px, py, N, alpha, repetitions);

        err = cudaFree(px);
        err = cudaFree(py);
    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

template <typename T>
void saxpy_cub(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::device gpu;
        cuda::error err;

        cout << "--- SAXPY CUB ---" << endl;

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


        run_saxpy_cub(px, py, pz, N, alpha, repetitions);

        err = cudaFree(px);
        err = cudaFree(py);
        err = cudaFree(pz);

    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

void launch_saxpy(cuda::device& gpu, unsigned N, unsigned repetitions)
{

    saxpy_c<float>(gpu, N, repetitions);
    //saxpy_cpp<float>(gpu, N, repetitions);
    saxpy_cublas<float>(gpu, N, repetitions);
    saxpy_cub<float>(gpu, N, repetitions);

    saxpy_c<double>(gpu, N, repetitions);
    //saxpy_cpp<double>(gpu, N, repetitions);
    saxpy_cublas<double>(gpu, N, repetitions);
    saxpy_cub<double>(gpu, N, repetitions);
}
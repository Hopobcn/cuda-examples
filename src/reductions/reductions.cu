#include <reductions/reductions.hpp>
#include <iostream>
#include <util/grid_stride.hpp>
#include <util/cuda_grid_config.hpp>
#include <util/cuda_error.hpp>


using cuda::grid_stride_range;
using cuda::util::getGridDimensions;
using cuda::util::lang::range;

struct Reduction_sum {
public:
    inline __device__ __host__
    Reduction_sum() {
        sum = 0;
    };

    inline __device__ __host__
    Reduction_sum& operator+=(float elem)
    {
        sum += elem;
        return *this;
    }

    inline __device__ __host__
    volatile Reduction_sum& operator+=(float elem) volatile
    {
        sum += elem;
        return *this;
    }

    inline __device__ __host__
    Reduction_sum& operator+=(const Reduction_sum &other)
    {
        sum += other.sum;
        return *this;
    }

    inline __device__ __host__
    volatile Reduction_sum& operator +=(const Reduction_sum& other) volatile
    {
        sum += other.sum;
        return *this;
    }

    float sum;
};

template <typename T>
inline bool operator!=( const Reduction_sum& lhs,
                        const Reduction_sum& rhs )
{
    return lhs.sum != rhs.sum;
}

template <typename T>
struct SharedMemory
{
    __device__ inline operator T*()
    {
        extern __shared__ T __smem[];
        return (T*) (void*) __smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ T __smem[];
        return (T*) (void*) __smem;
    }
};

template <>
struct SharedMemory<Reduction_sum>
{
    __device__ inline operator Reduction_sum*()
    {
        extern __shared__ Reduction_sum __smem_Reduction_sum[];
        return (Reduction_sum*) (void*) __smem_Reduction_sum;
    }

    __device__ inline operator const Reduction_sum*() const
    {
        extern __shared__ Reduction_sum __smem_Reduction_sum[];
        return (Reduction_sum*) (void*) __smem_Reduction_sum;
    }
};


template <typename T>
__device__
inline void warpReduceSmem(volatile T* warp_smem, const unsigned tid) {
    if (blockDim.x >= 64) warp_smem[tid] += warp_smem[tid + 32];
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300))
    T sum = warp_smem[tid];
    sum += __shfl_down(sum, 16);
    sum += __shfl_down(sum,  8);
    sum += __shfl_down(sum,  4);
    sum += __shfl_down(sum,  2);
    sum += __shfl_down(sum,  1);
#else
    warp_smem[tid] += warp_smem[tid + 16];
    warp_smem[tid] += warp_smem[tid + 8];
    warp_smem[tid] += warp_smem[tid + 4];
    warp_smem[tid] += warp_smem[tid + 2];
    warp_smem[tid] += warp_smem[tid + 1];
#endif
}

template <typename ReductionType>
__device__
inline void warpReduceSmem(SharedMemory<ReductionType> warp_smem, const unsigned tid) {
    if (blockDim.x >= 64) warp_smem[tid] += warp_smem[tid + 32];
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300))
    ReductionType sum = warp_smem[tid];
    sum += __shfl_down(sum, 16);
    sum += __shfl_down(sum,  8);
    sum += __shfl_down(sum,  4);
    sum += __shfl_down(sum,  2);
    sum += __shfl_down(sum,  1);
#else
    warp_smem[tid] += warp_smem[tid + 16];
    warp_smem[tid] += warp_smem[tid + 8];
    warp_smem[tid] += warp_smem[tid + 4];
    warp_smem[tid] += warp_smem[tid + 2];
    warp_smem[tid] += warp_smem[tid + 1];
#endif
}

template <typename T, unsigned blockDimx>
__device__
inline void warpReduceSmem(volatile T* warp_smem, const unsigned tid) {
    if (blockDimx >= 64) warp_smem[tid] += warp_smem[tid + 32];
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300))
    T sum = warp_smem[tid];
    if (blockDimx >= 32) sum += __shfl_down(sum, 16);
    if (blockDimx >= 16) sum += __shfl_down(sum,  8);
    if (blockDimx >=  8) sum += __shfl_down(sum,  4);
    if (blockDimx >=  4) sum += __shfl_down(sum,  2);
    if (blockDimx >=  2) sum += __shfl_down(sum,  1);
    warp_smem[tid];
#else
    if (blockDimx >= 32) warp_smem[tid] += warp_smem[tid + 16];
    if (blockDimx >= 16) warp_smem[tid] += warp_smem[tid + 8];
    if (blockDimx >=  8) warp_smem[tid] += warp_smem[tid + 4];
    if (blockDimx >=  4) warp_smem[tid] += warp_smem[tid + 2];
    if (blockDimx >=  2) warp_smem[tid] += warp_smem[tid + 1];
#endif
}

template <typename T, bool atomicOp>
__device__
inline void storeToMem(volatile T* warp_smem, T* out, const unsigned tid) {
    if (tid == 0) {
        if (atomicOp)
            atomicAdd(out, warp_smem[tid]);
        else
            out[blockIdx.x] = warp_smem[tid];
    }
}

template <typename T, bool atomics>
__global__
void reduction1_c_array(const T* __restrict__ in, T* out, unsigned N) {
    extern __shared__ T storage_smem[];

    T sum = 0;
    const unsigned tid = threadIdx.x;
    for (unsigned i = blockIdx.x * blockDim.x + tid;
                  i < N;
                  i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    storage_smem[tid] = sum;
    __syncthreads();

    for (unsigned activeThreads = blockDim.x>>1;
                  activeThreads > 0;
                  activeThreads >>= 1) {
        if (tid < activeThreads)
            storage_smem[tid] += storage_smem[tid + activeThreads];
        __syncthreads();
    }

    storeToMem<T, atomics>(storage_smem, out, tid);
}

template <typename T, bool atomics>
__global__
void reduction2_c_array(const T* __restrict__ in, T* out, unsigned N) {
    extern __shared__ T storage_smem[];

    T sum = 0;
    const unsigned tid = threadIdx.x;
    for (unsigned i = blockIdx.x * blockDim.x + tid;
         i < N;
         i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    storage_smem[tid] = sum;
    __syncthreads();

    for (unsigned activeThreads = blockDim.x>>1;
         activeThreads > warpSize;
         activeThreads >>= 1) {
        if (tid < activeThreads)
            storage_smem[tid] += storage_smem[tid + activeThreads];
        __syncthreads();
    }
    // warp synchronous at the end
    if (tid < warpSize) {
        warpReduceSmem(storage_smem, tid);
        storeToMem<T,atomics>(storage_smem, out, tid);
    }
}

template <typename T, unsigned blockDimx, bool atomics>
__global__
void reduction3_c_array(const T* __restrict__ in, T* out, unsigned N) {
    extern __shared__ T storage_smem[];

    T sum = 0;
    const unsigned tid = threadIdx.x;
    for (unsigned i = blockIdx.x * blockDim.x + tid;
         i < N;
         i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    storage_smem[tid] = sum;
    __syncthreads();

    if (blockDimx >= 1024) {
        if (tid < 512)
            storage_smem[tid] += storage_smem[tid + 512];
        __syncthreads();
    }
    if (blockDimx >= 512) {
        if (tid < 256)
            storage_smem[tid] += storage_smem[tid + 256];
        __syncthreads();
    }
    if (blockDimx >= 256) {
        if (tid < 128)
            storage_smem[tid] += storage_smem[tid + 128];
        __syncthreads();
    }
    if (blockDimx >= 128) {
        if (tid < 64)
            storage_smem[tid] += storage_smem[tid + 64];
        __syncthreads();
    }
    // warp synchronous at the end
    if (tid < warpSize) {
        warpReduceSmem<T,blockDimx>(storage_smem, tid);
        storeToMem<T,atomics>(storage_smem, out, tid);
    }
}

template <typename ReductionType, typename T, bool atomics>
__global__
void reduction2_c_array_templated(const T* __restrict__ in, ReductionType* out, unsigned N) {
    SharedMemory<ReductionType> storage_smem;
    ReductionType sum = 0;
    const unsigned tid = threadIdx.x;
    for (unsigned i = blockIdx.x * blockDim.x + tid;
         i < N;
         i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    storage_smem[tid] = sum;
    __syncthreads();

    for (unsigned activeThreads = blockDim.x>>1;
         activeThreads > warpSize;
         activeThreads >>= 1) {
        if (tid < activeThreads)
            storage_smem[tid] += storage_smem[tid + activeThreads];
        __syncthreads();
    }
    // warp synchronous at the end
    if (tid < warpSize) {
        warpReduceSmem(storage_smem, tid);
        storeToMem<T,atomics>(storage_smem, out, tid);
    }
}

template <typename T>
void reduction1_c_array_two_pass(const T* in, T* partial, T* out, unsigned N) {
    unsigned block_size_x = 128;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );
    unsigned sharedSize = block_size_x * sizeof(T);

    std::cout << "Launching two pass reduction 1" << std::endl;
    std::cout << "Grid  [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Block [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "SMEM  [" << sharedSize << "] bytes" << std::endl;

    reduction1_c_array<T,false><<<dimGrid,dimBlock,sharedSize>>>(in,      partial, N);
    reduction1_c_array<T,false><<<      1,dimBlock,sharedSize>>>(partial, out,     dimBlock.x);
}

template <typename T>
void reduction2_c_array_two_pass(const T* in, T* partial, T* out, unsigned N) {
    unsigned block_size_x = 128;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );
    unsigned sharedSize = block_size_x * sizeof(T);

    std::cout << "Launching two pass reduction 2" << std::endl;
    std::cout << "Grid  [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Block [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "SMEM  [" << sharedSize << "] bytes" << std::endl;

    reduction2_c_array<T,false><<<dimGrid,dimBlock,sharedSize>>>(in,      partial, N);
    reduction2_c_array<T,false><<<      1,dimBlock,sharedSize>>>(partial, out,     dimBlock.x);
}

template <typename T, unsigned blockDimx>
void reduction3_c_array_two_pass_unroll(const T *in, T *partial, T *out, unsigned N, dim3 dimGrid, dim3 dimBlock,
                                        unsigned sharedSize) {
    reduction3_c_array<T,blockDimx,false><<<dimGrid,dimBlock,sharedSize>>>(in,      partial, N);
    reduction3_c_array<T,blockDimx,false><<<      1,dimBlock,sharedSize>>>(partial, out,     blockDimx);
}

template <typename T>
void reduction3_c_array_two_pass(const T* in, T* partial, T* out, unsigned N) {
    unsigned block_size_x = 128;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );
    unsigned sharedSize = block_size_x * sizeof(T);

    std::cout << "Launching two pass reduction 2" << std::endl;
    std::cout << "Grid  [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Block [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "SMEM  [" << sharedSize << "] bytes" << std::endl;

    switch (dimBlock.x) {
        case 1024 : return reduction3_c_array_two_pass_unroll<T, 1024>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case  512 : return reduction3_c_array_two_pass_unroll<T,  512>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case  256 : return reduction3_c_array_two_pass_unroll<T,  256>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case  128 : return reduction3_c_array_two_pass_unroll<T,  128>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case   64 : return reduction3_c_array_two_pass_unroll<T,   64>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case   32 : return reduction3_c_array_two_pass_unroll<T,   32>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case   16 : return reduction3_c_array_two_pass_unroll<T,   16>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case    8 : return reduction3_c_array_two_pass_unroll<T,    8>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case    4 : return reduction3_c_array_two_pass_unroll<T,    4>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case    2 : return reduction3_c_array_two_pass_unroll<T,    2>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
        case    1 : return reduction3_c_array_two_pass_unroll<T,    1>(in, partial, out, N, dimGrid, dimBlock, sharedSize);
    }
}

template <typename ReductionType, typename T>
void reduction4_c_array_two_pass(const T* in, ReductionType* partial, ReductionType* out, unsigned N) {
    unsigned block_size_x = 128;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );
    unsigned sharedSize = block_size_x * sizeof(T);

    std::cout << "Launching one pass reduction 2" << std::endl;
    std::cout << "Grid  [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Block [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "SMEM  [" << sharedSize << "] bytes" << std::endl;

    reduction2_c_array_templated<ReductionType, T,false><<<dimGrid,dimBlock,sharedSize>>>(in,     partial, N);
    reduction2_c_array_templated<ReductionType, T,false><<<dimGrid,dimBlock,sharedSize>>>(partial, out,    dimBlock.x);
}


template <typename T>
void reduction1_c_array_one_pass(const T* in, T* out, unsigned N) {
    unsigned block_size_x = 128;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );
    unsigned sharedSize = block_size_x * sizeof(T);

    std::cout << "Launching one pass reduction 1" << std::endl;
    std::cout << "Grid  [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Block [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "SMEM  [" << sharedSize << "] bytes" << std::endl;

    reduction1_c_array<T,true><<<dimGrid,dimBlock,sharedSize>>>(in, out, N);
}


template <typename T>
void reduction2_c_array_one_pass(const T* in, T* out, unsigned N) {
    unsigned block_size_x = 128;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );
    unsigned sharedSize = block_size_x * sizeof(T);

    std::cout << "Launching one pass reduction 2" << std::endl;
    std::cout << "Grid  [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Block [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "SMEM  [" << sharedSize << "] bytes" << std::endl;

    reduction2_c_array<T,true><<<dimGrid,dimBlock,sharedSize>>>(in, out, N);
}

template <typename T, unsigned blockDimx>
void reduction3_c_array_one_pass_unroll(const T *in, T *out, unsigned N, dim3 dimGrid, dim3 dimBlock,
                                        unsigned sharedSize) {
    reduction3_c_array<T,blockDimx,true><<<dimGrid,dimBlock,sharedSize>>>(in, out, N);
}

template <typename T>
void reduction3_c_array_one_pass(const T* in, T* out, unsigned N) {
    unsigned block_size_x = 128;
    unsigned block_size_y = 1;
    unsigned block_size_z = 1;
    dim3 dimGrid = getGridDimensions(N, 1, 1, block_size_x, block_size_y, block_size_z);
    dim3 dimBlock( block_size_x, block_size_y, block_size_z );
    unsigned sharedSize = block_size_x * sizeof(T);

    std::cout << "Launching one pass reduction 2" << std::endl;
    std::cout << "Grid  [" << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << "]" << std::endl;
    std::cout << "Block [" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << "]" << std::endl;
    std::cout << "SMEM  [" << sharedSize << "] bytes" << std::endl;

    switch (dimBlock.x) {
        case 1024 : return reduction3_c_array_one_pass_unroll<T, 1024>(in, out, N, dimGrid, dimBlock, sharedSize);
        case  512 : return reduction3_c_array_one_pass_unroll<T,  512>(in, out, N, dimGrid, dimBlock, sharedSize);
        case  256 : return reduction3_c_array_one_pass_unroll<T,  256>(in, out, N, dimGrid, dimBlock, sharedSize);
        case  128 : return reduction3_c_array_one_pass_unroll<T,  128>(in, out, N, dimGrid, dimBlock, sharedSize);
        case   64 : return reduction3_c_array_one_pass_unroll<T,   64>(in, out, N, dimGrid, dimBlock, sharedSize);
        case   32 : return reduction3_c_array_one_pass_unroll<T,   32>(in, out, N, dimGrid, dimBlock, sharedSize);
        case   16 : return reduction3_c_array_one_pass_unroll<T,   16>(in, out, N, dimGrid, dimBlock, sharedSize);
        case    8 : return reduction3_c_array_one_pass_unroll<T,    8>(in, out, N, dimGrid, dimBlock, sharedSize);
        case    4 : return reduction3_c_array_one_pass_unroll<T,    4>(in, out, N, dimGrid, dimBlock, sharedSize);
        case    2 : return reduction3_c_array_one_pass_unroll<T,    2>(in, out, N, dimGrid, dimBlock, sharedSize);
        case    1 : return reduction3_c_array_one_pass_unroll<T,    1>(in, out, N, dimGrid, dimBlock, sharedSize);
    }
}



void run_reduce(const cuda::vector<float>& in,
                      cuda::vector<float>& partial,
                      cuda::vector<float>& out,
                const float* pin,
                      float* ppartial,
                      float* pout,
                unsigned N,
                unsigned repetitions) {
    cuda::error err;
    for (int i = 0; i < repetitions; i++) {
        reduction1_c_array_two_pass(pin, ppartial, pout, N);
        err = cudaGetLastError();

        reduction2_c_array_two_pass(pin, ppartial, pout, N);
        err = cudaGetLastError();

        reduction3_c_array_two_pass(pin, ppartial, pout, N);
        err = cudaGetLastError();

        reduction4_c_array_two_pass(pin, ppartial, pout, N);
        err = cudaGetLastError();

        reduction1_c_array_one_pass(pin, pout, N);
        err = cudaGetLastError();

        reduction2_c_array_one_pass(pin, pout, N);
        err = cudaGetLastError();

        reduction3_c_array_one_pass(pin, pout, N);
        err = cudaGetLastError();
    }
}
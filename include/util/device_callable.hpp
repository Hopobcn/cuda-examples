#pragma once

#ifdef __CUDACC__
#define DEVICE_CALLABLE             __device__
#define DEVICE_CALLABLE_INLINE      __device__ __forceinline__
#define HOST_DEVICE_CALLABLE        __host__ __device__
#define HOST_DEVICE_CALLABLE_INLINE __host__ __device__ __forceinline__
#else
#define DEVICE_CALLABLE
#define DEVICE_CALLABLE_INLINE
#define HOST_DEVICE_CALLABLE
#define HOST_DEVICE_CALLABLE_INLINE
#endif
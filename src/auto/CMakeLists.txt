include_directories("../../include" SYSTEM ${INCLUDE_DIRS})

set(CUDA_SEPARATE_COMPILATION OFF)


list(APPEND CMAKE_NVCC_FLAGS "-std=c++11 -O3 -Xptxas -v")
list(APPEND CMAKE_NVCC_FLAGS "-Xcompiler '-march=native' -lineinfo -use_fast_math")
list(APPEND CMAKE_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")


cuda_add_library(cuda_auto
                 1d_pattern.cu
                 OPTIONS -DUSE_CUDA="__USE_CUDA__"
                 RELEASE -std=c++11 -DNDEBUG -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -O3 -use_fast_math
                 DEBUG   -std=c++11 -DDEBUG  -gencode arch=compute_35,code=sm_35 -g -G )

cuda_add_cublas_to_target(cuda_auto)
target_link_libraries(cuda_auto cuda_util)

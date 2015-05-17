include_directories("../../include" SYSTEM ${INCLUDE_DIRS})

set(CUDA_SEPARATE_COMPILATION OFF)


list(APPEND CMAKE_NVCC_FLAGS "-std=c++11 -O3 -Xptxas -v")
list(APPEND CMAKE_NVCC_FLAGS "-Xcompiler '-march=native' -lineinfo -use_fast_math")
list(APPEND CMAKE_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")


cuda_add_library(cuda_streaming
                 saxpy.cu
                 OPTIONS -DUSE_CUDA="__USE_CUDA__"
                 RELEASE --use_fast_math -DNDEBUG
                 DEBUG -g -DDEBUG -std=c++11)
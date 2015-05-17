if(USEMPI)
  set(ENV{CC}  mpicc ) # C compiler for parallel build
  set(ENV{CXX} mpicxx) # C++ compiler for parallel build
else()
  set(ENV{CC}  gcc ) # C compiler for serial build
  set(ENV{CXX} g++ ) # C++ compiler for serial build
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native -Wall")
set(CMAKE_NVCC_FLAGS "-std=c++11 -O3 -Xptxas -v -lineinfo -use_fast_math -gencode arch=compute_52,code=sm_52")

set(LIBS m)
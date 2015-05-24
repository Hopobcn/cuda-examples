include_directories("../../include" "/usr/local/cuda/include" SYSTEM ${INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_library(cuda_util cuda_grid_config.cpp)
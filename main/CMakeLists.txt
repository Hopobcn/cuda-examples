
include_directories(${INCLUDE_DIRS} "../include")
include_directories(${INCLUDE_DIRS} "../include/streaming")
include_directories(${INCLUDE_DIRS} "../include/reductions")
include_directories(${INCLUDE_DIRS} "../include/util")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

cuda_add_executable(cuda_examples main.cpp)
target_link_libraries(cuda_examples cuda_streaming cuda_reductions ${LIBS} m)
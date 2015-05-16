
include_directories(${INCLUDE_DIRS} "../include")

cuda_add_executable(cuda_examples main.cpp)
target_link_libraries(cuda_examples cuda_examples_lib ${LIBS} m)
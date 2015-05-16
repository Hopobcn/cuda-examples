cmake_minimum_required(VERSION 3.1)
project(cuda_examples)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/config)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(CUDA)

add_subdirectory(src)
add_subdirectory(main)
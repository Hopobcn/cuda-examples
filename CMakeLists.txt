cmake_minimum_required(VERSION 3.0)
project(cuda_examples)

find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

include(ExternalProject REQUIRED)


set( CMAKE_MODULE_PATH              ${PROJECT_SOURCE_DIR}/config ${CMAKE_MODULE_PATH} )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )
set( EXTERNAL_PROJECTS_DIRECTORY    ${PROJECT_SOURCE_DIR}/ext )


add_subdirectory( ${EXTERNAL_PROJECTS_DIRECTORY}/gtest )
add_subdirectory( ${EXTERNAL_PROJECTS_DIRECTORY}/gmock )

add_subdirectory(src)
add_subdirectory(main)
add_subdirectory(test)
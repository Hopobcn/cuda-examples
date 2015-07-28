
file(GLOB TEST_UTIL_SRC *.cpp)

include_directories( ${GTEST_INCLUDE_DIRS} ${GMOCK_INCLUDE_DIRS} )
add_executable(utiltest ${TEST_UTIL_SRC})
add_dependencies(utiltest gtest gmock)

target_link_libraries(utiltest cuda_util libgtest libgmock)

install(TARGETS utiltest DESTINATION bin)
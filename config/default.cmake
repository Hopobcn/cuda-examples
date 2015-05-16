if(USEMPI)
  set(ENV{CC}  mpicc ) # C compiler for parallel build
  set(ENV{CXX} mpicxx) # C++ compiler for parallel build
else()
  set(ENV{CC}  gcc ) # C compiler for serial build
  set(ENV{CXX} g++ ) # C++ compiler for serial build
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(LIBS m)
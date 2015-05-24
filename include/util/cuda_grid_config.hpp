#pragma once
#include <vector_types.h>

namespace cuda { namespace util {

dim3 getGridDimensions(unsigned  sx, unsigned  sy, unsigned  sz,
                       unsigned& bx, unsigned& by, unsigned& bz);

}}

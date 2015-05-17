#pragma once

#include "cuda_error.hpp"

namespace cuda {

class managed {
public:

    void *operator new(std::size_t len) {
        void* ptr;
        error err = cudaMallocManaged(&ptr, len);
        err.update();
        return ptr;
    };

    void operator delete(void *ptr) {
        error err = cudaFree(ptr);
        err.update();
    }
};

}
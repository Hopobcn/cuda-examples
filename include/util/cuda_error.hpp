#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include "cuda_exceptions.hpp"

namespace cuda {

class error {
public:
    error()
            : err{cudaSuccess} { };

    error(cudaError_t err)
            : err{err}
    {
        checkError();
    }
    error(cudaError_t* err)
            : err{*err}
    {
        checkError();
    }

    error(const error &rhs) {
        this->err = rhs.err;
        checkError();
    }
    error &operator=(const error& rhs) {
        this->err = rhs.err;
        checkError();
        return *this;
    }

    void update() {
        err = cudaGetLastError();
        checkError();
    }

private:
    void checkError() {
        if (err != cudaSuccess)
            throw cuda::cuda_exception(cudaGetErrorString(err));
    }

    cudaError_t err;
};

}
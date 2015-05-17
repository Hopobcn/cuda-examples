#pragma once

#include <iostream>
#include <driver_types.h>
#include "cuda_error.hpp"
#include "range.hpp"

using namespace std;

namespace cuda {

class device {

public:
    device() {
        error err;
        err = cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 0) {
            selected_device = 0;
            err = cudaSetDevice(selected_device);
        } else {
            throw cuda::cuda_exception("No devices found!");
        }
    }
    device(int hint) {
        error err;
        err = cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 0) {
            selected_device = hint % deviceCount;
            err = cudaSetDevice(selected_device);
        } else {
            throw cuda::cuda_exception("No devices found!");
        }
    }

    void list_devices() {
        for (auto i :  cuda::util::lang::range(0, deviceCount)) {
            cudaDeviceProp deviceProp;
            error err = cudaGetDeviceProperties(&deviceProp, i);
            err.update();
            cout << "Device " << i << " has compute capability: " << deviceProp.major << "." << deviceProp.minor << endl;
        }
    }
private:
    int selected_device;
    int deviceCount;
};

}
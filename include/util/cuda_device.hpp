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
        error err;
        int driverVersion = 0, runtimeVersion = 0;

        err = cudaDriverGetVersion(&driverVersion);
        err = cudaRuntimeGetVersion(&runtimeVersion);
        cout << "CUDA Driver Version:  " << driverVersion/1000 << "." << (driverVersion%100)/10 << endl;
        cout << "CUDA Runtime Version: " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << endl << endl;

        for (auto i :  cuda::util::lang::range(0, deviceCount)) {
            cudaDeviceProp deviceProp;

            err = cudaSetDevice(i);
            err = cudaGetDeviceProperties(&deviceProp, i);

            cout << "Device " << i << "                 " << deviceProp.name << endl;
            cout << "-Compute capability:               " << deviceProp.major << "." << deviceProp.minor << endl;
            cout << "-Multiprocessors:                  " << deviceProp.multiProcessorCount << endl;
            //cout << "-CUDA Cores/MP:      " << _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor) << endl;
            //cout << "-CUDA Cores:         " << _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor) << endl;
            cout << "-Shared Mem/MP:                    " << deviceProp.sharedMemPerBlock << endl;
            cout << "-Num Registers/MP:                 " << deviceProp.regsPerBlock << endl;
            cout << "-WarpSize:                         " << deviceProp.warpSize <<  endl;
            cout << "-Max number of threads per MP:     " << deviceProp.maxThreadsPerMultiProcessor << endl;
            cout << "-Max number of threads per block:  " << deviceProp.maxThreadsPerBlock << endl;
            cout << "-Max dim size of a thread block:   " << deviceProp.maxThreadsDim[0] << " x "
                                                          << deviceProp.maxThreadsDim[1] << " x "
                                                          << deviceProp.maxThreadsDim[2] << endl;
            cout << "-Max dim size of a grid size:      " << deviceProp.maxGridSize[0] << " x "
                                                          << deviceProp.maxGridSize[1] << " x "
                                                          << deviceProp.maxGridSize[2] << endl;


        }

        cout << endl;
    }


    int get_device_id() {
        error err;
        int device;
        err = cudaGetDevice(&device);
        return device;
    }

    void getMemInfo(size_t& free, size_t& total) {
        error err;
        err = cudaMemGetInfo(&free, &total);
        err.update();
    }

private:
    int selected_device;
    int deviceCount;
};

}
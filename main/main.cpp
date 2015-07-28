#include <iostream>
#include <cuda_device.hpp>
#include <cuda_vector.hpp>
#include <auto.hpp>
#include <saxpy.hpp>
#include <vecadd.hpp>
#include <reductions.hpp>

using namespace std;



int main() {
    unsigned N = 1 * 1024 * 1024;
    unsigned rep = 100;

    cuda::device gpu;

    gpu.list_devices();

    //cuda_cpp11_testing(gpu, N);

    launch_saxpy(gpu, N, rep);
    //launch_vecadd(gpu, N, rep);

    return 0;
}
#include <iostream>
#include <cuda_device.hpp>
#include <cuda_vector.hpp>
#include <saxpy.hpp>

using namespace std;


int main() {
    unsigned N = 3 * 1024 * 1024;
    using type = float;

    try {
        cuda::device gpu;
        cuda::error err;
        cuda::vector<type> *a = new cuda::vector<type>(N);
        cuda::vector<type> *b = new cuda::vector<type>(N);
        cuda::vector<type> *c = new cuda::vector<type>(N);
        type *px, *py, *pz;
        err = cudaMallocManaged((void**)&px, N * sizeof(type));
        err = cudaMallocManaged((void**)&py, N * sizeof(type));
        err = cudaMallocManaged((void**)&pz, N * sizeof(type));
        type alpha = 0.8;

        gpu.list_devices();

        err = cudaDeviceSynchronize();
        for (auto i : cuda::util::lang::range<unsigned>(0, N)) {
            (*a)[i] = 0;
            (*b)[i] = 0;
        }

        run_saxpy(*a, *b, *c, px, py, pz, N, alpha);


    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }

    return 0;
}
#include <iostream>
#include <cuda_device.hpp>
#include <cuda_vector.hpp>
#include <saxpy.hpp>

using namespace std;


int main() {
    unsigned N = 100 * 1024 * 1024;
    using type = float;

    //try {
        cuda::device gpu;
        cuda::error err;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        cuda::vector<type> *a = new cuda::vector<type>(N);
        cuda::vector<type> *b = new cuda::vector<type>(N);
        cuda::vector<type> *c = new cuda::vector<type>(N);
        type *px, *py, *pz;
        err = cudaMallocManaged((void**)&px, N * sizeof(type));
        err = cudaMallocManaged((void**)&py, N * sizeof(type));
        err = cudaMalloc((void**)&pz, N * sizeof(type));
        type alpha = 0.8;

        gpu.list_devices();
        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t est_program_alloc = 2 * 3 * N * sizeof(type);
        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Difference "                       << factor << endl;

        err = cudaDeviceSynchronize();
        for (auto i : cuda::util::lang::range<unsigned>(0, N)) {
            (*a)[i] = 0;
            (*b)[i] = 0;
        }

        run_saxpy(*a, *b, *c, px, py, pz, N, alpha);

        err = cudaFree(px);
        err = cudaFree(py);
        err = cudaFree(pz);

    //} catch(cuda::cuda_exception error) {
    //    std::cout << error.what() << std::endl;
    //}

    return 0;
}
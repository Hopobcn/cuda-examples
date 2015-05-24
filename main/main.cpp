#include <iostream>
#include <cuda_device.hpp>
#include <cuda_vector.hpp>
#include <saxpy.hpp>
#include <reductions.hpp>

using namespace std;

template <typename T>
void saxpy_c(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::device gpu;
        cuda::error err;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        T *px, *py, *pz;
        err = cudaMallocManaged((void**)&px, N * sizeof(T));
        err = cudaMallocManaged((void**)&py, N * sizeof(T));
        err = cudaMalloc((void**)&pz, N * sizeof(T));
        T alpha = 0.8;

        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t est_program_alloc = 3 * N * sizeof(T);
        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Difference "                       << factor << endl;


        run_saxpy_c(px, py, pz, N, alpha, repetitions);

        err = cudaFree(px);
        err = cudaFree(py);
        err = cudaFree(pz);

    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

template <typename T>
void saxpy_cpp(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::error err;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        cuda::vector<T> *a = new cuda::vector<T>(N);
        cuda::vector<T> *b = new cuda::vector<T>(N);
        cuda::vector<T> *c = new cuda::vector<T>(N);
        T alpha = 0.8;

        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t est_program_alloc = 3 * N * sizeof(T);
        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Difference "                       << factor << endl;

        run_saxpy_cpp(*a, *b, *c, N, alpha, repetitions);

        delete(a);
        delete(b);
        delete(c);
    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

template <typename T>
void saxpy_cublas(cuda::device& gpu, unsigned N, unsigned repetitions) {
    try {
        cuda::device gpu;
        cuda::error err;

        size_t free, total, allocated_by_os;
        gpu.getMemInfo(free, total);
        allocated_by_os = total - free;

        T *px, *py;
        err = cudaMallocManaged((void**)&px, N * sizeof(T));
        err = cudaMallocManaged((void**)&py, N * sizeof(T));
        const T alpha = 0.8;

        gpu.getMemInfo(free, total);
        cout << "Free mem: " << free/(1024*1024) << " / " << total/(1024*1024) << " MB" << endl;

        size_t est_program_alloc = 2 * N * sizeof(T);
        size_t real_program_alloc = total - free - allocated_by_os;
        float  factor = (real_program_alloc - est_program_alloc)/ static_cast<float>(real_program_alloc);

        cout << "Mem allocated by os: "             << allocated_by_os/(1024*1024)      << " MB" << endl;
        cout << "Mem allocated by program[Real]: "  << real_program_alloc/(1024*1024)   << " MB" << endl;
        cout << "Mem allocated by program[Est]: "   << est_program_alloc/(1024*1024)    << " MB" << endl;
        cout << "Difference "                       << factor << endl;


        run_saxpy_cublas(px, py, N, alpha, repetitions);

        err = cudaFree(px);
        err = cudaFree(py);
    } catch(cuda::cuda_exception error) {
        std::cout << error.what() << std::endl;
    }
}

int main() {
    unsigned N = 50 * 1024 * 1024;
    unsigned rep = 10;
    cuda::device gpu;

    gpu.list_devices();

    saxpy_c<float>(gpu, N, rep);
    saxpy_cpp<float>(gpu, N, rep);
    saxpy_cublas<float>(gpu, N, rep);

    saxpy_c<double>(gpu, N, rep);
    saxpy_cpp<double>(gpu, N, rep);
    saxpy_cublas<double>(gpu, N, rep);

    return 0;
}
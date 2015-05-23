#pragma once

#include <util/cuda_vector.hpp>

void run_saxpy(const cuda::vector<float>& x,
               const cuda::vector<float>& y,
                     cuda::vector<float>& z,
               const float* px,
               const float* py,
                     float* pz,
               unsigned N,
               float alpha,
               unsigned repetitions);
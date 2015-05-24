#pragma once

#include <util/cuda_vector.hpp>

void run_reduce(const cuda::vector<float>& in,
                      cuda::vector<float>& partial,
                      cuda::vector<float>& out,
                const float* pin,
                      float* ppartial,
                      float* pout,
                unsigned N,
                unsigned repetitions);
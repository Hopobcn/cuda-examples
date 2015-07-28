#pragma once

#include <util/cuda_device.hpp>


void launch_vecadd(cuda::device& gpu, unsigned N, unsigned repetitions);